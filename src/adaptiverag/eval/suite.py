"""EvalSuite — orchestrates the eval framework.

Wires together the dataset, the existing pipeline (router → DIRECT/RAG/
MULTI_STEP), the metric functions, and result persistence.

Heavy DI: every component is injected by the caller, matching the
project's existing pattern. The suite never constructs its own pipeline
— the caller's __main__ entry or scratch script wires it up, ideally
with the same instances the live UI uses (so metrics measure what the
user actually experiences).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from .dataset import EvalSample, load_dataset
from .metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ..reason.router import QueryRouter, QueryRoute
from ..reason.chain import RAGChain, MultiStepChain
from ..reason.grounding import GroundingValidator
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class SampleRun:
    """Per-sample runtime state — pipeline output plus metric scores.

    Working object that flows through run → score → aggregate → save.
    Projected to a leaner shape before being written to disk (full
    context texts are too verbose to persist; we keep counts and
    previews).

    Attributes:
        sample:        The labeled input from the dataset.
        actual_route:  What the router actually classified ("DIRECT" |
                       "RAG" | "MULTI_STEP"). May differ from
                       sample.expected_route — that's the router-accuracy
                       metric.
        answer:        Pipeline-produced answer text.
        contexts:      Sources returned by RAGChain/MultiStepChain
                       (empty for DIRECT). Each dict has 'source',
                       'full_text', 'text_preview', etc.
        scores:        Metric name → score in [0, 1], or None if the
                       metric was skipped or returned None for this
                       sample.
        error:         If the pipeline crashed on this sample, the
                       message is captured here so other samples still
                       run. Successful samples have error=None.
    """
    sample: EvalSample
    actual_route: str = ""
    answer: str = ""
    contexts: list[dict] = field(default_factory=list)
    scores: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None

def _fmt(value: float | None) -> str:
    """Format a metric value to 2 decimals, or '  – ' for None."""
    return "  – " if value is None else f"{value:.2f}"


def _find_worst(
    results: list[SampleRun],
    metric: str,
) -> tuple[SampleRun, float] | None:
    """Find the lowest-scoring sample for a given metric.

    Returns (sample_run, score), or None if no samples have a non-None
    score for this metric (e.g. context_precision when every question
    was DIRECT).
    """
    candidates: list[tuple[SampleRun, float]] = []
    for r in results:
        s = r.scores.get(metric)
        if s is not None:
            candidates.append((r, s))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])

class EvalSuite:
    """End-to-end eval runner.

    Typical usage (will become possible after the next legos add
    methods):

        suite = EvalSuite(
            dataset_path="data/eval/qa_pairs.json",
            router=router,
            rag_chain=rag_chain,
            multi_step_chain=multi_step_chain,
            llm_client=llm_client,
            embedder=embedder,
            validator=validator,
        )
        results = suite.run()
        suite.save(results)
        suite.print_scorecard(results)

    Design contract:
        - Stateless across runs: calling run() twice produces
          independent results.
        - Per-sample errors are captured (stored in SampleRun.error),
          not raised — one bad sample should not abort the whole eval.
        - The corpus referenced by the dataset is assumed to already
          be ingested into the vector store. The suite does not ingest.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        router: QueryRouter,
        rag_chain: RAGChain,
        multi_step_chain: MultiStepChain,
        llm_client,
        embedder,
        validator: GroundingValidator,
        results_dir: str | Path = "eval_results",
    ):
        self.dataset_path = Path(dataset_path)
        self.router = router
        self.rag_chain = rag_chain
        self.multi_step_chain = multi_step_chain
        self.llm_client = llm_client
        self.embedder = embedder
        self.validator = validator
        self.results_dir = Path(results_dir)

        # Eager load — fail fast on a malformed dataset, before any
        # LLM calls happen.
        self.samples = load_dataset(self.dataset_path)
        logger.info(
            "Loaded %d eval samples from %s",
            len(self.samples), self.dataset_path,
        )

    def _run_one_sample(self, sample: EvalSample) -> SampleRun:
        """Run one labeled question through the live pipeline.

        Routes via QueryRouter, dispatches to the appropriate chain
        (DIRECT → llm_client, RAG → rag_chain, MULTI_STEP →
        multi_step_chain), and returns a SampleRun with the actual route,
        answer, and contexts populated. The scores dict stays empty —
        metric scoring happens in a separate step so we can re-score an
        existing run without re-burning LLM calls on the pipeline.

        Errors:
            Per-sample exceptions are caught, recorded in SampleRun.error,
            and logged. We do not re-raise — one bad sample (e.g. a
            transient LLM timeout) should not abort the whole eval. Samples
            with errors will fail-soft when scored: metrics that need an
            answer will return None.
        """
        run = SampleRun(sample=sample)

        try:
            # 1. Classify
            route_result = self.router.classify(sample.question)
            run.actual_route = route_result.route.name  # "DIRECT" | "RAG" | "MULTI_STEP"

            # 2. Dispatch — same branching logic the Streamlit UI uses
            if route_result.route == QueryRoute.DIRECT:
                run.answer = self.llm_client.generate(sample.question)
                run.contexts = []

            elif route_result.route == QueryRoute.MULTI_STEP:
                response = self.multi_step_chain.query(sample.question)
                run.answer = response["answer"]
                run.contexts = response["sources"]

            else:  # RAG (also the router's fallback path)
                response = self.rag_chain.query(sample.question)
                run.answer = response["answer"]
                run.contexts = response["sources"]

            logger.info(
                "[%s] route=%s, %d sources, answer=%d chars",
                sample.id, run.actual_route, len(run.contexts), len(run.answer),
            )

        except Exception as e:
            run.error = f"{type(e).__name__}: {e}"
            logger.warning("[%s] pipeline error: %s", sample.id, run.error)

        return run
    
    def _score_one_sample(self, run: SampleRun) -> None:
        """Apply all metrics to a populated SampleRun, mutating its
        `scores` dict in place.

        Skipping rules, in priority order:
        1. If the pipeline crashed (run.error is set), every metric is
            None — there's no answer to score.
        2. If a metric name is in sample.skip_metrics, it's None for
            this sample (e.g. q012's "answer_relevancy" and
            "context_recall" — see qa_pairs.json).
        3. Otherwise the metric runs. The metric itself may still return
            None for its own edge cases (no contexts, empty answer, etc.).
            All three "N/A" reasons land as None; the aggregator excludes
            them from the mean.

        Returns None — mutates `run.scores` in place. Same convention as
        list.sort() / list.append().
        """
        metric_names = (
            "router_accuracy",
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
        )

        # Pipeline failure: nothing to score, but still record the sample
        # with all-None scores so it shows up in the result file.
        if run.error is not None:
            for name in metric_names:
                run.scores[name] = None
            return

        sample = run.sample
        skip = set(sample.skip_metrics)

        # Router accuracy — free, no LLM/embedding cost.
        run.scores["router_accuracy"] = (
            None if "router_accuracy" in skip
            else float(run.actual_route == sample.expected_route)
        )

        # Faithfulness — answer claims supported by contexts.
        run.scores["faithfulness"] = (
            None if "faithfulness" in skip
            else faithfulness(run.answer, run.contexts, self.validator)
        )

        # Answer relevancy — answer addresses the question.
        run.scores["answer_relevancy"] = (
            None if "answer_relevancy" in skip
            else answer_relevancy(
                sample.question, run.answer, self.llm_client, self.embedder
            )
        )

        # Context recall — contexts cover the ground truth.
        run.scores["context_recall"] = (
            None if "context_recall" in skip
            else context_recall(sample.ground_truth, run.contexts, self.validator)
        )

        # Context precision — relevant contexts ranked early.
        run.scores["context_precision"] = (
            None if "context_precision" in skip
            else context_precision(run.contexts, sample.relevant_sources)
        )

        logger.info(
            "[%s] scored: %s",
            sample.id,
            ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}=N/A"
                for k, v in run.scores.items()
            ),
        )
    def run(self, sample_ids: list[str] | None = None) -> list[SampleRun]:
        """Run the eval over all samples (or a filtered subset).

        Each sample goes through two stages:
        1. _run_one_sample  — pipeline (router + chain) → answer, contexts
        2. _score_one_sample — apply metrics → scores dict

        Args:
            sample_ids: If provided, run only samples whose id is in this
                        list. Useful for debugging a single sample (e.g.
                        rerun q012 after a fix) or for a cheap smoke test
                        on the DIRECT samples before launching a full run.
                        None = run everything.

        Returns:
            list[SampleRun] in dataset order (filtering preserves order),
            one per executed sample. Each has its scores populated; samples
            whose pipeline crashed have scores=all-None and a non-None
            .error message.

        Notes:
            - Progress is logged at INFO level. Set the root logger to
            INFO to see it.
            - A bad sample never aborts the run — errors are captured
            per-sample inside _run_one_sample.
        """
        # Filter (or run everything)
        if sample_ids is None:
            samples_to_run = self.samples
        else:
            wanted = set(sample_ids)
            samples_to_run = [s for s in self.samples if s.id in wanted]

            # Typo guard: warn loudly if any requested id wasn't found.
            found = {s.id for s in samples_to_run}
            missing = wanted - found
            if missing:
                logger.warning(
                    "Requested sample_ids not in dataset: %s",
                    sorted(missing),
                )

        n = len(samples_to_run)
        logger.info("Starting eval run: %d samples", n)
        started = time.time()

        results: list[SampleRun] = []
        for i, sample in enumerate(samples_to_run, 1):
            logger.info("---- [%d/%d] %s ----", i, n, sample.id)
            sample_run = self._run_one_sample(sample)
            self._score_one_sample(sample_run)
            results.append(sample_run)

        elapsed = time.time() - started
        logger.info(
            "Eval run complete: %d samples in %.1fs (%.2fs/sample avg)",
            n, elapsed, elapsed / n if n else 0.0,
        )

        return results
    
    def _aggregate(self, results: list[SampleRun]) -> dict:
        """Compute summary statistics from a completed run.

        For each metric, two views:
        - Overall: mean / min / max / n across all non-None scores.
        - Per-route: same (mean + n only) bucketed by actual_route,
            so you can see whether a regression is global or path-specific.

        Errored samples (run.error is not None) contribute to error_count
        but are excluded from all metric means — their scores are all None
        by construction in _score_one_sample.

        Bucketing uses actual_route (where the question actually went),
        not expected_route (where it should have gone). The routing miss
        is captured separately by the router_accuracy metric, so we keep
        "did routing work?" and "did the path that ran work?" as separate
        diagnostic signals.

        Returns:
            JSON-serializable dict with keys:
            - total_samples:           int
            - error_count:             int
            - per_metric:              metric_name → {n, mean, min, max}
            - per_metric_per_route:    metric_name → route → {n, mean}
            Empty buckets have n=0 and mean=None (so empty != zero).
        """
        metric_names = (
            "router_accuracy",
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
        )
        routes = ("DIRECT", "RAG", "MULTI_STEP")

        error_count = sum(1 for r in results if r.error is not None)

        # ── Per-metric overall ──
        per_metric = {}
        for name in metric_names:
            valid = [
                s for s in (r.scores.get(name) for r in results)
                if s is not None
            ]
            per_metric[name] = {
                "n": len(valid),
                "mean": (sum(valid) / len(valid)) if valid else None,
                "min": min(valid) if valid else None,
                "max": max(valid) if valid else None,
            }

        # ── Per-metric per-route ──
        per_metric_per_route = {}
        for name in metric_names:
            per_route = {}
            for route in routes:
                valid = [
                    s for s in (
                        r.scores.get(name) for r in results
                        if r.actual_route == route
                    )
                    if s is not None
                ]
                per_route[route] = {
                    "n": len(valid),
                    "mean": (sum(valid) / len(valid)) if valid else None,
                }
            per_metric_per_route[name] = per_route

        return {
            "total_samples": len(results),
            "error_count": error_count,
            "per_metric": per_metric,
            "per_metric_per_route": per_metric_per_route,
        }

    def save(self, results: list[SampleRun]) -> Path:
        """Persist a completed run to a timestamped JSON file under
        self.results_dir.

        Filename format: {YYYY-MM-DD}T{HH-MM-SS}Z.json (UTC, hyphens
        instead of colons so it's safe on Windows). Z suffix marks UTC,
        so result files from different machines/timezones sort correctly
        on the timeline.

        Persisted shape per sample is leaner than the in-memory SampleRun:
        - full context texts are dropped (lives in the corpus)
        - ground_truth is dropped (lives in qa_pairs.json)
        - the answer IS kept, so post-hoc inspection of bad scores
            doesn't require re-running the pipeline
        - diagnostic_intent is kept, so a result file is interpretable
            weeks later without cross-referencing the dataset

        Returns:
            Path to the written file. Caller can log it.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
        out_path = self.results_dir / f"{stamp}.json"

        payload = {
            "timestamp": now.isoformat(),
            "dataset_path": str(self.dataset_path),
            "summary": self._aggregate(results),
            "samples": [
                {
                    "id": r.sample.id,
                    "question": r.sample.question,
                    "expected_route": r.sample.expected_route,
                    "actual_route": r.actual_route,
                    "answer": r.answer,
                    "context_count": len(r.contexts),
                    "context_sources": [c.get("source", "") for c in r.contexts],
                    "scores": r.scores,
                    "error": r.error,
                    "diagnostic_intent": r.sample.diagnostic_intent,
                }
                for r in results
            ],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("Eval results written to %s", out_path)
        return out_path
    
    def print_scorecard(self, results: list[SampleRun]) -> None:
        """Print a human-readable scorecard to stdout.

        Sections:
            1. Header  — timestamp, total samples, error count, route mix
            2. Per-metric overall — n, mean, min, max
            3. Per-route table  — mean per (metric × route)
            4. Worst sample per metric — id, score, diagnostic_intent

        For machine-readable output use save(); this is terminal-only.
        """
        summary = self._aggregate(results)

        # Route distribution from the actual run (not the dataset)
        route_counts = {"DIRECT": 0, "RAG": 0, "MULTI_STEP": 0}
        for r in results:
            route_counts[r.actual_route] = route_counts.get(r.actual_route, 0) + 1

        bar = "═" * 72
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── Header ──
        print(bar)
        print(f" AdaptiveRAG eval — {ts}")
        print(bar)
        print(f" Samples: {summary['total_samples']}    Errors: {summary['error_count']}")
        print(
            " By route (actual): "
            + "    ".join(f"{rt}={n}" for rt, n in route_counts.items())
        )

        # ── Per-metric overall ──
        print()
        print(f" {'Metric':<19} {'n':>4}   {'mean':>5}   {'min':>5}   {'max':>5}")
        print(" " + "─" * 19 + "  " + "───" + "   " + "─────" + "   " + "─────" + "   " + "─────")
        for name, stats in summary["per_metric"].items():
            print(
                f" {name:<19} {stats['n']:>4}   "
                f"{_fmt(stats['mean']):>5}   "
                f"{_fmt(stats['min']):>5}   "
                f"{_fmt(stats['max']):>5}"
            )

        # ── Per-route table ──
        print()
        print(" Per-route means")
        print(f" {'Metric':<19} {'DIRECT':>7}   {'RAG':>7}   {'MULTI_STEP':>10}")
        print(" " + "─" * 19 + "  " + "─" * 7 + "   " + "─" * 7 + "   " + "─" * 10)
        for name, by_route in summary["per_metric_per_route"].items():
            d = _fmt(by_route["DIRECT"]["mean"])
            r = _fmt(by_route["RAG"]["mean"])
            m = _fmt(by_route["MULTI_STEP"]["mean"])
            print(f" {name:<19} {d:>7}   {r:>7}   {m:>10}")

        # ── Worst sample per metric ──
        print()
        print(" Worst sample per metric")
        for name in summary["per_metric"]:
            worst = _find_worst(results, name)
            if worst is None:
                print(f"   {name:<19} (no scored samples)")
                continue
            run, score = worst
            intent = run.sample.diagnostic_intent
            if len(intent) > 50:
                intent = intent[:50].rstrip() + "…"
            print(f"   {name:<19} {run.sample.id} ({score:.2f}) — {intent}")

        print(bar)

# ────────────────────────────────────────────────────────────────────
# CLI entry point — `python -m adaptiverag.eval.suite [sample_id ...]`
# ────────────────────────────────────────────────────────────────────
# NOTE: pipeline wiring below duplicates init_pipeline() from ui/app.py.
# A shared wire_pipeline() factory belongs to the upcoming UI rewrite
# block (per pre-deploy blockers in CLAUDE.md), not here.

if __name__ == "__main__":
    import sys

    # Lazy imports — only needed when this file is run as a script,
    # not when EvalSuite is imported from elsewhere.
    from adaptiverag.config import settings
    from adaptiverag.ingest.embedder import create_embedder
    from adaptiverag.ingest.chunker import RecursiveChunker
    from adaptiverag.ingest.loader import DocumentLoader
    from adaptiverag.ingest.pipeline import IngestPipeline
    from adaptiverag.retrieve.vector_store import create_vector_store
    from adaptiverag.retrieve.query_expander import QueryExpander
    from adaptiverag.llm_client import AzureLLMClient
    from adaptiverag.ingest.summarizer import CorpusSummarizer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Project root: src/adaptiverag/eval/suite.py → ../../../..
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

    # ── Embedder (shared with the live app: same model, same vectors) ──
    embedder = create_embedder("local", model_name="all-MiniLM-L6-v2")

    # ── Vector store: dedicated eval collection, hermetic ──
    persist_dir = PROJECT_ROOT / "data" / "chroma_eval"
    vector_store = create_vector_store(
        backend="chroma",
        collection_name="eval_collection",
        persist_directory=str(persist_dir),
    )

    # ── LLM client (built early — summarizer + chains both need it) ──
    llm_client = AzureLLMClient(
        endpoint=settings.azure.endpoint,
        api_key=settings.azure.api_key,
        deployment=settings.azure.deployment,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )

    # ── Corpus summarizer (sidecar next to the eval chroma store) ──
    summarizer = CorpusSummarizer(
        llm_client=llm_client,
        persist_path=persist_dir / "_corpus_summary.txt",
    )

    # Auto-ingest the eval corpus on first run (or if the dir is wiped)
    if vector_store.count() == 0:
        logger.info("Eval vector store is empty — ingesting corpus...")
        chunker = RecursiveChunker(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
        loader = DocumentLoader()
        ingest_pipeline = IngestPipeline(
            loader, chunker, embedder, vector_store, summarizer=summarizer,
        )
        corpus_dir = PROJECT_ROOT / "data" / "eval" / "corpus"
        result = ingest_pipeline.ingest(str(corpus_dir))
        logger.info(
            "Ingested %d files, %d chunks",
            result["files_processed"], result["total_chunks"],
        )
        logger.info(
            "Corpus summary: %s",
            "generated" if result.get("corpus_summary") else "skipped/failed",
        )

    query_expander = QueryExpander(llm_client)

    rag_chain = RAGChain(
        vector_store=vector_store,
        embedder=embedder,
        llm_client=llm_client,
        top_k=settings.retrieval.top_k,
        query_expander=query_expander,
    )

    router = QueryRouter(
        llm_client=llm_client,
        examples=settings.routing.examples,
        corpus_summary=summarizer.load(),
    )

    multi_step_chain = MultiStepChain(
        rag_chain=rag_chain,
        llm_client=llm_client,
        max_sub_questions=4,
    )

    validator = GroundingValidator(
        llm_client=llm_client,
        threshold=0.6,
    )

    # ── Suite + run ──
    suite = EvalSuite(
        dataset_path=PROJECT_ROOT / "data" / "eval" / "qa_pairs.json",
        router=router,
        rag_chain=rag_chain,
        multi_step_chain=multi_step_chain,
        llm_client=llm_client,
        embedder=embedder,
        validator=validator,
        results_dir=PROJECT_ROOT / "eval_results",
    )

    # CLI: positional args = sample IDs to filter, none = run everything.
    sample_ids = sys.argv[1:] if len(sys.argv) > 1 else None

    results = suite.run(sample_ids=sample_ids)
    suite.save(results)
    suite.print_scorecard(results)