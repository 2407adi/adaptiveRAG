"""Block 5.3: Ablation benchmark — prove every layer pays rent.

Runs the finished system against public QA benchmarks (HotpotQA multi-hop,
SQuAD single-hop) across a ladder of config presets, each enabling exactly
ONE more capability layer. Same questions, same corpus, one lever flipped
at a time — a textbook ablation study.

Reuses the Block 2.4 metric functions unchanged, and adds the benchmarks'
own official string metrics (Exact Match + token F1 — free, no LLM calls).

Datasets are NOT bundled (gitignored). Download once into data/benchmarks/:
    HotpotQA dev (distractor):
        http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    SQuAD v1.1 dev:
        https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

Entry point (from the project root):
    python -m adaptiverag.eval.benchmark --dataset hotpotqa --sample 20
    python -m adaptiverag.eval.benchmark --dataset squad --sample 20 --rungs 0_dense 3_reranker
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import string
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ..reason.router import QueryRoute

logger = logging.getLogger(__name__)

# Metric columns, in scorecard order. exact_match/f1 are the benchmarks'
# official metrics; the rest are Block 2.4's. router_accuracy only applies
# to the "routed" dispatch rung (None elsewhere).
METRIC_NAMES = (
    "exact_match",
    "f1",
    "answer_relevancy",
    "faithfulness",
    "context_recall",
    "context_precision",
    "router_accuracy",
)

# Concise mode (default ON for benchmarks): appended to the question sent
# into the pipeline — NOT to the question used for scoring. Gold answers are
# short spans ("Denver Broncos"); without this, EM/F1 punish the system for
# writing good paragraphs, i.e. they measure verbosity instead of correctness.
# Applied identically on every rung, so it's a constant, never a confound.
# Known tradeoff: retrieval embeds the suffixed question too (RAGChain uses
# one string for both retrieve and generate) — the shift is small, generic,
# and identical across rungs, so deltas stay valid.
CONCISE_INSTRUCTION = (
    "\n\nIMPORTANT: Reply with ONLY the short answer span itself — a few words "
    "at most (a name, date, number, or phrase). No explanation, no full "
    "sentences, no citations."
)

DATA_FILES = {
    "hotpotqa": "hotpot_dev_distractor_v1.json",
    "squad": "dev-v1.1.json",
}
DOWNLOAD_URLS = {
    "hotpotqa": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "squad": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


# ────────────────────────────────────────────────────────────────────
# Dataset loading
# ────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkSample:
    """One benchmark question, normalized to the shape the runner needs.

    Same fields the Block 2.4 EvalSample carries, minus the hand-written
    extras (diagnostic_intent, skip_metrics) that only make sense for a
    curated dataset.
    """
    id: str
    question: str
    ground_truth: str                                 # the gold short answer
    relevant_sources: list[str] = field(default_factory=list)  # corpus filenames holding the answer
    expected_route: str = "RAG"                       # convention: multi-hop = MULTI_STEP, single-hop = RAG


def _safe_filename(title: str) -> str:
    """Wikipedia title → filesystem-safe corpus filename (.txt).

    Must be deterministic AND collision-averse: relevant_sources labels are
    matched against Path(source).name by context_precision, so the exact
    same function must name the corpus files and the labels.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return f"{cleaned[:120] or 'untitled'}.txt"


def load_hotpotqa(
    path: str | Path,
    sample_size: int,
    seed: int,
) -> tuple[list[BenchmarkSample], dict[str, str]]:
    """Load a seeded subset of HotpotQA dev (distractor setting).

    Each raw item carries: _id, question, answer, supporting_facts
    ([title, sent_idx] pairs — the 2 gold paragraphs), and context
    (10 [title, [sentences]] paragraphs: 2 gold + 8 distractors).

    Returns:
        (samples, corpus) where corpus maps filename → paragraph text.
        The corpus is the union of ALL 10 paragraphs of every sampled
        question (deduped by title) — gold needles plus distractor hay.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rng = random.Random(seed)               # seeded → same subset every run
    picked = rng.sample(raw, min(sample_size, len(raw)))

    samples: list[BenchmarkSample] = []
    corpus: dict[str, str] = {}
    for item in picked:
        for title, sentences in item["context"]:
            corpus.setdefault(_safe_filename(title), "".join(sentences))
        gold_titles = {t for t, _ in item["supporting_facts"]}
        samples.append(BenchmarkSample(
            id=item["_id"],
            question=item["question"],
            ground_truth=item["answer"],
            relevant_sources=sorted(_safe_filename(t) for t in gold_titles),
            expected_route="MULTI_STEP",    # multi-hop by construction
        ))
    return samples, corpus


def load_squad(
    path: str | Path,
    sample_size: int,
    seed: int,
) -> tuple[list[BenchmarkSample], dict[str, str]]:
    """Load a seeded subset of SQuAD v1.1 dev (single-hop control group).

    SQuAD nests article → paragraphs → questions. We flatten to one row
    per question, sample with the seed, and build the corpus from each
    sampled question's gold paragraph — every OTHER sampled paragraph
    then acts as a natural distractor in the haystack.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    flat = []   # (qa_id, question, gold_answer, doc_name, paragraph_text)
    for article in raw["data"]:
        for p_idx, para in enumerate(article["paragraphs"]):
            doc_name = _safe_filename(f"{article['title']}_p{p_idx}")
            for qa in para["qas"]:
                if qa.get("answers"):
                    flat.append((qa["id"], qa["question"],
                                 qa["answers"][0]["text"],
                                 doc_name, para["context"]))

    rng = random.Random(seed)
    picked = rng.sample(flat, min(sample_size, len(flat)))

    samples: list[BenchmarkSample] = []
    corpus: dict[str, str] = {}
    for qa_id, question, answer, doc_name, context in picked:
        corpus.setdefault(doc_name, context)
        samples.append(BenchmarkSample(
            id=qa_id,
            question=question,
            ground_truth=answer,
            relevant_sources=[doc_name],
            expected_route="RAG",           # single-hop by construction
        ))
    return samples, corpus


def write_corpus(corpus: dict[str, str], corpus_dir: str | Path) -> int:
    """Materialize the corpus as one .txt file per paragraph.

    Idempotent: skips writing when the directory already holds exactly
    these files (same sample ids + seed → same corpus → same dir).
    """
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in corpus.items():
        target = corpus_dir / filename
        if not target.exists():
            target.write_text(text, encoding="utf-8")
    return len(corpus)


# ────────────────────────────────────────────────────────────────────
# Official benchmark metrics: Exact Match + token F1 (SQuAD-style)
# ────────────────────────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    """Standard SQuAD normalization: lowercase, strip punctuation,
    articles (a/an/the), and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 iff the normalized prediction equals the normalized gold answer.

    Honest caveat for the report: EM was designed for short-span answers;
    our chains write paragraphs, so EM stays near zero on ALL rungs. It's
    included because it's the official benchmark number — F1 is the
    fairer comparison for long-form answers.
    """
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def answer_f1(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 between prediction and gold answer (SQuAD official).

    F1 = harmonic mean of precision (fraction of predicted tokens that
    appear in the gold answer) and recall (fraction of gold tokens that
    appear in the prediction).
    """
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)   # multiset intersection
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ────────────────────────────────────────────────────────────────────
# Presets — the ladder
# ────────────────────────────────────────────────────────────────────

@dataclass
class RungPreset:
    """One rung of the ablation ladder — the three levers plus metadata."""
    name: str
    adds: str                       # human-readable: what this rung enables
    dispatch: str                   # forced_rag | routed | agent | supervisor
    expand: bool                    # query-expansion call-time flag
    overrides: dict = field(default_factory=dict)   # dotted-path Settings overrides


def load_presets(path: str | Path) -> tuple[dict, list[RungPreset]]:
    """Parse config/ablation.yaml → (constants, ordered rung presets)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rungs = [
        RungPreset(
            name=r["name"],
            adds=r.get("adds", ""),
            dispatch=r["dispatch"],
            expand=bool(r["expand"]),
            overrides=r.get("overrides", {}) or {},
        )
        for r in cfg["rungs"]
    ]
    return cfg.get("constants", {}) or {}, rungs


def apply_overrides(settings, overrides: dict):
    """Return a DEEP COPY of Settings with dotted-path overrides applied.

    "retrieval.rerank.enabled": True walks settings.retrieval.rerank and
    sets .enabled. Deep copy so the caller's base Settings is never
    mutated — every rung starts from the same pristine base.

    Raises AttributeError on unknown paths (typo in the YAML should fail
    loudly, not silently benchmark the wrong config).
    """
    patched = deepcopy(settings)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = patched
        for part in parents:
            node = getattr(node, part)
        if not hasattr(node, leaf):
            raise AttributeError(f"Unknown settings path: {dotted!r}")
        setattr(node, leaf, value)
    return patched


# ────────────────────────────────────────────────────────────────────
# Recording retriever — captures what agent rungs actually retrieved
# ────────────────────────────────────────────────────────────────────

class RecordingRetriever:
    """Wraps a RAGChain, recording every chunk retrieved through it.

    Agent/supervisor answers arrive without a sources list (the tool
    returns formatted text to the LLM, not structured results). This
    wrapper sits where the real chain would — the search_documents tool
    duck-types `.retrieve` — so context-based metrics (faithfulness,
    recall, precision) still work on rungs 5–6.

    Also pins the rung's `expand` flag: the tool calls retrieve(query)
    one-arg, so the flag must be baked in here.
    """

    def __init__(self, rag_chain, expand: bool = False):
        self._chain = rag_chain
        self._expand = expand
        self._collected: dict[str, object] = {}    # chunk_id → SearchResult (dedupe)

    def retrieve(self, query: str, scopes=None, **kwargs):
        results = self._chain.retrieve(query, expand=self._expand, scopes=scopes)
        for r in results:
            self._collected.setdefault(r.chunk_id, r)
        return results

    def reset(self) -> None:
        """Clear between samples so contexts never leak across questions."""
        self._collected = {}

    @property
    def collected(self) -> list:
        return list(self._collected.values())


def package_sources(results: list) -> list[dict]:
    """SearchResults → the citation-dict shape the Block 2.4 metrics expect
    (mirrors RAGChain._package_sources)."""
    return [
        {
            "chunk_id": r.chunk_id,
            "source": Path(r.metadata.get("source", "unknown")).name,
            "score": round(r.score, 4),
            "text_preview": r.text[:200],
            "full_text": r.text,
        }
        for r in results
    ]


# ────────────────────────────────────────────────────────────────────
# Per-rung stack + per-sample run/score (pure-ish, fake-friendly)
# ────────────────────────────────────────────────────────────────────

@dataclass
class RungStack:
    """Everything one rung needs at question time. Built by
    build_rung_stack() for live runs; tests inject fakes directly."""
    preset: RungPreset
    rag_chain: object
    router: object
    multi_step_chain: object
    llm_client: object
    embedder: object
    validator: object
    agent: object = None            # only on dispatch="agent"
    supervisor: object = None       # only on dispatch="supervisor"
    recorder: RecordingRetriever | None = None
    vector_store: object = None
    ingest: object = None


def build_rung_stack(base_settings, constants: dict, preset: RungPreset,
                     collection_name: str, persist_dir: str | Path) -> RungStack:
    """Wire one rung's pipeline: constants + rung overrides → wire_pipeline.

    Heavy imports live here (not module top) so tests and report-only
    usage never pull chromadb/langgraph/sentence-transformers.
    """
    from ..pipeline import wire_pipeline
    from ..agents.tools import build_default_registry
    from ..agents.executor import AgentExecutor
    from ..agents.supervisor import SupervisorAgent

    rung_settings = apply_overrides(base_settings, {**constants, **preset.overrides})
    bundle = wire_pipeline(rung_settings, collection_name, persist_dir)

    agent = supervisor = recorder = None
    if preset.dispatch in ("agent", "supervisor"):
        # Build our OWN registry around the recorder (wire_pipeline binds
        # the raw chain, which would leave rungs 5–6 without contexts).
        recorder = RecordingRetriever(bundle.rag_chain, expand=preset.expand)
        registry = build_default_registry(
            recorder, rung_settings.tools,
            hmac_key=os.getenv("AUDIT_HMAC_KEY") or "benchmark-local-key",
            tavily_api_key=None,            # web search stays off (see ablation.yaml)
        )
        if preset.dispatch == "agent":
            agent = AgentExecutor(
                bundle.llm_client, registry,
                max_iterations=rung_settings.agent.max_iterations,
                require_approval=[],        # unattended run — no warrants
                memory_manager=None,        # no cross-sample leakage
            )
        else:
            supervisor = SupervisorAgent(
                bundle.llm_client, registry,
                max_handoffs=rung_settings.agent.max_handoffs,
                worker_iterations=rung_settings.agent.worker_iterations,
                require_approval=[],
                memory_manager=None,
            )

    return RungStack(
        preset=preset,
        rag_chain=bundle.rag_chain,
        router=bundle.router,
        multi_step_chain=bundle.multi_step_chain,
        llm_client=bundle.llm_client,
        embedder=bundle.embedder,
        validator=bundle.grounding_validator,
        agent=agent,
        supervisor=supervisor,
        recorder=recorder,
        vector_store=bundle.vector_store,
        ingest=bundle.ingest,
    )


def run_sample(stack: RungStack, sample: BenchmarkSample,
               concise: bool = False) -> dict:
    """Push one question through this rung's dispatch path.

    concise=True appends CONCISE_INSTRUCTION to the question the PIPELINE
    sees. Routing still classifies the bare question (the instruction would
    skew classification), and scoring always uses the bare question too.

    Returns {"actual_route", "answer", "contexts", "error"} — errors are
    captured, never raised (Block 2.4 convention: one bad sample must not
    abort a rung, let alone the sweep).
    """
    preset = stack.preset
    query = sample.question + (CONCISE_INSTRUCTION if concise else "")
    out: dict = {"actual_route": "", "answer": "", "contexts": [], "error": None}
    try:
        if preset.dispatch == "forced_rag":
            # Rungs 0–3: router bypassed on purpose — these rungs measure
            # retrieval quality, so every question takes the same path.
            resp = stack.rag_chain.query(query, expand=preset.expand)
            out["answer"], out["contexts"] = resp["answer"], resp["sources"]
            out["actual_route"] = "RAG"

        elif preset.dispatch == "routed":
            # Rung 4: same branching the API/UI uses (EvalSuite logic).
            # Classify the BARE question — the concise instruction is a
            # formatting concern, not part of what's being asked.
            route_result = stack.router.classify(sample.question)
            out["actual_route"] = route_result.route.name
            if route_result.route == QueryRoute.DIRECT:
                out["answer"] = stack.llm_client.generate(query)
            elif route_result.route == QueryRoute.MULTI_STEP:
                resp = stack.multi_step_chain.query(query, expand=preset.expand)
                out["answer"], out["contexts"] = resp["answer"], resp["sources"]
            else:
                resp = stack.rag_chain.query(query, expand=preset.expand)
                out["answer"], out["contexts"] = resp["answer"], resp["sources"]

        elif preset.dispatch in ("agent", "supervisor"):
            # Rungs 5–6: contexts come from the recorder, answer from run().
            assert stack.recorder is not None
            stack.recorder.reset()
            runner = stack.agent if preset.dispatch == "agent" else stack.supervisor
            result = runner.run(
                query,
                approver=lambda request: True,          # auto-sign every warrant
                conversation_id=f"bench-{sample.id}",   # unique per sample
            )
            out["answer"] = result.get("answer") or ""
            out["contexts"] = package_sources(stack.recorder.collected)
            out["actual_route"] = preset.dispatch.upper()

        else:
            raise ValueError(f"Unknown dispatch mode: {preset.dispatch!r}")

    except Exception as e:  # noqa: BLE001 — deliberate catch-all, per-sample
        out["error"] = f"{type(e).__name__}: {e}"
        logger.warning("[%s] pipeline error: %s", sample.id, out["error"])
    return out


def score_sample(out: dict, sample: BenchmarkSample, stack: RungStack) -> dict:
    """Apply all metrics to one run_sample() output. Errored samples score
    all-None (excluded from means, counted in error_count)."""
    if out["error"] is not None:
        return {name: None for name in METRIC_NAMES}

    answer, contexts = out["answer"], out["contexts"]
    scores: dict[str, float | None] = {}
    scores["exact_match"] = exact_match(answer, sample.ground_truth)
    scores["f1"] = answer_f1(answer, sample.ground_truth)
    scores["answer_relevancy"] = answer_relevancy(
        sample.question, answer, stack.llm_client, stack.embedder)
    scores["faithfulness"] = faithfulness(answer, contexts, stack.validator)
    scores["context_recall"] = context_recall(
        sample.ground_truth, contexts, stack.validator)
    scores["context_precision"] = context_precision(
        contexts, sample.relevant_sources)
    # Only meaningful when a router actually made a decision.
    scores["router_accuracy"] = (
        float(out["actual_route"] == sample.expected_route)
        if stack.preset.dispatch == "routed" else None
    )
    return scores


def aggregate(sample_rows: list[dict]) -> dict:
    """Per-metric {n, mean} over non-None scores (Block 2.4 convention:
    None = not applicable, excluded — so empty ≠ zero)."""
    agg = {}
    for name in METRIC_NAMES:
        valid = [row["scores"][name] for row in sample_rows
                 if row["scores"].get(name) is not None]
        agg[name] = {
            "n": len(valid),
            "mean": (sum(valid) / len(valid)) if valid else None,
        }
    return agg


# ────────────────────────────────────────────────────────────────────
# The runner — sweeps the ladder
# ────────────────────────────────────────────────────────────────────

class AblationRunner:
    """Sweeps every rung over the same seeded sample set and corpus.

    Results land incrementally under results_dir/ablation/<run_id>/ —
    manifest.json first, then one rung_<name>.json per completed rung —
    so a crash at rung 5 never loses rungs 0–4.
    """

    def __init__(self, *, base_settings, constants: dict,
                 presets: list[RungPreset],
                 samples: list[BenchmarkSample],
                 corpus_dir: str | Path,
                 collection_name: str, persist_dir: str | Path,
                 results_dir: str | Path, dataset: str, seed: int,
                 concise: bool = True):
        self.base_settings = base_settings
        self.constants = constants
        self.presets = presets
        self.samples = samples
        self.corpus_dir = Path(corpus_dir)
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.results_dir = Path(results_dir)
        self.dataset = dataset
        self.seed = seed
        self.concise = concise

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self.run_dir = self.results_dir / "ablation" / f"{stamp}_{dataset}"

    def _ensure_ingested(self, stack: RungStack) -> None:
        """Ingest the corpus into the shared collection once (first stack
        whose store is empty). Chunking/embeddings never change across
        rungs, so all 7 rungs share the same chunks; only the retrieval
        wiring on top differs."""
        if stack.vector_store.count() == 0:
            logger.info("Benchmark store empty — ingesting corpus from %s", self.corpus_dir)
            result = stack.ingest.ingest(str(self.corpus_dir))
            logger.info("Ingested %d files, %d chunks",
                        result["files_processed"], result["total_chunks"])

    def _write_manifest(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset,
            "seed": self.seed,
            "sample_size": len(self.samples),
            "sample_ids": [s.id for s in self.samples],   # ← reproducibility receipt
            "corpus_dir": str(self.corpus_dir),
            "collection_name": self.collection_name,
            "constants": self.constants,
            "concise": self.concise,
            "rungs": [p.name for p in self.presets],
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    def run(self, rung_names: list[str] | None = None) -> Path:
        """Sweep the ladder (or a named subset). Returns the run directory."""
        presets = self.presets
        if rung_names:
            wanted = set(rung_names)
            presets = [p for p in self.presets if p.name in wanted]
            missing = wanted - {p.name for p in presets}
            if missing:
                logger.warning("Requested rungs not in ablation.yaml: %s", sorted(missing))

        self._write_manifest()

        for preset in presets:
            logger.info("════ Rung %s — %s (dispatch=%s, expand=%s) ════",
                        preset.name, preset.adds, preset.dispatch, preset.expand)
            started = time.time()
            stack = build_rung_stack(
                self.base_settings, self.constants, preset,
                self.collection_name, self.persist_dir)
            self._ensure_ingested(stack)

            rows = []
            for i, sample in enumerate(self.samples, 1):
                logger.info("[%s %d/%d] %s", preset.name, i, len(self.samples), sample.id)
                out = run_sample(stack, sample, concise=self.concise)
                scores = score_sample(out, sample, stack)
                rows.append({
                    "id": sample.id,
                    "question": sample.question,
                    "ground_truth": sample.ground_truth,
                    "expected_route": sample.expected_route,
                    "actual_route": out["actual_route"],
                    "answer": out["answer"],
                    "context_sources": [c.get("source", "") for c in out["contexts"]],
                    "scores": scores,
                    "error": out["error"],
                })

            payload = {
                "rung": preset.name,
                "adds": preset.adds,
                "dispatch": preset.dispatch,
                "expand": preset.expand,
                "overrides": preset.overrides,
                "elapsed_seconds": round(time.time() - started, 1),
                "error_count": sum(1 for r in rows if r["error"] is not None),
                "aggregate": aggregate(rows),
                "samples": rows,
            }
            out_path = self.run_dir / f"rung_{preset.name}.json"
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            logger.info("Rung %s saved → %s (%.1fs)",
                        preset.name, out_path, payload["elapsed_seconds"])

        return self.run_dir


# ────────────────────────────────────────────────────────────────────
# CLI — python -m adaptiverag.eval.benchmark
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Block 5.3 ablation benchmark — one config rung per capability layer.")
    parser.add_argument("--dataset", choices=("hotpotqa", "squad"), required=True)
    parser.add_argument("--sample", type=int, default=20,
                        help="number of questions (default 20 — mind the token bill)")
    parser.add_argument("--seed", type=int, default=42,
                        help="sampling seed (default 42; same seed = same questions)")
    parser.add_argument("--rungs", nargs="*", default=None,
                        help="subset of rung names to run (default: the whole ladder)")
    parser.add_argument("--data-file", default=None,
                        help="path to the raw dataset JSON (default: data/benchmarks/<official name>)")
    parser.add_argument("--no-report", action="store_true",
                        help="skip report generation after the sweep")
    parser.add_argument("--no-concise", action="store_true",
                        help="don't ask for short-span answers (f1 will then "
                             "measure verbosity, not correctness)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from ..config import settings, PROJECT_ROOT   # heavy-ish; CLI-only

    data_file = Path(args.data_file) if args.data_file else (
        PROJECT_ROOT / "data" / "benchmarks" / DATA_FILES[args.dataset])
    if not data_file.exists():
        print(f"Dataset file not found: {data_file}")
        print("Download it once (it is gitignored) and re-run:")
        print(f"  mkdir -p {data_file.parent}")
        print(f"  curl -L -o '{data_file}' '{DOWNLOAD_URLS[args.dataset]}'")
        return 1

    loader = load_hotpotqa if args.dataset == "hotpotqa" else load_squad
    samples, corpus = loader(data_file, args.sample, args.seed)

    # Corpus + collection are keyed by (dataset, seed, n): same key → same
    # questions → same corpus → the ingested collection is reused as-is.
    key = f"{args.dataset}_s{args.seed}_n{len(samples)}"
    corpus_dir = PROJECT_ROOT / "data" / "benchmarks" / f"corpus_{key}"
    n_docs = write_corpus(corpus, corpus_dir)
    print(f"{args.dataset}: {len(samples)} questions, {n_docs} corpus docs → {corpus_dir}")

    constants, presets = load_presets(PROJECT_ROOT / "config" / "ablation.yaml")

    runner = AblationRunner(
        base_settings=settings,
        constants=constants,
        presets=presets,
        samples=samples,
        corpus_dir=corpus_dir,
        collection_name=f"bench_{key}",
        persist_dir=PROJECT_ROOT / "data" / "chroma_bench",
        results_dir=PROJECT_ROOT / "eval_results",
        dataset=args.dataset,
        seed=args.seed,
        concise=not args.no_concise,
    )
    run_dir = runner.run(rung_names=args.rungs)
    print(f"Run complete → {run_dir}")

    if not args.no_report:
        from .report import generate_report
        report_path = generate_report(run_dir, copy_latest_to=PROJECT_ROOT / "eval_results")
        print(f"Report → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
