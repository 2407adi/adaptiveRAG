"""Offline tests for Block 5.3 — ablation benchmark + report.

No network, no LLM, no vector store: loaders run on tiny fixture JSONs,
the runner's per-sample functions run on fakes, and the report runs on a
synthetic results directory. The preset test loads the REAL
config/ablation.yaml and enforces the study's core rule: each rung flips
exactly one lever.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from adaptiverag.eval.benchmark import (
    BenchmarkSample,
    RecordingRetriever,
    RungPreset,
    RungStack,
    _safe_filename,
    aggregate,
    answer_f1,
    apply_overrides,
    exact_match,
    load_hotpotqa,
    load_presets,
    load_squad,
    package_sources,
    run_sample,
    score_sample,
    write_corpus,
)
from adaptiverag.eval.report import build_markdown, generate_report, load_run
from adaptiverag.reason.router import QueryRoute, RouteResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────
# Fixtures — tiny files in the official schemas
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def hotpot_file(tmp_path):
    """Two items in HotpotQA dev-distractor schema (trimmed to 3 paragraphs)."""
    data = [
        {
            "_id": "h001",
            "question": "Which country is the birthplace of the author of Book A?",
            "answer": "France",
            "supporting_facts": [["Author One", 0], ["Book A", 1]],
            "context": [
                ["Author One", ["Author One was born in France. ", "They wrote many books."]],
                ["Book A", ["Book A is a novel. ", "It was written by Author One."]],
                ["Distractor Topic", ["Nothing relevant here at all."]],
            ],
        },
        {
            "_id": "h002",
            "question": "What year did Event X happen?",
            "answer": "1999",
            "supporting_facts": [["Event X", 0]],
            "context": [
                ["Event X", ["Event X happened in 1999."]],
                ["Author One", ["Author One was born in France. ", "They wrote many books."]],
            ],
        },
    ]
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def squad_file(tmp_path):
    """One article, two paragraphs, three questions in SQuAD v1.1 schema."""
    data = {
        "data": [
            {
                "title": "Test Article",
                "paragraphs": [
                    {
                        "context": "The tower is 300 metres tall. It opened in 1889.",
                        "qas": [
                            {"id": "s001", "question": "How tall is the tower?",
                             "answers": [{"text": "300 metres", "answer_start": 13}]},
                            {"id": "s002", "question": "When did the tower open?",
                             "answers": [{"text": "1889", "answer_start": 43}]},
                        ],
                    },
                    {
                        "context": "The bridge spans the river and carries six lanes.",
                        "qas": [
                            {"id": "s003", "question": "How many lanes does the bridge carry?",
                             "answers": [{"text": "six", "answer_start": 41}]},
                        ],
                    },
                ],
            }
        ]
    }
    path = tmp_path / "squad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _search_result(chunk_id, text="some text", source="doc.txt", score=0.9):
    """SearchResult stand-in (duck-typed: chunk_id/text/score/metadata)."""
    return SimpleNamespace(chunk_id=chunk_id, text=text, score=score,
                           metadata={"source": source})


# ────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────

class TestLoaders:
    def test_safe_filename(self):
        assert _safe_filename("Kiss and Tell (1945 film)") == "Kiss_and_Tell_1945_film.txt"
        assert _safe_filename("...") == "untitled.txt"
        assert len(_safe_filename("x" * 500)) <= 124  # 120 chars + ".txt"

    def test_hotpot_shapes(self, hotpot_file):
        samples, corpus = load_hotpotqa(hotpot_file, sample_size=2, seed=1)
        assert len(samples) == 2
        s = {x.id: x for x in samples}["h001"]
        assert s.ground_truth == "France"
        assert s.expected_route == "MULTI_STEP"
        assert sorted(s.relevant_sources) == ["Author_One.txt", "Book_A.txt"]
        # gold labels must point at files that exist in the corpus
        assert set(s.relevant_sources) <= set(corpus)

    def test_hotpot_corpus_dedupes_shared_paragraphs(self, hotpot_file):
        # "Author One" appears in both items' contexts → one corpus doc
        _, corpus = load_hotpotqa(hotpot_file, sample_size=2, seed=1)
        assert len(corpus) == 4  # Author One, Book A, Distractor Topic, Event X

    def test_hotpot_seed_reproducible(self, hotpot_file):
        a, _ = load_hotpotqa(hotpot_file, sample_size=1, seed=42)
        b, _ = load_hotpotqa(hotpot_file, sample_size=1, seed=42)
        assert [s.id for s in a] == [s.id for s in b]

    def test_squad_shapes(self, squad_file):
        samples, corpus = load_squad(squad_file, sample_size=3, seed=1)
        assert len(samples) == 3
        s = {x.id: x for x in samples}["s003"]
        assert s.ground_truth == "six"
        assert s.expected_route == "RAG"
        assert s.relevant_sources == ["Test_Article_p1.txt"]
        assert set(s.relevant_sources) <= set(corpus)
        assert len(corpus) == 2  # two distinct paragraphs

    def test_sample_size_capped_at_dataset(self, squad_file):
        samples, _ = load_squad(squad_file, sample_size=100, seed=1)
        assert len(samples) == 3

    def test_write_corpus_idempotent(self, tmp_path):
        corpus = {"a.txt": "alpha", "b.txt": "beta"}
        d = tmp_path / "corpus"
        assert write_corpus(corpus, d) == 2
        assert write_corpus(corpus, d) == 2          # second call: no-op
        assert (d / "a.txt").read_text() == "alpha"


# ────────────────────────────────────────────────────────────────────
# EM / F1
# ────────────────────────────────────────────────────────────────────

class TestStringMetrics:
    def test_exact_match_normalization(self):
        assert exact_match("The Eiffel Tower!", "eiffel tower") == 1.0
        assert exact_match("Eiffel Tower", "Louvre") == 0.0

    def test_f1_perfect_and_zero(self):
        assert answer_f1("300 metres", "300 metres") == 1.0
        assert answer_f1("completely unrelated", "300 metres") == 0.0

    def test_f1_partial_overlap(self):
        # pred 6 tokens, gold 2, overlap 2 → p=1/3, r=1 → f1=0.5
        score = answer_f1("the tower is 300 metres tall indeed", "300 metres")
        assert score == pytest.approx(0.5)

    def test_f1_long_answer_containing_gold(self):
        # paragraph answers still earn recall credit — why F1 > EM for us
        long_answer = "According to the documents, the tower stands 300 metres tall [1]."
        assert answer_f1(long_answer, "300 metres") > 0.3
        assert exact_match(long_answer, "300 metres") == 0.0

    def test_empty_prediction(self):
        assert answer_f1("", "300 metres") == 0.0
        assert exact_match("", "300 metres") == 0.0


# ────────────────────────────────────────────────────────────────────
# Presets — the ladder's core rule
# ────────────────────────────────────────────────────────────────────

def _levers(preset: RungPreset) -> dict:
    """Flatten a preset into comparable lever→value pairs."""
    flat = {"dispatch": preset.dispatch, "expand": preset.expand}
    for k, v in preset.overrides.items():
        flat[f"overrides.{k}"] = v
    return flat


class TestPresets:
    def test_ladder_loads_seven_rungs_in_order(self):
        constants, rungs = load_presets(PROJECT_ROOT / "config" / "ablation.yaml")
        assert [r.name for r in rungs] == [
            "0_dense", "1_expansion", "2_hybrid", "3_reranker",
            "4_routed", "5_agent", "6_supervisor",
        ]
        # the fixed lab environment is present and NOT a lever
        assert constants["tools.tavily.enabled"] is False
        assert constants["memory.enabled"] is False

    def test_each_rung_flips_exactly_one_lever(self):
        _, rungs = load_presets(PROJECT_ROOT / "config" / "ablation.yaml")
        for prev, cur in zip(rungs, rungs[1:]):
            a, b = _levers(prev), _levers(cur)
            changed = [k for k in (a.keys() | b.keys()) if a.get(k) != b.get(k)]
            assert len(changed) == 1, (
                f"{prev.name} → {cur.name} flips {changed} — an ablation rung "
                f"must flip exactly ONE lever")

    def test_apply_overrides_deep_copies(self):
        base = SimpleNamespace(
            retrieval=SimpleNamespace(
                mode="hybrid",
                rerank=SimpleNamespace(enabled=True),
            ),
        )
        patched = apply_overrides(base, {"retrieval.mode": "dense",
                                         "retrieval.rerank.enabled": False})
        assert patched.retrieval.mode == "dense"
        assert patched.retrieval.rerank.enabled is False
        # the base is untouched — every rung starts pristine
        assert base.retrieval.mode == "hybrid"
        assert base.retrieval.rerank.enabled is True

    def test_apply_overrides_rejects_typos(self):
        base = SimpleNamespace(retrieval=SimpleNamespace(mode="hybrid"))
        with pytest.raises(AttributeError):
            apply_overrides(base, {"retrieval.moed": "dense"})

    def test_ladder_paths_exist_on_real_settings(self):
        """Every dotted path in ablation.yaml must resolve on the real
        Settings schema — catches YAML/config drift."""
        from adaptiverag.config import Settings
        constants, rungs = load_presets(PROJECT_ROOT / "config" / "ablation.yaml")
        for preset in rungs:
            apply_overrides(Settings(), {**constants, **preset.overrides})


# ────────────────────────────────────────────────────────────────────
# RecordingRetriever
# ────────────────────────────────────────────────────────────────────

class TestRecordingRetriever:
    def _chain(self, results):
        chain = SimpleNamespace()
        chain.calls = []

        def retrieve(query, expand=False, scopes=None):
            chain.calls.append({"query": query, "expand": expand, "scopes": scopes})
            return results
        chain.retrieve = retrieve
        return chain

    def test_records_and_dedupes(self):
        chain = self._chain([_search_result("c1"), _search_result("c2")])
        rec = RecordingRetriever(chain)
        rec.retrieve("q1")
        rec.retrieve("q2")                      # same chunks again
        assert sorted(r.chunk_id for r in rec.collected) == ["c1", "c2"]

    def test_reset_clears_between_samples(self):
        chain = self._chain([_search_result("c1")])
        rec = RecordingRetriever(chain)
        rec.retrieve("q")
        rec.reset()
        assert rec.collected == []

    def test_pins_expand_flag(self):
        chain = self._chain([])
        rec = RecordingRetriever(chain, expand=True)
        rec.retrieve("q")                       # tool calls one-arg style
        assert chain.calls[0]["expand"] is True

    def test_package_sources_shape(self):
        pkg = package_sources([_search_result("c1", text="body", source="/tmp/a.txt")])
        assert pkg[0]["source"] == "a.txt"      # filename only, like RAGChain
        assert pkg[0]["full_text"] == "body"


# ────────────────────────────────────────────────────────────────────
# run_sample / score_sample / aggregate — with fakes
# ────────────────────────────────────────────────────────────────────

def _fake_stack(preset, **kwargs):
    rag = SimpleNamespace(query=lambda q, expand=False: {
        "answer": "France [1]", "sources": package_sources([_search_result("c1", source="Author_One.txt")])})
    multi = SimpleNamespace(query=lambda q, expand=False: {
        "answer": "multi answer", "sources": []})
    llm = SimpleNamespace(generate=lambda p: '["What country?"]')
    embedder = SimpleNamespace(embed=lambda t: [1.0, 0.0])
    validator = SimpleNamespace(validate=lambda a, c: SimpleNamespace(score=0.8))
    defaults = dict(preset=preset, rag_chain=rag, router=None,
                    multi_step_chain=multi, llm_client=llm,
                    embedder=embedder, validator=validator)
    defaults.update(kwargs)
    return RungStack(**defaults)


def _sample():
    return BenchmarkSample(id="q1", question="Which country?",
                           ground_truth="France",
                           relevant_sources=["Author_One.txt"],
                           expected_route="MULTI_STEP")


class TestRunSample:
    def test_forced_rag_bypasses_router(self):
        preset = RungPreset("0_dense", "", "forced_rag", expand=False)
        out = run_sample(_fake_stack(preset), _sample())
        assert out["error"] is None
        assert out["actual_route"] == "RAG"
        assert out["answer"] == "France [1]"
        assert len(out["contexts"]) == 1

    def test_routed_dispatches_multi_step(self):
        preset = RungPreset("4_routed", "", "routed", expand=True)
        router = SimpleNamespace(classify=lambda q: RouteResult(
            route=QueryRoute.MULTI_STEP, confidence="high", reasoning=""))
        out = run_sample(_fake_stack(preset, router=router), _sample())
        assert out["actual_route"] == "MULTI_STEP"
        assert out["answer"] == "multi answer"

    def test_agent_contexts_come_from_recorder(self):
        preset = RungPreset("5_agent", "", "agent", expand=True)
        chain = SimpleNamespace(retrieve=lambda q, expand=False, scopes=None:
                                [_search_result("c9", source="Book_A.txt")])
        recorder = RecordingRetriever(chain, expand=True)
        agent = SimpleNamespace(run=lambda q, approver=None, conversation_id=None:
                                {"status": "done", "answer": "agent answer"})
        stack = _fake_stack(preset, agent=agent, recorder=recorder)
        recorder.retrieve("warm-up")            # stale chunks from a previous sample
        recorder_before = len(recorder.collected)
        assert recorder_before == 1
        out = run_sample(stack, _sample())
        assert out["answer"] == "agent answer"
        # reset() ran first — the warm-up chunk must NOT leak in
        assert out["contexts"] == []
        assert out["error"] is None

    def test_errors_captured_not_raised(self):
        preset = RungPreset("0_dense", "", "forced_rag", expand=False)
        rag = SimpleNamespace(query=lambda q, expand=False: 1 / 0)
        out = run_sample(_fake_stack(preset, rag_chain=rag), _sample())
        assert out["error"] is not None
        assert "ZeroDivisionError" in out["error"]

    def test_unknown_dispatch_is_an_error(self):
        preset = RungPreset("x", "", "teleport", expand=False)
        out = run_sample(_fake_stack(preset), _sample())
        assert out["error"] is not None


class TestConciseMode:
    def _recording_rag(self):
        rag = SimpleNamespace(seen=[])

        def query(q, expand=False):
            rag.seen.append(q)
            return {"answer": "France", "sources": []}
        rag.query = query
        return rag

    def test_suffix_appended_to_pipeline_query(self):
        from adaptiverag.eval.benchmark import CONCISE_INSTRUCTION
        preset = RungPreset("0_dense", "", "forced_rag", expand=False)
        rag = self._recording_rag()
        run_sample(_fake_stack(preset, rag_chain=rag), _sample(), concise=True)
        assert rag.seen[0] == _sample().question + CONCISE_INSTRUCTION

    def test_off_by_default_and_when_disabled(self):
        preset = RungPreset("0_dense", "", "forced_rag", expand=False)
        rag = self._recording_rag()
        stack = _fake_stack(preset, rag_chain=rag)
        run_sample(stack, _sample())                      # default
        run_sample(stack, _sample(), concise=False)       # explicit off
        assert rag.seen == [_sample().question, _sample().question]

    def test_router_classifies_bare_question(self):
        # The concise instruction is a formatting concern — it must never
        # influence which route the classifier picks.
        preset = RungPreset("4_routed", "", "routed", expand=True)
        classified = []

        def classify(q):
            classified.append(q)
            return RouteResult(route=QueryRoute.RAG, confidence="high", reasoning="")
        stack = _fake_stack(preset, router=SimpleNamespace(classify=classify),
                            rag_chain=self._recording_rag())
        run_sample(stack, _sample(), concise=True)
        assert classified == [_sample().question]         # bare — no suffix
        assert stack.rag_chain.seen[0] != _sample().question  # but the chain got the suffix


class TestScoreSample:
    def test_scores_populated(self):
        preset = RungPreset("0_dense", "", "forced_rag", expand=False)
        stack = _fake_stack(preset)
        out = run_sample(stack, _sample())
        scores = score_sample(out, _sample(), stack)
        assert scores["f1"] > 0                       # "France [1]" overlaps "France"
        assert scores["faithfulness"] == 0.8          # fake validator
        assert scores["context_precision"] == 1.0     # gold file ranked first
        assert scores["router_accuracy"] is None      # no router on this rung

    def test_router_accuracy_only_on_routed(self):
        preset = RungPreset("4_routed", "", "routed", expand=True)
        router = SimpleNamespace(classify=lambda q: RouteResult(
            route=QueryRoute.MULTI_STEP, confidence="high", reasoning=""))
        stack = _fake_stack(preset, router=router)
        out = run_sample(stack, _sample())
        scores = score_sample(out, _sample(), stack)
        assert scores["router_accuracy"] == 1.0       # expected MULTI_STEP, got it

    def test_errored_sample_scores_all_none(self):
        preset = RungPreset("0_dense", "", "forced_rag", expand=False)
        stack = _fake_stack(preset)
        scores = score_sample({"error": "boom", "answer": "", "contexts": [],
                               "actual_route": ""}, _sample(), stack)
        assert all(v is None for v in scores.values())


class TestAggregate:
    def test_none_excluded_from_mean(self):
        rows = [
            {"scores": {"f1": 0.5, "router_accuracy": None}},
            {"scores": {"f1": 1.0, "router_accuracy": None}},
        ]
        agg = aggregate(rows)
        assert agg["f1"] == {"n": 2, "mean": 0.75}
        assert agg["router_accuracy"] == {"n": 0, "mean": None}


# ────────────────────────────────────────────────────────────────────
# Report
# ────────────────────────────────────────────────────────────────────

def _synthetic_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {"timestamp": "2026-07-14T00:00:00+00:00", "dataset": "hotpotqa",
                "seed": 42, "sample_size": 2, "sample_ids": ["h001", "h002"],
                "corpus_dir": "x", "collection_name": "y",
                "constants": {}, "rungs": ["0_dense", "1_expansion"]}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    for name, f1 in (("0_dense", 0.40), ("1_expansion", 0.48)):
        payload = {
            "rung": name, "adds": "layer", "dispatch": "forced_rag",
            "expand": name != "0_dense", "overrides": {},
            "elapsed_seconds": 10.0, "error_count": 0,
            "aggregate": {
                "f1": {"n": 2, "mean": f1},
                "faithfulness": {"n": 2, "mean": 0.9},
                "router_accuracy": {"n": 0, "mean": None},
            },
            "samples": [],
        }
        (run_dir / f"rung_{name}.json").write_text(json.dumps(payload))
    return run_dir


class TestReport:
    def test_load_run_orders_rungs(self, tmp_path):
        run_dir = _synthetic_run_dir(tmp_path)
        _, rungs = load_run(run_dir)
        assert [r["rung"] for r in rungs] == ["0_dense", "1_expansion"]

    def test_markdown_contains_table_and_deltas(self, tmp_path):
        manifest, rungs = load_run(_synthetic_run_dir(tmp_path))
        md = build_markdown(manifest, rungs, chart_files=[])
        assert "| Metric | 0_dense | 1_expansion |" in md
        assert "0.400" in md and "0.480" in md
        assert "+0.080" in md                     # the delta table
        assert "seed 42" in md
        # router_accuracy all-None → rendered as – not crashing
        assert "router_accuracy" in md

    def test_generate_report_writes_and_copies_latest(self, tmp_path):
        run_dir = _synthetic_run_dir(tmp_path)
        latest = tmp_path / "latest"
        path = generate_report(run_dir, copy_latest_to=latest)
        assert path.exists()
        assert (latest / "ablation_report.md").exists()
        # charts are optional (matplotlib may be absent) — but if any were
        # produced they must be referenced by the markdown
        text = path.read_text(encoding="utf-8")
        for png in run_dir.glob("*.png"):
            assert png.name in text
