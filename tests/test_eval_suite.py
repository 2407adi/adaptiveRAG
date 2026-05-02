"""Tests for adaptiverag.eval.suite — orchestration + aggregation + save."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from adaptiverag.eval.dataset import EvalSample
from adaptiverag.eval.suite import EvalSuite, SampleRun
from adaptiverag.reason.router import RouteResult, QueryRoute


# ──────── helpers ────────

def _write_dataset(tmp_path: Path, samples: list[dict]) -> Path:
    p = tmp_path / "qa.json"
    p.write_text(json.dumps(samples))
    return p


def _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator) -> EvalSuite:
    dataset_path = _write_dataset(tmp_path, [{
        "id": "s1", "question": "?", "ground_truth": "!",
        "expected_route": "RAG", "relevant_sources": ["a.md"],
    }])
    return EvalSuite(
        dataset_path=dataset_path,
        router=Mock(), rag_chain=Mock(), multi_step_chain=Mock(),
        llm_client=fake_llm, embedder=fake_embedder, validator=fake_validator,
        results_dir=tmp_path / "results",
    )


def _make_run(route="RAG", scores=None, error=None,
              relevant_sources=None, skip_metrics=None,
              expected_route=None) -> SampleRun:
    return SampleRun(
        sample=EvalSample(
            id="s1", question="?", ground_truth="!",
            expected_route=expected_route or route,
            relevant_sources=relevant_sources or [],
            skip_metrics=skip_metrics or [],
        ),
        actual_route=route,
        answer="some answer",
        contexts=[{"source": "a.md", "full_text": "verbose chunk text"}],
        scores=scores or {},
        error=error,
    )


# ──────── _aggregate ────────

def test_aggregate_empty_results(tmp_path, fake_llm, fake_embedder, fake_validator):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    agg = suite._aggregate([])
    assert agg["total_samples"] == 0
    assert agg["error_count"] == 0
    for stats in agg["per_metric"].values():
        assert stats["n"] == 0
        assert stats["mean"] is None


def test_aggregate_empty_bucket_has_mean_none_not_zero(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    """Critical: empty route bucket reports mean=None, NOT 0.0 — because
    0.0 is also a valid score (e.g. complete retrieval miss)."""
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    runs = [_make_run(route="RAG", scores={"faithfulness": 0.8})]
    direct = suite._aggregate(runs)["per_metric_per_route"]["faithfulness"]["DIRECT"]
    assert direct["n"] == 0
    assert direct["mean"] is None


def test_aggregate_per_metric_overall_mean(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    runs = [
        _make_run(scores={"faithfulness": 0.8}),
        _make_run(scores={"faithfulness": 0.6}),
        _make_run(scores={"faithfulness": None}),  # excluded
    ]
    f = suite._aggregate(runs)["per_metric"]["faithfulness"]
    assert f["n"] == 2
    assert f["mean"] == pytest.approx(0.7)
    assert f["min"] == 0.6
    assert f["max"] == 0.8


def test_aggregate_buckets_by_actual_route(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    runs = [
        _make_run(route="RAG", scores={"faithfulness": 0.8}),
        _make_run(route="MULTI_STEP", scores={"faithfulness": 0.4}),
    ]
    by_route = suite._aggregate(runs)["per_metric_per_route"]["faithfulness"]
    assert by_route["RAG"]["mean"] == pytest.approx(0.8)
    assert by_route["MULTI_STEP"]["mean"] == pytest.approx(0.4)


def test_aggregate_error_count(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    runs = [_make_run(error="boom"), _make_run(error=None), _make_run(error="kaboom")]
    assert suite._aggregate(runs)["error_count"] == 2


# ──────── _score_one_sample ────────

def test_score_error_yields_all_none(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    run = _make_run(error="something blew up")
    suite._score_one_sample(run)
    assert all(v is None for v in run.scores.values())
    assert set(run.scores) >= {
        "router_accuracy", "faithfulness", "answer_relevancy",
        "context_recall", "context_precision",
    }


def test_score_respects_skip_metrics(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    run = _make_run(
        relevant_sources=["a.md"],
        skip_metrics=["answer_relevancy", "context_recall"],
    )
    suite._score_one_sample(run)
    assert run.scores["answer_relevancy"] is None
    assert run.scores["context_recall"] is None
    assert run.scores["router_accuracy"] is not None
    assert run.scores["faithfulness"] is not None
    assert run.scores["context_precision"] is not None


def test_score_router_accuracy_match_and_mismatch(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)

    matched = _make_run(route="RAG", expected_route="RAG", relevant_sources=["a.md"])
    suite._score_one_sample(matched)
    assert matched.scores["router_accuracy"] == 1.0

    mismatched = _make_run(route="MULTI_STEP", expected_route="RAG",
                           relevant_sources=["a.md"])
    suite._score_one_sample(mismatched)
    assert mismatched.scores["router_accuracy"] == 0.0


# ──────── save ────────

def test_save_creates_timestamped_json(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    out = suite.save([_make_run(scores={"faithfulness": 0.5})])
    assert out.exists()
    assert out.suffix == ".json"
    assert "T" in out.stem and out.stem.endswith("Z")


def test_save_does_not_persist_full_text(
    tmp_path, fake_llm, fake_embedder, fake_validator,
):
    """Result files must not bloat with corpus text."""
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    out = suite.save([_make_run(scores={"faithfulness": 0.8})])
    assert "verbose chunk text" not in out.read_text()


def test_save_round_trip(tmp_path, fake_llm, fake_embedder, fake_validator):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    run = _make_run(scores={"faithfulness": 0.8, "router_accuracy": 1.0})
    data = json.loads(suite.save([run]).read_text())
    assert data["samples"][0]["scores"]["faithfulness"] == 0.8
    assert data["samples"][0]["context_count"] == 1
    assert data["samples"][0]["context_sources"] == ["a.md"]
    assert "summary" in data
    assert "timestamp" in data


# ──────── run() filtering ────────

def test_run_filters_by_sample_ids(tmp_path, fake_llm, fake_embedder, fake_validator):
    dataset_path = _write_dataset(tmp_path, [
        {"id": "a", "question": "?", "ground_truth": "!", "expected_route": "DIRECT"},
        {"id": "b", "question": "?", "ground_truth": "!", "expected_route": "DIRECT"},
    ])
    router = Mock()
    router.classify.return_value = RouteResult(
        route=QueryRoute.DIRECT, confidence="high", reasoning="test",
    )
    suite = EvalSuite(
        dataset_path=dataset_path, router=router, rag_chain=Mock(),
        multi_step_chain=Mock(), llm_client=fake_llm, embedder=fake_embedder,
        validator=fake_validator, results_dir=tmp_path / "results",
    )

    results = suite.run(sample_ids=["a"])
    assert len(results) == 1
    assert results[0].sample.id == "a"


def test_run_warns_on_unknown_sample_id(
    tmp_path, fake_llm, fake_embedder, fake_validator, caplog,
):
    suite = _make_suite(tmp_path, fake_llm, fake_embedder, fake_validator)
    suite.router.classify = Mock(return_value=RouteResult(
        route=QueryRoute.RAG, confidence="high", reasoning="test",
    ))
    suite.rag_chain.query = Mock(return_value={"answer": "ok", "sources": []})

    with caplog.at_level("WARNING"):
        suite.run(sample_ids=["s1", "qoo1"])

    assert any("qoo1" in record.message for record in caplog.records)