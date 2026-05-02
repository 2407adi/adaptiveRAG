"""Tests for adaptiverag.eval.dataset."""

import json
from pathlib import Path

import pytest

from adaptiverag.eval.dataset import EvalSample, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_PAIRS = PROJECT_ROOT / "data" / "eval" / "qa_pairs.json"


def test_load_dataset_returns_25_samples():
    samples = load_dataset(QA_PAIRS)
    assert len(samples) == 25


def test_each_sample_has_required_fields():
    samples = load_dataset(QA_PAIRS)
    for s in samples:
        assert isinstance(s, EvalSample)
        assert s.id and s.question and s.ground_truth
        assert s.expected_route in {"DIRECT", "RAG", "MULTI_STEP"}


def test_skip_metrics_default_empty():
    by_id = {s.id: s for s in load_dataset(QA_PAIRS)}
    assert by_id["q001"].skip_metrics == []
    assert by_id["q014"].skip_metrics == []


def test_q012_skips_relevancy_and_recall():
    """The trick-gap question must skip the metrics that misfire on
    noncommittal answers."""
    q012 = next(s for s in load_dataset(QA_PAIRS) if s.id == "q012")
    assert "answer_relevancy" in q012.skip_metrics
    assert "context_recall" in q012.skip_metrics


def test_route_distribution():
    """3 DIRECT + 15 RAG + 7 MULTI_STEP = 25."""
    routes = [s.expected_route for s in load_dataset(QA_PAIRS)]
    assert routes.count("DIRECT") == 3
    assert routes.count("RAG") == 15
    assert routes.count("MULTI_STEP") == 7


def test_missing_required_field_raises_keyerror(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"id": "x", "question": "?", "ground_truth": "!"}
        # missing expected_route
    ]))
    with pytest.raises(KeyError):
        load_dataset(bad)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nonexistent.json")