"""Tests for adaptiverag.eval.metrics."""

import pytest

from adaptiverag.eval.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    _cosine,
    _parse_questions,
)


# ──────── faithfulness ────────

def test_faithfulness_none_for_empty_contexts(fake_validator):
    assert faithfulness("any answer", [], fake_validator) is None


def test_faithfulness_delegates_to_validator(fake_validator):
    fake_validator.score = 0.75
    assert faithfulness("answer", [{"full_text": "x"}], fake_validator) == 0.75
    assert len(fake_validator.calls) == 1


# ──────── answer_relevancy ────────

def test_answer_relevancy_none_for_empty_answer(fake_llm, fake_embedder):
    assert answer_relevancy("Q?", "", fake_llm, fake_embedder) is None
    assert answer_relevancy("Q?", "   ", fake_llm, fake_embedder) is None


def test_answer_relevancy_none_when_parsing_fails(fake_embedder):
    llm = type("X", (), {"generate": lambda self, p: "no questions here"})()
    assert answer_relevancy("Q?", "answer", llm, fake_embedder) is None


def test_answer_relevancy_higher_for_on_topic(fake_embedder):
    on_llm = type("On", (), {
        "generate": lambda self, p: '["What is the capital of France?"]'
    })()
    off_llm = type("Off", (), {
        "generate": lambda self, p: '["What company makes robots?"]'
    })()
    on = answer_relevancy("What is the capital of France?", "Paris.",
                         on_llm, fake_embedder, n_generated=1)
    off = answer_relevancy("What is the capital of France?", "Paris.",
                          off_llm, fake_embedder, n_generated=1)
    assert on is not None and off is not None
    assert on > off


# ──────── context_recall ────────

def test_context_recall_none_for_empty_contexts(fake_validator):
    assert context_recall("ground truth", [], fake_validator) is None


def test_context_recall_none_for_empty_ground_truth(fake_validator):
    assert context_recall("", [{"full_text": "x"}], fake_validator) is None
    assert context_recall("   ", [{"full_text": "x"}], fake_validator) is None


def test_context_recall_delegates_to_validator(fake_validator):
    fake_validator.score = 0.6
    assert context_recall("gt", [{"full_text": "x"}], fake_validator) == 0.6


# ──────── context_precision (no mocks needed — pure math) ────────

def test_context_precision_perfect_ranking():
    score = context_precision(
        contexts=[{"source": "a.md"}, {"source": "a.md"}, {"source": "x.md"}],
        relevant_sources=["a.md"],
    )
    assert score == pytest.approx(1.0)


def test_context_precision_bad_ranking():
    score = context_precision(
        contexts=[{"source": "x.md"}, {"source": "y.md"}, {"source": "a.md"}],
        relevant_sources=["a.md"],
    )
    assert score == pytest.approx(1 / 3)


def test_context_precision_interleaved():
    # labels=[1,0,1,0,0]: P@1=1.0, P@3=2/3
    score = context_precision(
        contexts=[
            {"source": "a.md"}, {"source": "x.md"},
            {"source": "a.md"}, {"source": "x.md"}, {"source": "x.md"},
        ],
        relevant_sources=["a.md"],
    )
    assert score == pytest.approx((1.0 + 2 / 3) / 2)


def test_context_precision_no_relevant_returns_zero():
    score = context_precision(
        contexts=[{"source": "x.md"}, {"source": "y.md"}],
        relevant_sources=["a.md"],
    )
    assert score == 0.0


def test_context_precision_empty_contexts_none():
    assert context_precision([], ["a.md"]) is None


def test_context_precision_empty_relevant_sources_none():
    assert context_precision([{"source": "x.md"}], []) is None


def test_context_precision_multiple_relevant_sources():
    # labels=[1,0,1]: P@1=1.0, P@3=2/3
    score = context_precision(
        contexts=[{"source": "a.md"}, {"source": "x.md"}, {"source": "b.md"}],
        relevant_sources=["a.md", "b.md"],
    )
    assert score == pytest.approx((1.0 + 2 / 3) / 2)


# ──────── private helpers (worth testing in isolation) ────────

def test_cosine_identical():
    assert _cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert _cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert _cosine([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_no_div_error():
    """Guards the +1e-10-style div-by-zero handling in _cosine."""
    assert _cosine([0, 0, 0], [1, 1, 1]) == 0.0


def test_parse_questions_clean_json():
    assert _parse_questions('["Q1?", "Q2?"]') == ["Q1?", "Q2?"]


def test_parse_questions_strips_markdown_fences():
    assert _parse_questions('```json\n["Q1?"]\n```') == ["Q1?"]


def test_parse_questions_fallback_to_lines():
    assert _parse_questions("1. What is X?\n2. Who founded Y?") == [
        "What is X?", "Who founded Y?",
    ]


def test_parse_questions_filters_non_questions():
    assert _parse_questions("Just a statement.") == []