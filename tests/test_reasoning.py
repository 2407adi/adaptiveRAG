# tests/test_reasoning.py

"""Tests for MultiStepChain — chain-of-thought reasoning over documents."""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from adaptiverag.reason.chain import MultiStepChain, RAGChain


# ── Helpers ──

def _make_mock_rag_chain(answers: dict[str, str] | None = None):
    """Create a mock RAGChain whose .query() returns canned answers.

    Args:
        answers: mapping of sub-question substring → answer text.
                 If a sub-question contains the key, that answer is returned.
                 Falls back to a generic answer.
    """
    answers = answers or {}
    mock = MagicMock(spec=RAGChain)

    def fake_query(question, expand=False):
        for keyword, answer_text in answers.items():
            if keyword.lower() in question.lower():
                return {
                    "answer": answer_text,
                    "sources": [{"chunk_id": f"chunk_{keyword}",
                                 "source": "test.pdf", "page": 1,
                                 "chunk_index": 0, "score": 0.95,
                                 "text_preview": answer_text[:200],
                                 "full_text": answer_text}],
                }
        return {
            "answer": "Generic answer.",
            "sources": [{"chunk_id": "chunk_generic",
                         "source": "test.pdf", "page": 1,
                         "chunk_index": 0, "score": 0.8,
                         "text_preview": "Generic", "full_text": "Generic answer."}],
        }

    mock.query.side_effect = fake_query
    return mock


def _make_mock_llm(decompose_response: str, synthesize_response: str = "Final answer."):
    """Create a mock LLM that returns different outputs on successive calls.

    Call 1 (decompose): returns decompose_response
    Call 2 (synthesize): returns synthesize_response
    """
    mock = MagicMock()
    mock.generate.side_effect = [decompose_response, synthesize_response]
    return mock

# ── Decomposition tests ──

class TestDecomposition:

    def test_decomposes_into_sub_questions(self):
        sub_qs = ["What is the US policy?", "What is the UK policy?"]
        llm = _make_mock_llm(json.dumps(sub_qs))
        rag = _make_mock_rag_chain()
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        result = chain._decompose("Compare US vs UK policy")

        assert result == sub_qs

    def test_handles_markdown_code_fences(self):
        sub_qs = ["Question A?", "Question B?"]
        fenced = f"```json\n{json.dumps(sub_qs)}\n```"
        llm = _make_mock_llm(fenced)
        rag = _make_mock_rag_chain()
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        result = chain._decompose("some complex question")

        assert result == sub_qs

    def test_fallback_on_invalid_json(self):
        llm = _make_mock_llm("This is not JSON at all")
        rag = _make_mock_rag_chain()
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        result = chain._decompose("Compare things")

        assert result == ["Compare things"]  # falls back to original

    def test_fallback_on_too_many_sub_questions(self):
        too_many = json.dumps(["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"])
        llm = _make_mock_llm(too_many)
        rag = _make_mock_rag_chain()
        chain = MultiStepChain(rag_chain=rag, llm_client=llm, max_sub_questions=4)

        result = chain._decompose("Big question")

        assert result == ["Big question"]  # 5 > max of 4, fallback

    def test_fallback_on_single_sub_question(self):
        one = json.dumps(["Only one question?"])
        llm = _make_mock_llm(one)
        rag = _make_mock_rag_chain()
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        result = chain._decompose("Something")

        assert result == ["Something"]  # less than 2, fallback

# ── Full query cycle tests ──

class TestMultiStepQuery:

    def test_full_cycle_returns_expected_keys(self):
        sub_qs = ["What is the US policy?", "What is the UK policy?"]
        llm = _make_mock_llm(
            decompose_response=json.dumps(sub_qs),
            synthesize_response="The US offers 15 days, the UK offers 25 days.",
        )
        rag = _make_mock_rag_chain({
            "us": "US employees get 15 vacation days.",
            "uk": "UK employees get 25 vacation days.",
        })
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        result = chain.query("Compare US vs UK vacation policy")

        assert "answer" in result
        assert "sources" in result
        assert "reasoning_steps" in result

    def test_reasoning_steps_match_sub_questions(self):
        sub_qs = ["What is A?", "What is B?"]
        llm = _make_mock_llm(json.dumps(sub_qs), "Synthesized.")
        rag = _make_mock_rag_chain()
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        result = chain.query("Compare A and B")

        assert len(result["reasoning_steps"]) == 2
        assert result["reasoning_steps"][0]["sub_question"] == "What is A?"
        assert result["reasoning_steps"][1]["sub_question"] == "What is B?"

    def test_each_sub_question_calls_rag_chain(self):
        sub_qs = ["First?", "Second?", "Third?"]
        llm = _make_mock_llm(json.dumps(sub_qs), "Done.")
        rag = _make_mock_rag_chain()
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        chain.query("Complex question")

        assert rag.query.call_count == 3

    def test_sources_are_deduplicated(self):
        sub_qs = ["About topic?", "Also about topic?"]
        llm = _make_mock_llm(json.dumps(sub_qs), "Merged.")
        # Both sub-questions will match "topic" → same chunk_id
        rag = _make_mock_rag_chain({"topic": "Shared info."})
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        result = chain.query("Tell me about topic from two angles")

        chunk_ids = [s["chunk_id"] for s in result["sources"]]
        assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates

    def test_expand_flag_passes_through(self):
        sub_qs = ["Sub Q?"]  # will fallback since < 2, but that's fine
        llm = _make_mock_llm(json.dumps(["Sub A?", "Sub B?"]), "Done.")
        rag = _make_mock_rag_chain()
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        chain.query("Complex question", expand=True)

        # Verify expand=True was passed to each rag_chain.query call
        for c in rag.query.call_args_list:
            assert c.kwargs.get("expand") is True or c[1].get("expand") is True

    def test_synthesis_receives_sub_answers(self):
        sub_qs = ["What is X?", "What is Y?"]
        llm = _make_mock_llm(json.dumps(sub_qs), "X and Y combined.")
        rag = _make_mock_rag_chain({"x": "X is 10.", "y": "Y is 20."})
        chain = MultiStepChain(rag_chain=rag, llm_client=llm)

        chain.query("Compare X and Y")

        # The second LLM call (synthesize) should contain sub-answers
        synthesize_prompt = llm.generate.call_args_list[1][0][0]
        assert "X is 10." in synthesize_prompt
        assert "Y is 20." in synthesize_prompt