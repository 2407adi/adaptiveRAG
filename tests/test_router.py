# tests/test_router.py

"""Tests for the QueryRouter — LLM-based query classification."""

import pytest
from unittest.mock import MagicMock
from adaptiverag.reason.router import QueryRouter, QueryRoute, RouteResult


# ── Sample few-shot examples (same shape as config/default.yaml) ──

ROUTING_EXAMPLES = [
    {"query": "What is 2+2?", "route": "direct", "reason": "Simple math"},
    {"query": "Hello!", "route": "direct", "reason": "Greeting"},
    {"query": "What does section 3.2 say?", "route": "rag", "reason": "References a document section"},
    {"query": "What are the key risks in the report?", "route": "rag", "reason": "Needs document content"},
    {"query": "Compare terms across all contracts", "route": "multi_step", "reason": "Cross-document comparison"},
    {"query": "Summarize themes across all files", "route": "multi_step", "reason": "Multi-document synthesis"},
]


def _make_mock_llm(route: str, confidence: str = "high", reason: str = "test") -> MagicMock:
    """Create a mock LLM client that returns a fixed routing response."""
    mock = MagicMock()
    mock.generate.return_value = (
        f"ROUTE: {route}\n"
        f"CONFIDENCE: {confidence}\n"
        f"REASON: {reason}"
    )
    return mock

# ── Core classification tests ──

class TestQueryRouter:

    def test_routes_to_direct(self):
        llm = _make_mock_llm("DIRECT", "high", "General knowledge question")
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("What is 2+2?")

        assert result.route == QueryRoute.DIRECT
        assert result.confidence == "high"

    def test_routes_to_rag(self):
        llm = _make_mock_llm("RAG", "high", "Needs document content")
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("What does section 3.2 say?")

        assert result.route == QueryRoute.RAG

    def test_routes_to_multi_step(self):
        llm = _make_mock_llm("MULTI_STEP", "high", "Cross-document comparison")
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("Compare the terms across all contracts")

        assert result.route == QueryRoute.MULTI_STEP

    # ── Fallback and error handling ──

    def test_unknown_route_falls_back_to_rag(self):
        llm = _make_mock_llm("BANANA")  # LLM returns nonsense
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("some query")

        assert result.route == QueryRoute.RAG  # safe fallback

    def test_llm_exception_falls_back_to_rag(self):
        llm = MagicMock()
        llm.generate.side_effect = Exception("API timeout")
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("anything")

        assert result.route == QueryRoute.RAG
        assert result.confidence == "low"

    def test_empty_response_falls_back_to_rag(self):
        llm = MagicMock()
        llm.generate.return_value = ""
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("anything")

        assert result.route == QueryRoute.RAG

    # ── Parse quality tests ──

    def test_confidence_is_parsed(self):
        llm = _make_mock_llm("RAG", "medium", "Somewhat related to docs")
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("maybe about the document?")

        assert result.confidence == "medium"

    def test_reason_with_colon_is_preserved(self):
        llm = _make_mock_llm("RAG", "high", "Needs docs: section 3 was mentioned")
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("tell me about section 3")

        assert "section 3 was mentioned" in result.reasoning

    def test_case_insensitive_route_parsing(self):
        llm = _make_mock_llm("rag")  # lowercase
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        result = router.classify("some query")

        assert result.route == QueryRoute.RAG

    def test_prompt_includes_examples(self):
        llm = _make_mock_llm("DIRECT")
        router = QueryRouter(llm_client=llm, examples=ROUTING_EXAMPLES)

        router.classify("test query")

        # Check the prompt that was sent to the LLM
        prompt_sent = llm.generate.call_args[0][0]
        assert "What is 2+2?" in prompt_sent
        assert "section 3.2" in prompt_sent

    def test_works_with_no_examples(self):
        llm = _make_mock_llm("DIRECT")
        router = QueryRouter(llm_client=llm, examples=[])

        result = router.classify("Hello")

        assert result.route == QueryRoute.DIRECT