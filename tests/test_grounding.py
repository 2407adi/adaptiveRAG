"""Tests for answer grounding and hallucination detection."""

import pytest
from unittest.mock import MagicMock
from adaptiverag.reason.grounding import (
    ClaimStatus,
    ClaimVerdict,
    GroundingResult,
    GroundingValidator,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_sources():
    """Simulates what RAGChain.query() returns in its sources list."""
    return [
        {
            "chunk_id": "doc1::chunk-0",
            "source": "report.pdf",
            "page": "1",
            "chunk_index": 0,
            "score": 0.85,
            "text_preview": "Revenue grew 15% in Q3 2024...",
            "full_text": (
                "Revenue grew 15% in Q3 2024, reaching $12M. "
                "The growth was driven by enterprise contracts."
            ),
        },
        {
            "chunk_id": "doc1::chunk-1",
            "source": "report.pdf",
            "page": "2",
            "chunk_index": 1,
            "score": 0.78,
            "text_preview": "The company employs 250 people...",
            "full_text": (
                "The company employs 250 people across 3 offices. "
                "Headquarters is in Austin, Texas."
            ),
        },
    ]


# ── Claim Decomposition Tests ────────────────────────────

class TestDecomposeClaims:

    def test_json_response(self, mock_llm):
        """LLM returns clean JSON list."""
        mock_llm.generate.return_value = (
            '["Revenue grew 15%", "The company has 250 employees"]'
        )
        validator = GroundingValidator(mock_llm)
        claims = validator._decompose_claims("some answer")

        assert len(claims) == 2
        assert "Revenue grew 15%" in claims

    def test_numbered_list_fallback(self, mock_llm):
        """LLM returns numbered list instead of JSON."""
        mock_llm.generate.return_value = (
            "1. Revenue grew 15%\n"
            "2. The company has 250 employees"
        )
        validator = GroundingValidator(mock_llm)
        claims = validator._decompose_claims("some answer")

        assert len(claims) == 2
        assert "Revenue grew 15%" in claims

    def test_short_fragments_filtered(self, mock_llm):
        """Fragments under 10 chars are dropped."""
        mock_llm.generate.return_value = (
            '["Revenue grew 15%", "Yes", "OK"]'
        )
        validator = GroundingValidator(mock_llm)
        claims = validator._decompose_claims("some answer")

        assert len(claims) == 1

    def test_empty_answer(self, mock_llm):
        """No claims from a hedging answer."""
        mock_llm.generate.return_value = '[]'
        validator = GroundingValidator(mock_llm)
        claims = validator._decompose_claims("I'm not sure.")

        assert claims == []


# ── Entailment Tests ─────────────────────────────────────

class TestCheckEntailment:

    def test_supported_claim(self, mock_llm, mock_sources):
        """Claim clearly supported by a source gets SUPPORTED."""
        mock_llm.generate.return_value = "SUPPORTED"
        validator = GroundingValidator(mock_llm)

        verdict = validator._check_entailment("Revenue grew 15%", mock_sources)

        assert verdict.status == ClaimStatus.SUPPORTED
        assert verdict.max_score == 1.0
        assert verdict.supporting_source is not None

    def test_unsupported_claim(self, mock_llm, mock_sources):
        """Fabricated claim gets UNSUPPORTED."""
        mock_llm.generate.return_value = "UNSUPPORTED"
        validator = GroundingValidator(mock_llm)

        verdict = validator._check_entailment(
            "The CEO resigned in January", mock_sources
        )

        assert verdict.status == ClaimStatus.UNSUPPORTED
        assert verdict.max_score == 0.0
        assert verdict.supporting_source is None

    def test_contradicted_claim(self, mock_llm, mock_sources):
        """Claim that contradicts a source gets CONTRADICTED."""
        mock_llm.generate.return_value = "CONTRADICTED"
        validator = GroundingValidator(mock_llm)

        verdict = validator._check_entailment(
            "Revenue declined 15%", mock_sources
        )

        assert verdict.status == ClaimStatus.CONTRADICTED

    def test_empty_sources(self, mock_llm):
        """No sources means UNSUPPORTED."""
        validator = GroundingValidator(mock_llm)

        verdict = validator._check_entailment("Any claim", [])

        assert verdict.status == ClaimStatus.UNSUPPORTED
        assert verdict.max_score == 0.0


# ── Full Validation Tests ────────────────────────────────

class TestValidate:

    def test_well_grounded_answer(self, mock_llm, mock_sources):
        """Answer where all claims are supported scores 1.0."""
        # First call: decompose claims, subsequent calls: entailment checks
        mock_llm.generate.side_effect = [
            '["Revenue grew 15%", "The company has 250 employees"]',
            "SUPPORTED",  # claim 1
            "SUPPORTED",  # claim 2
        ]
        validator = GroundingValidator(mock_llm)

        result = validator.validate("some answer", mock_sources)

        assert result.score == 1.0
        assert result.is_grounded is True
        assert result.total_claims == 2
        assert result.grounded_claims == 2

    def test_hallucinated_answer(self, mock_llm, mock_sources):
        """Answer where no claims are supported scores 0.0."""
        mock_llm.generate.side_effect = [
            '["The CEO resigned", "Offices closed in March"]',
            "UNSUPPORTED",  # claim 1
            "UNSUPPORTED",  # claim 2
        ]
        validator = GroundingValidator(mock_llm)

        result = validator.validate("some answer", mock_sources)

        assert result.score == 0.0
        assert result.is_grounded is False
        assert result.grounded_claims == 0

    def test_partial_grounding_passes(self, mock_llm, mock_sources):
        """2 out of 3 claims grounded = 0.67, passes 0.6 threshold."""
        mock_llm.generate.side_effect = [
            '["Revenue grew 15%", "Company has 250 employees", '
            '"CEO resigned in January"]',
            "SUPPORTED",    # claim 1
            "SUPPORTED",    # claim 2
            "UNSUPPORTED",  # claim 3
        ]
        validator = GroundingValidator(mock_llm)

        result = validator.validate("some answer", mock_sources)

        assert result.total_claims == 3
        assert result.grounded_claims == 2
        assert round(result.score, 2) == 0.67
        assert result.is_grounded is True

    def test_partial_grounding_fails(self, mock_llm, mock_sources):
        """1 out of 3 claims grounded = 0.33, fails threshold."""
        mock_llm.generate.side_effect = [
            '["Revenue grew 15%", "CEO resigned", "Offices closed"]',
            "SUPPORTED",    # claim 1
            "UNSUPPORTED",  # claim 2
            "UNSUPPORTED",  # claim 3
        ]
        validator = GroundingValidator(mock_llm)

        result = validator.validate("some answer", mock_sources)

        assert result.grounded_claims == 1
        assert result.is_grounded is False

    def test_no_claims_is_grounded(self, mock_llm, mock_sources):
        """Hedging answer with no factual claims treated as grounded."""
        mock_llm.generate.return_value = '[]'
        validator = GroundingValidator(mock_llm)

        result = validator.validate(
            "I don't have enough information.", mock_sources
        )

        assert result.score == 1.0
        assert result.is_grounded is True
        assert result.total_claims == 0
