"""Answer grounding and hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class ClaimStatus(Enum):
    """Entailment verdict for a single claim."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


@dataclass
class ClaimVerdict:
    """Result of checking one claim against the source chunks."""
    claim: str
    status: ClaimStatus
    max_score: float                        # highest entailment prob across all sources
    supporting_source: dict | None = None   # the chunk that best supports it (if any)


@dataclass
class GroundingResult:
    """Aggregate result of validating an entire answer."""
    score: float                            # grounded_claims / total_claims
    is_grounded: bool                       # score >= threshold
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    total_claims: int = 0
    grounded_claims: int = 0

class GroundingValidator:
    """Validates whether an LLM answer is grounded in retrieved sources."""

    def __init__(
        self,
        llm_client,
        threshold: float = 0.6,
    ):
        self.llm_client = llm_client
        self.threshold = threshold

    def _decompose_claims(self, answer: str) -> list[str]:
        """Extract atomic factual claims from an answer."""
        prompt = (
            "Extract every substantive factual claim from the following answer.\n"
            "Rules:\n"
            "- Each claim should be a single, self-contained sentence.\n"
            "- Skip greetings, filler, hedging, and meta-commentary "
            "(e.g. 'Based on the documents', 'In summary').\n"
            "- Skip opinions or subjective statements.\n"
            "- Treat quoted statements as claims if they assert facts.\n"
            "- Include ALL factual assertions, even ones labeled as examples "
            "or hypothetical.\n"
            "- Return ONLY a JSON list of strings. No explanation.\n\n"
            f"Answer:\n{answer}\n\n"
            "Claims:"
        )
        raw = self.llm_client.generate(prompt)

        try:
            claims = json.loads(raw)
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return [c.strip() for c in claims if len(c.strip()) > 10]
        except json.JSONDecodeError:
            pass

        # Fallback: split by newlines, strip numbering
        lines = raw.strip().splitlines()
        claims = []
        for line in lines:
            line = line.strip().lstrip("0123456789.-) ").strip('" ')
            if len(line) > 10:  # skip tiny fragments
                claims.append(line)
        return claims
    
    def _check_entailment(self, claim: str, sources: list[dict]) -> ClaimVerdict:
        """Check if any source chunk supports the claim using LLM-as-judge."""
        if not sources:
            return ClaimVerdict(
                claim=claim,
                status=ClaimStatus.UNSUPPORTED,
                max_score=0.0,
            )

        # Combine all source texts into one context block
        source_texts = []
        for i, source in enumerate(sources):
            text = source.get("full_text", source.get("text_preview", ""))
            source_texts.append(f"[Source {i+1}]: {text}")
        combined_sources = "\n\n".join(source_texts)

        prompt = (
            "You are a fact-checking assistant. Determine whether the following "
            "claim is supported by, contradicted by, or not mentioned in the "
            "source documents.\n\n"
            f"Sources:\n{combined_sources}\n\n"
            f"Claim: {claim}\n\n"
            "Respond with EXACTLY one of these three words, nothing else:\n"
            "SUPPORTED — if the sources contain information that backs this claim\n"
            "CONTRADICTED — if the sources contain information that directly "
            "contradicts this claim\n"
            "UNSUPPORTED — if the sources do not mention this topic at all"
        )

        raw = self.llm_client.generate(prompt).strip().upper()

        # Parse verdict — check UNSUPPORTED before SUPPORTED
        # since "UNSUPPORTED" contains "SUPPORT" as a substring
        if "CONTRADICT" in raw:
            status = ClaimStatus.CONTRADICTED
            score = 0.0
        elif "UNSUPPORT" in raw:
            status = ClaimStatus.UNSUPPORTED
            score = 0.0
        elif "SUPPORT" in raw:
            status = ClaimStatus.SUPPORTED
            score = 1.0
        else:
            status = ClaimStatus.UNSUPPORTED
            score = 0.0

        # Find best matching source (first one for supported claims)
        best_source = sources[0] if status == ClaimStatus.SUPPORTED else None

        return ClaimVerdict(
            claim=claim,
            status=status,
            max_score=score,
            supporting_source=best_source,
        )
    
    def validate(self, answer: str, sources: list[dict]) -> GroundingResult:
        """Validate whether an answer is grounded in the retrieved sources."""
        # Step 1: Decompose answer into atomic claims
        claims = self._decompose_claims(answer)

        if not claims:
            return GroundingResult(
                score=1.0,
                is_grounded=True,
                verdicts=[],
                total_claims=0,
                grounded_claims=0,
            )

        # Step 2: Check each claim against sources
        verdicts = []
        for claim in claims:
            verdict = self._check_entailment(claim, sources)
            verdicts.append(verdict)

        # Step 3: Aggregate
        grounded = sum(
            1 for v in verdicts if v.status == ClaimStatus.SUPPORTED
        )
        score = grounded / len(verdicts)

        return GroundingResult(
            score=score,
            is_grounded=score >= self.threshold,
            verdicts=verdicts,
            total_claims=len(verdicts),
            grounded_claims=grounded,
        )