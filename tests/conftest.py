"""Shared test fixtures for the adaptiverag.eval module."""

import pytest


class FakeLLM:
    """Mock AzureLLMClient. Records calls; returns canned responses by trigger."""

    def __init__(self, responses=None, default=""):
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        for trigger, response in self.responses.items():
            if trigger in prompt:
                return response
        return self.default


class FakeEmbedder:
    """Returns word-presence vectors over a fixed keyword set, so similar
    texts get similar vectors — good enough for answer_relevancy smoke tests."""

    KEYWORDS = ("paris", "france", "capital", "city", "robot", "company",
                "founded", "year")

    def embed(self, text: str) -> list[float]:
        text_lower = text.lower()
        return [1.0 if kw in text_lower else 0.0 for kw in self.KEYWORDS]


class FakeValidator:
    """Mock GroundingValidator. Returns a canned GroundingResult."""

    def __init__(self, score: float = 1.0):
        self.score = score
        self.calls: list[tuple] = []

    def validate(self, answer, sources):
        from adaptiverag.reason.grounding import GroundingResult
        self.calls.append((answer, sources))
        return GroundingResult(
            score=self.score,
            is_grounded=self.score >= 0.6,
            verdicts=[],
            total_claims=1,
            grounded_claims=1 if self.score >= 0.5 else 0,
        )


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_validator():
    return FakeValidator(score=1.0)