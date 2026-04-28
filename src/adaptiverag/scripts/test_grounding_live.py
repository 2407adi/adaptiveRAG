"""Live test: feed the GroundingValidator a known-good and known-bad answer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptiverag.config import settings
from adaptiverag.llm_client import AzureLLMClient
from adaptiverag.reason.grounding import GroundingValidator

llm_client = AzureLLMClient(
    endpoint=settings.azure.endpoint,
    api_key=settings.azure.api_key,
    deployment=settings.azure.deployment,
)

validator = GroundingValidator(llm_client, threshold=0.6)

# Fake sources (simulating what RAGChain returns)
sources = [
    {
        "chunk_id": "doc1::chunk-0",
        "source": "report.pdf",
        "full_text": (
            "Revenue grew 15% in Q3 2024, reaching $12M. "
            "The growth was driven by enterprise contracts. "
            "The company employs 250 people across 3 offices. "
            "Headquarters is in Austin, Texas."
        ),
    },
]

# Test 1: Well-grounded answer
good_answer = (
    "Revenue grew 15% in Q3 2024, reaching $12M. "
    "The company is headquartered in Austin, Texas."
)

# Test 2: Answer with hallucinated claims
bad_answer = (
    "Revenue grew 15% in Q3 2024, reaching $12M. "
    "The CEO announced plans to open offices in Tokyo and Berlin. "
    "The company went public in 2023 with a valuation of $500M."
)

print("=== Test 1: Well-grounded answer ===")
result = validator.validate(good_answer, sources)
print(f"Score: {result.score:.2f} | Grounded: {result.is_grounded}")
for v in result.verdicts:
    print(f"  {v.status.value:13s} (score {v.max_score:.2f}) — {v.claim}")

print("\n=== Test 2: Hallucinated answer ===")
result = validator.validate(bad_answer, sources)
print(f"Score: {result.score:.2f} | Grounded: {result.is_grounded}")
for v in result.verdicts:
    print(f"  {v.status.value:13s} (score {v.max_score:.2f}) — {v.claim}")