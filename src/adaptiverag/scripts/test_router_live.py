# src/adaptiverag/scripts/test_router_live.py

"""
Quick manual test — routes real queries through the QueryRouter
using your Azure OpenAI connection.

Run from project root:
    python -m adaptiverag.scripts.test_router_live
"""

from adaptiverag.config import settings
from adaptiverag.reason.router import QueryRouter

from adaptiverag.llm_client import AzureLLMClient

def main():
    # Set up LLM and router
    llm = AzureLLMClient(
        endpoint=settings.azure.endpoint,
        api_key=settings.azure.api_key,
        deployment=settings.azure.deployment,
        temperature=0.0,  # deterministic for testing
    )
    router = QueryRouter(llm_client=llm, examples=settings.routing.examples)

    # Test queries — expected routes in comments
    test_queries = [
        ("What is 2+2?",                                    "DIRECT"),
        ("Hello, how are you?",                              "DIRECT"),
        ("Define photosynthesis",                            "DIRECT"),
        ("What does section 3.2 say?",                       "RAG"),
        ("What are the key risks mentioned in the report?",  "RAG"),
        ("How is capital ratio defined in the document?",    "RAG"),
        ("Compare the terms across all contracts",           "MULTI_STEP"),
        ("Summarize key themes across all uploaded files",   "MULTI_STEP"),
        ("What contradictions exist between the reports?",   "MULTI_STEP"),
        ("How do documents A and B differ on risk?",         "MULTI_STEP"),
    ]

    print("=" * 70)
    print("QUERY ROUTER — LIVE TEST")
    print("=" * 70)

    correct = 0
    for query, expected in test_queries:
        result = router.classify(query)
        match = "✓" if result.route.value.upper() == expected.upper().replace(" ", "_") else "✗"
        if match == "✓":
            correct += 1

        print(f"\n  Q: {query}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result.route.value.upper()}  ({result.confidence})")
        print(f"  Reason:   {result.reasoning}")
        print(f"  {match}")

    print(f"\n{'=' * 70}")
    print(f"  Score: {correct}/{len(test_queries)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()