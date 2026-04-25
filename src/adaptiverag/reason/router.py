from enum import Enum
from dataclasses import dataclass


class QueryRoute(Enum):
    """The three execution paths a query can take."""
    DIRECT = "direct"          # LLM answers from its own knowledge
    RAG = "rag"                # Retrieve from documents first
    MULTI_STEP = "multi_step"  # Complex reasoning across documents


@dataclass
class RouteResult:
    """What the router returns — the route plus the reasoning."""
    route: QueryRoute
    confidence: str   # "high", "medium", "low" — from the LLM's self-assessment
    reasoning: str    # One-line explanation of why this route was chosen


class QueryRouter:
    """
    LLM-based query classifier.
    Uses few-shot examples to route queries to the right execution path.
    """

    FALLBACK_ROUTE = QueryRoute.RAG  # Safe default — retrieval never hurts

    def __init__(self, llm_client, examples: list[dict] | None = None):
        """
        Args:
            llm_client: Your existing AzureLLMClient (or any object with .generate(prompt) -> str)
            examples: Few-shot examples from config. Each dict has 'query', 'route', 'reason'.
        """
        self.llm = llm_client
        self.examples = examples or []

    def _build_prompt(self, query: str) -> str:
        """Build the few-shot classification prompt."""

        # Format each example as a line: Q: "..." → ROUTE (reason)
        example_lines = []
        for ex in self.examples:
            example_lines.append(
                f'Q: "{ex["query"]}" → {ex["route"].upper()} ({ex["reason"]})'
            )
        examples_block = "\n".join(example_lines)

        return f"""You are a query routing classifier for a document Q&A system.
    The user has uploaded documents. Classify their query into exactly one category:

    DIRECT — General knowledge, math, greetings, definitions. No document context needed.
    RAG — Needs information from the uploaded documents. Single lookup or factual extraction.
    MULTI_STEP — Requires comparing, synthesizing, or analyzing across multiple parts or documents.

    Examples:
    {examples_block}

    Rules:
    - If the query mentions specific sections, pages, or "the document", it is RAG.
    - If the query asks to compare, contrast, summarize across, or analyze multiple things, it is MULTI_STEP.
    - If unsure between RAG and MULTI_STEP, choose RAG.
    - If unsure between DIRECT and RAG, choose RAG.

    Respond in EXACTLY this format, nothing else:
    ROUTE: <category>
    CONFIDENCE: <high/medium/low>
    REASON: <one sentence>

    Q: "{query}" →"""

    def _parse_response(self, response: str) -> RouteResult:
        """Parse the LLM's structured response into a RouteResult."""

        route = self.FALLBACK_ROUTE
        confidence = "low"
        reasoning = "Could not parse LLM response"

        for line in response.strip().split("\n"):
            line = line.strip()

            if line.upper().startswith("ROUTE:"):
                route_str = line.split(":", 1)[1].strip().upper()
                # Try to match against valid routes
                route_map = {
                    "DIRECT": QueryRoute.DIRECT,
                    "RAG": QueryRoute.RAG,
                    "MULTI_STEP": QueryRoute.MULTI_STEP,
                }
                route = route_map.get(route_str, self.FALLBACK_ROUTE)

            elif line.upper().startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().lower()

            elif line.upper().startswith("REASON:"):
                reasoning = line.split(":", 1)[1].strip()

        return RouteResult(route=route, confidence=confidence, reasoning=reasoning)

    def classify(self, query: str) -> RouteResult:
        """Classify a query into DIRECT, RAG, or MULTI_STEP."""

        try:
            prompt = self._build_prompt(query)
            response = self.llm.generate(prompt)
            return self._parse_response(response)

        except Exception:
            # Network error, LLM timeout, anything unexpected — fall back safely
            return RouteResult(
                route=self.FALLBACK_ROUTE,
                confidence="low",
                reasoning="Classification failed, falling back to RAG",
            )