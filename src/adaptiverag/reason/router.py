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

    def __init__(
        self,
        llm_client,
        examples: list[dict] | None = None,
        corpus_summary: str | None = None,
    ):
        self.llm = llm_client
        self.examples = examples or []
        self.corpus_summary = corpus_summary

    def _build_prompt(self, query: str) -> str:
        """Build the few-shot classification prompt."""

        example_lines = []
        for ex in self.examples:
            example_lines.append(
                f'Q: "{ex["query"]}" → {ex["route"].upper()} ({ex["reason"]})'
            )
        examples_block = "\n".join(example_lines)

        corpus_block = (
            "\n<corpus_summary>\n"
            "The user has uploaded documents to this Q&A system. The paragraph "
            "below summarizes the topics, entities, and key facts those documents "
            "contain. Use it to judge whether the question's subject is actually "
            "present in the corpus.\n\n"
            f"{self.corpus_summary}\n"
            "</corpus_summary>\n"
            if self.corpus_summary else ""
        )

        return f"""<task>
    You are a query routing classifier for a document Q&A system. The user has uploaded their own documents and is now asking questions. Classify each incoming question into exactly one of three execution paths.
    </task>

    <routes>
    DIRECT — Answer from your own general knowledge. Greetings, math, definitions, or questions whose subject is NOT covered by the uploaded documents.
    RAG — Retrieve from the uploaded documents. Single-fact lookup or factual extraction where the subject IS present in the documents.
    MULTI_STEP — Multi-document reasoning. Comparing, synthesizing, or analyzing across multiple parts or documents.
    </routes>
    {corpus_block}
    <examples>
    {examples_block}
    </examples>

    <rules>
    - If the question's subject appears in <corpus_summary>, prefer RAG (or MULTI_STEP if compositional).
    - If the question's subject is clearly NOT in <corpus_summary>, prefer DIRECT — the documents will not contain the answer.
    - If the question mentions specific sections, pages, or "the document", it is RAG.
    - If the question asks to compare, contrast, summarize across multiple things, or analyze relationships, it is MULTI_STEP.
    - If unsure between RAG and MULTI_STEP, choose RAG.
    - If <corpus_summary> is absent and the question doesn't obviously need general knowledge, choose RAG.
    </rules>

    <question>
    {query}
    </question>

    Respond in EXACTLY this format, nothing else:
    ROUTE: <DIRECT|RAG|MULTI_STEP>
    CONFIDENCE: <high|medium|low>
    REASON: <one sentence>"""

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