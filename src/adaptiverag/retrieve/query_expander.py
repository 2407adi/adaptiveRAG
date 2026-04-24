"""
Pre-retrieval query expansion.

Rewrites the user's question into a richer search query using an LLM,
adding synonyms and technical terms to improve retrieval recall.
This is NOT HyDE — we rewrite the *question*, not generate a fake answer.
"""

EXPANSION_PROMPT = (
    "You are a search query optimizer. Your job is to rewrite the user's "
    "question into a more detailed search query that will retrieve better "
    "results from a document collection.\n\n"
    "Rules:\n"
    "- Add relevant synonyms, technical terms, and related concepts\n"
    "- Keep the original intent — do NOT change what the user is asking\n"
    "- Return ONLY the rewritten query, no explanation or preamble\n"
    "- Keep it under 50 words — it's a search query, not an essay\n\n"
    "User question: {question}\n\n"
    "Rewritten search query:"
)


class QueryExpander:
    """Expands user queries for better retrieval using LLM rewriting."""

    def __init__(self, llm_client):
        """
        Args:
            llm_client: Any object with a generate(prompt) -> str method.
                        Works with AzureLLMClient or any future LLM client.
        """
        self.llm_client = llm_client

    def expand(self, question: str) -> str:
        """
        Rewrite a user question into a richer search query.

        Args:
            question: The user's original question.

        Returns:
            An expanded query string with added synonyms and terms.
        """
        prompt = EXPANSION_PROMPT.format(question=question)
        expanded = self.llm_client.generate(prompt).strip()

        # Safety: if the LLM returns something absurdly long or empty,
        # fall back to the original
        if not expanded or len(expanded) > 500:
            return question

        return expanded