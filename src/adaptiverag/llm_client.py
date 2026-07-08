"""Thin LLM wrapper that satisfies RAGChain's .generate() interface."""

from openai import AzureOpenAI
from collections.abc import Iterator      # top of file: the type of "a thing you can loop over"

class AzureLLMClient:
    """Wraps Azure OpenAI to expose a simple .generate(prompt) -> str."""

    def __init__(self, endpoint: str, api_key: str, deployment: str,
                 temperature: float = 0.7, max_tokens: int = 4096):
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-06-01",
        )
        self._deployment = deployment
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Send a prompt, return the completion text."""
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""
    
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Same call as generate(), but yields the text piece-by-piece."""
        stream = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,                                 # dictate through the door, don't mail
        )
        for chunk in stream:                             # each chunk = a few characters of answer
            if not chunk.choices:                        # Azure sends occasional housekeeping
                continue                                 #   chunks with no text — skip them
            delta = chunk.choices[0].delta.content       # the NEW characters only (not cumulative)
            if delta:                                    # final chunk's delta is None
                yield delta                              # slide this piece out; pause here till asked again