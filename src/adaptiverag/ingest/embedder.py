from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from openai import OpenAI


class Embedder(ABC):
    """Base class for all embedding implementations."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns one vector per text."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of vectors this embedder produces."""
        ...


class LocalEmbedder(Embedder):
    """Embedding using a local sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        assert dim is not None, f"Model {model_name} did not report embedding dimension"
        self._dimension = dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts)
        return vectors.tolist()

class OpenAIEmbedder(Embedder):
    """Embedding using OpenAI's API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self._client = OpenAI(api_key=api_key)  # falls back to OPENAI_API_KEY env var
        self._model_name = model_name
        self._dimension = 1536  # text-embedding-3-small default

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text,
            model=self._model_name,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model_name,
        )
        # OpenAI returns embeddings in same order as input
        return [item.embedding for item in response.data]
    

def create_embedder(embedder_type: str = "local", **kwargs) -> Embedder:
    """Factory to create an embedder by type name.
    
    Args:
        embedder_type: "openai" or "local"
        **kwargs: passed to the embedder constructor (e.g. api_key, model_name)
    """
    embedders = {
        "openai": OpenAIEmbedder,
        "local": LocalEmbedder,
    }

    if embedder_type not in embedders:
        raise ValueError(f"Unknown embedder type: {embedder_type}. Choose from: {list(embedders.keys())}")

    return embedders[embedder_type](**kwargs)