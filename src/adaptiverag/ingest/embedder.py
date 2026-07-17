import logging
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)


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
    

class AzureEmbedder(Embedder):
    """Embedding via the Azure OpenAI API.

    The heavy math runs on Azure's GPUs, not this container's fractional
    vCPU — a ~1,000-chunk document embeds in under a minute for a fraction
    of a cent, instead of ~10 minutes of local CPU grinding. Requires an
    EMBEDDING deployment on the Azure OpenAI resource (separate from the
    chat deployment), named by AZURE_OPENAI_EMBED_DEPLOYMENT.
    """

    # Known model dimensions; anything unknown falls back to a live probe.
    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    _MAX_BATCH = 2048        # Azure's per-request input cap

    def __init__(self, endpoint: str, api_key: str, deployment: str,
                 api_version: str = "2024-06-01"):
        self._client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version,
        )
        self._deployment = deployment
        self._dimension = self._DIMENSIONS.get(deployment, 0)   # 0 → probe lazily

    @property
    def dimension(self) -> int:
        if self._dimension == 0:                 # unknown model name: ask the API once
            self._dimension = len(self.embed("dimension probe"))
        return self._dimension

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text, model=self._deployment,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # One API call per _MAX_BATCH inputs (the ingest pipeline sends 32
        # at a time anyway, but be correct for any caller).
        out: list[list[float]] = []
        for start in range(0, len(texts), self._MAX_BATCH):
            response = self._client.embeddings.create(
                input=texts[start:start + self._MAX_BATCH],
                model=self._deployment,
            )
            out.extend(item.embedding for item in response.data)
        return out


def create_embedder(embedder_type: str = "local", **kwargs) -> Embedder:
    """Factory to create an embedder by type name.
    
    Args:
        embedder_type: "openai" or "local"
        **kwargs: passed to the embedder constructor (e.g. api_key, model_name)
    """
    embedders = {
        "openai": OpenAIEmbedder,
        "azure_openai": AzureEmbedder,
        "local": LocalEmbedder,
    }

    if embedder_type not in embedders:
        raise ValueError(f"Unknown embedder type: {embedder_type}. Choose from: {list(embedders.keys())}")

    return embedders[embedder_type](**kwargs)


def build_embedder_from_settings(settings) -> Embedder:
    """Pick the embedding backend from config, falling back safely to local.

    - `embeddings.provider: azure_openai` + full Azure creds + an embed
      deployment → AzureEmbedder (production: fast, offloaded, ~free).
    - Anything missing → LocalEmbedder (dev Macs, the future fully-local
      build, CI runners without the embed secret). The fallback is logged
      loudly because mixing backends mid-store corrupts retrieval: vectors
      of different dimensions can't share a collection.
    """
    provider = getattr(settings.embeddings, "provider", "local")
    if provider == "azure_openai":
        az = settings.azure
        embed_deployment = getattr(az, "embed_deployment", "")
        if az.endpoint and az.api_key and embed_deployment:
            logger.info("Embedder: azure_openai (deployment=%s)", embed_deployment)
            return AzureEmbedder(endpoint=az.endpoint, api_key=az.api_key,
                                 deployment=embed_deployment)
        logger.warning(
            "Embedder: azure_openai configured but endpoint/api_key/"
            "AZURE_OPENAI_EMBED_DEPLOYMENT missing — falling back to LOCAL "
            "embeddings. Do NOT mix backends against the same vector store.")
    return LocalEmbedder(model_name="all-MiniLM-L6-v2")