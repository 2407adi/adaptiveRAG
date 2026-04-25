"""
Configuration loader — merges .env secrets + YAML defaults
into a single Settings object the rest of the framework imports.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv
from typing import Optional

# ── Locate project root (two levels up from this file) ──
# src/adaptiverag/config.py  →  adaptiverag/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env() -> None:
    """Load .env file into environment variables."""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)


def _load_yaml() -> dict:
    """Load default.yaml and return as a dict."""
    yaml_path = PROJECT_ROOT / "config" / "default.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# ── Nested config dataclasses ───────────────────────────

@dataclass
class LLMConfig:
    provider: str = "azure_openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class EmbeddingsConfig:
    provider: str = "azure_openai"
    model: str = "text-embedding-ada-002"


@dataclass
class ChunkingConfig:
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class VectorDBConfig:
    provider: str = "chroma"
    collection_name: str = "default"


@dataclass
class HybridConfig:
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_sparse: float = 1.0


@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.7
    mode: str = "hybrid"
    hybrid: Optional[HybridConfig] = None

    def __post_init__(self):
        if self.hybrid is None:
            self.hybrid = HybridConfig()
        elif isinstance(self.hybrid, dict):
            self.hybrid = HybridConfig(**self.hybrid)


@dataclass
class AzureConfig:
    endpoint: str = ""
    api_key: str = ""
    deployment: str = ""

@dataclass
class RoutingConfig:
    fallback: str = "rag"
    examples: Optional[list[dict]] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


def load_settings() -> Settings:
    """Build Settings from .env + YAML."""

    # 1. Load secrets into env vars
    _load_env()

    # 2. Load YAML defaults
    cfg = _load_yaml()

    # 3. Build Settings — YAML values, with env overrides for secrets
    return Settings(
        llm=LLMConfig(**cfg.get("llm", {})),
        embeddings=EmbeddingsConfig(**cfg.get("embeddings", {})),
        chunking=ChunkingConfig(**cfg.get("chunking", {})),
        vector_db=VectorDBConfig(**cfg.get("vector_db", {})),
        retrieval=RetrievalConfig(**cfg.get("retrieval", {})),
        azure=AzureConfig(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        ),
        routing=RoutingConfig(**cfg.get("routing", {})),
    )


# ── Module-level singleton — import this everywhere ─────
settings = load_settings()