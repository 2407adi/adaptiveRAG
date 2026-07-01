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
class RerankConfig:
    enabled: bool = False
    backend: str = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fetch_k: int = 20

@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.7
    mode: str = "hybrid"
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)

    def __post_init__(self):
        if self.hybrid is None:
            self.hybrid = HybridConfig()
        elif isinstance(self.hybrid, dict):
            self.hybrid = HybridConfig(**self.hybrid)

        if self.rerank is None:
            self.rerank = RerankConfig()
        elif isinstance(self.rerank, dict):
            self.rerank = RerankConfig(**self.rerank)


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
class AgentConfig:
    max_iterations: int = 6                     # the detective's move budget — REASON checks it, ACT ticks it
    require_approval: list[str] = field(        # specialists that need the supervisor's warrant before ACT
        default_factory=lambda: ["run_python", "web_search"]
    )
    # No __post_init__ needed: YAML hands these over as a plain int and a
    # plain list-of-strings — already the right types, nothing to coerce.
    # (Contrast ToolsConfig, which DOES coerce because its values are nested dicts.)

@dataclass
class SandboxConfig:
    timeout: float = 5.0        # wall-clock seconds before the locked room is evicted
    cpu_seconds: int = 2        # CPU-time cap the building manager (OS) enforces
    max_memory_mb: int = 256    # the room's whiteboard (memory) cap, in megabytes


@dataclass
class TavilyConfig:
    enabled: bool = True        # whether to offer web_search (it still degrades safely without a key)
    max_results: int = 3        # how many web snippets the researcher returns


@dataclass
class AuditConfig:
    path: str = "data/audit/tool_calls.jsonl"   # where the tamper-evident logbook file is written


@dataclass
class ToolsConfig:
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)   # house rules for the locked room
    tavily: TavilyConfig = field(default_factory=TavilyConfig)      # house rules for the web researcher
    audit: AuditConfig = field(default_factory=AuditConfig)         # where the logbook lives

    def __post_init__(self):
        # YAML hands nested blocks over as plain dicts; coerce each into its
        # dataclass — the SAME trick RetrievalConfig uses for hybrid/rerank.
        if isinstance(self.sandbox, dict):
            self.sandbox = SandboxConfig(**self.sandbox)
        if isinstance(self.tavily, dict):
            self.tavily = TavilyConfig(**self.tavily)
        if isinstance(self.audit, dict):
            self.audit = AuditConfig(**self.audit)


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)   # the detective's house rules


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
        tools=ToolsConfig(**cfg.get("tools", {})),
        agent=AgentConfig(**cfg.get("agent", {})),   # pull the `agent:` YAML block, fall back to defaults
    )


# ── Module-level singleton — import this everywhere ─────
settings = load_settings()