"""Single source of truth for pipeline construction.

wire_pipeline(settings, ...) builds and connects every component the RAG
pipeline needs and returns them in one Pipeline bundle. The UI, eval suite,
agent, and future API all call this instead of repeating the wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import os
import threading
import time

from .config import Settings
from .ingest.loader import DocumentLoader
from .ingest.chunker import RecursiveChunker
from .ingest.embedder import build_embedder_from_settings, Embedder
from .ingest.pipeline import IngestPipeline
from .ingest.summarizer import CorpusSummarizer
from .retrieve.vector_store import create_vector_store, VectorStore
from .retrieve.query_expander import QueryExpander
from .retrieve.reranker import build_reranker_from_settings
from .reason.chain import RAGChain, MultiStepChain
from .reason.router import QueryRouter
from .reason.grounding import GroundingValidator
from .llm_client import AzureLLMClient
from .retrieve.hybrid import BM25Retriever, HybridRetriever

from .agents.tools import build_default_registry, ToolRegistry
from .agents.executor import AgentExecutor
from .agents.supervisor import SupervisorAgent
from .agents.memory import MemoryManager    # was: BufferMemory, VectorMemory, ConversationMemory

logger = logging.getLogger(__name__)


def seed_bm25(bm25: BM25Retriever, vector_store) -> int:
    """Rebuild the in-RAM BM25 index from every chunk persisted in the store.

    Extracted from wire_pipeline so it can run either synchronously (eval
    suite, benchmark, Streamlit — where reproducibility demands the index
    is ready before the first query) or on a background thread (the API,
    where boot time is probed and a big store must never block /health).
    Returns the number of chunks indexed.
    """
    started = time.time()
    existing = vector_store.get_all()          # the slow part on a big store
    if existing:
        bm25.add(existing)
    logger.info("BM25 index seeded: %d chunks in %.1fs",
                len(existing), time.time() - started)
    return len(existing)


@dataclass
class Pipeline:
    """Every wired component, in one bundle.

    Callers pull out the pieces they need: the UI stashes them in
    session_state, the eval suite hands them to EvalSuite, the agent
    will grab rag_chain for retrieve().
    """
    embedder: Embedder
    vector_store: VectorStore
    llm_client: AzureLLMClient
    summarizer: CorpusSummarizer
    ingest: IngestPipeline           # the load→chunk→embed→store pipeline
    rag_chain: RAGChain
    multi_step_chain: MultiStepChain
    router: QueryRouter
    grounding_validator: GroundingValidator
    tool_registry: ToolRegistry | None = None      # Block 3.1 front desk (shared + audited)
    agent_executor: AgentExecutor | None = None    # Block 3.2 ReAct detective
    memory_manager: MemoryManager | None = None
    supervisor_agent: SupervisorAgent | None = None  # Block 3.4 the firm (Chief + 3 juniors)


def wire_pipeline(
    settings: Settings,
    collection_name: str,
    persist_directory: str | Path,
    lazy_bm25: bool = False,
) -> Pipeline:
    """Build and connect every pipeline component from settings.

    collection_name / persist_directory vary per caller (the UI uses a
    'streamlit_docs' store, the eval suite a hermetic 'eval_collection'),
    so they're explicit args; everything else comes from settings.

    lazy_bm25=True moves the BM25 boot seed onto a background daemon
    thread so wiring returns immediately no matter how big the store is.
    The API uses this: a fat store must never make boot slower than the
    startup probe's patience (the boot-loop failure mode). Until the seed
    finishes, hybrid search transparently serves dense-only results.
    Default False: eval/benchmark/Streamlit keep the synchronous seed —
    reproducible runs need the index ready before the first query.
    """
    persist_directory = str(persist_directory)

    # 1. Embedder — config-driven backend. Production: azure_openai (the
    #    math runs on Azure's GPUs; ~1,000 chunks in <1 min for pennies).
    #    Missing creds/deployment, dev Macs, the future fully-local build:
    #    falls back to local sentence-transformers automatically.
    #    ⚠ Backends have different vector dimensions (384 vs 1536) — never
    #    switch backends over an existing store; wipe + re-ingest instead.
    embedder = build_embedder_from_settings(settings)

    # 2. Vector store (Chroma, persistent)
    vector_store = create_vector_store(
        backend="chroma",
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    # 2b. Hybrid retrieval — built ONLY when config asks for it.
    #     One shared BM25 instance: seeded now from any chunks persisted
    #     in past sessions, kept in sync by the ingest pipeline (5c), and
    #     searched by the HybridRetriever. None in dense mode.
    bm25 = None
    hybrid_retriever = None
    if settings.retrieval.mode == "hybrid":
        bm25 = BM25Retriever()
        if lazy_bm25:
            # Background seed: boot finishes now, keyword search joins in
            # when ready. BM25Retriever is thread-safe (lock + dedupe), so
            # an ingest landing mid-seed can't corrupt the index.
            threading.Thread(
                target=seed_bm25, args=(bm25, vector_store),
                daemon=True, name="bm25-seed",
            ).start()
        else:
            seed_bm25(bm25, vector_store)   # restart-safe: rebuild BM25 from Chroma
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25=bm25,
            embedder=embedder,
            rrf_k=settings.retrieval.hybrid.rrf_k,
            weight_dense=settings.retrieval.hybrid.weight_dense,
            weight_sparse=settings.retrieval.hybrid.weight_sparse,
        )

    # 3. LLM client — built early; summarizer + chains all need it
    llm_client = AzureLLMClient(
        endpoint=settings.azure.endpoint,
        api_key=settings.azure.api_key,
        deployment=settings.azure.deployment,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )

    # 4. Corpus summarizer (sidecar file next to the chroma store)
    summarizer = CorpusSummarizer(
        llm_client=llm_client,
        persist_path=Path(persist_directory) / "_corpus_summary.txt",
    )

    # 5. Ingest pipeline (load → chunk → embed → store, + summary)
    ingest = IngestPipeline(
            loader=DocumentLoader(),
            chunker=RecursiveChunker(
                chunk_size=settings.chunking.chunk_size,
                chunk_overlap=settings.chunking.chunk_overlap,
            ),
            embedder=embedder,
            vector_store=vector_store,
            summarizer=summarizer,
            bm25=bm25,                 # syncs the keyword index on every ingest
        )

    # 6. Retrieval add-ons
    query_expander = QueryExpander(llm_client)
    reranker = build_reranker_from_settings(settings.retrieval.rerank)

    # 7. RAG chain (dense retrieval for now; hybrid wired in Step 5)
    rag_chain = RAGChain(
            vector_store=vector_store,
            embedder=embedder,
            llm_client=llm_client,
            top_k=settings.retrieval.top_k,
            query_expander=query_expander,
            reranker=reranker,
            fetch_k=settings.retrieval.rerank.fetch_k,
            hybrid_retriever=hybrid_retriever,   # None → dense-only
        )

    # 8. Router (corpus-aware: seeded from disk if a prior session summarized)
    router = QueryRouter(
        llm_client=llm_client,
        examples=settings.routing.examples,
        corpus_summary=summarizer.load(),
    )

    # 9. Multi-step chain (complex MULTI_STEP queries)
    multi_step_chain = MultiStepChain(
        rag_chain=rag_chain,
        llm_client=llm_client,
        max_sub_questions=4,
    )

    # 10. Grounding validator (hallucination detection)
    grounding_validator = GroundingValidator(llm_client=llm_client, threshold=0.6)

    # 11. Conversation memory (3.3 → 4.3a): the notebook RACK. One shared archive
    #     (its own Chroma drawer, cards stamped per chat) + one private clipboard
    #     per conversation, minted on first request. No more session-wide id.
    memory_manager = None
    if settings.memory.enabled:
        memory_store = create_vector_store(
            backend="chroma",
            collection_name=settings.memory.collection_name,
            persist_directory=persist_directory,        # same room, different drawer (unchanged)
        )
        memory_manager = MemoryManager(
            embedder=embedder,
            vector_store=memory_store,
            max_turns=settings.memory.max_turns,
            recall_k=settings.memory.recall_k,
            recall_score_threshold=settings.memory.recall_score_threshold,
        )

    # Wiring agents pipeline
    hmac_key = os.getenv("AUDIT_HMAC_KEY")
    tool_registry = None
    agent_executor = None
    supervisor_agent = None
    if hmac_key:
        tool_registry = build_default_registry(
            rag_chain, settings.tools,
            hmac_key=hmac_key,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )
        agent_executor = AgentExecutor(
            llm_client, tool_registry,
            max_iterations=settings.agent.max_iterations,
            require_approval=settings.agent.require_approval,
            memory_manager=memory_manager,   # the rack; the detective pulls per-ticket notebooks himself
        )
        # Block 3.4 — the firm. SAME registry (shared audit log), same approval
        # house rule; only the org chart differs. UI toggles between the two.
        # 4.3a: SAME rack too — toggle modes mid-chat and the memory follows,
        # because both desks pull the same per-ticket notebook.
        supervisor_agent = SupervisorAgent(
            llm_client, tool_registry,
            max_handoffs=settings.agent.max_handoffs,
            worker_iterations=settings.agent.worker_iterations,
            require_approval=settings.agent.require_approval,
            memory_manager=memory_manager,
        )

    return Pipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm_client=llm_client,
        summarizer=summarizer,
        ingest=ingest,
        rag_chain=rag_chain,
        multi_step_chain=multi_step_chain,
        router=router,
        grounding_validator=grounding_validator,
        tool_registry=tool_registry,
        agent_executor=agent_executor,
        memory_manager=memory_manager,
        supervisor_agent=supervisor_agent,
    )