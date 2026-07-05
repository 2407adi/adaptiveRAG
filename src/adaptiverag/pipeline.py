"""Single source of truth for pipeline construction.

wire_pipeline(settings, ...) builds and connects every component the RAG
pipeline needs and returns them in one Pipeline bundle. The UI, eval suite,
agent, and future API all call this instead of repeating the wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import uuid                                  # a per-session conversation id

from .config import Settings
from .ingest.loader import DocumentLoader
from .ingest.chunker import RecursiveChunker
from .ingest.embedder import create_embedder, Embedder
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
from .agents.memory import BufferMemory, VectorMemory, ConversationMemory

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
    memory: ConversationMemory | None = None        # Block 3.3 conversation-memory clerk


def wire_pipeline(
    settings: Settings,
    collection_name: str,
    persist_directory: str | Path,
) -> Pipeline:
    """Build and connect every pipeline component from settings.

    collection_name / persist_directory vary per caller (the UI uses a
    'streamlit_docs' store, the eval suite a hermetic 'eval_collection'),
    so they're explicit args; everything else comes from settings.
    """
    persist_directory = str(persist_directory)

    # 1. Embedder (local sentence-transformers, no API key)
    embedder = create_embedder("local", model_name="all-MiniLM-L6-v2")

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
        existing = vector_store.get_all()
        if existing:
            bm25.add(existing)      # restart-safe: rebuild BM25 from Chroma
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

    # 11. Conversation memory (Block 3.3): short-term buffer + long-term vector
    #     archive, behind the ConversationMemory clerk. Gated by config. Its OWN
    #     Chroma collection (separate drawer) in the SAME persist dir, so memory
    #     cards never mix with document chunks. Reuses the shared embedder.
    memory = None
    if settings.memory.enabled:
        memory_store = create_vector_store(
            backend="chroma",
            collection_name=settings.memory.collection_name,   # e.g. "conversation_memory"
            persist_directory=persist_directory,               # same room, different drawer
        )
        memory = ConversationMemory(
            buffer=BufferMemory(max_turns=settings.memory.max_turns),
            vector=VectorMemory(embedder=embedder, vector_store=memory_store),
            recall_k=settings.memory.recall_k,
            recall_score_threshold=settings.memory.recall_score_threshold,
            conversation_id=uuid.uuid4().hex,   # one id per wired session ≈ one conversation
        )

    # Wiring agents pipeline
    hmac_key = os.getenv("AUDIT_HMAC_KEY")
    tool_registry = None
    agent_executor = None
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
            memory=memory,        # Block 3.3 — agent reads the briefing + records each turn
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
        memory=memory,
    )