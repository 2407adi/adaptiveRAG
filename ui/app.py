"""AdaptiveRAG — Streamlit Chat UI."""

import sys
from pathlib import Path

# ── Make the src package importable from ui/ ──────────────
# ui/app.py lives at adaptiveRAG/ui/app.py
# src/adaptiverag/ lives at adaptiveRAG/src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st
import tempfile
import os

from adaptiverag.config import settings
from adaptiverag.ingest.loader import DocumentLoader
from adaptiverag.ingest.chunker import RecursiveChunker
from adaptiverag.ingest.embedder import create_embedder
from adaptiverag.ingest.pipeline import IngestPipeline
from adaptiverag.retrieve.vector_store import create_vector_store
from adaptiverag.reason.chain import RAGChain
from adaptiverag.retrieve.query_expander import QueryExpander

from llm_client import AzureLLMClient
from components import render_sources


def init_pipeline():
    """Initialize all RAG components once, store in session state."""

    if "initialized" in st.session_state:
        return  # already done — skip on re-runs

    # 1. Embedder (local sentence-transformers, no API key needed)
    embedder = create_embedder("local", model_name="all-MiniLM-L6-v2")

    # 2. Vector store (Chroma, persistent so docs survive re-runs)
    persist_dir = str(PROJECT_ROOT / "data" / "chroma_store")
    vector_store = create_vector_store(
        backend="chroma",
        collection_name="streamlit_docs",
        persist_directory=persist_dir,
    )

    # 3. Chunker (recursive, from your config defaults)
    chunker = RecursiveChunker(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )

    # 4. Loader
    loader = DocumentLoader()

    # 5. Ingest pipeline
    pipeline = IngestPipeline(loader, chunker, embedder, vector_store)

    # 6. LLM client (Azure OpenAI)
    llm_client = AzureLLMClient(
        endpoint=settings.azure.endpoint,
        api_key=settings.azure.api_key,
        deployment=settings.azure.deployment,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )

    # 7. Query expander (uses same LLM client)
    query_expander = QueryExpander(llm_client)

    # 8. RAG chain
    rag_chain = RAGChain(
        vector_store=vector_store,
        embedder=embedder,
        llm_client=llm_client,
        top_k=settings.retrieval.top_k,
        query_expander=query_expander,
    )

    # ── Stash everything in session state ──
    st.session_state.embedder = embedder
    st.session_state.vector_store = vector_store
    st.session_state.pipeline = pipeline
    st.session_state.rag_chain = rag_chain
    st.session_state.messages = []        # chat history
    st.session_state.ingested_files = set()  # track what's been uploaded
    st.session_state.initialized = True

def ingest_uploads(files):
    """Save uploaded files to static/uploads/, then run the ingest pipeline."""

    with st.spinner("Ingesting documents..."):
        # Persistent directory so files stay accessible for clickable links
        upload_dir = PROJECT_ROOT / "ui" / "static" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write uploaded files to disk
        for f in files:
            file_path = upload_dir / f.name
            file_path.write_bytes(f.getvalue())

        # 2. Run the pipeline on the uploads directory
        result = st.session_state.pipeline.ingest(str(upload_dir))

        # 3. Track what we've ingested
        for f in files:
            st.session_state.ingested_files.add(f.name)

    st.sidebar.success(
        f"Done! Processed {result['files_processed']} file(s), "
        f"{result['total_chunks']} chunks indexed."
    )

def render_sidebar():
    """Sidebar: file upload, ingestion trigger, and status."""

    with st.sidebar:
        st.header("📄 Document Upload")

        uploaded_files = st.file_uploader(
            "Drop your documents here",
            type=["pdf", "txt", "md", "csv", "html", "docx"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            # Filter out files we've already ingested
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.ingested_files
            ]

            if new_files:
                if st.button(f"Ingest {len(new_files)} new file(s)"):
                    ingest_uploads(new_files)
            else:
                st.info("All uploaded files already ingested.")

        # ── Status section ──
        st.divider()
        st.subheader("Status")
        chunk_count = st.session_state.vector_store.count()
        file_count = len(st.session_state.ingested_files)
        st.metric("Files ingested", file_count)
        st.metric("Chunks indexed", chunk_count)
        # ── Retrieval settings ──
        st.divider()
        st.subheader("Settings")
        st.session_state.expand_queries = st.toggle(
            "Smart query expansion",
            value=False,
            help="Uses the LLM to rewrite your question with synonyms "
                 "and technical terms before searching. May improve "
                 "results for vague or short queries.",
        )

def render_chat():
    """Main chat area: display history, handle new input."""

    st.title("AdaptiveRAG Chat")

    # ── Guard: need documents before chatting ──
    if st.session_state.vector_store.count() == 0:
        st.info("Upload and ingest some documents to start chatting.")
        return

    # ── Render conversation history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    # ── Handle new user input ──
    if user_input := st.chat_input("Ask a question about your documents..."):
        # 1. Show the user's message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "sources": [],
        })

        # 2. Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.query(
                    user_input,
                    expand=st.session_state.get("expand_queries", False),
                )

            st.markdown(response["answer"])
            if response["sources"]:
                render_sources(response["sources"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })

def main():
    """App entry point — configure page, init, render."""

    st.set_page_config(
        page_title="AdaptiveRAG",
        page_icon="🔍",
        layout="wide",
    )

    # Wire up pipeline (runs once per session)
    init_pipeline()

    # Render the two panels
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()