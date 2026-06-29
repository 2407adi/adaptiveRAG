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
from adaptiverag.reason.router import QueryRoute
from components import render_sources, render_grounding


from adaptiverag.pipeline import wire_pipeline
from adaptiverag.agents.tools import build_default_registry


def init_pipeline():
    """Initialize all RAG components once, store in session state."""
    if "initialized" in st.session_state:
        return  # already done — skip on re-runs

    bundle = wire_pipeline(
        settings,
        collection_name="streamlit_docs",
        persist_directory=PROJECT_ROOT / "data" / "chroma_store",
    )

    # ── Stash the wired components in session state ──
    st.session_state.embedder = bundle.embedder
    st.session_state.vector_store = bundle.vector_store
    st.session_state.pipeline = bundle.ingest          # ingest pipeline
    st.session_state.rag_chain = bundle.rag_chain
    st.session_state.tool_registry = build_default_registry(   # ← add this
    bundle.rag_chain,
    settings.tools,
    hmac_key=os.getenv("AUDIT_HMAC_KEY"),
    tavily_api_key=os.getenv("TAVILY_API_KEY"), )
    st.session_state.router = bundle.router
    st.session_state.multi_step_chain = bundle.multi_step_chain
    st.session_state.llm_client = bundle.llm_client
    st.session_state.grounding_validator = bundle.grounding_validator

    # ── UI-only state (not part of the pipeline) ──
    st.session_state.messages = []          # chat history
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

    # Push the fresh corpus summary onto the router so subsequent
    # queries route with up-to-date corpus awareness.
    if result.get("corpus_summary"):
        st.session_state.router.corpus_summary = result["corpus_summary"]

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

    # ── Render conversation history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("grounding"):
                render_grounding(msg["grounding"])

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

        # 2. Route and generate response
        with st.chat_message("assistant"):
            with st.spinner("Classifying question..."):
                route_result = st.session_state.router.classify(user_input)

            expand = st.session_state.get("expand_queries", False)

            if route_result.route == QueryRoute.DIRECT:
                with st.spinner("Thinking..."):
                    answer = st.session_state.llm_client.generate(user_input)
                    response = {"answer": answer, "sources": []}

            elif route_result.route == QueryRoute.MULTI_STEP:
                with st.spinner("Breaking down your question..."):
                    response = st.session_state.multi_step_chain.query(
                        user_input, expand=expand,
                    )

                # Show the reasoning trace
                if response.get("reasoning_steps"):
                    with st.expander("🧠 Reasoning Steps", expanded=False):
                        for i, step in enumerate(response["reasoning_steps"], 1):
                            st.markdown(f"**Step {i}: {step['sub_question']}**")
                            st.caption(step["answer"])
                            if i < len(response["reasoning_steps"]):
                                st.divider()
            else:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.query(
                        user_input, expand=expand,
                    )

            st.markdown(response["answer"])
            if response["sources"]:
                with st.spinner("Checking answer grounding..."):
                    grounding = st.session_state.grounding_validator.validate(
                        response["answer"], response["sources"]
                    )
                    response["grounding"] = grounding
                render_sources(response["sources"])
                render_grounding(response.get("grounding"))

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "grounding": response.get("grounding"),
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