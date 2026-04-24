"""Reusable UI components for AdaptiveRAG chat."""

import streamlit as st


def render_sources(sources: list[dict]):
    """Display RAG sources as expandable sections below an answer."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            source_file = src.get("source", "unknown")
            page = src.get("page", "")
            chunk_idx = src.get("chunk_index", "?")
            score = src.get("score", 0.0)
            full_text = src.get("full_text", src.get("text_preview", ""))

            # Build a clickable link for the source file
            page_fragment = f"#page={page}" if page else ""
            file_url = f"app/static/uploads/{source_file}{page_fragment}"
            page_label = f" · page {page}" if page else ""

            st.markdown(
                f"**[{i}] [{source_file}]({file_url})**{page_label} · "
                f"chunk {chunk_idx} · score: {score:.3f}"
            )

            with st.container():
                st.caption(full_text)

            if i < len(sources):
                st.divider()