"""Reusable UI components for AdaptiveRAG chat."""

import json

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

def render_grounding(grounding):
    """Display grounding validation results below an answer."""
    if grounding is None:
        return

    score = grounding.score
    total = grounding.total_claims
    grounded = grounding.grounded_claims

    # Color-coded header
    if grounding.is_grounded:
        icon = "🟢"
        label = "Well-grounded"
    elif score >= 0.4:
        icon = "🟡"
        label = "Partially grounded"
    else:
        icon = "🔴"
        label = "Poorly grounded"

    with st.expander(
        f"{icon} Grounding: {label} — {grounded}/{total} claims verified",
        expanded=False,
    ):
        st.progress(score)

        for v in grounding.verdicts:
            if v.status.value == "supported":
                st.markdown(f"✅ {v.claim}")
                if v.supporting_source:
                    st.caption(
                        f"Supported by: {v.supporting_source.get('source', '?')} "
                        f"(score: {v.max_score:.2f})"
                    )
            elif v.status.value == "contradicted":
                st.markdown(f"❌ {v.claim}")
                st.caption("Contradicts the source documents")
            else:
                st.markdown(f"⚠️ {v.claim}")
                st.caption("Not found in the source documents")


def _render_trace_entries(trace):
    """The Thought → Action → Observation lines themselves (no expander).
    Shared by render_agent_trace and render_supervisor_reports — Streamlit
    forbids nesting expanders, so the wrapper differs but the body is one."""
    for entry in trace:
        kind = entry.get("type")
        if kind == "thought":
            st.markdown(f"💭 **Thought:** {entry.get('content', '')}")
        elif kind == "action":
            st.markdown(f"🔧 **Action:** `{entry.get('tool', '')}`")
            args = entry.get("args") or {}
            if args:
                st.code(json.dumps(args, indent=2), language="json")
        elif kind == "observation":
            content = entry.get("content", "")
            if len(content) > 1500:                       # keep long tool output readable
                content = content[:1500] + " …"
            st.markdown("👁️ **Observation:**")
            st.caption(content)
        st.divider()


def render_agent_trace(trace, expanded=False):
    """Show the agent's Thought → Action → Observation trail in an expander.

    `trace` is the executor's scratchpad: a list of tagged dicts, each one of
    {"type": "thought"|"action"|"observation", ...}. This is the persisted
    "Thought Process" panel — it survives Streamlit reruns because it's stored
    on the chat message, not rebuilt from a transient expander.
    """
    if not trace:
        return

    with st.expander("🧠 Thought Process", expanded=expanded):
        _render_trace_entries(trace)


_AGENT_BADGES = {"retriever": "🔎", "reasoner": "🧮", "validator": "✅"}


def render_supervisor_reports(reports, expanded=False):
    """Block 3.4: the firm's case file — one section per junior dispatched,
    showing the polished report plus that junior's private notepad (trace).

    `reports` comes from SupervisorAgent results: a list of
    {"agent": name, "report": str, "trace": [scratchpad entries]}.
    Persisted on the chat message, same as render_agent_trace.
    """
    if not reports:
        return

    with st.expander(f"🕵️ Team Reports ({len(reports)})", expanded=expanded):
        for i, r in enumerate(reports, 1):
            agent = r.get("agent", "?")
            badge = _AGENT_BADGES.get(agent, "🕵️")
            st.markdown(f"#### {badge} {i}. {agent.capitalize()}'s report")
            st.markdown(r.get("report", "_(empty report)_"))
            trace = r.get("trace") or []
            if trace:
                st.markdown("**Notepad (working-out):**")
                _render_trace_entries(trace)
            if i < len(reports):
                st.divider()