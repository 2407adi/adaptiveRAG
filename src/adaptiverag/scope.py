"""Scope stamps + guest lists for chat-scoped retrieval (Block 4.2b).

Story: every book entering the library gets an ink stamp — "shared" for the
public shelf (seed/demo corpus), "chat:<id>" for that chat's private locker.
Retrieval requests carry a *guest list* of allowed stamps; both librarians
(dense + BM25) refuse to hand over books whose stamp isn't on the list.

This module is the ONLY place the stamp format lives. Filter code stays
ignorant of what kind of id sits inside a stamp, so a later "Projects"
feature can mint "workspace:<id>" stamps with zero changes to the machinery.
"""

from __future__ import annotations

from contextvars import ContextVar

SHARED_SCOPE = "shared"          # the public shelf: visible in every chat


def chat_scope(conversation_id: str) -> str:
    """Mint the locker stamp for one chat."""
    return f"chat:{conversation_id}"


def scopes_for(conversation_id: str | None) -> list[str] | None:
    """Build a request's guest list: the public shelf + this chat's locker.

    None in -> None out (no filter at all): the eval suite, the Streamlit
    dev harness, and every pre-4.2b caller keep seeing everything, exactly
    as before.
    """
    if conversation_id is None:
        return None
    return [SHARED_SCOPE, chat_scope(conversation_id)]


# Ambient guest list for the AGENT path. The LLM only ever fills in `query`
# when it calls search_documents — it must never see (or be able to invent)
# a scope. So the API layer sets this context variable per request, and the
# search_documents tool reads it at call time. Default None = unscoped
# (local dev, tests, Streamlit harness).
current_scopes: ContextVar[list[str] | None] = ContextVar(
    "current_scopes", default=None
)
