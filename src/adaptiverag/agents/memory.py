"""Conversation memory: what the detective remembers ACROSS questions.

Two tiers (Block 3.3):
  • BufferMemory — the desk CLIPBOARD: the last few exchanges, in RAM,
    wiped when we go home. A fixed-size sliding window.
  • VectorMemory — the basement ARCHIVE: every exchange filed by MEANING,
    on disk, recalled by relevance. (Next snippets.)

NB: this is NOT executor.py's `scratchpad`. The scratchpad is the detective's
rough working-out for ONE case (one question). This memory spans the whole
conversation (and, for the archive, across sessions). See CLAUDE.md Block 3.3.
"""

from __future__ import annotations

from collections import deque              # deque(maxlen=K) = the self-trimming clipboard
from dataclasses import dataclass, field
from datetime import datetime, timezone    # UTC timestamps, same style as audit.py

import uuid                                          # uuid4().hex = a unique catalog number per card
from ..retrieve.vector_store import StoredChunk, SearchResult      # the index-card shape you already use for docs


@dataclass
class Turn:
    """One page on the clipboard: a single thing that was said."""
    role: str                              # who spoke — "user" or "assistant"
    content: str                           # what they said (the message text)
    # When it was said. default_factory → a FRESH timestamp per Turn (never share
    # one mutable default across instances). isoformat string = JSON-friendly later.
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class BufferMemory:
    """The desk clipboard: the most recent K turns of THIS conversation.

    Short-term, in-RAM, forgetful by design — older pages slide off so the
    prompt never overflows the model's context window.
    """

    def __init__(self, max_turns: int = 10):
        # The clipboard itself. maxlen=K means: hold at most K pages; append a
        # K+1th and the OLDEST (front) page is dropped automatically — the
        # sliding window, for free. One exchange = 2 turns, so 10 ≈ 5 exchanges.
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        """Clip a new page onto the end. If the board is full, the oldest
        page slides off the front on its own (deque handles it)."""
        self._turns.append(Turn(role=role, content=content))

    def get(self) -> list[Turn]:
        """Read the whole board, oldest page first. list(...) takes a snapshot
        so the caller can't mutate our clipboard by holding the deque itself."""
        return list(self._turns)

    def recent(self, n: int) -> list[Turn]:
        """Just the last n pages (deques don't slice, so snapshot then slice).
        n larger than the board? [-n:] just returns everything — no error."""
        return list(self._turns)[-n:]

    def as_prompt(self) -> str:
        """Flatten the board into a readable transcript to paste into a prompt.

        One line per page: 'User: ...' / 'Assistant: ...'. Empty board → ""
        so the caller can cheaply skip the whole 'Conversation so far' section.
        """
        # .capitalize() so "user" reads as "User:" in the briefing.
        return "\n".join(f"{t.role.capitalize()}: {t.content}" for t in self._turns)

    def clear(self) -> None:
        """New client, fresh board — drop every page. (maxlen is preserved:
        deque.clear() empties it but keeps the same K limit.)"""
        self._turns.clear()

    def __len__(self) -> int:
        """How many pages are on the board right now — lets callers do
        `len(buffer)` and `if buffer:` naturally."""
        return len(self._turns)
    
class VectorMemory:
    """The basement archive: every turn filed by MEANING, on disk, recalled
    by relevance. Long-term, survives across sessions (Chroma persists to disk).

    The Embedder (stamp machine) and VectorStore (records room) are HANDED IN,
    never built here — same heavy-DI style as RAGChain / the eval suite. Keeps
    this domain-agnostic and easy to fake in tests.
    """

    def __init__(self, embedder, vector_store):
        self._embedder = embedder            # the meaning-stamp machine (.embed(text) -> vector)
        self._vector_store = vector_store    # the records cabinet (Chroma, persistent)

    def add(self, role: str, content: str, *, conversation_id: str | None = None) -> str:
        """File ONE turn as a meaning-stamped index card. Returns its catalog
        number (id) so a caller could fetch or delete it later."""
        # 1. Stamp the sentence with its meaning (a vector of numbers).
        vector = self._embedder.embed(content)

        # 2. Write the card's tab-labels. Chroma metadata must be scalars and
        #    can't be None — so we only pin conversation_id on when we have one.
        metadata: dict = {
            "role": role,                                          # "user" / "assistant" — lets us recall just questions later
            "timestamp": datetime.now(timezone.utc).isoformat(),   # same UTC style as Turn / audit.py
        }
        if conversation_id is not None:
            metadata["conversation_id"] = conversation_id

        # 3. Build the index card: readable text + its meaning-stamp + labels,
        #    under a unique catalog number so two identical sentences never clash.
        card_id = uuid.uuid4().hex
        chunk = StoredChunk(id=card_id, text=content, embedding=vector, metadata=metadata)

        # 4. Slot it into the cabinet. add() takes a LIST — we file one at a time.
        self._vector_store.add([chunk])
        return card_id
    
    def recall(
        self,
        query: str,
        k: int = 3,
        *,
        role: str | None = None,
        conversation_id: str | None = None,
        fetch_k: int | None = None,
    ) -> list["SearchResult"]:
        """Pull the k past turns closest in MEANING to `query`.

        Optional tab-filters (role / conversation_id) are applied AFTER the
        search, because the generic VectorStore only ranks by similarity — it
        can't filter by metadata. So we over-fetch, then keep matches. Returns
        SearchResult objects (reuse your existing shape: .text/.metadata/.score).
        """
        # 1. Speak the cabinet's language: stamp the query with its meaning.
        query_vector = self._embedder.embed(query)

        # 2. Over-fetch when we're going to filter — grab a wider stack so that
        #    after tossing off-tab cards we still have ~k left. No filter → just k.
        fetch = fetch_k or (k * 4 if (role or conversation_id) else k)
        results = self._vector_store.search(query_vector, fetch)

        # 3. Thumb through: keep only cards whose tab-labels match what we asked.
        kept = []
        for r in results:
            if role is not None and r.metadata.get("role") != role:
                continue                                  # wrong speaker — skip
            if conversation_id is not None and r.metadata.get("conversation_id") != conversation_id:
                continue                                  # different conversation — skip
            kept.append(r)

        # 4. Hand back the best k (search already returned them meaning-sorted).
        return kept[:k]
    

class ConversationMemory:
    """The memory clerk: one object over BOTH tiers, so callers never touch
    the buffer or vector store directly.

    Implements the Block 3.3 decision (auto-inject, Mem0 style):
      • add_turn()      — fan one turn into BOTH stores (parallel write)
      • build_context() — recency (buffer) + relevance (threshold-filtered
                          recall), as one ready-to-paste prompt block
    """

    def __init__(self, buffer: "BufferMemory", vector: "VectorMemory", *,
                 recall_k: int = 3, recall_score_threshold: float = 0.3,
                 conversation_id: str | None = None):
        self._buffer = buffer                          # the desk clipboard (short-term)
        self._vector = vector                          # the basement archive (long-term)
        self._recall_k = recall_k                      # how many cards the archivist pulls
        self._threshold = recall_score_threshold       # how relevant a card must be to make the briefing
        self._conversation_id = conversation_id        # which conversation these pages belong to

    def add_turn(self, role: str, content: str) -> None:
        """Clip the page on the desk AND file a copy in the archive — one call,
        both stores. The clipboard auto-drops its oldest; the archive keeps all."""
        self._buffer.add(role, content)
        self._vector.add(role, content, conversation_id=self._conversation_id)

    def build_context(self, query: str) -> str:
        """Staple the briefing for THIS query. Call it BEFORE add_turn(user, ...)
        so 'recent' doesn't echo the question we're about to answer."""
        sections = []

        # 1. RELEVANCE: archivist pulls cards close in meaning to the query, we
        #    keep only confident ones AND drop any that merely repeat the desk
        #    (no point injecting a card the clipboard already shows).
        on_desk = {t.content for t in self._buffer.get()}
        recalled = self._vector.recall(
            query, k=self._recall_k, conversation_id=self._conversation_id,
        )
        hits = [r for r in recalled if r.score >= self._threshold and r.text not in on_desk]
        if hits:
            lines = [f"- ({r.metadata.get('role', 'unknown')}) {r.text}" for r in hits]
            sections.append("Relevant earlier in this conversation:\n" + "\n".join(lines))

        # 2. RECENCY: the last few desk pages, verbatim.
        recent = self._buffer.as_prompt()
        if recent:
            sections.append("Recent conversation:\n" + recent)

        # Empty on turn one (nothing recalled, empty clipboard) → "" so the caller
        # can skip the whole 'what we already know' section cleanly.
        return "\n\n".join(sections)

    def clear(self) -> None:
        """New conversation: wipe the clipboard. The on-disk archive is untouched
        (that's the whole point of long-term memory)."""
        self._buffer.clear()