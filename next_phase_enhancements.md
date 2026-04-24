# Next Phase Enhancements — AdaptiveRAG

Compiled from ALL build sessions (Blocks 0.1 through 1.7), April 2026.
Every time something was flagged as "not now, do it later" during development.

For each item: what the issue is, which future phase (if any) addresses it in the project plan, and whether it needs separate attention.

---

## RAG Quality Improvements (Before Moving to Phase 2)

These are issues discovered during Phase 1 that affect retrieval quality. Aditya's preference is to have a solid RAG before moving to Phases 2+. These should be addressed first.

### 1. PDF Header/Footer Boilerplate Drowning Out Real Content

**Source:** Block 1.5 (RAG Chain) — end-to-end test with Basel III PDF.

**The problem:** When querying "What are the key requirements of Basel III?", the top 3 results were all chunk-0 from different pages — because every page starts with the same header text ("Basel III: A global regulatory framework for more resilient banks and banking sy..."). The header is a strong semantic match for the query, so it drowns out the actual substantive content.

**What to build:**
- Strip or de-weight header/footer boilerplate during PDF extraction (detect repeated text at start/end of pages)
- Alternatively, use MMR (Maximal Marginal Relevance) during retrieval to penalize redundant results — if result 2 is too similar to result 1, skip it and find something more diverse
- Could also add metadata filtering: "don't return multiple chunk-0s from different pages of the same file"

**Addressed in project plan?** Partially — Block 2.5 (Reranker) would help re-score results, and the reranker could learn to deprioritize boilerplate. But the root cause (boilerplate in the extracted text) should be fixed at the ingestion layer.

**Recommendation:** Fix this before Phase 2. Either strip repeated headers during PDF extraction, or add MMR to retrieval. This is a ~2 hour fix that dramatically improves retrieval quality.

### 2. OCR for Scanned PDFs

**Source:** Block 1.1 (Document Loader) — PDF extractor returns empty text for image-only pages.

**The problem:** `pypdf`'s `extract_text()` returns `None` for scanned pages. The current code silently skips them (`if text.strip():`), which means scanned documents are partially or fully invisible to the RAG system.

**What to build:**
- Add OCR fallback using `pytesseract` or `easyocr` when `extract_text()` returns empty
- Detect whether a page is text-based or scanned, and route accordingly

**Addressed in project plan?** No — not explicitly mentioned in any block. This is a gap.

**Recommendation:** Add this when you have a use case with scanned docs. Low priority if all your test documents are digital PDFs.

### 3. DOCX Table Extraction

**Source:** Block 1.1 (Document Loader) — DOCX extractor only reads paragraphs.

**The problem:** `python-docx` gives you `doc.paragraphs` and `doc.tables` separately. Current code only reads paragraphs, so any data in Word tables is invisible to the RAG system.

**What to build:**
- Iterate `doc.tables` and convert each table to readable text (similar to CSV approach — "Column1: Value | Column2: Value")
- Append table text to the document or create separate Document objects per table

**Addressed in project plan?** No — not mentioned in any block.

**Recommendation:** Quick fix (~30 min). Do it before Phase 2 since it's a data completeness issue.

---

## Phase 2 Enhancements (Reasoning & Evaluation)

These are items that align with Phase 2 blocks in the project plan.

### 4. Multi-turn Conversation Awareness

**Source:** Block 1.7 (Streamlit Chat UI) — chat interface stores history but RAGChain ignores it.

**The problem:** `RAGChain.query()` treats each question independently. Follow-up questions like "Can you elaborate on the second point?" fail because the chain has no context about what "the second point" refers to.

**What to build:**
- **Simple:** Append last N messages to `_build_prompt()` as chat context
- **Better:** Query rewriting — use the LLM to rewrite "elaborate on the second point" into a standalone query like "Explain the second requirement of Basel III section 3.2" before retrieval

**Addressed in project plan?** Partially — Block 2.1 (Query Router) routes queries by complexity, and Block 2.2 (Chain-of-Thought) decomposes complex queries. But neither specifically handles conversational context. Query rewriting should be added to the router or as a pre-processing step.

### 5. Knowledge Graph Memory Layer (Cognee-style ECL)

**Source:** Block 1.7 — discussion of LinkedIn post about multi-hop reasoning failures.

**The problem:** Vector search alone can't answer questions that require connecting facts across multiple hops (e.g., Mark → Grade 10 → March exams → library closing). The middle link gets missed because it's semantically distant from the query.

**What to build:**
- Graph store (NetworkX local, Neo4j production) alongside vector store
- Entity and relationship extraction during ingestion via LLM calls
- Hybrid retrieval: vector + graph queries merged

**Addressed in project plan?** Yes — Phase 5 mentions Knowledge Graphs in the skill coverage matrix, and the project plan's overall architecture mentions a reasoning layer. But no specific block is dedicated to building a knowledge graph. This would be a significant addition.

**Recommendation:** This is an impressive portfolio differentiator but adds 1-2 weeks of work. Consider it as a Phase 5 showcase feature rather than blocking Phase 2/3.

---

## Phase 3 Enhancements (Agents)

### 6. Conversation Memory (Short-term + Long-term)

**Source:** Block 1.7 — discussion of "LLM memory hype" and the four flavors of memory.

**The problem:** Currently no memory system exists. The agent can't remember past conversations or user preferences across sessions.

**Addressed in project plan?** Yes — Block 3.3 (Conversation Memory) explicitly covers this: BufferMemory (sliding window for current conversation) + VectorMemory (long-term, persisted across sessions).

**Recommendation:** This is already planned. No gap.

---

## Phase 4 Enhancements (Production)

### 7. FAISS Idempotent Upsert

**Source:** Block 1.4 (Vector Store) — noted that FAISS `add` is NOT idempotent.

**The problem:** If you add the same chunk twice to FAISS, it stores duplicate vectors. ChromaDB's `upsert` handles this gracefully, but FAISS doesn't. This means re-running ingestion on the same documents creates duplicates.

**What to build:**
- Check if a string ID already exists in `_str_to_int` before adding
- If it exists, update the vector and metadata instead of adding a new entry

**Addressed in project plan?** Not explicitly, but Block 4.1-4.5 (Production Hardening) is where this kind of robustness fix belongs.

**Recommendation:** Quick fix (~30 min). Do it when FAISS is actively being used. For now, ChromaDB is the default and handles this fine.

### 8. Per-User Collection Isolation

**Source:** Block 1.4 (Vector Store) — discussion about whether to use multiple ChromaDB collections.

**The problem:** When multiple users upload different documents, their data needs isolation. Options: one collection per user, or one collection with metadata filtering.

**Addressed in project plan?** Yes — Block 4.2 (Authentication & RBAC) handles user-level access control, and the architecture naturally supports per-user collections via the `collection_name` parameter.

**Recommendation:** Already designed for. The `ChromaStore(collection_name=...)` parameter makes this a config-level decision, not a code change.

---

## Architectural Decisions (Ruled Out for Now)

### 9. React/Next.js Frontend

**Source:** Block 1.7 — Aditya considered a JS/TS frontend.

**Decision:** Stick with Streamlit. A JS frontend adds 3-4x complexity (FastAPI backend, CORS, Node project, TypeScript, two deployments) for equivalent functionality.

**When to revisit:** If the project needs production-grade custom UI, real-time streaming, or auth flows beyond what Streamlit supports.

---

## Recommendations: What to Do Next

Given Aditya's preference to have a **solid, working RAG** before moving to Phase 2, here's the recommended order:

### Step 1: Finish & Verify Block 1.7 (Current)
Get the Streamlit UI running end-to-end. Upload docs → ingest → chat → see citations.

### Step 2: Fix RAG Quality (1-2 sessions, ~3-4 hours)
Before Phase 2, fix these Phase 1 gaps that directly hurt retrieval quality:

1. **PDF header/footer stripping or MMR** — This is the biggest bang-for-buck improvement. Your Basel III test showed 3 identical header chunks as top results. Fix this and retrieval quality jumps immediately.
2. **DOCX table extraction** — 30-minute fix, ensures no data is silently dropped.
3. **Add a pre-retrieval LLM layer** — Use the LLM to rewrite/expand the user's query before embedding and searching. This is a lightweight way to improve retrieval without building the full Query Router (Block 2.1). Example: user asks "what about capital requirements?" → LLM rewrites to "What are the minimum capital requirements and capital adequacy ratios specified in the Basel III framework?" → much better retrieval. This is sometimes called "HyDE" (Hypothetical Document Embeddings) or simply query expansion.

### Step 3: Phase 2 — Reasoning & Evaluation (Blocks 2.1–2.5)
With solid RAG in place, layer on:
- Query Router (2.1) — classify queries into DIRECT / RAG / MULTI_STEP
- Chain-of-Thought (2.2) — decompose complex queries
- Grounding Validator (2.3) — catch hallucinations
- Eval Suite (2.4) — measure quality with RAGAS metrics
- Reranker (2.5) — cross-encoder re-scoring

### Step 4: Phase 3 — Agents (Blocks 3.1–3.4)
This is where the project goes from "impressive" to "portfolio-defining."

### Knowledge Graph (Enhancement #5)
Consider adding after Phase 3 as a showcase feature. It maps to the "Knowledge Graphs" skill in the JD analysis (KPMG wants this) and would differentiate your project from every other RAG tutorial on GitHub.
