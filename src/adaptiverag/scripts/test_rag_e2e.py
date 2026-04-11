"""End-to-end smoke test: ingest sample docs → ask a question → get cited answer."""

from adaptiverag.ingest.loader import DocumentLoader
from adaptiverag.ingest.chunker import RecursiveChunker
from adaptiverag.ingest.embedder import LocalEmbedder
from adaptiverag.retrieve.vector_store import ChromaStore
from adaptiverag.ingest.pipeline import IngestPipeline
from adaptiverag.reason.chain import RAGChain


# ── Stub LLM (replace with real OpenAI client later) ──────
class StubLLM:
    """Echoes back the prompt so you can verify the chain wiring."""

    def generate(self, prompt: str) -> str:
        # Just return the last 500 chars of the prompt so you can see
        # the context + question that would be sent to the real LLM
        return f"[STUB LLM] Would send this to the model:\n\n{prompt[-500:]}"


# ── 1. Build components ───────────────────────────────────
loader = DocumentLoader()
chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
embedder = LocalEmbedder()  # uses all-MiniLM-L6-v2 locally, no API key
vector_store = ChromaStore(collection_name="e2e_test")

# ── 2. Ingest ─────────────────────────────────────────────
pipeline = IngestPipeline(loader, chunker, embedder, vector_store)
result = pipeline.ingest("data/sample")

print("=== INGEST ===")
print(f"Files processed: {result['files_processed']}")
print(f"Total chunks:    {result['total_chunks']}")
print(f"Chunks in store: {vector_store.count()}")
print()

# ── 3. Query ──────────────────────────────────────────────
chain = RAGChain(
    vector_store=vector_store,
    embedder=embedder,
    llm_client=StubLLM(),
    top_k=3,
)

question = "What are the key requirements of Basel III?"
response = chain.query(question)

print("=== QUERY ===")
print(f"Question: {question}")
print(f"\nAnswer:\n{response['answer']}")
print(f"\n=== SOURCES ({len(response['sources'])}) ===")
for s in response["sources"]:
    page_info = f", page {s['page']}" if s.get("page") else ""
    print(f"  - {s['source']} (chunk {s['chunk_index']}{page_info}, score: {s['score']})")
    print(f"    Preview: {s['text_preview'][:120]}...")