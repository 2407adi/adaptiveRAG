"""Async-ingestion drills: the throttled pipeline (batching, progress,
work cap) and the job store. All offline — fakes only, no models."""

from types import SimpleNamespace

import pytest

from adaptiverag.api.jobs import JobStore
from adaptiverag.ingest.exceptions import IngestTooLarge
from adaptiverag.ingest.pipeline import IngestPipeline


# ── fakes ───────────────────────────────────────────────────────────────────

class FakeLoader:
    def __init__(self, n_docs=1):
        self.n_docs = n_docs

    def load_directory(self, path):
        return [SimpleNamespace(text=f"doc {i} text", metadata={"source": f"doc{i}.txt"})
                for i in range(self.n_docs)]


class FakeChunker:
    """Emits a fixed number of chunks per document."""
    def __init__(self, per_doc=5):
        self.per_doc = per_doc

    def chunk(self, text, doc_id, metadata):
        return [SimpleNamespace(text=f"{doc_id} chunk {i}", doc_id=doc_id,
                                chunk_index=i, metadata=dict(metadata))
                for i in range(self.per_doc)]


class FakeEmbedder:
    """Counts embed_batch calls and records batch sizes — the batching pin."""
    def __init__(self):
        self.batch_sizes: list[int] = []

    def embed_batch(self, texts):
        self.batch_sizes.append(len(texts))
        return [[0.1, 0.2] for _ in texts]

    def embed(self, text):
        return [0.1, 0.2]


class FakeStore:
    def __init__(self):
        self.added = []

    def add(self, chunks):
        self.added.extend(chunks)


def make_pipeline(n_docs=1, per_doc=5, batch_size=32, throttle=0.0):
    embedder = FakeEmbedder()
    store = FakeStore()
    pipe = IngestPipeline(
        loader=FakeLoader(n_docs), chunker=FakeChunker(per_doc),
        embedder=embedder, vector_store=store,
        batch_size=batch_size, throttle_seconds=throttle,
    )
    return pipe, embedder, store


# ── throttled batching ──────────────────────────────────────────────────────

class TestBatchedEmbedding:
    def test_embeds_in_batches_not_one_giant_call(self):
        pipe, embedder, store = make_pipeline(n_docs=2, per_doc=50, batch_size=32)
        stats = pipe.ingest("whatever")
        assert stats["total_chunks"] == 100
        assert len(store.added) == 100
        assert embedder.batch_sizes == [32, 32, 32, 4]   # 100 chunks, batches of 32

    def test_result_shape_unchanged(self):
        # The receipt contract every existing caller depends on.
        pipe, _, _ = make_pipeline(n_docs=3, per_doc=2)
        stats = pipe.ingest("whatever", scope="chat:abc")
        assert stats["files_processed"] == 3
        assert stats["total_chunks"] == 6
        assert "corpus_summary" in stats

    def test_scope_still_stamped_on_every_chunk(self):
        pipe, _, store = make_pipeline(n_docs=1, per_doc=3)
        pipe.ingest("whatever", scope="chat:xyz")
        assert all(c.metadata["scope"] == "chat:xyz" for c in store.added)


class TestProgressCallback:
    def test_stages_and_counts_reported_in_order(self):
        pipe, _, _ = make_pipeline(n_docs=1, per_doc=70, batch_size=32)
        events = []
        pipe.ingest("whatever", progress_cb=lambda s, d, t: events.append((s, d, t)))
        stages = [e[0] for e in events]
        assert stages[0] == "loading" and stages[1] == "chunking"
        # embedding narrated batch by batch, cumulative counts, honest total
        assert [e for e in events if e[0] == "embedding"] == [
            ("embedding", 32, 70), ("embedding", 64, 70), ("embedding", 70, 70)]
        assert ("storing", 70, 70) in events

    def test_no_callback_is_fine(self):
        pipe, _, _ = make_pipeline()
        assert pipe.ingest("whatever")["total_chunks"] == 5


class TestWorkCap:
    def test_over_cap_raises_before_any_embedding(self):
        pipe, embedder, store = make_pipeline(n_docs=1, per_doc=200)
        with pytest.raises(IngestTooLarge) as exc:
            pipe.ingest("whatever", max_chunks=150)
        assert "200 chunks" in str(exc.value) and "150" in str(exc.value)
        assert embedder.batch_sizes == []        # cap fired BEFORE the expensive part
        assert store.added == []                 # and nothing was written

    def test_at_cap_passes(self):
        pipe, _, _ = make_pipeline(n_docs=1, per_doc=150)
        assert pipe.ingest("whatever", max_chunks=150)["total_chunks"] == 150

    def test_none_means_uncapped(self):
        pipe, _, _ = make_pipeline(n_docs=1, per_doc=200)
        assert pipe.ingest("whatever", max_chunks=None)["total_chunks"] == 200


# ── job store ───────────────────────────────────────────────────────────────

class TestJobStore:
    def test_create_get_update_roundtrip(self):
        jobs = JobStore()
        job = jobs.create(files=["a.pdf"])
        assert jobs.get(job.id).status == "queued"
        jobs.update(job.id, status="running", stage="embedding",
                    chunks_done=10, chunks_total=100)
        got = jobs.get(job.id)
        assert (got.status, got.stage, got.chunks_done) == ("running", "embedding", 10)

    def test_unknown_job_is_none_and_update_is_calm(self):
        jobs = JobStore()
        assert jobs.get("nope") is None
        jobs.update("nope", status="done")       # must not raise

    def test_finished_jobs_evicted_past_cap_running_kept(self):
        jobs = JobStore(keep=3)
        old = [jobs.create(files=[f"{i}.txt"]) for i in range(3)]
        for j in old[:2]:
            jobs.update(j.id, status="done")
        jobs.update(old[2].id, status="running")
        newest = jobs.create(files=["new.txt"])  # pushes past keep=3
        assert jobs.get(newest.id) is not None
        assert jobs.get(old[2].id) is not None   # running job survives eviction
        assert jobs.get(old[0].id) is None       # oldest finished job evicted
