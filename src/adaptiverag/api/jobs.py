"""In-memory job tracker for asynchronous ingestion.

One IngestJob per upload: /ingest creates it and returns 202 + the id;
a background thread runs the actual pipeline and updates the record;
GET /ingest/status/{id} reads it. Deliberately in-memory (a dict) — if
the container restarts mid-job, the job is gone and the status endpoint
returns 404, which the UI translates to "server restarted during
ingestion, please retry". Persisting jobs would imply resumable ingests,
which this box doesn't need.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class IngestJob:
    id: str
    files: list[str]
    status: str = "queued"        # queued | running | done | failed
    stage: str | None = None      # loading | chunking | embedding | storing | summarizing
    chunks_done: int = 0
    chunks_total: int = 0
    error: str | None = None
    result: dict | None = None    # IngestPipeline.ingest() receipt, set on done
    created_at: float = field(default_factory=time.time)


class JobStore:
    """Thread-safe registry of ingest jobs (background threads write,
    request handlers read). Keeps only the most recent `keep` jobs."""

    def __init__(self, keep: int = 50):
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def create(self, files: list[str]) -> IngestJob:
        job = IngestJob(id=uuid.uuid4().hex, files=files)
        with self._lock:
            self._jobs[job.id] = job
            # Evict the oldest finished jobs past the cap (running ones stay).
            if len(self._jobs) > self._keep:
                for jid, j in sorted(self._jobs.items(),
                                     key=lambda kv: kv[1].created_at):
                    if len(self._jobs) <= self._keep:
                        break
                    if j.status in ("done", "failed"):
                        del self._jobs[jid]
        return job

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:                      # evicted — nothing to update
                return
            for key, value in fields.items():
                setattr(job, key, value)
