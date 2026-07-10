"""Offline tests for the front desk: real windows, fake staff."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from adaptiverag.api.main import app
from adaptiverag.api.auth import RateLimiter
from adaptiverag.reason.router import QueryRoute

# ── the fake staff ──────────────────────────────────────────────────────────

SOURCE = {"chunk_id": "c1", "source": "doc.pdf", "page": "2", "chunk_index": "0",
          "score": 0.9, "text_preview": "Solstice raised...", "full_text": "Solstice raised $12M."}


class FakeRouter:
    def classify(self, q):
        # keyword triage: enough to steer each test down a chosen corridor
        if "hello" in q.lower():
            route = QueryRoute.DIRECT
        elif "compare" in q.lower():
            route = QueryRoute.MULTI_STEP
        else:
            route = QueryRoute.RAG
        return SimpleNamespace(route=route, confidence="high", reasoning="test")


class FakeLLM:
    def generate(self, prompt):
        return "direct answer"

    def generate_stream(self, prompt):
        yield "direct "
        yield "answer"


class FakeChain:                                     # serves as both RAG and MultiStep fake
    def query(self, q, expand=False, scopes=None):   # scopes: Block 4.2b guest list
        return {"answer": "grounded answer", "sources": [SOURCE],
                "reasoning_steps": [{"sub_question": "sq1", "answer": "a1", "sources": [SOURCE]}]}

    def query_stream(self, q, expand=False, scopes=None):
        yield {"type": "stage", "stage": "retrieving"}
        yield {"type": "sources", "sources": [SOURCE]}
        yield {"type": "token", "text": "grounded "}
        yield {"type": "token", "text": "answer"}
        yield {"type": "done", "answer": "grounded answer", "sources": [SOURCE]}


class FakeValidator:
    def validate(self, answer, sources):
        verdict = SimpleNamespace(claim="a claim", status=SimpleNamespace(value="SUPPORTED"),
                                  max_score=0.95)
        return SimpleNamespace(score=1.0, is_grounded=True,
                               total_claims=1, grounded_claims=1, verdicts=[verdict])


class FakeAgent:
    """Freezes on the first start(); completes on resume(). Stream twins match."""
    def start(self, question, thread_id=None):
        return {"status": "awaiting_approval", "thread_id": "T1", "answer": None,
                "trace": [{"type": "thought", "content": "hmm"}],
                "request": {"type": "approval_request", "tool": "run_python",
                            "args": {"code": "2+2"}, "message": "wants to run code"}}

    def resume(self, thread_id, decision):
        return {"status": "done", "thread_id": thread_id, "answer": "4",
                "trace": [{"type": "observation", "content": "4"}]}

    def start_stream(self, question, thread_id=None):
        yield {"type": "thought", "text": "hmm"}
        yield {"type": "approval", "request": {"tool": "run_python", "args": {},
                                               "message": "wants to run code"}, "thread_id": "T1"}

    def resume_stream(self, thread_id, decision):
        yield {"type": "token", "text": "4"}
        yield {"type": "done", "answer": "4", "thread_id": thread_id}


# ── the fixture: hand-place the fake staff on the real shelf ────────────────

@pytest.fixture()
def client():
    app.state.pipeline = SimpleNamespace(
        router=FakeRouter(), llm_client=FakeLLM(),
        rag_chain=FakeChain(), multi_step_chain=FakeChain(),
        grounding_validator=FakeValidator(),
        ingest=SimpleNamespace(ingest=lambda d, scope="shared": {"files_processed": 1, "total_chunks": 3,
                                                                 "corpus_summary": "test docs"}),
        vector_store=SimpleNamespace(count=lambda: 0),       # empty archive → caps never trip here
        agent_executor=FakeAgent(), supervisor_agent=None,   # None → 503 path testable
    )
    app.state.conversations = {}                     # fresh cabinet per test
    # Block 4.2: lifespan never runs (no `with`), so hand-place the doorman's
    # equipment on the shelf too — same trick as the fake staff above.
    app.state.settings = SimpleNamespace(auth=SimpleNamespace(
        enabled=True, rate_limit_per_minute=10_000,  # tally so generous it never trips
        max_upload_mb=20, max_total_chunks=50_000))
    app.state.api_keys = {"gold-test": "admin"}
    app.state.rate_limiter = RateLimiter(10_000)
    c = TestClient(app)                              # NO `with` → lifespan never runs
    c.headers["X-API-Key"] = "gold-test"             # every test flashes the gold card
    return c

def sse_events(response) -> list[dict]:
    """Unframe an SSE body back into event dicts (inverse of routes._sse)."""
    import json
    return [json.loads(line[len("data: "):])
            for line in response.text.splitlines() if line.startswith("data: ")]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_query_direct_has_zero_sources(client):
    # THE no-phantom-citations contract: small talk carries no evidence.
    r = client.post("/query", json={"question": "hello there"})
    body = r.json()
    assert r.status_code == 200
    assert body["route"] == "direct"
    assert body["sources"] == [] and body["grounding"] is None
    assert body["conversation_id"]                    # fresh ticket was issued


def test_query_rag_returns_sources_and_grounding(client):
    body = client.post("/query", json={"question": "what did Solstice raise?"}).json()
    assert body["route"] == "rag"
    assert body["sources"][0]["chunk_id"] == "c1"
    assert "full_text" not in body["sources"][0]      # envelopes stay light
    assert body["grounding"]["is_grounded"] is True


def test_query_multistep_returns_reasoning_steps(client):
    body = client.post("/query", json={"question": "compare US and UK policy"}).json()
    assert body["route"] == "multi_step"
    assert body["reasoning_steps"][0]["sub_question"] == "sq1"


def test_blank_question_bounced_at_the_window(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_conversation_drawer_fills_and_reads_back(client):
    conv = client.post("/query", json={"question": "hello"}).json()["conversation_id"]
    client.post("/query", json={"question": "hello again", "conversation_id": conv})
    turns = client.get(f"/conversations/{conv}").json()["turns"]
    assert len(turns) == 4 and turns[0]["role"] == "user"
    assert client.get("/conversations/nope").status_code == 404


def test_ingest_returns_receipt(client):
    r = client.post("/ingest", files={"files": ("a.txt", b"hello world", "text/plain")})
    assert r.json() == {"files_processed": 1, "total_chunks": 3, "corpus_summary": "test docs"}


def test_chat_stream_event_order(client):
    r = client.post("/chat/stream", json={"question": "what did Solstice raise?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r)
    types = [e["type"] for e in events]
    assert types[0] == "route" and types[-1] == "done"          # our done, not the chain's
    assert "".join(e["text"] for e in events if e["type"] == "token") == "grounded answer"
    assert types.index("sources") < types.index("token")        # evidence before answer
    assert types.index("grounding") > types.index("token")      # badge after the text settles


def test_agent_approval_round_trip(client):
    # Visit 1: case freezes, ticket issued.
    r1 = client.post("/agent/start", json={"question": "run some code"}).json()
    assert r1["status"] == "awaiting_approval" and r1["request"]["tool"] == "run_python"
    # Visit 2: ticket + decision → verdict.
    r2 = client.post("/agent/resume",
                     json={"thread_id": r1["thread_id"], "approved": True}).json()
    assert r2["status"] == "done" and r2["answer"] == "4"


def test_unstaffed_desk_returns_503(client):
    r = client.post("/agent/start", json={"question": "hi", "supervisor": True})
    assert r.status_code == 503


def test_agent_stream_freezes_then_resumes(client):
    e1 = sse_events(client.post("/agent/start/stream", json={"question": "run code"}))
    assert e1[-1]["type"] == "approval" and e1[-1]["thread_id"] == "T1"
    e2 = sse_events(client.post("/agent/resume/stream",
                                json={"thread_id": "T1", "approved": True}))
    assert e2[-1]["type"] == "done" and e2[-1]["answer"] == "4"