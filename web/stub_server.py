"""Stub backend for UI development — speaks the exact same wire protocol as the
real API (endpoints, SSE event types, auth) but with canned data and zero LLM
cost. Handy for frontend tinkering without burning tokens.

Run:  python web/stub_server.py   (serves web/dist + fake API on :8000)
Keys: any key works EXCEPT "bad" (use that to test the 401 → landing bounce).
"""
import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AdaptiveRAG stub API")

CONVS: dict[str, dict] = {}  # id -> {"title": str|None, "owner": str|None, "turns": [...]}


def _auth(key: str | None):
    if not key or key == "bad":
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def _mine(cid: str, owner: str | None) -> bool:
    return owner is None or CONVS.get(cid, {}).get("owner") == owner


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _record(conv_id: str, question: str, answer: str, owner: str | None = None):
    c = CONVS.setdefault(conv_id, {"title": None, "owner": owner, "turns": []})
    c["turns"] += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if c["title"] is None:
        c["title"] = question[:40] + ("…" if len(question) > 40 else "")


SOURCES = [
    {"chunk_id": "c1", "source": "solstice_funding_memo.md", "page": None, "chunk_index": 3,
     "score": 8.2, "text_preview": "…closed a $12M Series A led by Meridian Ventures in…"},
    {"chunk_id": "c2", "source": "investor_update_q1.md", "page": None, "chunk_index": 1,
     "score": 6.9, "text_preview": "…following our $15M Series A round, headcount grew…"},
]
GROUNDING = {
    "type": "grounding", "score": 0.67, "is_grounded": True, "total_claims": 3, "grounded_claims": 2,
    "verdicts": [
        {"claim": "Solstice Robotics raised a Series A round.", "status": "supported", "max_score": 0.9},
        {"claim": "The round was led by Meridian Ventures.", "status": "supported", "max_score": 0.8},
        {"claim": "The round size was $12M.", "status": "contradicted", "max_score": 0.7},
    ],
}
ANSWER = ("Solstice Robotics raised a $12M Series A led by Meridian Ventures [1]. "
          "Worth flagging: the Q1 investor update states $15M [2] — the two documents "
          "contradict each other on the round size.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/conversations")
def list_convs(x_api_key: str | None = Header(None), x_client_id: str | None = Header(None)):
    _auth(x_api_key)
    return {"conversations": [
        {"id": cid, "title": c["title"], "turns": len(c["turns"])}
        for cid, c in reversed(CONVS.items()) if _mine(cid, x_client_id)
    ]}


@app.get("/conversations/{cid}")
def get_conv(cid: str, x_api_key: str | None = Header(None), x_client_id: str | None = Header(None)):
    _auth(x_api_key)
    if cid not in CONVS or not _mine(cid, x_client_id):
        raise HTTPException(status_code=404, detail="unknown conversation_id")
    return {"id": cid, "title": CONVS[cid]["title"], "turns": CONVS[cid]["turns"]}


@app.delete("/conversations")
def clear_convs(x_api_key: str | None = Header(None), x_client_id: str | None = Header(None)):
    _auth(x_api_key)
    if not x_client_id:
        raise HTTPException(status_code=400, detail="X-Client-Id header required")
    mine = [cid for cid, c in CONVS.items() if c.get("owner") == x_client_id]
    for cid in mine:
        del CONVS[cid]
    return {"deleted": len(mine)}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...), conversation_id: str | None = Form(None),
                 x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    await asyncio.sleep(1.5)
    return {"files_processed": len(files), "total_chunks": 17 * len(files), "corpus_summary": None}


@app.post("/chat/stream")
async def chat_stream(body: dict, x_api_key: str | None = Header(None),
                      x_client_id: str | None = Header(None)):
    _auth(x_api_key)
    conv_id = body.get("conversation_id") or str(uuid.uuid4())
    multi = len(body.get("question", "")) > 90 or "compare" in body.get("question", "").lower()

    async def notes():
        route = "multi_step" if multi else "rag"
        yield _sse({"type": "route", "route": route, "conversation_id": conv_id})
        await asyncio.sleep(0.4)
        if multi:
            yield _sse({"type": "stage", "stage": "decomposing"})
            await asyncio.sleep(0.6)
            for i, sq in enumerate(["What does contract v1 commit to?",
                                    "What does contract v2 commit to?",
                                    "How much downtime does each allow?"], 1):
                yield _sse({"type": "stage", "stage": "sub_question", "index": i, "total": 3, "text": sq})
                await asyncio.sleep(0.5)
        else:
            yield _sse({"type": "stage", "stage": "retrieving"})
            await asyncio.sleep(0.5)
        yield _sse({"type": "sources", "sources": SOURCES})
        if multi:
            yield _sse({"type": "stage", "stage": "synthesizing"})
        await asyncio.sleep(0.3)
        for i in range(0, len(ANSWER), 6):
            yield _sse({"type": "token", "text": ANSWER[i:i + 6]})
            await asyncio.sleep(0.02)
        yield _sse(GROUNDING)
        _record(conv_id, body.get("question", ""), ANSWER, owner=x_client_id)
        yield _sse({"type": "done", "conversation_id": conv_id})

    return StreamingResponse(notes(), media_type="text/event-stream")


@app.post("/agent/start/stream")
async def agent_start(body: dict, x_api_key: str | None = Header(None),
                      x_client_id: str | None = Header(None)):
    _auth(x_api_key)
    sup = body.get("supervisor", False)
    thread_id = str(uuid.uuid4())

    async def lone():
        yield _sse({"type": "thought", "text": "I should find the SLA percentages, then verify the math with code. "})
        await asyncio.sleep(0.6)
        yield _sse({"type": "trace", "entry": {"type": "thought", "content": "I should find the SLA percentages, then verify the math with code."}})
        yield _sse({"type": "trace", "entry": {"type": "action", "tool": "search_documents", "args": {"query": "SLA uptime commitment"}}})
        await asyncio.sleep(0.8)
        yield _sse({"type": "trace", "entry": {"type": "observation", "content": "3 chunks — v1: 99.5%, v2: 99.9%, credit cap 10%."}})
        await asyncio.sleep(0.5)
        yield _sse({"type": "approval", "thread_id": thread_id,
                    "request": {"type": "approval_request", "tool": "run_python",
                                "args": {"code": "(1-0.999)*8760, (1-0.995)*8760"},
                                "message": "The agent wants to run 'run_python'. Approve?"}})

    async def firm():
        answer = "One material inconsistency: the Series A is $12M in two documents but $15M in the investor update."
        for agent, report in [("retriever", "Funding references found in 3 documents; two say $12M, one says $15M."),
                              ("reasoner", "The $15M figure likely bundles the $3M venture-debt facility."),
                              ("validator", "All claims entailed except the bundling explanation — marked as inference.")]:
            yield _sse({"type": "handoff", "agent": agent})
            await asyncio.sleep(0.4)
            yield _sse({"type": "trace", "agent": agent, "entry": {"type": "action", "tool": "search_documents", "args": {"query": "Series A"}}})
            await asyncio.sleep(0.5)
            for i in range(0, len(report), 8):
                yield _sse({"type": "report_token", "agent": agent, "text": report[i:i + 8]})
                await asyncio.sleep(0.02)
            yield _sse({"type": "report", "agent": agent, "report": report})
        await asyncio.sleep(0.4)
        for i in range(0, len(answer), 6):
            yield _sse({"type": "token", "text": answer[i:i + 6]})
            await asyncio.sleep(0.02)
        conv = body.get("conversation_id")
        if conv:
            _record(conv, body.get("question", ""), answer, owner=x_client_id)
        yield _sse({"type": "done", "answer": answer, "thread_id": thread_id})

    return StreamingResponse(firm() if sup else lone(), media_type="text/event-stream")


@app.post("/agent/resume/stream")
async def agent_resume(body: dict, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    approved = body.get("approved", False)

    async def notes():
        if approved:
            yield _sse({"type": "trace", "entry": {"type": "observation", "content": "run_python → (8.76, 43.8) · logged to audit chain"}})
            answer = "Verified with code: 99.9% → 8.76 h/yr and 99.5% → 43.8 h/yr. The math holds."
        else:
            yield _sse({"type": "trace", "entry": {"type": "observation", "content": "Action 'run_python' was rejected by the reviewer. Choose a different approach."}})
            answer = "Without running code: v1 = 99.5% and v2 = 99.9% uptime; roughly 44h and 9h of allowed annual downtime."
        await asyncio.sleep(0.5)
        for i in range(0, len(answer), 6):
            yield _sse({"type": "token", "text": answer[i:i + 6]})
            await asyncio.sleep(0.02)
        yield _sse({"type": "done", "answer": answer, "thread_id": body.get("thread_id")})

    return StreamingResponse(notes(), media_type="text/event-stream")


_dist = Path(__file__).resolve().parent / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
