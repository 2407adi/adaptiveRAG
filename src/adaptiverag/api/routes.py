"""The service windows."""
import json
import shutil
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from adaptiverag.api.auth import require_api_key, require_role
from adaptiverag.ingest.exceptions import IngestTooLarge
from adaptiverag.scope import (          # Block 4.2b: stamps + guest lists
    SHARED_SCOPE, chat_scope, scopes_for, current_scopes,
)

from adaptiverag.api.models import (
    QueryRequest, QueryResponse, Source, GroundingReport,
    ClaimVerdict, ReasoningStep, IngestResponse,
    IngestAccepted, IngestJobStatus,
    AgentStartRequest, AgentResumeRequest, AgentResponse,
    AgentReport, ApprovalRequest,
)

from adaptiverag.reason.router import QueryRoute


# Two routers now: `public` has no doorman (Azure's probe carries no card);
# `router` gets the doorman ONCE, and every window on it inherits him.
public = APIRouter()
router = APIRouter(dependencies=[Depends(require_api_key)])

def _owner(request: Request) -> str | None:
    """Browser-local tenancy (post-4.3b): the SPA mints an anonymous client_id
    into localStorage and sends it as X-Client-Id on every call. It stamps new
    folders and filters listings — per-BROWSER privacy without a login flow.
    No header (curl, tests, old clients) → ownerless, unfiltered, old behavior."""
    return request.headers.get("x-client-id")


def _record_exchange(request: Request, conv_id: str, question: str, answer: str,
                     *, make_title: bool = True) -> None:
    """File one full exchange into the cabinet, then make sure the folder has a tab label.
    Streaming endpoints pass make_title=False: the titler is an LLM call, and it must
    never keep the stream open after `done` — they run _ensure_title as a background
    task once the response has fully closed instead."""
    store = request.app.state.store
    owner = _owner(request)
    store.append_turn(conv_id, "user", question, owner=owner)     # page 1: what they asked
    store.append_turn(conv_id, "assistant", answer, owner=owner)  # page 2: what we said
    if make_title and store.get_title(conv_id) is None:   # blank tab = first exchange
        _auto_title(request, conv_id, question)

def _record_agent_answer(request: Request, conv_id: str | None, answer: str | None,
                         *, make_title: bool = True) -> None:
    """File the verdict page + label the tab. Only called when a case reaches 'done'.
    No ticket or no answer → nothing to file (bare API calls stay folder-less)."""
    if not (conv_id and answer):
        return
    store = request.app.state.store
    store.append_turn(conv_id, "assistant", answer, owner=_owner(request))
    if make_title and store.get_title(conv_id) is None:   # blank tab = this chat's first exchange
        turns = store.get_turns(conv_id)                  # the question page was filed at case-open —
        first_user = next((t["content"] for t in turns if t["role"] == "user"), answer)
        _auto_title(request, conv_id, first_user)         # — so read it back for the labeler


def _ensure_title(request: Request, conv_id: str | None) -> None:
    """Background labeling: runs AFTER a stream has closed, so the title LLM call
    never makes the client stare at a blinking cursor. No folder / no user turn /
    already titled → quietly does nothing."""
    if not conv_id:
        return
    store = request.app.state.store
    if store.get_title(conv_id) is not None:
        return
    turns = store.get_turns(conv_id)
    first_user = next((t["content"] for t in turns if t["role"] == "user"), None)
    if first_user:
        _auto_title(request, conv_id, first_user)


def _auto_title(request: Request, conv_id: str, question: str) -> None:
    """The labeling clerk: skim the first question, write 3–6 words on the tab.
    NEVER allowed to break chatting — any failure falls back to a dumb-but-safe label."""
    try:
        title = request.app.state.pipeline.llm_client.generate(
            "Name this conversation in 3 to 6 plain words based on the user's "
            f"first message. Reply with the title only, no quotes.\n\nMessage: {question}"
        ).strip().strip('"')[:60]                          # belt-and-braces: cap runaway titles
    except Exception:
        title = ""                                        # LLM had a bad day — shrug
    if not title:                                         # empty reply or failure → fallback
        title = question[:40] + ("…" if len(question) > 40 else "")
    request.app.state.store.set_title(conv_id, title)


@public.get("/health")
def health() -> dict:
    # The "we're open" light. Azure pings this to know the container is alive.
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    pipe = request.app.state.pipeline                    # the staff, off the shelf
    conv_id = req.conversation_id or str(uuid4())        # no ticket? issue a fresh one

    route = pipe.router.classify(req.question).route     # triage desk decides the path

    if route == QueryRoute.DIRECT:                       # small talk: no detective, no evidence
        answer = pipe.llm_client.generate(req.question)
        _record_exchange(request, conv_id, req.question, answer)
        return QueryResponse(answer=answer, route=route.value, conversation_id=conv_id)
    
    chain = pipe.multi_step_chain if route == QueryRoute.MULTI_STEP else pipe.rag_chain
    # 4.2b: guest list = public shelf + THIS chat's locker. A fresh ticket
    # (new conv_id) means a fresh, empty locker — new chats see only `shared`.
    result = chain.query(req.question, expand=req.expand, scopes=scopes_for(conv_id))

    g = pipe.grounding_validator.validate(result["answer"], result["sources"])  # Auditor stamps it
    
    _record_exchange(request, conv_id, req.question, result["answer"])

    return QueryResponse(
        answer=result["answer"], route=route.value, conversation_id=conv_id,
        sources=[Source(**s) for s in result["sources"]],
        grounding=GroundingReport(
            score=g.score, is_grounded=g.is_grounded,
            total_claims=g.total_claims, grounded_claims=g.grounded_claims,
            verdicts=[ClaimVerdict(claim=v.claim, status=v.status.value, max_score=v.max_score) for v in g.verdicts],
        ),
        reasoning_steps=[ReasoningStep(sub_question=s["sub_question"], answer=s["answer"])
                         for s in result.get("reasoning_steps", [])],
    )


# One heavy ingestion at a time, server-wide: the box has a fraction of a CPU,
# and two concurrent embedding jobs would starve serving traffic completely.
# Later jobs wait here (their status shows "queued"), then run in turn.
_INGEST_LOCK = threading.Lock()


def _run_ingest_job(app, job_id: str, tmpdir: str, scope: str) -> None:
    """The background worker: runs the real pipeline, narrates progress into
    the job record, and always cleans up the loading dock. Runs on a daemon
    thread — completely detached from any HTTP connection, so neither the
    ingress timeout nor an impatient browser can kill it."""
    jobs = app.state.ingest_jobs
    pipe = app.state.pipeline
    cfg = app.state.settings.auth
    try:
        with _INGEST_LOCK:
            jobs.update(job_id, status="running", stage="loading")
            stats = pipe.ingest.ingest(
                tmpdir, scope=scope,
                progress_cb=lambda stage, done, total: jobs.update(
                    job_id, stage=stage, chunks_done=done, chunks_total=total),
                max_chunks=cfg.max_upload_chunks,        # the work cap — fails fast, pre-embedding
            )
            jobs.update(job_id, status="done", stage="done", result=stats)
    except IngestTooLarge as e:
        jobs.update(job_id, status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 — job must record ANY failure, never vanish
        jobs.update(job_id, status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)        # demolish the dock, success or not


@router.post("/ingest", response_model=IngestAccepted, status_code=202,
             dependencies=[Depends(require_role("admin"))])   # the stricter doorman: gold cards only
def ingest(
    request: Request,
    files: list[UploadFile] = File(...),
    conversation_id: str | None = Form(None),    # 4.2b: which locker? None = public shelf
) -> IngestAccepted:
    # The drop-off window — now ASYNC: accept the files, hand back a claim
    # ticket (job_id) in under a second, do the heavy work on a background
    # thread. The UI polls /ingest/status/{job_id} for live progress.
    pipe = request.app.state.pipeline
    cfg = request.app.state.settings.auth

    # Cheap gate checks stay synchronous — instant answers, proper HTTP codes.
    # The archive ceiling: gold cards go public for demos, so the CAP — not the
    # role — is what actually protects the store. Applies to everyone, admin too.
    if pipe.vector_store.count() >= cfg.max_total_chunks:
        raise HTTPException(status_code=507,             # Insufficient Storage
                            detail="document store is full; ingestion is closed")

    # A loading dock that does NOT self-demolish: the job outlives this
    # request, so the worker thread owns the cleanup (see _run_ingest_job).
    max_bytes = cfg.max_upload_mb * 1024 * 1024
    tmpdir = tempfile.mkdtemp(prefix="ingest_")
    names: list[str] = []
    try:
        for f in files:
            content = f.file.read()
            if len(content) > max_bytes:                 # the dock scale: weigh every package
                raise HTTPException(status_code=413,     # Payload Too Large
                                    detail=f"{f.filename}: exceeds {cfg.max_upload_mb} MB upload limit")
            name = Path(f.filename or "upload.bin").name   # unnamed file? give it a boring name
            (Path(tmpdir) / name).write_bytes(content)
            names.append(name)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)        # rejected at the dock → clean up now
        raise

    # 4.2b: the clerk's ink stamp — a conversation_id means these books go
    # into that chat's locker; without one they land on the public shelf.
    scope = chat_scope(conversation_id) if conversation_id else SHARED_SCOPE

    job = request.app.state.ingest_jobs.create(files=names)
    threading.Thread(
        target=_run_ingest_job,
        args=(request.app, job.id, tmpdir, scope),
        daemon=True,
    ).start()
    return IngestAccepted(job_id=job.id, status=job.status, files=names)


@router.get("/ingest/status/{job_id}", response_model=IngestJobStatus,
            dependencies=[Depends(require_role("admin"))])
def ingest_status(job_id: str, request: Request) -> IngestJobStatus:
    # The claim-ticket window: instant read of the job record.
    job = request.app.state.ingest_jobs.get(job_id)
    if job is None:
        # Jobs live in memory — a missing id usually means the container
        # restarted mid-job. The UI shows this message verbatim.
        raise HTTPException(status_code=404,
                            detail="unknown ingest job (the server may have "
                                   "restarted during processing — please retry the upload)")
    return IngestJobStatus(
        job_id=job.id, status=job.status, stage=job.stage,
        chunks_done=job.chunks_done, chunks_total=job.chunks_total,
        files=job.files, error=job.error,
        result=IngestResponse(**job.result) if job.result else None,
    )



@router.get("/conversations")
def list_conversations(request: Request) -> dict:
    # The sidebar list — with an X-Client-Id, ONLY that browser's drawer.
    return {"conversations": request.app.state.store.list_conversations(owner=_owner(request))}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, request: Request) -> dict:
    store = request.app.state.store
    # A stranger's folder 404s exactly like a missing one — no drawer-peeking,
    # and no oracle that reveals which ids exist.
    if not store.exists(conversation_id, owner=_owner(request)):
        raise HTTPException(status_code=404, detail="unknown conversation_id")
    return {"id": conversation_id,
            "title": store.get_title(conversation_id),    # new: the 4.3b sidebar wants this
            "turns": store.get_turns(conversation_id)}


@router.delete("/conversations")
def clear_conversations(request: Request) -> dict:
    # "Clear my chats": shreds ONLY the caller's drawer. Deliberately requires a
    # client id — there is no anonymous 'shred the whole cabinet' button.
    owner = _owner(request)
    if not owner:
        raise HTTPException(status_code=400, detail="X-Client-Id header required")
    return {"deleted": request.app.state.store.delete_owned(owner)}


def _sse(event: dict) -> str:
    # Fold one event into SSE framing: "data: {...}\n\n" — blank line = end of note.
    return f"data: {json.dumps(event)}\n\n"


@router.post("/chat/stream")
def chat_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    pipe = request.app.state.pipeline
    conv_id = req.conversation_id or str(uuid4())

    def notes() -> Iterator[str]:                        # the open slot in the window
        try:
            route = pipe.router.classify(req.question).route
            yield _sse({"type": "route", "route": route.value,
                        "conversation_id": conv_id})     # first note: which desk took the case

            if route == QueryRoute.DIRECT:               # small talk: tokens straight from the LLM
                parts: list[str] = []
                for piece in pipe.llm_client.generate_stream(req.question):
                    parts.append(piece)
                    yield _sse({"type": "token", "text": piece})
                answer = "".join(parts)
            else:
                chain = pipe.multi_step_chain if route == QueryRoute.MULTI_STEP else pipe.rag_chain
                final: dict = {}
                for event in chain.query_stream(req.question, expand=req.expand,
                                                scopes=scopes_for(conv_id)):  # 4.2b guest list
                    if event["type"] == "done":
                        final = event                    # keep for grounding; don't forward —
                    else:                                #   WE decide when it's truly done
                        yield _sse(event)
                answer = final["answer"]

                g = pipe.grounding_validator.validate(answer, final["sources"])
                yield _sse({"type": "grounding",         # the late-arriving Auditor badge
                            "score": g.score, "is_grounded": g.is_grounded,
                            "total_claims": g.total_claims, "grounded_claims": g.grounded_claims,
                            "verdicts": [{"claim": v.claim, "status": v.status.value,
                                          "max_score": v.max_score} for v in g.verdicts]})

            _record_exchange(request, conv_id, req.question, answer,
                             make_title=False)           # filing is fast; titling is NOT —
            yield _sse({"type": "done", "conversation_id": conv_id})   # ours, after ALL work

        except Exception as exc:                         # 200 already sent — errors become a note
            yield _sse({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(notes(), media_type="text/event-stream",
                             background=BackgroundTask(_ensure_title, request, conv_id))

def _agent_envelope(result: dict) -> AgentResponse:
    # Translate executor dict → sealed envelope. One translator for both desks and both endpoints.
    approval = result.get("request")                     # present only when frozen mid-case
    return AgentResponse(
        status=result["status"],
        thread_id=result["thread_id"],
        answer=result.get("answer"),
        trace=result.get("trace") or [],                 # lone-detective scratchpad
        reports=[AgentReport(**r) for r in result.get("reports") or []],  # Chief's junior reports
        request=ApprovalRequest(tool=approval["tool"], args=approval["args"],
                                message=approval["message"]) if approval else None,
    )


def _pick_agent(request: Request, supervisor: bool):
    # Route the visitor to the right desk; 503 if that desk was never staffed.
    pipe = request.app.state.pipeline
    agent = pipe.supervisor_agent if supervisor else pipe.agent_executor
    if agent is None:
        raise HTTPException(status_code=503, detail="agent not configured")
    return agent


def _scoped(events: Iterator[dict], scopes: list[str] | None) -> Iterator[dict]:
    # 4.2b, streaming-only subtlety: Starlette pulls each event of a sync
    # generator in a FRESH context copy, so pinning the guest list once up
    # front would be forgotten by the second pull. Re-pin it inside every
    # pull, so whatever tool call fires during next(events) sees the list.
    while True:
        current_scopes.set(scopes)               # pin the guest list for THIS pull
        try:
            event = next(events)                 # agent may call search_documents in here
        except StopIteration:
            return
        yield event


@router.post("/agent/start", response_model=AgentResponse)
def agent_start(req: AgentStartRequest, request: Request) -> AgentResponse:
    agent = _pick_agent(request, req.supervisor)
    current_scopes.set(scopes_for(req.conversation_id))
    if req.conversation_id:                               # question page: filed NOW, certain
        request.app.state.store.append_turn(req.conversation_id, "user", req.question,
                                            owner=_owner(request))
    result = agent.start(req.question, conversation_id=req.conversation_id)   # 4.3a: the ticket
    if result["status"] == "done":                        # frozen case → no answer page yet
        _record_agent_answer(request, req.conversation_id, result.get("answer"))
    return _agent_envelope(result)


@router.post("/agent/resume", response_model=AgentResponse)
def agent_resume(req: AgentResumeRequest, request: Request) -> AgentResponse:
    # Visitor returns with the claim ticket and a decision; the case thaws mid-step.
    agent = _pick_agent(request, req.supervisor)
    current_scopes.set(scopes_for(req.conversation_id))   # 4.2b: same guest list on thaw
    decision = {"approved": req.approved, "reason": req.reason} if req.reason else req.approved
    result = agent.resume(req.thread_id, decision, conversation_id=req.conversation_id)
    if result["status"] == "done":
        _record_agent_answer(request, req.conversation_id, result.get("answer"))
    return _agent_envelope(result)

@router.post("/agent/start/stream")
def agent_start_stream(req: AgentStartRequest, request: Request) -> StreamingResponse:
    # Open a case with live narration. Stream ends at done OR at a freeze —
    # the approval event carries the thread_id, so the client can come back.
    agent = _pick_agent(request, req.supervisor)
    scopes = scopes_for(req.conversation_id)              # 4.2b guest list

    def notes() -> Iterator[str]:
        try:
            if req.conversation_id:                       # question page at case-open, same as non-stream
                request.app.state.store.append_turn(req.conversation_id, "user", req.question,
                                                    owner=_owner(request))
            for event in _scoped(agent.start_stream(req.question,
                                                    conversation_id=req.conversation_id), scopes):
                if event.get("type") == "done":           # verdict flowing past → file the answer page
                    _record_agent_answer(request, req.conversation_id, event.get("answer"),
                                         make_title=False)   # titling happens post-close
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(notes(), media_type="text/event-stream",
                             background=BackgroundTask(_ensure_title, request, req.conversation_id))


@router.post("/agent/resume/stream")
def agent_resume_stream(req: AgentResumeRequest, request: Request) -> StreamingResponse:
    # Return with the ticket + decision; narration picks up mid-case.
    agent = _pick_agent(request, req.supervisor)
    scopes = scopes_for(req.conversation_id)              # 4.2b guest list
    decision = {"approved": req.approved, "reason": req.reason} if req.reason else req.approved

    def notes() -> Iterator[str]:
        try:
            for event in _scoped(agent.resume_stream(req.thread_id, decision, conversation_id=req.conversation_id), scopes):
                if event.get("type") == "done":           # verdict flowing past → file the answer page
                    _record_agent_answer(request, req.conversation_id, event.get("answer"),
                                         make_title=False)   # titling happens post-close
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(notes(), media_type="text/event-stream",
                             background=BackgroundTask(_ensure_title, request, req.conversation_id))
