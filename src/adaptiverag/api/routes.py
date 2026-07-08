"""The service windows."""
"""The service windows."""
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from adaptiverag.api.models import (
    QueryRequest, QueryResponse, Source, GroundingReport,
    ClaimVerdict, ReasoningStep, IngestResponse,
    AgentStartRequest, AgentResumeRequest, AgentResponse,
    AgentReport, ApprovalRequest,
)

from adaptiverag.reason.router import QueryRoute


router = APIRouter()


@router.get("/health")
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
        _turns(request, conv_id).append({"role": "user", "content": req.question})
        _turns(request, conv_id).append({"role": "assistant", "content": answer})
        return QueryResponse(answer=answer, route=route.value, conversation_id=conv_id)
    
    chain = pipe.multi_step_chain if route == QueryRoute.MULTI_STEP else pipe.rag_chain
    result = chain.query(req.question, expand=req.expand)

    g = pipe.grounding_validator.validate(result["answer"], result["sources"])  # Auditor stamps it
    
    _turns(request, conv_id).append({"role": "user", "content": req.question})
    _turns(request, conv_id).append({"role": "assistant", "content": result["answer"]})

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


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: Request, files: list[UploadFile] = File(...)) -> IngestResponse:
    # The drop-off window: visitor hands over documents, gets a receipt.
    pipe = request.app.state.pipeline
    with tempfile.TemporaryDirectory() as tmpdir:        # a loading dock that self-demolishes
        for f in files:
            name = Path(f.filename or "upload.bin").name   # unnamed file? give it a boring name
            (Path(tmpdir) / name).write_bytes(f.file.read())
        stats = pipe.ingest.ingest(tmpdir)               # loader→chunker→embedder→Chroma, unchanged
    return IngestResponse(**stats)                       # dock demolished; chunks live in Chroma


def _turns(request: Request, conv_id: str) -> list[dict]:
    # Open the drawer for this ticket; create it on first visit.
    return request.app.state.conversations.setdefault(conv_id, [])


@router.get("/conversations")
def list_conversations(request: Request) -> dict:
    cabinet = request.app.state.conversations
    return {"conversations": [{"id": cid, "turns": len(t)} for cid, t in cabinet.items()]}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, request: Request) -> dict:
    cabinet = request.app.state.conversations
    if conversation_id not in cabinet:
        raise HTTPException(status_code=404, detail="unknown conversation_id")  # no such drawer
    return {"id": conversation_id, "turns": cabinet[conversation_id]}

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
                for event in chain.query_stream(req.question, expand=req.expand):
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

            _turns(request, conv_id).append({"role": "user", "content": req.question})
            _turns(request, conv_id).append({"role": "assistant", "content": answer})
            yield _sse({"type": "done", "conversation_id": conv_id})   # ours, after ALL work

        except Exception as exc:                         # 200 already sent — errors become a note
            yield _sse({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(notes(), media_type="text/event-stream")

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


@router.post("/agent/start", response_model=AgentResponse)
def agent_start(req: AgentStartRequest, request: Request) -> AgentResponse:
    # Open a case. Returns either the verdict, or a frozen case + claim ticket.
    agent = _pick_agent(request, req.supervisor)
    return _agent_envelope(agent.start(req.question))


@router.post("/agent/resume", response_model=AgentResponse)
def agent_resume(req: AgentResumeRequest, request: Request) -> AgentResponse:
    # Visitor returns with the claim ticket and a decision; the case thaws mid-step.
    agent = _pick_agent(request, req.supervisor)
    decision = {"approved": req.approved, "reason": req.reason} if req.reason else req.approved
    return _agent_envelope(agent.resume(req.thread_id, decision))

@router.post("/agent/start/stream")
def agent_start_stream(req: AgentStartRequest, request: Request) -> StreamingResponse:
    # Open a case with live narration. Stream ends at done OR at a freeze —
    # the approval event carries the thread_id, so the client can come back.
    agent = _pick_agent(request, req.supervisor)

    def notes() -> Iterator[str]:
        try:
            for event in agent.start_stream(req.question):
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(notes(), media_type="text/event-stream")


@router.post("/agent/resume/stream")
def agent_resume_stream(req: AgentResumeRequest, request: Request) -> StreamingResponse:
    # Return with the ticket + decision; narration picks up mid-case.
    agent = _pick_agent(request, req.supervisor)
    decision = {"approved": req.approved, "reason": req.reason} if req.reason else req.approved

    def notes() -> Iterator[str]:
        try:
            for event in agent.resume_stream(req.thread_id, decision):
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(notes(), media_type="text/event-stream")