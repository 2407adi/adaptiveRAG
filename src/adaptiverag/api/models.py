"""Forms and envelopes for the front desk (request/response contracts)."""
from pydantic import BaseModel, Field


# QueryRequest is the intake form a visitor fills to ask the agency a question.
class QueryRequest(BaseModel):
    # The intake form a visitor fills to ask the agency a question.
    question: str = Field(min_length=1)          # blank forms bounced at the window
    conversation_id: str | None = None           # claim ticket; None = start a new conversation
    expand: bool = False                         # tick-box: use query expansion in retrieval


# For RAG and MULTI_STEP, the answer is accompanied by a list of sources and a grounding report.
class Source(BaseModel):
    # One evidence slip stapled to the answer (matches rag_chain's sources dicts).
    chunk_id: str
    source: str                                  # filename
    page: str | int | None = None
    chunk_index: str | int | None = None
    score: float
    text_preview: str


class ClaimVerdict(BaseModel):
    # Auditor's stamp on one claim: SUPPORTED / UNSUPPORTED / CONTRADICTED.
    claim: str
    status: str
    max_score: float


class GroundingReport(BaseModel):
    # The Auditor's cover sheet (mirrors GroundingResult).
    score: float
    is_grounded: bool
    total_claims: int
    grounded_claims: int
    verdicts: list[ClaimVerdict] = []


class ReasoningStep(BaseModel):
    # One sub-question the Analyst worked through (MULTI_STEP only).
    sub_question: str
    answer: str


class QueryResponse(BaseModel):
    # The sealed envelope handed back at the /query window.
    answer: str
    route: str                                   # "direct" | "rag" | "multi_step"
    conversation_id: str                         # ticket echoed back (or freshly issued)
    sources: list[Source] = []                   # DIRECT ⇒ stays empty
    grounding: GroundingReport | None = None     # DIRECT ⇒ None (nothing to audit)
    reasoning_steps: list[ReasoningStep] = []    # MULTI_STEP only


# For Agent requests and response, the answer is accompanied by a list of sources and a grounding report.

class AgentStartRequest(BaseModel):
    # Form to open a new case at the agent window.
    question: str = Field(min_length=1)
    supervisor: bool = False                     # False = lone detective, True = the Chief's firm
    conversation_id: str | None = None           # 4.2b: scopes search_documents to this chat's docs


class AgentResumeRequest(BaseModel):
    # Form the visitor brings back after seeing the approval modal.
    thread_id: str                               # the claim ticket from start()
    approved: bool
    reason: str | None = None                    # optional note ("denied: don't run code")
    supervisor: bool = False                     # must match start() — resume the SAME desk
    conversation_id: str | None = None           # 4.2b: must match start() — same guest list on thaw


class ApprovalRequest(BaseModel):
    # The note the detective slides out when frozen mid-case (interrupt payload).
    tool: str                                    # e.g. "run_python"
    args: dict
    message: str                                 # human-readable "agent wants to ..."


class AgentReport(BaseModel):
    # One junior's filed report (supervisor mode only).
    agent: str                                   # "retriever" | "reasoner" | "validator"
    report: str
    trace: list[dict] = []                       # that junior's Thought/Action/Observation log


class AgentResponse(BaseModel):
    # Envelope for BOTH /agent/start and /agent/resume — same shape, so approvals can loop.
    status: str                                  # "done" | "awaiting_approval"
    thread_id: str                               # always returned; needed to resume
    answer: str | None = None                    # filled only when status == "done"
    trace: list[dict] = []                       # scratchpad (lone-detective mode)
    reports: list[AgentReport] = []              # juniors' reports (supervisor mode)
    request: ApprovalRequest | None = None       # filled only when awaiting_approval


# For files ingesting
class IngestResponse(BaseModel):
    # Receipt for a document drop-off (mirrors IngestPipeline.ingest()'s return).
    files_processed: int
    total_chunks: int
    corpus_summary: str | None = None