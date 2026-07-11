/* Shared types — mirrors the FastAPI contracts (api/models.py) and the
   typed SSE wire protocol (api/routes.py, Block 4.1). */

export interface Source {
  chunk_id?: string;
  source: string;
  page?: string | number | null;
  chunk_index?: string | number | null;
  score: number;
  text_preview: string;
  full_text?: string;
}

export interface Verdict {
  claim: string;
  status: "supported" | "unsupported" | "contradicted";
  max_score: number;
}

export interface Grounding {
  score: number;
  is_grounded: boolean;
  total_claims: number;
  grounded_claims: number;
  verdicts: Verdict[];
}

export interface TraceEntry {
  type: "thought" | "action" | "observation";
  content?: string;
  tool?: string;
  args?: Record<string, unknown>;
}

export interface ApprovalRequest {
  tool: string;
  args: Record<string, unknown>;
  message: string;
}

/* One SSE note from any streaming endpoint. */
export interface SseEvent {
  type:
    | "route" | "stage" | "sources" | "token" | "grounding" | "done" | "error"
    | "thought" | "trace" | "approval"
    | "handoff" | "report_token" | "report";
  // route
  route?: string;
  conversation_id?: string;
  // stage
  stage?: string;
  index?: number;
  total?: number;
  text?: string;
  // sources
  sources?: Source[];
  // grounding (flattened onto the event)
  score?: number;
  is_grounded?: boolean;
  total_claims?: number;
  grounded_claims?: number;
  verdicts?: Verdict[];
  // agent
  entry?: TraceEntry;
  request?: ApprovalRequest & { type?: string };
  thread_id?: string;
  agent?: string;
  report?: string;
  answer?: string;
  // error
  detail?: string;
}

export interface WorkerReport {
  agent: string;
  report: string;      // the filed report (final)
  draft: string;       // report_token accumulation while streaming
  thinking: string;    // live "thought" dictation
  trace: TraceEntry[];
}

/* Client-side chat message model. Assistant messages accumulate the rich
   panels as SSE events arrive; they persist for the session in a cache. */
export type Mode = "chat" | "agent" | "sup";

export interface UserMsg { role: "user"; text: string; }

export interface IngestMsg {
  role: "ingest";
  files: string[];
  status: "uploading" | "processing" | "done" | "error";
  progress?: number;
  detail?: string;
  chunks?: number;
}

export interface AssistantMsg {
  role: "assistant";
  kind: Mode;
  text: string;
  streaming: boolean;
  route?: string;
  stages?: string[];
  subQuestions?: string[];
  sources?: Source[];
  grounding?: Grounding;
  // lone agent
  trace?: TraceEntry[];
  liveThought?: string;
  // supervisor
  handoffs?: string[];
  reports?: WorkerReport[];
  // approval freeze
  pendingApproval?: { request: ApprovalRequest; threadId: string };
  denied?: string | null;
  error?: string;
}

export type Msg = UserMsg | IngestMsg | AssistantMsg;

export interface ConvListItem { id: string; title: string | null; turns: number; }

export interface ServerTurn { role: string; content: string; }
