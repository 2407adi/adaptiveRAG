/* API client — every request stamps X-API-Key (fetch-based SSE POSTs included,
   which is exactly why we don't use native EventSource: it can't set headers).
   Any 401 fires onUnauthorized so the app can bounce back to the landing page
   (that's also the key-rotation kill switch working as designed). */

import type { ConvListItem, ServerTurn, SseEvent } from "./types";

const KEY_STORAGE = "adaptiverag_api_key";
const CLIENT_STORAGE = "adaptiverag_client_id";

export function getStoredKey(): string | null {
  return localStorage.getItem(KEY_STORAGE);
}
export function storeKey(key: string) {
  localStorage.setItem(KEY_STORAGE, key);
}
export function clearKey() {
  localStorage.removeItem(KEY_STORAGE);
}

/* Anonymous browser identity: minted once per browser, sent as X-Client-Id on
   every request. The server stamps new conversations with it and filters the
   sidebar by it — per-browser privacy without a login flow. */
export function clientId(): string {
  let id = localStorage.getItem(CLIENT_STORAGE);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(CLIENT_STORAGE, id);
  }
  return id;
}

/* Explicit logout rotates the identity, so the next visitor at this machine
   starts with a clean sidebar. (The 401 bounce does NOT rotate — same person,
   new key, history should survive.) */
export function rotateClientId() {
  localStorage.removeItem(CLIENT_STORAGE);
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn;
}

function headers(json = true): Record<string, string> {
  const h: Record<string, string> = { "X-Client-Id": clientId() };
  const key = getStoredKey();
  if (key) h["X-API-Key"] = key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

async function check(resp: Response): Promise<Response> {
  if (resp.ok) return resp;
  if (resp.status === 401) onUnauthorized();
  let detail = resp.statusText;
  try {
    const body = await resp.json();
    if (body?.detail) detail = String(body.detail);
  } catch { /* non-JSON error body — keep statusText */ }
  throw new ApiError(resp.status, detail);
}

/* ---------- plain JSON endpoints ---------- */

export async function validateKey(key: string): Promise<boolean> {
  // No dedicated auth endpoint — a cheap authenticated GET is the check.
  const resp = await fetch("/conversations", {
    headers: { "X-API-Key": key, "X-Client-Id": clientId() },
  });
  if (resp.status === 401 || resp.status === 403) return false;
  return resp.ok;
}

export async function clearMyConversations(): Promise<number> {
  const resp = await check(await fetch("/conversations", { method: "DELETE", headers: headers(false) }));
  const body = await resp.json();
  return body.deleted as number;
}

export async function listConversations(): Promise<ConvListItem[]> {
  const resp = await check(await fetch("/conversations", { headers: headers(false) }));
  const body = await resp.json();
  return body.conversations as ConvListItem[];
}

export async function getConversation(id: string): Promise<{ title: string | null; turns: ServerTurn[] }> {
  const resp = await check(await fetch(`/conversations/${id}`, { headers: headers(false) }));
  return (await resp.json()) as { title: string | null; turns: ServerTurn[] };
}

/* ---------- ingest (async: upload → claim ticket → poll) ---------- */

export interface IngestResult { files_processed: number; total_chunks: number; corpus_summary?: string | null; }

export interface IngestAccepted { job_id: string; status: string; files: string[]; }

export interface IngestJobStatus {
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
  stage?: string | null;
  chunks_done: number;
  chunks_total: number;
  files: string[];
  error?: string | null;
  result?: IngestResult | null;
}

/* Upload the files; resolves as soon as the server accepts the job (202). */
export function ingestFiles(
  files: File[],
  conversationId: string,
  onProgress: (fraction: number) => void,
): Promise<IngestAccepted> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("conversation_id", conversationId);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/ingest");
    const key = getStoredKey();
    if (key) xhr.setRequestHeader("X-API-Key", key);
    xhr.setRequestHeader("X-Client-Id", clientId());
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText) as IngestAccepted);
      } else {
        if (xhr.status === 401) onUnauthorized();
        let detail = `upload failed (${xhr.status})`;
        try {
          const body = JSON.parse(xhr.responseText);
          if (body?.detail) detail = String(body.detail);
        } catch { /* keep default */ }
        if (xhr.status === 403) detail = "Your key can't upload documents (user role) — ask for an admin key.";
        reject(new ApiError(xhr.status, detail));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "network error during upload"));
    xhr.send(form);
  });
}

/* One poll of the job record. 404 = the server restarted mid-job. */
export async function ingestStatus(jobId: string): Promise<IngestJobStatus> {
  const resp = await check(await fetch(`/ingest/status/${jobId}`, { headers: headers(false) }));
  return (await resp.json()) as IngestJobStatus;
}

/* Poll until the job finishes; onUpdate fires on every tick. */
export async function waitForIngest(
  jobId: string,
  onUpdate: (st: IngestJobStatus) => void,
  intervalMs = 1500,
  timeoutMs = 30 * 60 * 1000,          // a monster doc on a tiny CPU is slow — be patient
): Promise<IngestJobStatus> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const st = await ingestStatus(jobId);
    onUpdate(st);
    if (st.status === "done" || st.status === "failed") return st;
    if (Date.now() > deadline) throw new ApiError(0, "ingestion timed out — check back later");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

/* ---------- SSE streams ---------- */

/* Parse a text/event-stream body: notes are "data: {...}\n\n". */
async function* readSse(resp: Response): AsyncGenerator<SseEvent> {
  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) yield JSON.parse(line.slice(6)) as SseEvent;
      }
    }
  }
}

async function streamPost(path: string, body: unknown): Promise<AsyncGenerator<SseEvent>> {
  const resp = await check(await fetch(path, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
  }));
  return readSse(resp);
}

export function chatStream(question: string, conversationId: string | null) {
  return streamPost("/chat/stream", {
    question,
    conversation_id: conversationId,
    expand: false,
  });
}

export function agentStartStream(question: string, conversationId: string, supervisor: boolean) {
  return streamPost("/agent/start/stream", {
    question,
    conversation_id: conversationId,
    supervisor,
  });
}

export function agentResumeStream(
  threadId: string,
  approved: boolean,
  reason: string | null,
  conversationId: string,
  supervisor: boolean,
) {
  return streamPost("/agent/resume/stream", {
    thread_id: threadId,
    approved,
    reason: reason || null,
    conversation_id: conversationId,
    supervisor,
  });
}
