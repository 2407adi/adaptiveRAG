/* The chat engine: conversation list + per-conversation message cache +
   SSE consumption for all three layers (chat / agent / supervisor).

   Design notes:
   - Sidebar switching is blocked while a stream is live (`busy`), so every
     in-flight event can safely target the ACTIVE conversation.
   - Rich panels (sources, grounding, traces, reports) live in a client-side
     cache per conversation id — they persist for the whole browser session.
     Turns restored from the server (after a reload) are plain text, by design:
     the 4.3a store persists role+content only.
   - New chats: chat mode lets the SERVER mint the conversation_id (it arrives
     on the `route` event); agent mode mints a client-side UUID because the
     agent endpoints only file turns when given a ticket up front. */

import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api";
import type {
  ApprovalRequest, AssistantMsg, ConvListItem, Mode, Msg, SseEvent, WorkerReport,
} from "../types";

const NEW_KEY = "__new__";

const STAGE_LABELS: Record<string, string> = {
  retrieving: "retrieving evidence",
  decomposing: "decomposing the question",
  synthesizing: "synthesizing the answer",
};

interface PendingApproval {
  request: ApprovalRequest;
  threadId: string;
  supervisor: boolean;
  convId: string;
}

export function useChat() {
  const [items, setItems] = useState<ConvListItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [approval, setApproval] = useState<PendingApproval | null>(null);
  const cache = useRef(new Map<string, Msg[]>());

  const keyOf = (id: string | null) => id ?? NEW_KEY;

  /* keep the cache mirroring whatever is on screen */
  useEffect(() => {
    cache.current.set(keyOf(activeId), msgs);
  }, [activeId, msgs]);

  const refreshList = useCallback(async () => {
    try {
      setItems(await api.listConversations());
    } catch { /* sidebar refresh is best-effort; a 401 already bounced us */ }
  }, []);

  useEffect(() => { void refreshList(); }, [refreshList]);

  /* ---------- message patch helpers ---------- */

  const patchLast = useCallback((fn: (m: AssistantMsg) => AssistantMsg) => {
    setMsgs((ms) => {
      const last = ms[ms.length - 1];
      if (!last || last.role !== "assistant") return ms;
      return [...ms.slice(0, -1), fn(last)];
    });
  }, []);

  const patchWorker = useCallback((agent: string, fn: (w: WorkerReport) => WorkerReport) => {
    patchLast((m) => {
      const reports = [...(m.reports ?? [])];
      let i = reports.findIndex((r) => r.agent === agent);
      if (i < 0) {
        reports.push({ agent, report: "", draft: "", thinking: "", trace: [] });
        i = reports.length - 1;
      }
      reports[i] = fn(reports[i]);
      return { ...m, reports };
    });
  }, [patchLast]);

  /* when the server (or we) mint an id for a fresh chat, move the draft over */
  const adoptId = useCallback((id: string | undefined) => {
    if (!id) return;
    setActiveId((current) => {
      if (current === id) return current;
      cache.current.delete(NEW_KEY);
      return id;
    });
  }, []);

  /* ---------- event application ---------- */

  const applyChatEvent = useCallback((ev: SseEvent) => {
    switch (ev.type) {
      case "route":
        adoptId(ev.conversation_id);
        patchLast((m) => ({ ...m, route: ev.route }));
        break;
      case "stage":
        if (ev.stage === "sub_question") {
          patchLast((m) => ({ ...m, subQuestions: [...(m.subQuestions ?? []), ev.text ?? ""] }));
        } else {
          const label = STAGE_LABELS[ev.stage ?? ""] ?? ev.stage ?? "";
          patchLast((m) => ({ ...m, stages: [...(m.stages ?? []), label] }));
        }
        break;
      case "sources":
        patchLast((m) => ({ ...m, sources: ev.sources }));
        break;
      case "token":
        patchLast((m) => ({ ...m, text: m.text + (ev.text ?? "") }));
        break;
      case "grounding":
        patchLast((m) => ({
          ...m,
          grounding: {
            score: ev.score ?? 0,
            is_grounded: ev.is_grounded ?? false,
            total_claims: ev.total_claims ?? 0,
            grounded_claims: ev.grounded_claims ?? 0,
            verdicts: ev.verdicts ?? [],
          },
        }));
        break;
      case "error":
        patchLast((m) => ({ ...m, error: ev.detail }));
        break;
      case "done":
        adoptId(ev.conversation_id);
        break;
      default:
        break;
    }
  }, [adoptId, patchLast]);

  /* returns "frozen" if the stream ended on an approval interrupt */
  const consumeAgent = useCallback(async (
    gen: AsyncGenerator<SseEvent>,
    supervisor: boolean,
    convId: string,
  ): Promise<"done" | "frozen"> => {
    for await (const ev of gen) {
      switch (ev.type) {
        case "thought":
          if (supervisor && ev.agent) {
            patchWorker(ev.agent, (w) => ({ ...w, thinking: w.thinking + (ev.text ?? "") }));
          } else {
            patchLast((m) => ({ ...m, liveThought: (m.liveThought ?? "") + (ev.text ?? "") }));
          }
          break;
        case "trace": {
          const entry = ev.entry!;
          if (supervisor && ev.agent) {
            patchWorker(ev.agent, (w) => ({
              ...w,
              trace: [...w.trace, entry],
              thinking: entry.type === "thought" ? "" : w.thinking,  // final form got filed
            }));
          } else {
            patchLast((m) => ({
              ...m,
              trace: [...(m.trace ?? []), entry],
              liveThought: entry.type === "thought" ? "" : m.liveThought,
            }));
          }
          break;
        }
        case "token":
          patchLast((m) => ({ ...m, text: m.text + (ev.text ?? "") }));
          break;
        case "report_token":
          if (ev.agent) patchWorker(ev.agent, (w) => ({ ...w, draft: w.draft + (ev.text ?? "") }));
          break;
        case "report":
          if (ev.agent) patchWorker(ev.agent, (w) => ({ ...w, report: ev.report ?? w.draft, draft: "", thinking: "" }));
          break;
        case "handoff":
          if (ev.agent) {
            patchLast((m) => ({ ...m, handoffs: [...(m.handoffs ?? []), ev.agent!] }));
            patchWorker(ev.agent, (w) => w);   // make sure the card exists immediately
          }
          break;
        case "approval": {
          const request: ApprovalRequest = {
            tool: ev.request?.tool ?? "?",
            args: ev.request?.args ?? {},
            message: ev.request?.message ?? "The agent wants to use a tool.",
          };
          const pending = { request, threadId: ev.thread_id ?? "", supervisor, convId };
          setApproval(pending);
          patchLast((m) => ({ ...m, pendingApproval: { request, threadId: pending.threadId } }));
          return "frozen";
        }
        case "done":
          // supervisor's budget-exhausted answer can arrive ONLY via done
          patchLast((m) => ({ ...m, text: m.text || ev.answer || "", liveThought: "" }));
          break;
        case "error":
          patchLast((m) => ({ ...m, error: ev.detail }));
          break;
        default:
          break;
      }
    }
    return "done";
  }, [patchLast, patchWorker]);

  const finalize = useCallback(() => {
    patchLast((m) => ({ ...m, streaming: false, liveThought: "" }));
    setBusy(false);
    void refreshList();
  }, [patchLast, refreshList]);

  /* ---------- public actions ---------- */

  const send = useCallback(async (question: string, mode: Mode) => {
    const q = question.trim();
    if (!q || busy) return;
    setBusy(true);

    const placeholder: AssistantMsg = { role: "assistant", kind: mode, text: "", streaming: true };
    setMsgs((ms) => [...ms, { role: "user", text: q }, placeholder]);

    try {
      if (mode === "chat") {
        const gen = await api.chatStream(q, activeId);
        for await (const ev of gen) applyChatEvent(ev);
        finalize();
      } else {
        const supervisor = mode === "sup";
        const convId = activeId ?? crypto.randomUUID();  // agent endpoints need the ticket up front
        adoptId(convId);
        const gen = await api.agentStartStream(q, convId, supervisor);
        const result = await consumeAgent(gen, supervisor, convId);
        if (result === "done") finalize();
        // frozen: busy stays true; the approval modal owns the next move
      }
    } catch (e) {
      patchLast((m) => ({ ...m, error: e instanceof Error ? e.message : String(e) }));
      finalize();
    }
  }, [activeId, adoptId, applyChatEvent, busy, consumeAgent, finalize, patchLast]);

  const decideApproval = useCallback(async (approved: boolean, reason: string) => {
    const a = approval;
    if (!a) return;
    setApproval(null);
    patchLast((m) => ({ ...m, pendingApproval: undefined, denied: approved ? null : (reason || "denied") }));
    try {
      const gen = await api.agentResumeStream(a.threadId, approved, reason || null, a.convId, a.supervisor);
      const result = await consumeAgent(gen, a.supervisor, a.convId);
      if (result === "done") finalize();
    } catch (e) {
      patchLast((m) => ({ ...m, error: e instanceof Error ? e.message : String(e) }));
      finalize();
    }
  }, [approval, consumeAgent, finalize, patchLast]);

  const openConversation = useCallback(async (id: string) => {
    if (busy || id === activeId) return;
    const cached = cache.current.get(id);
    if (cached) {
      setActiveId(id);
      setMsgs(cached);
      return;
    }
    try {
      const conv = await api.getConversation(id);
      const restored: Msg[] = conv.turns.map((t) =>
        t.role === "user"
          ? { role: "user", text: t.content }
          : { role: "assistant", kind: "chat", text: t.content, streaming: false },
      );
      setActiveId(id);
      setMsgs(restored);
    } catch { /* 404/401 handled globally; stay where we are */ }
  }, [activeId, busy]);

  const newChat = useCallback(() => {
    if (busy) return;
    cache.current.delete(NEW_KEY);
    setActiveId(null);
    setMsgs([]);
  }, [busy]);

  const clearAll = useCallback(async () => {
    if (busy) return;
    try {
      await api.clearMyConversations();   // shreds ONLY this browser's drawer
    } catch { /* 401 handled globally */ }
    cache.current.clear();
    setActiveId(null);
    setMsgs([]);
    void refreshList();
  }, [busy, refreshList]);

  const upload = useCallback(async (files: File[]) => {
    if (!files.length || busy) return;
    setBusy(true);
    const convId = activeId ?? crypto.randomUUID();
    adoptId(convId);
    const names = files.map((f) => f.name);
    setMsgs((ms) => [...ms, { role: "ingest", files: names, status: "uploading", progress: 0 }]);

    const patchIngest = (fn: (m: Msg & { role: "ingest" }) => Msg) =>
      setMsgs((ms) => {
        const i = ms.map((m) => m.role).lastIndexOf("ingest");
        if (i < 0) return ms;
        const copy = [...ms];
        copy[i] = fn(copy[i] as Msg & { role: "ingest" });
        return copy;
      });

    try {
      const res = await api.ingestFiles(files, convId, (frac) =>
        patchIngest((m) => ({ ...m, progress: frac, status: frac >= 1 ? "processing" : "uploading" })),
      );
      patchIngest((m) => ({ ...m, status: "done", chunks: res.total_chunks }));
    } catch (e) {
      patchIngest((m) => ({ ...m, status: "error", detail: e instanceof Error ? e.message : String(e) }));
    } finally {
      setBusy(false);
      void refreshList();
    }
  }, [activeId, adoptId, busy, refreshList]);

  const title = activeId ? (items.find((i) => i.id === activeId)?.title ?? "Untitled chat") : "New chat";

  return {
    items, activeId, msgs, busy, approval, title,
    send, decideApproval, openConversation, newChat, clearAll, upload,
  };
}
