import { useEffect, useRef, useState } from "react";
import type { AssistantMsg, Grounding, IngestMsg, Msg, Source, TraceEntry, WorkerReport } from "../types";
import {
  BulbIcon, CheckCircleIcon, CheckIcon, ChevronIcon, FileIcon,
  SearchIcon, WarnIcon, XIcon, ZapIcon,
} from "../icons";

/* ---------- small building blocks ---------- */

const ROUTE_STYLES: Record<string, string> = {
  direct: "bg-mut/10 text-mut",
  rag: "bg-accent/10 text-accent",
  multi_step: "bg-accent/10 text-accent",
  agent: "bg-accent/10 text-accent",
  supervisor: "bg-accent/10 text-accent",
};

function RouteBadge({ label }: { label: string }) {
  return (
    <span className={`inline-block text-[10.5px] font-bold tracking-wide px-2 py-0.5 rounded-md mr-1.5 mb-2 uppercase ${ROUTE_STYLES[label] ?? ROUTE_STYLES.rag}`}>
      {label === "multi_step" ? "route: multi-step" : label === "direct" || label === "rag" ? `route: ${label}` : label}
    </span>
  );
}

function Panel({ head, children, defaultOpen = true }: {
  head: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="my-2.5 border border-line rounded-xl overflow-hidden">
      <button
        className="w-full px-3.5 py-2.5 bg-panel2 text-xs font-semibold text-mut flex items-center gap-2 select-none"
        onClick={() => setOpen((o) => !o)}
      >
        {head}
        <span className="ml-auto text-dim"><ChevronIcon size={12} open={open} /></span>
      </button>
      {open && <div className="px-3.5 py-3">{children}</div>}
    </div>
  );
}

/* Animated "what's happening" line: pulsing dot + shimmering label.
   Shown whenever the stream is waiting on the server between phases —
   replaces the bare blinking cursor that made waits feel broken. */
function StatusLine({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2 py-0.5 text-[13px] text-mut select-none">
      <span className="relative flex h-2 w-2 shrink-0">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-60" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-accent" />
      </span>
      <span className="animate-pulse">{text}</span>
    </div>
  );
}

function Stages({ stages }: { stages: string[] }) {
  return (
    <div className="flex flex-wrap gap-1.5 mb-2.5">
      {stages.map((s, i) => (
        <span
          key={i}
          className={`text-[11.5px] px-2.5 py-0.5 rounded-full border ${
            i === stages.length - 1
              ? "text-ink border-line2 bg-panel2"
              : "text-accent border-accent/35 bg-panel2"
          }`}
        >
          {i === stages.length - 1 ? `… ${s}` : `✓ ${s}`}
        </span>
      ))}
    </div>
  );
}

function SourcesList({ sources }: { sources: Source[] }) {
  return (
    <div className="mt-3 pt-2.5 border-t border-line">
      <h4 className="text-[10.5px] uppercase tracking-wider text-dim font-bold mb-1.5">Sources · retrieved &amp; reranked</h4>
      {sources.map((s, i) => (
        <div key={i} className="flex items-baseline gap-2.5 px-2.5 py-1.5 rounded-lg bg-panel2 mb-1 text-xs min-w-0">
          <span className="text-accent font-bold shrink-0">[{i + 1}]</span>
          <span className="font-semibold whitespace-nowrap shrink-0 max-w-[45%] truncate">{s.source}</span>
          <span className="text-mut truncate flex-1">{s.text_preview}</span>
          <span className="text-dim text-[11px] whitespace-nowrap shrink-0 hidden sm:inline">score {s.score.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function GroundingBadge({ g }: { g: Grounding }) {
  const [open, setOpen] = useState(false);
  const cls = g.score >= 0.99 ? "bg-ok/10 text-ok border-ok/35"
    : g.score >= 0.4 ? "bg-warn/10 text-warn border-warn/35"
    : "bg-bad/10 text-bad border-bad/35";
  const Icon = g.score >= 0.99 ? CheckIcon : g.score >= 0.4 ? WarnIcon : XIcon;
  const label = g.score >= 0.99 ? "Grounded" : g.score >= 0.4 ? "Partially grounded" : "Poorly grounded";
  const STATUS = {
    supported: ["SUPPORTED", "bg-ok/15 text-ok"],
    contradicted: ["CONTRADICTED", "bg-bad/15 text-bad"],
    unsupported: ["UNSUPPORTED", "bg-warn/15 text-warn"],
  } as const;
  return (
    <div className="mt-2.5">
      <button className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold border ${cls}`} onClick={() => setOpen((o) => !o)}>
        <Icon size={13} /> {label} — {g.grounded_claims}/{g.total_claims} claims ({Math.round(g.score * 100)}%)
        <ChevronIcon size={11} open={open} />
      </button>
      {open && (
        <div className="mt-2 space-y-1">
          {g.verdicts.map((v, i) => {
            const [txt, chip] = STATUS[v.status] ?? STATUS.unsupported;
            return (
              <div key={i} className="flex items-baseline gap-2 px-2.5 py-1.5 rounded-lg bg-panel2 text-xs">
                <span className={`text-[10px] font-bold px-2 py-px rounded shrink-0 ${chip}`}>{txt}</span>
                <span className="text-mut">{v.claim}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ---------- agent trace ---------- */

const KIND_STYLE: Record<string, string> = {
  thought: "text-mut",
  action: "text-accent",
  observation: "text-ok",
};

function TraceLine({ entry }: { entry: TraceEntry }) {
  return (
    <div className="flex gap-2.5 mb-2 text-xs">
      <span className={`shrink-0 w-[86px] font-bold text-[10.5px] uppercase tracking-wide pt-0.5 ${KIND_STYLE[entry.type] ?? "text-mut"}`}>
        {entry.type}
      </span>
      <span className="text-mut min-w-0 break-words">
        {entry.type === "action" ? (
          <code className="bg-[#111] border border-line rounded px-1.5 py-px text-[11.5px] text-[#dcdcdc] break-all">
            {entry.tool}({entry.args ? JSON.stringify(entry.args) : ""})
          </code>
        ) : (
          (entry.content ?? "").length > 700 ? `${entry.content!.slice(0, 700)} …` : entry.content
        )}
      </span>
    </div>
  );
}

function TracePanel({ trace, liveThought, pending, denied }: {
  trace: TraceEntry[]; liveThought?: string; pending?: boolean; denied?: string | null;
}) {
  return (
    <Panel head={<><SearchIcon size={13} /> Reasoning trace{pending ? " — frozen, awaiting your decision" : ""}</>}>
      {trace.map((t, i) => <TraceLine key={i} entry={t} />)}
      {liveThought && (
        <div className="flex gap-2.5 mb-2 text-xs">
          <span className="shrink-0 w-[86px] font-bold text-[10.5px] uppercase tracking-wide pt-0.5 text-mut">thought</span>
          <span className="text-mut caret">{liveThought}</span>
        </div>
      )}
      {pending && (
        <div className="flex gap-2.5 text-xs">
          <span className="shrink-0 w-[86px] font-bold text-[10.5px] uppercase tracking-wide pt-0.5 text-warn">pending</span>
          <span className="text-warn">Tool call awaiting your approval — see the dialog.</span>
        </div>
      )}
      {denied && (
        <div className="flex gap-2.5 text-xs">
          <span className="shrink-0 w-[86px] font-bold text-[10.5px] uppercase tracking-wide pt-0.5 text-bad">denied</span>
          <span className="text-mut">Tool call denied{denied !== "denied" ? ` ("${denied}")` : ""} — the agent replans without it.</span>
        </div>
      )}
    </Panel>
  );
}

/* ---------- supervisor ---------- */

const WORKER_ICONS: Record<string, React.ReactNode> = {
  retriever: <SearchIcon size={13} />,
  reasoner: <ZapIcon size={13} />,
  validator: <CheckCircleIcon size={13} />,
};

function ReportCard({ r }: { r: WorkerReport }) {
  return (
    <Panel head={<>{WORKER_ICONS[r.agent] ?? <SearchIcon size={13} />} <span className="capitalize font-bold text-ink">{r.agent}</span></>}>
      {r.trace.map((t, i) => <TraceLine key={i} entry={t} />)}
      {r.thinking && (
        <div className="text-xs text-mut caret mb-2">{r.thinking}</div>
      )}
      {(r.report || r.draft) && (
        <div className="text-xs">
          <b className="text-ink">Report filed:</b>{" "}
          <span className={`text-mut ${r.report ? "" : "caret"}`}>{r.report || r.draft}</span>
        </div>
      )}
    </Panel>
  );
}

function Handoffs({ agents, reports }: { agents: string[]; reports: WorkerReport[] }) {
  const active = new Set(reports.map((r) => r.agent));
  const seen: string[] = [];
  for (const a of agents) if (!seen.includes(a)) seen.push(a);
  return (
    <div className="flex flex-wrap items-center gap-1.5 mb-2 text-xs text-mut">
      Chief dispatching:
      {seen.map((a, i) => (
        <span key={a} className="flex items-center gap-1.5">
          {i > 0 && <span className="text-dim">→</span>}
          <span className={`inline-flex items-center gap-1 px-2.5 py-px rounded-full border font-semibold capitalize ${active.has(a) ? "border-accent/45 bg-panel2 opacity-100" : "border-line bg-panel2 opacity-40"}`}>
            {WORKER_ICONS[a]} {a}
          </span>
        </span>
      ))}
    </div>
  );
}

/* ---------- ingest card ---------- */

function IngestCard({ m }: { m: IngestMsg }) {
  return (
    <div className="border border-line rounded-xl bg-panel px-4 py-3 mb-4 text-[13px]">
      <b>Uploading to this chat</b>{" "}
      <span className="text-dim text-[11.5px]">(private — other chats can't see these)</span>
      <div className="flex flex-wrap gap-2 my-2">
        {m.files.map((f) => (
          <span key={f} className="inline-flex items-center gap-1.5 bg-panel2 border border-line rounded-md px-2.5 py-0.5 text-xs">
            <FileIcon size={12} /> {f}
          </span>
        ))}
      </div>
      {m.status !== "done" && m.status !== "error" && (
        <>
          <div className="h-1.5 rounded bg-panel2 overflow-hidden">
            <i
              className="block h-full bg-gradient-to-r from-[#ec8450] to-[#c85a28] transition-all"
              style={{ width: `${Math.round((m.progress ?? 0) * 100)}%` }}
            />
          </div>
          <div className="text-mut text-xs mt-2">
            {m.status === "uploading"
              ? "uploading…"
              : `processing — ${m.stage ?? "queued"}${
                  m.chunksTotal ? ` (${m.chunksDone ?? 0}/${m.chunksTotal} chunks)` : ""
                }…`}
          </div>
        </>
      )}
      {m.status === "done" && (
        <div className="text-ok text-xs mt-1 flex items-center gap-1.5">
          <CheckIcon size={12} /> Ingested <b>{m.files.length} file{m.files.length > 1 ? "s" : ""} → {m.chunks} chunks</b> · ready to query
        </div>
      )}
      {m.status === "error" && (
        <div className="text-bad text-xs mt-1 flex items-center gap-1.5"><XIcon size={12} /> {m.detail}</div>
      )}
    </div>
  );
}

/* ---------- assistant bubble ---------- */

function AssistantBubble({ m }: { m: AssistantMsg }) {
  const badge = m.kind === "agent" ? "agent" : m.kind === "sup" ? "supervisor" : m.route;
  const isSup = m.kind === "sup";
  return (
    <div className="flex mb-4">
      <div className="bg-panel border border-line rounded-2xl rounded-tl-[4px] px-4 py-3 max-w-[94%] min-w-0 text-sm leading-relaxed">
        {badge && <RouteBadge label={badge} />}
        {m.stages && m.stages.length > 0 && <Stages stages={m.stages} />}
        {m.subQuestions && m.subQuestions.length > 0 && (
          <Panel head={<><BulbIcon size={13} /> Thought process — {m.subQuestions.length} sub-question{m.subQuestions.length > 1 ? "s" : ""}</>}>
            {m.subQuestions.map((q, i) => (
              <div key={i} className="mb-2 text-[13px]"><span className="text-accent font-semibold">{i + 1}.</span> <span className="text-mut">{q}</span></div>
            ))}
          </Panel>
        )}
        {isSup && m.handoffs && m.handoffs.length > 0 && <Handoffs agents={m.handoffs} reports={m.reports ?? []} />}
        {isSup && (m.reports ?? []).map((r) => <ReportCard key={r.agent} r={r} />)}
        {m.kind === "agent" && (m.trace?.length || m.liveThought || m.pendingApproval || m.denied) ? (
          <TracePanel trace={m.trace ?? []} liveThought={m.liveThought} pending={Boolean(m.pendingApproval)} denied={m.denied} />
        ) : null}
        {isSup && m.text && <div className="mb-1"><b className="text-accent">Chief's verdict:</b>{" "}</div>}
        {m.streaming && m.status && !m.pendingApproval && <StatusLine text={m.status} />}
        {(m.text || !m.streaming) && (
          <div className={`whitespace-pre-wrap break-words ${m.streaming && !m.pendingApproval && !m.status ? "caret" : ""}`}>{m.text}</div>
        )}
        {m.sources && m.sources.length > 0 && <SourcesList sources={m.sources} />}
        {m.grounding && <GroundingBadge g={m.grounding} />}
        {m.error && (
          <div className="mt-2 text-xs text-bad flex items-center gap-1.5"><WarnIcon size={13} /> {m.error}</div>
        )}
      </div>
    </div>
  );
}

/* ---------- the list ---------- */

export default function Messages({ msgs }: { msgs: Msg[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [msgs]);

  return (
    <div className="max-w-[800px] mx-auto px-4 sm:px-6 min-w-0">
      {msgs.map((m, i) =>
        m.role === "user" ? (
          <div key={i} className="flex justify-end mb-4">
            <div className="bg-panel3 rounded-2xl rounded-br-[4px] px-4 py-2.5 max-w-[85%] text-sm whitespace-pre-wrap break-words">{m.text}</div>
          </div>
        ) : m.role === "ingest" ? (
          <IngestCard key={i} m={m} />
        ) : (
          <AssistantBubble key={i} m={m} />
        ),
      )}
      <div ref={endRef} />
    </div>
  );
}
