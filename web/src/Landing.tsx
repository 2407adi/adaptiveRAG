import { useEffect, useRef, useState } from "react";
import { storeKey, validateKey } from "./api";
import { BookIcon, BotIcon, BranchIcon, KeyIcon, WarnIcon } from "./icons";

/* ============================================================
   Landing page — the approved mockup design (black + burnt orange),
   fully responsive: two-column hero collapses to one on phones.
============================================================ */

const HERO_ANSWER =
  "Solstice Robotics raised a **$12M Series A** led by Meridian Ventures [1]. Heads up — the Q1 investor update says **$15M** [2]: the two documents contradict each other.";

function useTypewriter(text: string, startDelay = 1200, speed = 18) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    let i = 0;
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      i += 1;
      setShown(i);
      if (i < text.length) timer = setTimeout(tick, speed);
    };
    timer = setTimeout(tick, startDelay);
    return () => clearTimeout(timer);
  }, [text, startDelay, speed]);
  return { visible: text.slice(0, shown), done: shown >= text.length };
}

/* tiny markdown-ish renderer for the hero demo: **bold** and [n] citations */
function RichLine({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d+\])/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("**") ? <b key={i}>{p.slice(2, -2)}</b>
        : /^\[\d+\]$/.test(p) ? <span key={i} className="text-accent font-semibold text-[0.92em] cursor-pointer">{p}</span>
        : <span key={i}>{p}</span>,
      )}
    </>
  );
}

function HeroDemoCard() {
  const { visible, done } = useTypewriter(HERO_ANSWER);
  return (
    <div className="relative z-10">
      {/* warm glow behind the card — plays the template's photo role */}
      <div
        className="absolute -inset-x-6 -inset-y-9 rounded-[30px] blur-[28px] -z-10"
        style={{
          background:
            "radial-gradient(60% 50% at 72% 18%, rgba(224,112,58,.30), transparent 70%), radial-gradient(50% 42% at 18% 85%, rgba(200,90,40,.20), transparent 70%)",
        }}
      />
      <div className="relative rounded-[22px] border border-line bg-panel overflow-hidden shadow-[0_30px_80px_rgba(0,0,0,.45)]">
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-line bg-bg2">
          <i className="w-2.5 h-2.5 rounded-full bg-bad/80" />
          <i className="w-2.5 h-2.5 rounded-full bg-warn/80" />
          <i className="w-2.5 h-2.5 rounded-full bg-ok/80" />
          <span className="ml-2 text-dim text-xs truncate">adaptiverag.app — your documents, 30 seconds after upload</span>
        </div>
        <div className="p-5 text-[13.5px]">
          <div className="flex justify-end mb-3">
            <div className="bg-panel3 px-3.5 py-2 rounded-[13px] rounded-br-[4px]">What was the size of the Series A?</div>
          </div>
          <div className="bg-bg2 border border-line rounded-[13px] rounded-tl-[4px] px-4 py-3">
            <span className="inline-block text-[10.5px] font-bold tracking-wide px-2 py-0.5 rounded-md mb-2 bg-accent/10 text-accent">ROUTE: RAG</span>
            <div className={done ? "" : "caret"}>
              <RichLine text={visible} />
            </div>
            {done && (
              <div className="mt-2.5">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold bg-warn/10 text-warn border border-warn/35">
                  <WarnIcon size={13} /> Partially grounded — 2/3 claims
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
      {/* floating card overlapping the corner (template style) */}
      <div className="lg:absolute lg:-right-3 lg:-bottom-8 mt-3 lg:mt-0 lg:w-[238px] bg-panel2 border border-line2 rounded-2xl px-4 py-3.5 shadow-[0_18px_50px_rgba(0,0,0,.5)] text-xs text-mut">
        <b className="block text-ink text-[13px] mb-1">Grounded by design</b>
        Every claim is checked against your sources — contradictions get flagged, not glossed over.
      </div>
    </div>
  );
}

function LoginModal({ open, notice, onClose, onAuthed }: {
  open: boolean; notice: string | null; onClose: () => void; onAuthed: () => void;
}) {
  const [key, setKey] = useState("");
  const [err, setErr] = useState<string | null>(notice);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setErr(notice); }, [notice]);
  useEffect(() => { if (open) setTimeout(() => inputRef.current?.focus(), 50); }, [open]);

  if (!open) return null;

  const submit = async () => {
    const k = key.trim();
    if (!k) { setErr("Key required."); return; }
    setBusy(true); setErr(null);
    try {
      if (await validateKey(k)) {
        storeKey(k);
        onAuthed();
      } else {
        setErr("That key was rejected (401). Check for typos or ask for a fresh key.");
      }
    } catch {
      setErr("Couldn't reach the server — is the API running?");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-[3px] p-5" onClick={onClose}>
      <div className="w-[430px] max-w-full bg-panel border border-line2 rounded-2xl p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-base font-bold flex items-center gap-2 mb-1.5">
          <span className="text-accent"><KeyIcon size={16} /></span> Enter your access key
        </h3>
        <p className="text-mut text-[13px] mb-3.5">
          Your key determines what you can do — <b className="text-ok">admin</b> keys can upload documents,{" "}
          <b className="text-accent">guest</b> keys can ask questions.
        </p>
        <input
          ref={inputRef}
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !busy && submit()}
          placeholder="paste your key…"
          className="w-full bg-bg2 border border-line2 rounded-lg px-3.5 py-2.5 text-sm outline-none focus:border-accent mb-1.5"
        />
        <div className="text-bad text-xs min-h-4 mb-2">{err}</div>
        <div className="flex justify-end gap-2.5">
          <button className="px-4 py-2 rounded-lg border border-line2 bg-panel2 text-sm font-semibold hover:border-accent" onClick={onClose}>
            Cancel
          </button>
          <button
            className="px-4 py-2 rounded-lg text-sm font-bold text-white bg-gradient-to-br from-[#ec8450] to-[#c85a28] disabled:opacity-50"
            disabled={busy}
            onClick={submit}
          >
            {busy ? "Checking…" : "Continue →"}
          </button>
        </div>
        <p className="text-[11.5px] text-dim mt-2.5">Stored locally in your browser. A rotated key simply brings you back here.</p>
      </div>
    </div>
  );
}

/* ---------- static landing sections ---------- */

const LAYERS = [
  {
    icon: <BookIcon size={20} />,
    title: "Grounded Q&A",
    body: "Hybrid retrieval (dense + BM25 + reranker) finds the evidence; every answer carries inline citations and a grounding badge that judges each claim as supported, unsupported, or contradicted.",
    ex: '"What does the SLA commit to?" → answer + [1][2] + ✓ Grounded 4/4',
  },
  {
    icon: <BranchIcon size={20} />,
    title: "Multi-step reasoning",
    body: "Complex questions get decomposed into sub-questions, each answered from your documents, then synthesized — with the thought process shown in the chat, not thrown away.",
    ex: '"Compare both contracts and compute max downtime" → 3 sub-questions → synthesis',
  },
  {
    icon: <BotIcon size={20} />,
    title: "Autonomous agents",
    body: "A ReAct agent (or a supervised multi-agent team) that searches your docs, runs Python in a locked sandbox, and reaches for the web — pausing for your approval before any risky tool fires.",
    ex: '"Verify the math yourself" → search → approve run_python → verified',
  },
];

const STEPS = [
  { n: 1, t: "Sign in", d: "Paste your access key. Admin keys can upload documents; guest keys can ask questions." },
  { n: 2, t: "Drop your files", d: "PDF, Word, Markdown, CSV, HTML, PowerPoint, plain text. Ingestion takes ~30 seconds and your uploads stay private to your chat." },
  { n: 3, t: "Ask anything", d: "Answers stream in live with citations. Flip to Agent mode when you want the system to work a multi-tool case for you." },
];

const FEATURES = [
  ["Token streaming", "on every path (SSE)"],
  ["Hallucination detection", "— per-claim entailment"],
  ["Hybrid retrieval", "+ cross-encoder reranker"],
  ["Human-in-the-loop", "tool approval"],
  ["Multi-agent supervisor", "with worker reports"],
  ["Audit-logged", "tool calls (HMAC chain)"],
  ["Chat-private uploads", "— scoped retrieval"],
  ["Role-based access", "+ rate limits + caps"],
];

export default function Landing({ notice, onAuthed }: { notice: string | null; onAuthed: () => void }) {
  const [login, setLogin] = useState(Boolean(notice));

  return (
    <div className="min-h-full overflow-x-hidden relative bg-bg text-ink">
      {/* ambient glows */}
      <div className="absolute w-[640px] h-[640px] rounded-full blur-[110px] opacity-10 bg-accent -top-56 -right-36 pointer-events-none" />
      <div className="absolute w-[520px] h-[520px] rounded-full blur-[110px] opacity-[.08] bg-[#8a3d16] top-[620px] -left-44 pointer-events-none" />

      {/* nav */}
      <nav className="max-w-[1180px] mx-auto px-6 py-5 flex items-center gap-3 relative z-10">
        <div className="text-[17px] font-extrabold flex items-center gap-2 tracking-tight">
          <span className="w-[11px] h-[11px] rounded-full bg-accent shadow-[0_0_10px_var(--color-accent)]" />
          AdaptiveRAG
        </div>
        <a href="#how" className="hidden sm:block text-mut text-[12.5px] border border-line2 rounded-full px-4 py-1.5 whitespace-nowrap hover:text-ink hover:border-accent">How it works</a>
        <a href="#layers" className="hidden sm:block text-mut text-[12.5px] border border-line2 rounded-full px-4 py-1.5 whitespace-nowrap hover:text-ink hover:border-accent">Capabilities</a>
        <div className="flex-1" />
        <button className="px-4 py-2 rounded-lg border border-line2 bg-panel2 text-sm font-semibold whitespace-nowrap hover:border-accent" onClick={() => setLogin(true)}>
          Log in
        </button>
        <button className="px-4 py-2 rounded-lg text-sm font-bold text-white whitespace-nowrap bg-gradient-to-br from-[#ec8450] to-[#c85a28] shadow-[0_4px_24px_rgba(224,112,58,.18)]" onClick={() => setLogin(true)}>
          Get started
        </button>
      </nav>

      {/* hero */}
      <header className="max-w-[1180px] mx-auto px-6 pt-10 lg:pt-14 pb-8 grid lg:grid-cols-[1.05fr_.95fr] gap-10 lg:gap-14 items-center relative z-10">
        <div>
          <div className="text-accent italic font-semibold text-[12.5px] mb-4">RAG · Reasoning · Agents — one engine, any documents</div>
          <h1 className="text-[31px] sm:text-[40px] lg:text-[52px] leading-[1.08] font-extrabold tracking-[-0.03em] mb-5">
            Drop in any documents.
            <br />
            <span className="bg-gradient-to-br from-[#ec8450] to-[#c85a28] bg-clip-text text-transparent">Start asking questions.</span>
          </h1>
          <p className="text-mut text-[15.5px] leading-relaxed max-w-[520px] mb-5">
            AdaptiveRAG ingests whatever you upload, figures out the domain on its own, and answers with citations,
            verified grounding, and autonomous agents. No setup. No configuration. No domain picker.
          </p>
          <div className="flex flex-wrap gap-2 mb-6">
            {["Contracts & SLAs", "Research papers", "Financial reports", "Meeting notes"].map((u) => (
              <span key={u} className="border border-line2 rounded-full px-3.5 py-1 text-xs text-mut">{u}</span>
            ))}
          </div>
          <div className="flex flex-wrap gap-3 mb-5">
            <button
              className="px-6 py-3 rounded-xl text-[15px] font-bold text-white whitespace-nowrap bg-gradient-to-br from-[#ec8450] to-[#c85a28] shadow-[0_4px_24px_rgba(224,112,58,.18)]"
              onClick={() => setLogin(true)}
            >
              Open the app →
            </button>
            <a href="#layers" className="px-6 py-3 rounded-xl text-[15px] font-semibold border border-line2 bg-panel2 whitespace-nowrap hover:border-accent">
              See what it can do
            </a>
          </div>
          <div className="text-xs text-dim">Access key required — this is a personal demo deployment with capped usage.</div>
          <div className="mt-6 flex items-center gap-4 max-w-[520px] bg-panel border border-line rounded-2xl px-5 py-4">
            <div className="text-2xl font-extrabold tracking-tight text-accent whitespace-nowrap">~30s</div>
            <p className="text-mut text-xs">
              <b className="text-ink">from file-drop to first grounded answer</b> — loading, chunking, embedding and
              indexing all run automatically on upload.
            </p>
          </div>
        </div>
        <HeroDemoCard />
      </header>

      {/* stats row */}
      <section className="max-w-[1180px] mx-auto px-6 mt-14 lg:mt-20 grid grid-cols-2 lg:grid-cols-4 gap-4 relative z-10">
        {[
          ["100%", "domain-agnostic — no per-domain config, the model infers it from your files"],
          ["7", "file formats ingested — pdf, docx, md, txt, csv, html, pptx"],
          ["3", "capability layers — grounded Q&A, multi-step reasoning, autonomous agents"],
          ["0", "tool calls without your approval — each one gated and audit-logged"],
        ].map(([n, d]) => (
          <div key={n} className="bg-panel border border-line rounded-2xl px-5 py-5">
            <div className="text-3xl font-extrabold text-accent tracking-tight">{n}</div>
            <p className="text-mut text-xs mt-1">{d}</p>
          </div>
        ))}
      </section>

      {/* capability layers */}
      <section id="layers" className="max-w-[1180px] mx-auto px-6 pt-20 relative z-10">
        <div className="grid lg:grid-cols-[1.1fr_.9fr] gap-3 lg:gap-11 items-end mb-9">
          <h2 className="text-[30px] lg:text-[38px] font-extrabold tracking-tight leading-[1.12]">
            Three capability layers,
            <br className="hidden lg:block" /> one question box
          </h2>
          <p className="text-mut text-sm max-w-[440px] lg:justify-self-end">
            Every question is routed automatically to the cheapest layer that can answer it well — small talk stays
            cheap, hard questions get the full machinery.
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-4">
          {LAYERS.map((l) => (
            <div key={l.title} className="bg-panel border border-line rounded-[20px] p-7 transition hover:border-line2 hover:-translate-y-0.5">
              <div className="w-11 h-11 rounded-xl bg-accent/10 text-accent flex items-center justify-center mb-4">{l.icon}</div>
              <h3 className="font-bold text-[16.5px] mb-2">{l.title}</h3>
              <p className="text-mut text-[13.5px]">{l.body}</p>
              <div className="mt-3.5 pt-3 border-t border-dashed border-line text-xs text-dim">
                <b className="text-mut font-semibold">Ask:</b> {l.ex}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* how it works */}
      <section id="how" className="max-w-[1180px] mx-auto px-6 pt-20 relative z-10">
        <div className="grid lg:grid-cols-[1.1fr_.9fr] gap-3 lg:gap-11 items-end mb-9">
          <h2 className="text-[30px] lg:text-[38px] font-extrabold tracking-tight leading-[1.12]">
            From folder to conversation
            <br className="hidden lg:block" /> in under a minute
          </h2>
          <p className="text-mut text-sm max-w-[440px] lg:justify-self-end">
            No onboarding flow, no schema setup, no prompt engineering — sign in, drop files, ask.
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-4">
          {STEPS.map((s) => (
            <div key={s.n} className="text-center px-4 py-6">
              <div className="w-10 h-10 mx-auto mb-3.5 rounded-xl text-white font-extrabold text-[17px] flex items-center justify-center bg-gradient-to-br from-[#ec8450] to-[#c85a28]">
                {s.n}
              </div>
              <h3 className="font-bold text-[15.5px] mb-1.5">{s.t}</h3>
              <p className="text-mut text-[13px]">{s.d}</p>
            </div>
          ))}
        </div>
      </section>

      {/* production strip */}
      <section className="max-w-[1180px] mx-auto px-6 pt-20 relative z-10">
        <div className="grid lg:grid-cols-[1.1fr_.9fr] gap-3 lg:gap-11 items-end mb-9">
          <h2 className="text-[30px] lg:text-[38px] font-extrabold tracking-tight leading-[1.12]">Built like a production system</h2>
          <p className="text-mut text-sm max-w-[440px] lg:justify-self-end">
            Because it is one — a full-stack AI engineering showcase, not a notebook demo.
          </p>
        </div>
        <div className="flex flex-wrap justify-center gap-2.5">
          {FEATURES.map(([b, rest]) => (
            <span key={b} className="border border-line bg-panel rounded-full px-5 py-2 text-[13px] text-mut">
              <b className="text-ink font-semibold">{b}</b> {rest}
            </span>
          ))}
        </div>
      </section>

      <footer className="mt-20 border-t border-line px-6 py-7 text-center text-dim text-[12.5px] relative z-10">
        AdaptiveRAG — a capstone build by Aditya Chauhan.
        <div className="mt-1.5">
          {["FastAPI", "LangGraph", "ChromaDB", "React + Vite", "Azure Container Apps"].map((s, i) => (
            <span key={s}>{i > 0 && <span className="mx-2">·</span>}{s}</span>
          ))}
        </div>
      </footer>

      <LoginModal open={login} notice={notice} onClose={() => setLogin(false)} onAuthed={onAuthed} />
    </div>
  );
}
