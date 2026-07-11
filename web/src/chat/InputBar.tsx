import { useRef, useState } from "react";
import type { Mode } from "../types";
import { BotIcon, ChatIcon, ClipIcon, UsersIcon } from "../icons";

const HINTS: Record<Mode, string> = {
  chat: "Auto-routed: simple → direct · factual → RAG · complex → multi-step reasoning",
  agent: "ReAct loop — searches docs, runs sandboxed Python, hits the web. Pauses for your approval.",
  sup: "Multi-agent team — the Chief dispatches Retriever, Reasoner & Validator, then files a verdict.",
};

export default function InputBar({ mode, setMode, busy, onSend, onPickFiles }: {
  mode: Mode;
  setMode: (m: Mode) => void;
  busy: boolean;
  onSend: (text: string) => void;
  onPickFiles: (files: File[]) => void;
}) {
  const [text, setText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const send = () => {
    const t = text.trim();
    if (!t || busy) return;
    setText("");
    onSend(t);
  };

  const modeBtn = (m: Mode, icon: React.ReactNode, label: string) => (
    <button
      className={`inline-flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
        mode === m ? "bg-panel3 text-accent" : "text-mut hover:text-ink"
      }`}
      onClick={() => setMode(m)}
    >
      {icon} {label}
    </button>
  );

  return (
    <div className="border-t border-line bg-bg2 px-4 sm:px-6 pt-3 pb-2.5">
      <div className="max-w-[800px] mx-auto flex items-center gap-2.5 mb-2 min-w-0">
        <div className="inline-flex bg-panel border border-line rounded-[10px] p-0.5 shrink-0">
          {modeBtn("chat", <ChatIcon size={13} />, "Chat")}
          {modeBtn("agent", <BotIcon size={13} />, "Agent")}
          {modeBtn("sup", <UsersIcon size={13} />, "Supervisor")}
        </div>
        <span className="text-[11.5px] text-dim truncate hidden md:block">{HINTS[mode]}</span>
      </div>
      <div className="max-w-[800px] mx-auto flex items-center gap-2.5 bg-panel border border-line2 rounded-[13px] px-3.5 py-2.5 focus-within:border-accent">
        <input
          ref={fileRef}
          type="file"
          multiple
          className="hidden"
          accept=".pdf,.docx,.md,.txt,.csv,.html,.pptx"
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            e.target.value = "";
            if (files.length) onPickFiles(files);
          }}
        />
        <button
          className="text-dim hover:text-ink disabled:opacity-40"
          title="Upload documents to this chat"
          onClick={() => fileRef.current?.click()}
          disabled={busy}
        >
          <ClipIcon size={17} />
        </button>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask anything about your documents…"
          className="flex-1 bg-transparent outline-none text-sm min-w-0"
        />
        <button
          className="bg-accent text-white rounded-lg px-4 sm:px-5 py-1.5 text-[13px] font-bold disabled:opacity-40 whitespace-nowrap"
          onClick={send}
          disabled={busy || !text.trim()}
        >
          {busy ? "…" : "Send"}
        </button>
      </div>
      <div className="max-w-[800px] mx-auto mt-1.5 text-[11px] text-dim text-center hidden sm:block">
        Answers stream over SSE · evidence arrives before the answer · grounding badge after · tool calls need your approval
      </div>
    </div>
  );
}
