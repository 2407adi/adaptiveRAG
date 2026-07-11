import { useCallback, useState } from "react";
import { useChat } from "./useChat";
import Sidebar from "./Sidebar";
import Messages from "./Messages";
import InputBar from "./InputBar";
import ApprovalModal from "./ApprovalModal";
import type { Mode } from "../types";
import { BotIcon, MenuIcon, UsersIcon, XIcon } from "../icons";

function EmptyState({ onSuggest, onPickFiles, busy }: {
  onSuggest: (q: string, mode: Mode) => void;
  onPickFiles: (files: File[]) => void;
  busy: boolean;
}) {
  const suggestions: Array<{ q: string; mode: Mode; icon?: React.ReactNode }> = [
    { q: "What are the main topics covered in these documents?", mode: "chat" },
    { q: "Compare the documents and point out any disagreements between them.", mode: "chat" },
    { q: "Verify any numeric claims in the documents with code.", mode: "agent", icon: <BotIcon size={12} /> },
    { q: "Audit these documents and flag every inconsistency.", mode: "sup", icon: <UsersIcon size={12} /> },
  ];
  return (
    <div className="text-center pt-[6vh] px-5">
      <h2 className="text-2xl font-extrabold tracking-tight mb-2">What do you want to know?</h2>
      <p className="text-mut text-sm mb-6">This chat sees the shared corpus plus anything you upload here.</p>
      <label
        className={`block max-w-[560px] mx-auto border-[1.5px] border-dashed border-line2 rounded-2xl px-5 py-7 text-mut text-[13.5px] cursor-pointer hover:border-accent hover:bg-accent/[.03] ${busy ? "pointer-events-none opacity-60" : ""}`}
      >
        <input
          type="file" multiple className="hidden" accept=".pdf,.docx,.md,.txt,.csv,.html,.pptx"
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            e.target.value = "";
            if (files.length) onPickFiles(files);
          }}
        />
        <b className="text-ink">Drop documents here</b> or click to browse
        <div className="text-xs text-dim mt-1">pdf · docx · md · txt · csv · html · pptx — private to this chat, ingested in ~30s</div>
      </label>
      <div className="flex flex-wrap justify-center gap-2 mt-6">
        {suggestions.map((s) => (
          <button
            key={s.q}
            className="inline-flex items-center gap-1.5 border border-line bg-panel text-mut text-xs px-4 py-2 rounded-full hover:border-accent hover:text-ink disabled:opacity-50"
            onClick={() => onSuggest(s.q, s.mode)}
            disabled={busy}
          >
            {s.icon} {s.q.length > 46 ? `${s.q.slice(0, 46)}…` : s.q}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ChatApp({ onLogout }: { onLogout: () => void }) {
  const chat = useChat();
  const [mode, setMode] = useState<Mode>("chat");
  const [drawer, setDrawer] = useState(false);
  const [dragging, setDragging] = useState(false);

  const suggest = useCallback((q: string, m: Mode) => {
    setMode(m);
    void chat.send(q, m);
  }, [chat]);

  const openConv = useCallback((id: string) => {
    setDrawer(false);
    void chat.openConversation(id);
  }, [chat]);

  return (
    <div className="h-full flex flex-col bg-bg text-ink">
      <div className="flex-1 flex min-h-0">
        {/* sidebar — fixed on desktop, slide-over drawer on mobile */}
        <div className="hidden md:block shrink-0">
          <Sidebar items={chat.items} activeId={chat.activeId} busy={chat.busy}
                   onNew={chat.newChat} onOpen={openConv} onClearAll={() => void chat.clearAll()} onLogout={onLogout} />
        </div>
        {drawer && (
          <div className="fixed inset-0 z-40 md:hidden" onClick={() => setDrawer(false)}>
            <div className="absolute inset-0 bg-black/60" />
            <div className="absolute inset-y-0 left-0" onClick={(e) => e.stopPropagation()}>
              <Sidebar items={chat.items} activeId={chat.activeId} busy={chat.busy}
                       onNew={() => { setDrawer(false); chat.newChat(); }} onOpen={openConv}
                       onClearAll={() => { setDrawer(false); void chat.clearAll(); }} onLogout={onLogout} />
            </div>
            <button className="absolute top-3 left-[276px] text-mut" onClick={() => setDrawer(false)}><XIcon size={18} /></button>
          </div>
        )}

        {/* main column */}
        <div
          className="flex-1 flex flex-col min-w-0 relative"
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false); }}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const files = Array.from(e.dataTransfer.files ?? []);
            if (files.length && !chat.busy) void chat.upload(files);
          }}
        >
          {dragging && (
            <div className="absolute inset-2 z-30 border-2 border-dashed border-accent rounded-2xl bg-accent/5 flex items-center justify-center pointer-events-none">
              <span className="text-accent font-bold">Drop to ingest into this chat</span>
            </div>
          )}

          {/* top bar */}
          <div className="px-4 sm:px-5 py-3 border-b border-line bg-bg2 flex items-center gap-3">
            <button className="md:hidden text-mut" onClick={() => setDrawer(true)}><MenuIcon size={18} /></button>
            <span className="font-bold text-sm truncate">{chat.title}</span>
            <div className="flex-1" />
            <span className="hidden sm:inline text-[10.5px] font-bold tracking-wide uppercase text-mut border border-line2 rounded-full px-2.5 py-0.5">
              {chat.busy ? "working…" : "ready"}
            </span>
          </div>

          {/* messages */}
          <div className="flex-1 overflow-y-auto py-5">
            {chat.msgs.length === 0 ? (
              <EmptyState onSuggest={suggest} onPickFiles={(f) => void chat.upload(f)} busy={chat.busy} />
            ) : (
              <Messages msgs={chat.msgs} />
            )}
          </div>

          <InputBar mode={mode} setMode={setMode} busy={chat.busy}
                    onSend={(t) => void chat.send(t, mode)} onPickFiles={(f) => void chat.upload(f)} />
        </div>
      </div>

      {chat.approval && (
        <ApprovalModal request={chat.approval.request} onDecide={(a, r) => void chat.decideApproval(a, r)} />
      )}
    </div>
  );
}
