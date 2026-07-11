import { useState } from "react";
import type { ConvListItem } from "../types";
import { KeyIcon, PlusIcon, TrashIcon } from "../icons";

export default function Sidebar({ items, activeId, busy, onNew, onOpen, onClearAll, onLogout }: {
  items: ConvListItem[];
  activeId: string | null;
  busy: boolean;
  onNew: () => void;
  onOpen: (id: string) => void;
  onClearAll: () => void;
  onLogout: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  return (
    <div className="w-[268px] h-full bg-bg2 border-r border-line flex flex-col">
      <div className="px-4 pt-4 pb-3 text-[15.5px] font-extrabold flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full bg-accent shadow-[0_0_8px_var(--color-accent)]" /> AdaptiveRAG
      </div>
      <button
        className="mx-3 mb-3 py-2.5 rounded-[10px] border border-line2 bg-panel2 text-[13.5px] font-semibold hover:border-accent disabled:opacity-50 flex items-center justify-center gap-1.5"
        onClick={onNew}
        disabled={busy}
      >
        <PlusIcon size={14} /> New chat
      </button>
      <div className="flex-1 overflow-y-auto px-2">
        <div className="text-[10.5px] font-bold tracking-widest uppercase text-dim px-2.5 py-1.5">Conversations</div>
        {items.map((c) => (
          <button
            key={c.id}
            className={`w-full text-left px-3 py-2 rounded-lg mb-0.5 text-[13px] truncate block disabled:opacity-60 ${
              c.id === activeId ? "bg-panel2 text-ink shadow-[inset_2px_0_0_var(--color-accent)]" : "text-mut hover:bg-panel2"
            }`}
            onClick={() => onOpen(c.id)}
            disabled={busy}
            title={c.title ?? undefined}
          >
            {c.title ?? "Untitled chat"}
          </button>
        ))}
        {items.length === 0 && (
          <div className="px-3 py-2 text-xs text-dim">No conversations yet — start one on the right.</div>
        )}
      </div>
      <div className="p-3 border-t border-line space-y-2">
        {items.length > 0 && (
          confirming ? (
            <div className="px-3 py-2 bg-panel2 rounded-lg text-xs text-mut flex items-center gap-2">
              Delete {items.length} chat{items.length > 1 ? "s" : ""}?
              <button className="ml-auto text-bad font-semibold" onClick={() => { setConfirming(false); onClearAll(); }}>yes</button>
              <button className="text-dim hover:text-ink" onClick={() => setConfirming(false)}>no</button>
            </div>
          ) : (
            <button
              className="w-full px-3 py-2 rounded-lg text-xs text-dim hover:text-bad hover:bg-panel2 flex items-center gap-2 disabled:opacity-50"
              onClick={() => setConfirming(true)}
              disabled={busy}
            >
              <TrashIcon size={12} /> Clear my chats
            </button>
          )
        )}
        <div className="px-3 py-2 bg-panel2 rounded-lg text-xs text-mut flex items-center gap-2">
          <KeyIcon size={13} /> key active
          <button className="ml-auto text-dim hover:text-bad text-[11.5px]" onClick={onLogout}>log out</button>
        </div>
      </div>
    </div>
  );
}
