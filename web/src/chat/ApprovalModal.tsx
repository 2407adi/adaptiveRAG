import { useState } from "react";
import type { ApprovalRequest } from "../types";
import { PauseIcon } from "../icons";

export default function ApprovalModal({ request, onDecide }: {
  request: ApprovalRequest;
  onDecide: (approved: boolean, reason: string) => void;
}) {
  const [reason, setReason] = useState("");

  const argsPretty = Object.entries(request.args ?? {})
    .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("\n");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-[3px] p-5">
      <div className="w-[440px] max-w-full bg-panel border border-line2 rounded-2xl p-6 shadow-2xl">
        <h3 className="text-base font-bold flex items-center gap-2 mb-1.5">
          <span className="text-warn"><PauseIcon size={16} /></span> Tool call requires your approval
        </h3>
        <p className="text-mut text-[13px] mb-3.5">{request.message} Nothing executes until you decide.</p>
        <pre className="bg-[#111] border border-line rounded-lg px-3.5 py-3 text-xs text-[#dcdcdc] mb-2 overflow-x-auto leading-relaxed whitespace-pre-wrap break-words">
{`tool: ${request.tool}
${argsPretty}`}
        </pre>
        {request.tool === "run_python" && (
          <p className="text-[11.5px] text-dim mb-2.5">Sandbox: no files · no network · no imports · CPU + memory limits · call audit-logged.</p>
        )}
        <input
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="Optional reason if denying (sent back to the agent)"
          className="w-full bg-bg2 border border-line2 rounded-lg px-3.5 py-2.5 text-sm outline-none focus:border-accent mb-3"
        />
        <div className="flex justify-end gap-2.5">
          <button className="px-4 py-2 rounded-lg border border-line2 bg-panel2 text-sm font-semibold hover:border-bad" onClick={() => onDecide(false, reason.trim())}>
            Deny
          </button>
          <button className="px-4 py-2 rounded-lg text-sm font-bold text-[#111] bg-ok" onClick={() => onDecide(true, "")}>
            Approve &amp; run
          </button>
        </div>
      </div>
    </div>
  );
}
