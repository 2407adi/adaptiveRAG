"""The supervisor's warrant: a human-in-the-loop approval gate.

Before the detective runs a side-effecting tool (run_python, web_search),
the graph FREEZES here and surfaces the proposed move for sign-off. Approve
→ on to ACT. Reject → the detective replans without it. Which tools need a
warrant is config-driven (settings.agent.require_approval). See Block 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langgraph.types import interrupt   # the call that freezes the graph mid-run

if TYPE_CHECKING:                       # False at runtime → no real import → no circular import
    from .executor import AgentState    # visible to Pylance only, for annotations below


@dataclass
class ApprovalPolicy:
    """The house rule: which specialists may not act without a signed warrant."""
    require_approval: list[str]                    # tool names, from settings.agent.require_approval

    def needs(self, tool: str) -> bool:
        """Does this move need the supervisor's signature before ACT?"""
        return tool in self.require_approval


def _is_approved(decision) -> tuple[bool, str]:
    """Read whatever the reviewer sent back via Command(resume=...).
    Tolerant: accepts a bool, a yes/no string, or {"approved": bool, "reason": str}."""
    if isinstance(decision, dict):
        return bool(decision.get("approved", False)), str(decision.get("reason", ""))
    if isinstance(decision, str):
        return decision.strip().lower() in {"y", "yes", "approve", "approved", "true"}, ""
    return bool(decision), ""                       # plain truthy/falsy fallback


def make_human_gate(policy: ApprovalPolicy):
    """Factory: the SUPERVISOR'S desk. Freezes the case, surfaces the proposed
    risky move, resumes on the reviewer's decision."""

    # `state` is the AgentState whiteboard. We annotate it `dict` (not AgentState)
    # only to dodge a circular import — executor.py imports THIS file.
    def human_gate(state: AgentState) -> dict:
        action = state["pending_action"]            # the move REASON staged (read-only — see note)
        if action is None:                      # defensive: the gate is only reached with a staged move
            return {}                           # no-op → router sends the case back to REASON
        # ── from here Pylance knows `action` is a dict, so action["tool"] is fine ──

        # ── FREEZE ──────────────────────────────────────────────────────────
        # interrupt() stops the ENTIRE graph here and hands this payload back to
        # whoever invoked it ("may I run this?"). It resumes only when the caller
        # re-invokes with Command(resume=<decision>) — and THEN this function
        # RE-RUNS FROM THE TOP, with interrupt() now returning <decision>.
        # ⇒ Everything ABOVE this line must be side-effect-free, or it runs twice.
        decision = interrupt({
            "type": "approval_request",
            "tool": action["tool"],
            "args": action["args"],
            "message": f"The agent wants to run '{action['tool']}'. Approve?",
        })
        # ── RESUMED HERE with the reviewer's answer ─────────────────────────

        approved, reason = _is_approved(decision)
        if approved:
            return {}                               # leave pending_action set → router sends it to ACT

        # Rejected: clear the staged move and leave a note so the detective
        # replans WITHOUT this tool on the next REASON turn.
        note = f"Action '{action['tool']}' was rejected by the reviewer."
        note += f" Reason: {reason}" if reason else ""
        note += " Choose a different approach."
        return {
            "pending_action": None,                 # cleared → the router will send the case to REASON
            "scratchpad": [{"type": "observation", "content": note}],
        }

    return human_gate