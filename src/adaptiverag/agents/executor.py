"""The ReAct agent executor: the detective who works a case using the
Block 3.1 talent agency (ToolRegistry). Thinks, dispatches a specialist,
reads the result, repeats — pausing for a supervisor's warrant before
any risky move. State machine built on LangGraph. See CLAUDE.md Block 3.2.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict
import json   # json.dumps turns a Python dict into a JSON string for the prompt
from dataclasses import dataclass


# This class is kind of a "memory" that the agent has about what all has happoened so far (Scratchpad)
# plus what is pending, the final answer and total iterations that happened.
# Literally the "memory" is actually scratchpad, that saves all the steps that has happened till now.

# This agentstate is basically the link between the brain "LLM" and the body "Tools"
# LLM will read the question and the scratchpad and post its next action to pending action
# Which will be read by the tools and results will be posted back to the scratchpad.

class AgentState(TypedDict):
    """The case file on the whiteboard — passed desk to desk, edited in place."""

    question: str                                   # the case to crack (written once, never changes)

    # The notepad of moves. Annotated[...] = "a list, WITH a special rule attached".
    # That rule is operator.add: when a desk returns scratchpad entries, LangGraph
    # does old_list + new_list (append) instead of overwriting. So the trail grows.
    scratchpad: Annotated[list[dict], operator.add]

    pending_action: Optional[dict]                  # the move the detective wants next: {"tool","args"} — or None
    answer: Optional[str]                           # the cracked case (None until REASON declares a Final Answer)
    iterations: int                                 # tally marks: how many tool calls (moves) we've spent



# The exact output contract we ask the model to follow. Kept as a module
# constant so the prompt (here) and the parser (next step) agree on ONE format.
_REACT_INSTRUCTIONS = """\
Work in a loop of Thought -> Action -> Observation. Each turn, output EXACTLY
ONE of these two blocks and nothing else.

To use a tool:
Thought: <why you're taking this step>
Action: <one tool name from the list above>
Action Input: <a single-line JSON object of arguments, e.g. {"query": "..."}>

When you can answer the user's question:
Thought: <your final reasoning>
Final Answer: <the complete answer for the user>

Rules:
- Choose only from the listed tools; copy the name exactly.
- Action Input MUST be valid JSON on one line.
- Take ONE step at a time. Never invent an Observation — those are given to you.
- Prefer searching the user's documents before the web."""


def _build_reason_prompt(question: str, scratchpad: list[dict],
                         tools: list[dict]) -> str:
    """Brief the detective: the case, the roster of specialists, and the moves
    so far — then ask for the next single move (or the verdict)."""

    # 1. The roster — one business card per specialist (from registry.list_tools()).
    tool_lines = []
    for t in tools:                                                  # walk each card
        params = json.dumps(t["parameters"].get("properties", {}))  # "what I need", compactly
        tool_lines.append(f"- {t['name']}: {t['description']} | args: {params}")
    tools_block = "\n".join(tool_lines)

    # 2. Replay the notepad so the detective "remembers" the case so far.
    trail_lines = []
    for entry in scratchpad:                          # walk the moves in order
        if entry["type"] == "thought":
            trail_lines.append(f"Thought: {entry['content']}")
        elif entry["type"] == "action":
            trail_lines.append(f"Action: {entry['tool']}")
            trail_lines.append(f"Action Input: {json.dumps(entry['args'])}")
        elif entry["type"] == "observation":
            trail_lines.append(f"Observation: {entry['content']}")
    trail_block = "\n".join(trail_lines) if trail_lines else "(no moves yet)"

    # 3. Assemble the full briefing. The trailing "Your next step:" cue invites
    #    the model to continue the transcript with exactly one block.
    return f"""You are a problem-solving agent that answers the user's question \
by using tools. You reason step by step and call one tool at a time.

Available tools:
{tools_block}

{_REACT_INSTRUCTIONS}

User question: {question}

Work so far:
{trail_block}

Your next step:"""


# Reasonstep is just a structure of output of the reasoning step.
@dataclass
class ReasonStep:
    """What the think desk decided this turn."""
    thought: str                     # the detective's stated reasoning
    is_final: bool                   # True = verdict reached; False = wants a tool
    tool: Optional[str] = None       # which specialist to dispatch (when not final)
    args: Optional[dict] = None      # the arguments for that specialist
    answer: Optional[str] = None     # the verdict text (when final)


def _section_after(label: str, text: str) -> Optional[str]:
    """Return the text right after `label` (e.g. 'Thought:'), stopping at the
    next known label so a Thought doesn't swallow the Action. None if absent."""
    idx = text.lower().find(label.lower())
    if idx == -1:
        return None
    rest = text[idx + len(label):]
    rl = rest.lower()
    cut = len(rest)
    for s in ("thought:", "action input:", "action:", "final answer:", "observation:"):
        j = rl.find(s)
        if j != -1:
            cut = min(cut, j)        # stop before whichever label comes first
    return rest[:cut].strip()


def _loads_lenient(s: str) -> dict:
    """Parse a JSON object, tolerating ``` fences and stray prose. {} on failure
    (a bad-args call then just returns a tool error → the detective replans)."""
    s = s.strip()
    if s.startswith("```"):                              # strip a ```json fence, like chain.py does
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = s.find("{"), s.rfind("}")               # keep only the outermost {...}
    if start != -1 and end > start:
        s = s[start:end + 1]
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_reason_output(raw: str) -> ReasonStep:
    """Read the detective's reply: dispatch a specialist, or call the case.
    A garbled reply degrades to a Final Answer so the loop can't spin forever."""
    text = (raw or "").strip()
    thought = _section_after("Thought:", text) or ""

    # Verdict path: everything after 'Final Answer:' is the answer (to end of text).
    fa = text.lower().find("final answer:")
    if fa != -1:
        answer = text[fa + len("final answer:"):].strip()
        return ReasonStep(thought=thought, is_final=True, answer=answer)

    # Action path: pull the tool name (one line) and its JSON args.
    tool = _section_after("Action:", text)
    if tool:
        tool = tool.splitlines()[0].strip()              # the name is a single token/line
        args = _loads_lenient(_section_after("Action Input:", text) or "{}")
        return ReasonStep(thought=thought, is_final=False, tool=tool, args=args)

    # Neither parsed → treat the whole reply as the answer (safe degrade).
    return ReasonStep(thought=thought, is_final=True, answer=text)


def make_reason_node(llm_client, registry, max_iterations: int):
    """Factory: build the THINK desk with its llm / roster / budget baked in
    (closure — same trick as make_run_python in tools.py)."""

    def reason_node(state: AgentState) -> dict:
        # Egg timer: out of moves? Stop with an honest note instead of looping.
        if state["iterations"] >= max_iterations:
            return {
                "answer": f"Stopped after {max_iterations} steps without a final answer.",
                "pending_action": None,
            }

        # 1. Brief the detective: question + notepad + roster of business cards.
        prompt = _build_reason_prompt(
            state["question"], state["scratchpad"], registry.list_tools(),
        )
        # 2. The detective thinks out loud — one LLM call.
        reply = llm_client.generate(prompt)
        # 3. Read the reply: a dispatch order, or the verdict.
        step = _parse_reason_output(reply)

        # 4. Always jot the Thought on the notepad. NOTE: we return only the
        #    DELTA (new entries); the operator.add reducer appends it (Step 2).
        updates: dict = {"scratchpad": [{"type": "thought", "content": step.thought}],
                         "iterations": state["iterations"] + 1,} # one move spent (overwrite reducer)

        if step.is_final:
            updates["answer"] = step.answer          # verdict reached — REASON's final act
            updates["pending_action"] = None         # nothing staged for dispatch
        else:
            # Write the intended move on the notepad AND stage it on the board
            # for the gate / ACT desk to pick up.
            updates["scratchpad"].append(
                {"type": "action", "tool": step.tool, "args": step.args}
            )
            updates["pending_action"] = {"tool": step.tool, "args": step.args}

        return updates

    return reason_node

def make_act_node(registry):
    """Factory: build the DISPATCH desk, wired to the Block 3.1 front desk
    (the ToolRegistry). It runs the staged move and records what came back."""

    def act_node(state: AgentState) -> dict:
        action = state["pending_action"]            # the move REASON staged on the board
        if action is None:                          # defensive: edges shouldn't route here empty
            return {"pending_action": None}

        # Phone the front desk: it looks up the specialist, runs them, AND stamps
        # the tamper-evident logbook (Block 3.1). It never raises — a failure
        # comes back as an error STRING, which simply becomes the Observation,
        # so the detective reads it and replans instead of the loop crashing.
        observation = registry.call(action["tool"], action["args"])

        return {
            # Pin what the specialist brought back onto the notepad (append via reducer).
            "scratchpad": [{"type": "observation", "content": str(observation)}],
            "pending_action": None,                  # move done — clear the board
        }

    return act_node



from langgraph.graph import StateGraph, START, END        # the flowchart builder + entrance/exit markers
from langgraph.checkpoint.memory import MemorySaver        # in-RAM snapshot saver (needed for interrupt)

from .approval import ApprovalPolicy, make_human_gate
import uuid
from langgraph.types import Command          # the "un-freeze with this decision" wrapper
from langchain_core.runnables import RunnableConfig   # the TypedDict invoke() expects for `config`


class AgentExecutor:
    """The detective's whole operation: assembles the flowchart from the three
    desks and runs cases on it. Build once, reuse for every query."""

    def __init__(self, llm_client, registry, *,
                 max_iterations: int = 6,
                 require_approval: list[str] | None = None):
        self.registry = registry
        self.policy = ApprovalPolicy(require_approval or [])   # the warrant house-rule

        # Build the three desks (closures, deps baked in — Steps 5/6/7).
        self._reason = make_reason_node(llm_client, registry, max_iterations)
        self._act = make_act_node(registry)
        self._gate = make_human_gate(self.policy)

        self._graph = self._build_graph()                      # assemble + compile once

    def _build_graph(self):
        """Draw the flowchart: desks (nodes) joined by arrows (edges)."""
        builder = StateGraph(AgentState)        # hand LangGraph the whiteboard schema + its reducers

        builder.add_node("reason", self._reason)
        builder.add_node("gate", self._gate)
        builder.add_node("act", self._act)

        builder.add_edge(START, "reason")       # every case begins at the think desk

        # After REASON, fork based on the board: verdict / risky move / safe move.
        builder.add_conditional_edges(
            "reason", self._route_after_reason,
            {"needs_approval": "gate", "go": "act", END: END},
        )
        # After the warrant: approved move survived on the board → act; else replan.
        builder.add_conditional_edges(
            "gate", self._route_after_gate,
            {"go": "act", "think": "reason"},
        )
        builder.add_edge("act", "reason")       # after dispatching, loop back to think

        # interrupt() REQUIRES a checkpointer: it snapshots the frozen case so
        # resume() can pick it back up. MemorySaver keeps those snapshots in RAM.
        return builder.compile(checkpointer=MemorySaver())

    def _route_after_reason(self, state: AgentState):
        """Verdict in? go home. Risky move? get a warrant. Safe move? just act."""
        if state.get("answer") is not None:
            return END
        action = state.get("pending_action")
        if action is None:                      # no answer AND no move (shouldn't happen) → stop safely
            return END
        return "needs_approval" if self.policy.needs(action["tool"]) else "go"

    def _route_after_gate(self, state: AgentState):
        """Warrant signed → move still staged → ACT. Cleared → rejected → replan."""
        return "go" if state.get("pending_action") is not None else "think"


    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _package(self, state: dict, thread_id: str) -> dict:
        """Translate LangGraph's raw output into our result shape: paused or done."""
        interrupts = state.get("__interrupt__")          # present iff the graph froze at the gate
        if interrupts:
            return {
                "status": "awaiting_approval",
                "request": interrupts[0].value,          # the payload we passed to interrupt()
                "thread_id": thread_id,
                "trace": state.get("scratchpad", []),
            }
        return {                                         # the graph reached END
            "status": "done",
            "answer": state.get("answer"),
            "trace": state.get("scratchpad", []),
            "thread_id": thread_id,
        }

    def start(self, question: str, thread_id: str | None = None) -> dict:
        """Open a new case. Runs until the verdict OR the first warrant request."""
        thread_id = thread_id or uuid.uuid4().hex        # a fresh label per case
        initial: AgentState = {
            "question": question, "scratchpad": [],
            "pending_action": None, "answer": None, "iterations": 0,
        }
        state = self._graph.invoke(initial, self._config(thread_id))
        return self._package(state, thread_id)

    def resume(self, thread_id: str, decision) -> dict:
        """Hand the supervisor's decision to the frozen case and continue. May
        pause AGAIN if the detective's next move also needs a warrant."""
        state = self._graph.invoke(Command(resume=decision), self._config(thread_id))
        return self._package(state, thread_id)

    def run(self, question: str, approver=None) -> dict:
        """Drive a whole case to completion, auto-answering each warrant via
        approver(request) -> decision. Default approves everything. Great for
        tests / non-interactive callers; the UI uses start()+resume() directly."""
        approver = approver or (lambda request: True)
        result = self.start(question)
        while result["status"] == "awaiting_approval":   # keep going as long as it keeps stopping
            decision = approver(result["request"])
            result = self.resume(result["thread_id"], decision)
        return result