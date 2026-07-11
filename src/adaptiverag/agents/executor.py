"""The ReAct agent executor: the detective who works a case using the
Block 3.1 talent agency (ToolRegistry). Thinks, dispatches a specialist,
reads the result, repeats — pausing for a supervisor's warrant before
any risky move. State machine built on LangGraph. See CLAUDE.md Block 3.2.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict, cast
import json   # json.dumps turns a Python dict into a JSON string for the prompt
from dataclasses import dataclass
from collections.abc import Iterator

from langgraph.graph import StateGraph, START, END        # the flowchart builder + entrance/exit markers
from langgraph.checkpoint.memory import MemorySaver        # in-RAM snapshot saver (needed for interrupt)

from .approval import ApprovalPolicy, make_human_gate
import uuid
from langgraph.types import Command          # the "un-freeze with this decision" wrapper
from langchain_core.runnables import RunnableConfig   # the TypedDict invoke() expects for `config`

from langgraph.config import get_stream_writer   # a node's slot for pushing custom notes outward


# This class is kind of a "memory" that the agent has about what all has happoened so far (Scratchpad)
# plus what is pending, the final answer and total iterations that happened.
# Literally the "memory" is actually scratchpad, that saves all the steps that has happened till now.

# This agentstate is basically the link between the brain "LLM" and the body "Tools"
# LLM will read the question and the scratchpad and post its next action to pending action
# Which will be read by the tools and results will be posted back to the scratchpad.

class AgentState(TypedDict):
    """The case file on the whiteboard — passed desk to desk, edited in place."""

    question: str                                   # the case to crack (written once, never changes)
    conversation_context: str

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
                         tools: list[dict], conversation_context: str = "", role: str | None = None) -> str:
    """Brief the detective: the case, the roster of specialists, and the moves
    so far — then ask for the next single move (or the verdict)."""

    persona = role or ("You are a problem-solving agent that answers the user's "
                       "question by using tools. You reason step by step and call "
                       "one tool at a time.")

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

    # The clerk's briefing — only shown if there's anything to say (empty on turn one).
    memory_block = (
        f"What you already know from this conversation:\n{conversation_context}\n\n"
        if conversation_context else ""
    )

    # 3. Assemble the full briefing. The trailing "Your next step:" cue invites
    #    the model to continue the transcript with exactly one block.
    return f"""{persona}

Available tools:
{tools_block}

{_REACT_INSTRUCTIONS}

{memory_block}User question: {question}

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

_SECTIONS = {"thought:": "thought", "action:": None,      # None = don't stream this section
             "action input:": None, "final answer:": "token"}


def _relay_sections(pieces, writer, sections: dict | None = None) -> str:
    """Watch a dictation live; forward marked sections to the writer.
    Returns the full reply for the parser. Default sections = ReAct's."""
    sections = sections or _SECTIONS
    hold = max(len(m) for m in sections)       # marker may arrive split — withhold this tail
    buffer, pos, mode = "", 0, None
    for piece in pieces:
        buffer += piece
        while True:
            low = buffer.lower()
            nxt, which = None, None
            for m in sections:                 # ← was _SECTIONS
                i = low.find(m, pos)
                if i != -1 and (nxt is None or i < nxt):
                    nxt, which = i, m
            if nxt is None or which is None:
                break
            if mode and nxt > pos:
                writer({"type": mode, "text": buffer[pos:nxt]})
            mode, pos = sections[which], nxt + len(which)   # ← was _SECTIONS
        safe = len(buffer) - hold              # ← was _HOLD
        if mode and safe > pos:
            writer({"type": mode, "text": buffer[pos:safe]})
            pos = safe
    if mode and pos < len(buffer):
        writer({"type": mode, "text": buffer[pos:]})
    return buffer


def make_reason_node(llm_client, registry, max_iterations: int, role: str | None = None):
    """Factory: build the THINK desk with its llm / roster / budget baked in
    (closure — same trick as make_run_python in tools.py)."""

    def reason_node(state: AgentState) -> dict:
        # Egg timer fired? Don't throw the notepad away (all those Observations
        # were paid for!) — allow ONE last, tool-free call that files a report
        # from the findings so far.
        out_of_budget = state["iterations"] >= max_iterations

        # 1. Brief the detective: question + notepad + roster of business cards.
        prompt = _build_reason_prompt(
            state["question"], state["scratchpad"], registry.list_tools(),
            state.get("conversation_context", ""),
            role=role,                                   # NEW: pin the badge on the persona line
        )
        if out_of_budget:
            prompt += ("\nIMPORTANT: Your tool budget is exhausted. Do NOT choose "
                       "another Action. Write 'Final Answer:' NOW, summarizing every "
                       "useful finding from your Observations above (with sources).")

        # 2. The detective dictates; thoughts and verdict stream out live.
        if hasattr(llm_client, "generate_stream"):
            reply = _relay_sections(llm_client.generate_stream(prompt), get_stream_writer())
        else:                                  # test fakes without generate_stream still work
            reply = llm_client.generate(prompt)
        # 3. Read the reply: a dispatch order, or the verdict.
        step = _parse_reason_output(reply)

        # Out of budget AND it still tried to act? Hard stop — honest note.
        if out_of_budget and not step.is_final:
            return {
                "answer": f"Stopped after {max_iterations} steps without a final answer.",
                "pending_action": None,
            }

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



class AgentExecutor:
    """The detective's whole operation: assembles the flowchart from the three
    desks and runs cases on it. Build once, reuse for every query."""

    def __init__(self, llm_client, registry, *,
                 max_iterations: int = 6,
                 require_approval: list[str] | None = None, memory_manager=None):
        self.registry = registry
        self._manager = memory_manager
        self.policy = ApprovalPolicy(require_approval or [])   # the warrant house-rule

        # Build the three desks (closures, deps baked in — Steps 5/6/7).
        self._reason = make_reason_node(llm_client, registry, max_iterations)
        self._act = make_act_node(registry)
        self._gate = make_human_gate(self.policy)

        self._graph = self._build_graph()                      # assemble + compile once

    def _notebook(self, conversation_id: str | None):
        """Read the ticket, pull THIS chat's notebook off the rack. No rack → no memory."""
        return self._manager.for_conversation(conversation_id) if self._manager else None

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

    def _package(self, state: dict, thread_id: str, memory=None) -> dict:
        """Translate LangGraph's raw output into our result shape: paused or done."""
        interrupts = state.get("__interrupt__")
        if interrupts:                                # frozen at the gate — NOT done, don't file anything
            return {
                "status": "awaiting_approval",
                "request": interrupts[0].value,
                "thread_id": thread_id,
                "trace": state.get("scratchpad", []),
            }

        answer = state.get("answer")
        if memory and answer:                   # verdict reached → clerk files the detective's answer
            memory.add_turn("assistant", answer)

        return {
            "status": "done",
            "answer": answer,
            "trace": state.get("scratchpad", []),
            "thread_id": thread_id,
        }

    def start(self, question: str, thread_id: str | None = None, conversation_id: str | None = None) -> dict:
        """Open a new case. Runs until the verdict OR the first warrant request."""
        thread_id = thread_id or uuid.uuid4().hex        # a fresh label per case
        memory = self._notebook(conversation_id)

        # Clerk's briefing FIRST (so 'recent' excludes the question we're about to
        # ask), THEN log the client's question into both memory tiers.
        context = memory.build_context(question) if memory else ""
        if memory:
            memory.add_turn("user", question)


        initial: AgentState = {
            "question": question, "scratchpad": [], "conversation_context": context,
            "pending_action": None, "answer": None, "iterations": 0,
        }
        state = self._graph.invoke(initial, self._config(thread_id))
        return self._package(state, thread_id, memory)

    def resume(self, thread_id: str, decision, conversation_id: str | None = None) -> dict:
        """Hand the supervisor's decision to the frozen case and continue. May
        pause AGAIN if the detective's next move also needs a warrant."""
        state = self._graph.invoke(Command(resume=decision), self._config(thread_id))
        return self._package(state, thread_id, self._notebook(conversation_id))

    def run(self, question: str, approver=None, conversation_id: str | None = None) -> dict:
        """Drive a whole case to completion, auto-answering each warrant via
        approver(request) -> decision. Default approves everything. Great for
        tests / non-interactive callers; the UI uses start()+resume() directly."""
        approver = approver or (lambda request: True)
        result = self.start(question, conversation_id=conversation_id)
        while result["status"] == "awaiting_approval":   # keep going as long as it keeps stopping
            decision = approver(result["request"])
            result = self.resume(result["thread_id"], decision, conversation_id=conversation_id)
        return result
    
    def _stream_events(self, graph_input, thread_id: str, memory=None) -> Iterator[dict]:
        """Shared narrator for start/resume: desk updates AND live dictation."""
        answer = None
        for item in self._graph.stream(                   # two channels → (mode, payload) tuples
            graph_input, self._config(thread_id),
            stream_mode=["updates", "custom"],
        ):
            mode, payload = cast("tuple[str, dict]", item)   # stubs don't know list-mode ⇒ tuples
            if mode == "custom":                          # a note from get_stream_writer():
                yield payload                             #   already {"type": "thought"|"token", ...}
                continue

            # mode == "updates": one dict per desk that just finished
            if "__interrupt__" in payload:                # case froze at the gate
                yield {"type": "approval",
                       "request": payload["__interrupt__"][0].value,
                       "thread_id": thread_id}
                return

            for node_updates in payload.values():
                for entry in (node_updates or {}).get("scratchpad", []):
                    yield {"type": "trace", "entry": entry}
                if (node_updates or {}).get("answer") is not None:
                    answer = node_updates["answer"]       # verdict — hold for the end

        if memory and answer:                       # same filing rule as _package()
           memory.add_turn("assistant", answer)
        yield {"type": "done", "answer": answer, "thread_id": thread_id}

    def start_stream(self, question: str, thread_id: str | None = None, conversation_id: str | None = None) -> Iterator[dict]:
        """start(), but narrated live. Same memory bookkeeping, same freeze behavior."""
        thread_id = thread_id or uuid.uuid4().hex
        memory = self._notebook(conversation_id)
        context = memory.build_context(question) if memory else ""
        if memory:
            memory.add_turn("user", question)
        initial: AgentState = {
            "question": question, "scratchpad": [], "conversation_context": context,
            "pending_action": None, "answer": None, "iterations": 0,
        }
        yield from self._stream_events(initial, thread_id, memory)    # relay the narrator's notes

    def resume_stream(self, thread_id: str, decision, conversation_id: str | None = None) -> Iterator[dict]:
        """resume(), narrated. May freeze again — client just loops."""
        memory = self._notebook(conversation_id)
        yield from self._stream_events(Command(resume=decision), thread_id, memory)