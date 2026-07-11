"""The multi-agent firm: a Chief Detective (supervisor) who never leaves the
office, delegating to three juniors — Scout (retriever), Analyst (reasoner),
Auditor (validator). All juniors use the SAME Block 3.1 front desk (tools);
only their one-page role briefing (system prompt) differs. See CLAUDE.md 3.4.
"""

from __future__ import annotations

import operator                                      # operator.add = the "append, don't overwrite" rule
from typing import Annotated, Optional, TypedDict, cast
from typing import Hashable
from collections.abc import Iterator

from langgraph.graph import StateGraph, START, END
from .executor import (AgentState, _build_reason_prompt,
                       make_reason_node, make_act_node, _section_after,
                       _relay_sections)
from langgraph.config import get_stream_writer
from .approval import ApprovalPolicy, make_human_gate


class SupervisorState(TypedDict):
    """The case file in the Chief's office — passed hub ⇄ spoke, edited in place."""

    question: str                                    # the case (written once, never changes)

    # 4.3a: the memory clerk's briefing — what THIS conversation already covered
    # (recency + recall, from the same notebook rack the lone detective uses).
    # Written once at start(); read by the Chief. Juniors never see it — their
    # conversation_context slot carries colleagues' reports instead.
    conversation_context: str

    # The stack of pinned reports. Each entry is one junior's finished work:
    #   {"agent": "retriever", "report": "<polished summary for the Chief>",
    #    "trace": [<that junior's private Thought/Action/Observation notepad>]}
    # Annotated + operator.add = the pinning rule: a returning junior's report
    # is APPENDED to the stack, never allowed to overwrite a colleague's.
    reports: Annotated[list[dict], operator.add]

    next_agent: Optional[str]                        # the Chief's dispatch slip: "retriever" / "reasoner" / "validator" — or None
    answer: Optional[str]                            # the closing memo (None until the Chief declares FINISH)
    handoffs: int                                    # delegation budget spent — how many juniors dispatched so far

# ── The role briefings: each junior IS just one of these strings. ──────────
# They get prepended to the Block 3.2 ReAct prompt inside the worker loop;
# everything else (tools, format, parsing) is identical for all three.

WORKER_PROMPTS: dict[str, str] = {

    # The Scout: legwork only. Gathers evidence, never interprets it.
    # NOTE the explicit FINISH LINE: without it the model keeps re-searching
    # for a "perfect" chunk that doesn't exist (seen live: 4 wasted dispatches).
    "retriever": """\
You are the RETRIEVER, an evidence-gathering specialist.
Your ONLY job: find and collect passages relevant to the question.
- Use search_documents first (vary the wording meaningfully between searches — names, numbers, section titles — not just word order).
- Use web_search only if the documents clearly lack the information.
- Do NOT analyze, compare, or draw conclusions — that is another agent's job.
- Two or three searches are PLENTY. As soon as your observations contain relevant passages, STOP searching and write your Final Answer.
- Partial evidence is a valid report. NEVER finish without listing what you found.
Your final answer must be a tidy list of the evidence found, each item with its source.""",

    # The Analyst: desk work only. Works from evidence already in the case file.
    "reasoner": """\
You are the REASONER, an analysis specialist.
Your ONLY job: analyze the evidence already gathered (shown above) to answer the question.
- Compare, synthesize, and compute — use run_python for any arithmetic.
- Prefer the gathered evidence; search only if a small specific gap blocks you.
Your final answer must be your analysis with clear conclusions, citing the evidence.""",

    # The Auditor: trusts nothing. Re-checks the Analyst's claims against sources.
    # NOTE the PRIORITIZATION rule: one search per claim is correct method, but
    # with a limited budget the claims central to the user's question must be
    # checked FIRST (seen live: budget spent on peripheral claims, cut off
    # before the ones that mattered).
    "validator": """\
You are the VALIDATOR, a fact-checking specialist.
Your ONLY job: verify the claims in the analysis (shown above) against the documents.
- First list the claims, ordered by how central they are to the user's question. Verify in THAT order — you have a limited search budget and may not get to all of them.
- Verify at most 3 claims (one search_documents each); use run_python to re-check any calculations.
- Then STOP and write your verdicts. Mark claims you did not check as UNVERIFIED — do not assume them.
Your final answer must be a verdict per claim: SUPPORTED / UNSUPPORTED / CONTRADICTED / UNVERIFIED, with sources.""",
}


def _format_reports(reports: list[dict]) -> str:
    """Photocopy the case file for a junior: prior reports, tidy and readable."""
    if not reports:
        return ""
    blocks = [f"[{r['agent']} report]\n{r['report']}" for r in reports]
    return "Work done so far by other agents:\n\n" + "\n\n".join(blocks)




def _build_worker_graph(name: str, llm_client, registry,
                        policy: ApprovalPolicy, max_iterations: int = 3):
    """Factory: one junior's office — a scale replica of AgentExecutor's graph,
    with this junior's badge at the think desk. Returns a COMPILED mini-graph."""

    # Same furniture factories as Block 3.2 — only the badge differs.
    reason = make_reason_node(llm_client, registry, max_iterations,
                              role=WORKER_PROMPTS[name])          # the badge
    gate = make_human_gate(policy)                                # same warrant desk
    act = make_act_node(registry)                                 # same front-desk phone

    # The two corridor signs (routing rules), same logic as AgentExecutor's
    # methods — rewritten as closures since we have no `self` here.
    def route_after_reason(state: AgentState):
        if state.get("answer") is not None:                       # report written → office closes
            return END
        action = state.get("pending_action")
        if action is None:                                        # nothing staged → close safely
            return END
        return "needs_approval" if policy.needs(action["tool"]) else "go"

    def route_after_gate(state: AgentState):
        return "go" if state.get("pending_action") is not None else "think"

    # Assemble the replica — identical floor plan to Block 3.2.
    builder = StateGraph(AgentState)                    # NOTE: the WORKER's whiteboard schema
    builder.add_node("reason", reason)
    builder.add_node("gate", gate)
    builder.add_node("act", act)
    builder.add_edge(START, "reason")
    builder.add_conditional_edges("reason", route_after_reason,
                                  {"needs_approval": "gate", "go": "act", END: END})
    builder.add_conditional_edges("gate", route_after_gate,
                                  {"go": "act", "think": "reason"})
    builder.add_edge("act", "reason")

    # Compile WITHOUT a checkpointer: a subgraph must not own a filing cabinet.
    # It inherits the parent graph's at runtime — that's what lets the gate's
    # interrupt() freeze the whole firm, not just this office.
    return builder.compile()



def make_worker_wrapper(name: str, worker_graph):
    """Factory: the DOORMAN for one junior's office. From the firm's side this
    is a single node on SupervisorState; inside, it runs the whole subgraph."""

    def worker_node(state: SupervisorState) -> dict:
        # 1. Photocopy: firm's case file → a FRESH junior whiteboard.
        #    Colleagues' reports ride in the conversation_context slot (the
        #    briefing tray the 3.2 prompt already knows how to display).
        initial: AgentState = {
            "question": state["question"],
            "conversation_context": _format_reports(state["reports"]),
            "scratchpad": [],                 # clean notepad — his private working-out
            "pending_action": None,
            "answer": None,
            "iterations": 0,
        }

        # 2. Show him in. This runs the ENTIRE mini reason→gate→act graph.
        #    If his gate calls interrupt(), the freeze doesn't stop at this door —
        #    it bubbles up through the firm to whoever invoked the firm (the UI).
        final = worker_graph.invoke(initial)

        # 3. Translate his verdict → one pinned report (+ his notepad stapled on).
        report = final.get("answer") or "(the worker produced no report)"
        return {
            "reports": [{"agent": name, "report": report,
                         "trace": final.get("scratchpad", [])}],   # append via reducer
            "next_agent": None,                                    # dispatch slip cleared
            "handoffs": state["handoffs"] + 1,                     # one delegation spent
        }

    return worker_node


_WORKER_NAMES = list(WORKER_PROMPTS)      # ["retriever", "reasoner", "validator"] — one source of truth


def _build_supervisor_prompt(question: str, reports: list[dict],
                             conversation_context: str = "") -> str:
    """The Chief's desk view: the case, his team roster, the reports so far —
    and (4.3a) the memory clerk's briefing on what this chat already covered."""
    reports_block = _format_reports(reports) or "(no reports yet)"
    # Same shape as the lone detective's memory block: only shown when non-empty.
    memory_block = (
        f"What you already know from this conversation:\n{conversation_context}\n\n"
        if conversation_context else ""
    )
    return f"""You are the supervisor of a team of specialist agents answering a user's question. \
You never use tools yourself — you only decide who works next, or finish.

Your team:
- retriever: searches the user's documents and gathers evidence (the right FIRST pick when there are no reports yet)
- reasoner: analyzes, compares, and computes using the evidence already gathered
- validator: fact-checks the reasoner's claims against the documents

Rules:
- Dispatch ONE agent, or FINISH when the reports fully answer the question.
- Typical flow: retriever -> reasoner -> validator -> FINISH, but SKIP any agent that adds nothing (a simple lookup may need no reasoner).
- Re-dispatch the same agent ONLY if its report contains NO usable information. Partial evidence means move on to the next specialist — do not chase perfect reports.

Reply EXACTLY in this format:
Decision: <retriever | reasoner | validator | FINISH>
Final Answer: <only when Decision is FINISH — the complete answer for the user, grounded in the reports>

{memory_block}User question: {question}

{reports_block}

Your decision:"""


def _parse_supervisor_reply(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Read the dispatch slip -> (next_agent, answer). Exactly one is non-None.
    Garbled slip -> treat the whole reply as a final answer (safe degrade)."""
    text = (raw or "").strip()
    decision = (_section_after("Decision:", text) or "").strip().lower()   # reuse 3.2's label-reader

    for name in _WORKER_NAMES:                     # a clean dispatch: one known name
        if name in decision:
            return name, None

    fa = _section_after("Final Answer:", text)     # FINISH (or anything else) -> close the case
    return None, (fa or text)


def make_supervisor_node(llm_client, max_handoffs: int = 6):
    """Factory: the CHIEF's desk — reads the case file, writes a dispatch slip
    or the closing memo. Budget baked in via closure, like every other desk."""

    def supervisor_node(state: SupervisorState) -> dict:
        # Egg timer, firm edition: out of delegations? Close the case honestly —
        # one last LLM call composes the best answer FROM THE REPORTS WE HAVE.
        if state["handoffs"] >= max_handoffs:
            forced = llm_client.generate(
                f"Answer the user's question as best you can using ONLY these "
                f"reports.\n\nUser question: {state['question']}\n\n"
                f"{_format_reports(state['reports'])}\n\nAnswer:"
            )
            return {"answer": forced.strip(), "next_agent": None}

        # Normal turn: the Chief dictates. While deciding, nothing streams
        # (no marker appears); when FINISHing, the closing memo streams live.
        prompt = _build_supervisor_prompt(state["question"], state["reports"],
                                          state.get("conversation_context", ""))
        if hasattr(llm_client, "generate_stream"):
            reply = _relay_sections(
                llm_client.generate_stream(prompt), get_stream_writer(),
                {"decision:": None, "final answer:": "token"},   # Chief's own markers
            )
        else:
            reply = llm_client.generate(prompt)

        next_agent, answer = _parse_supervisor_reply(reply)

        if answer is not None:
            return {"answer": answer, "next_agent": None}    # closing memo — case done
        return {"next_agent": next_agent}                    # dispatch slip — router reads this

    return supervisor_node


import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig


class SupervisorAgent:
    """The whole firm: Chief + three specialist offices on a star graph.
    Same public face as AgentExecutor — the UI can swap one for the other."""

    def __init__(self, llm_client, registry, *,
                 max_handoffs: int = 6,            # the Chief's delegation budget
                 worker_iterations: int = 3,       # each junior's private tool budget
                 require_approval: list[str] | None = None,
                 memory_manager=None):             # 4.3a: the SAME notebook rack the lone detective uses
        policy = ApprovalPolicy(require_approval or [])   # same house rule, firm-wide
        self._manager = memory_manager             # None → the firm runs memoryless (old behavior)

        self._chief = make_supervisor_node(llm_client, max_handoffs)
        # Three offices + doormen, from the same factories — only the badge differs.
        self._workers = {
            name: make_worker_wrapper(
                name, _build_worker_graph(name, llm_client, registry,
                                          policy, worker_iterations))
            for name in _WORKER_NAMES
        }
        self._graph = self._build_graph()

    def _build_graph(self):
        """The floor plan: hub-and-spoke."""
        builder = StateGraph(SupervisorState)             # the FIRM's whiteboard schema
        builder.add_node("supervisor", self._chief)
        for name, node in self._workers.items():
            builder.add_node(name, node)                  # doorman+office = one node
            builder.add_edge(name, "supervisor")          # every junior reports back to the hub
        builder.add_edge(START, "supervisor")             # every case starts at the Chief

        # The Chief's fork: dispatch slip names a junior, or the case is closed.
        path_map: dict[Hashable, str] = {name: name for name in _WORKER_NAMES}
        path_map[END] = END                     # the "case closed" exit
        builder.add_conditional_edges(
            "supervisor", self._route_after_supervisor, path_map,
        )
        # ONE central filing cabinet: a freeze inside any office checkpoints the
        # whole firm (subgraphs inherit this — remember, they compiled without one).
        return builder.compile(checkpointer=MemorySaver())

    def _route_after_supervisor(self, state: SupervisorState):
        """Closing memo written? Case over. Else follow the dispatch slip."""
        if state.get("answer") is not None:
            return END
        nxt = state.get("next_agent")
        return nxt if nxt in self._workers else END       # unknown name → stop safely

    # ── The reception window: signature-identical to AgentExecutor ─────────
    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _notebook(self, conversation_id: str | None):
        """Read the ticket, pull THIS chat's notebook off the rack. No rack → no memory.
        Same rack as the lone detective — toggle modes mid-chat, the memory follows."""
        return self._manager.for_conversation(conversation_id) if self._manager else None

    def _package(self, state: dict, thread_id: str, memory=None) -> dict:
        interrupts = state.get("__interrupt__")
        if interrupts:                                    # frozen at some junior's gate
            return {"status": "awaiting_approval",
                    "request": interrupts[0].value,
                    "thread_id": thread_id,
                    "reports": state.get("reports", [])}
        answer = state.get("answer")
        if memory and answer:                             # closing memo → file it in THIS chat's notebook
            memory.add_turn("assistant", answer)
        return {"status": "done",
                "answer": answer,
                "reports": state.get("reports", []),      # per-agent report + trace, for the UI panel
                "thread_id": thread_id}

    def start(self, question: str, thread_id: str | None = None, conversation_id: str | None = None) -> dict:
        thread_id = thread_id or uuid.uuid4().hex
        memory = self._notebook(conversation_id)          # 4.3a: pull this chat's notebook
        context = memory.build_context(question) if memory else ""   # briefing BEFORE filing the question
        if memory:
            memory.add_turn("user", question)
        initial: SupervisorState = {
            "question": question, "conversation_context": context,
            "reports": [],
            "next_agent": None, "answer": None, "handoffs": 0,
        }
        return self._package(self._graph.invoke(initial, self._config(thread_id)), thread_id, memory)

    def resume(self, thread_id: str, decision, conversation_id: str | None = None) -> dict:
        return self._package(
            self._graph.invoke(Command(resume=decision), self._config(thread_id)), thread_id,
            self._notebook(conversation_id))              # same ticket → same notebook on thaw

    def run(self, question: str, approver=None, conversation_id: str | None = None) -> dict:
        approver = approver or (lambda request: True)
        result = self.start(question, conversation_id=conversation_id)
        while result["status"] == "awaiting_approval":
            decision = approver(result["request"])
            result = self.resume(result["thread_id"], decision, conversation_id=conversation_id)
        return result
    
    def _stream_events(self, graph_input, thread_id: str, memory=None) -> Iterator[dict]:
        """The firm narrated live: handoffs, each junior's thoughts and moves,
        filed reports, and the Chief's closing memo dictated token by token."""
        answer = None
        for item in self._graph.stream(
            graph_input, self._config(thread_id),
            stream_mode=["updates", "custom"],
            subgraphs=True,                       # ← also surface events from inside the offices
        ):
            ns, mode, payload = cast("tuple[tuple[str, ...], str, dict]", item)
            agent = ns[0].split(":")[0] if ns else None   # which office, e.g. "retriever"; None = firm floor

            if mode == "custom":                          # live dictation from some desk
                if agent is None:
                    yield payload                         # the Chief — "token" here IS the real answer
                elif payload.get("type") == "token":      # a junior's Final Answer = their REPORT,
                    yield {"type": "report_token",        #   relabel so the UI never confuses it
                           "agent": agent, "text": payload.get("text", "")}
                else:                                     # a junior thinking out loud
                    yield {"type": "thought", "agent": agent, "text": payload.get("text", "")}
                continue

            if "__interrupt__" in payload:                # some office's gate froze the whole firm
                yield {"type": "approval",
                       "request": payload["__interrupt__"][0].value,
                       "thread_id": thread_id}
                return

            for node_name, node_updates in payload.items():
                nu = node_updates or {}
                if agent is not None:                     # inside an office: their notepad, live
                    for entry in nu.get("scratchpad", []):
                        yield {"type": "trace", "agent": agent, "entry": entry}
                elif node_name == "supervisor":           # the Chief's desk
                    if nu.get("next_agent"):
                        yield {"type": "handoff", "agent": nu["next_agent"]}
                    if nu.get("answer") is not None:
                        answer = nu["answer"]             # closing memo — hold for done
                else:                                     # a doorman pinned a finished report
                    for r in nu.get("reports", []):
                        yield {"type": "report", "agent": r["agent"], "report": r["report"]}

        if memory and answer:                             # same filing rule as _package()
            memory.add_turn("assistant", answer)
        yield {"type": "done", "answer": answer, "thread_id": thread_id}

    def start_stream(self, question: str, thread_id: str | None = None, conversation_id: str | None = None) -> Iterator[dict]:
        """start(), narrated live. Same memory bookkeeping as the non-stream twin."""
        thread_id = thread_id or uuid.uuid4().hex
        memory = self._notebook(conversation_id)
        context = memory.build_context(question) if memory else ""
        if memory:
            memory.add_turn("user", question)
        initial: SupervisorState = {
            "question": question, "conversation_context": context,
            "reports": [],
            "next_agent": None, "answer": None, "handoffs": 0,
        }
        yield from self._stream_events(initial, thread_id, memory)

    def resume_stream(self, thread_id: str, decision, conversation_id: str | None = None) -> Iterator[dict]:
        """resume(), narrated. May freeze again at another gate — client loops."""
        yield from self._stream_events(Command(resume=decision), thread_id,
                                       self._notebook(conversation_id))