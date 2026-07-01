"""Block 3.2 — ReAct AgentExecutor. Offline: scripted LLM + fake tools,
so the loop, the trace, the budget, and the approval gate are all
deterministic. No network, no real sandbox. Run: pytest tests/test_executor.py -q
"""

import json

from adaptiverag.agents.executor import AgentExecutor
from adaptiverag.agents.tools import tool, ToolRegistry


# ── Test doubles ────────────────────────────────────────────────────────────

class ScriptedLLM:
    """Returns pre-written replies in order; sticks on the last one when the
    script runs out (handy for the 'loop forever' test)."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        if len(self._replies) > 1:
            return self._replies.pop(0)
        return self._replies[0]


def build_fake_registry():
    """A real ToolRegistry (no audit) with two fake tools that record their
    calls, so tests can assert exactly what ran."""
    calls = {"search_documents": [], "run_python": []}
    reg = ToolRegistry(audit_log=None)

    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded documents."""
        calls["search_documents"].append(query)
        return f"[1] (source: doc.txt) RESULT for {query}"

    @tool
    def run_python(code: str) -> str:
        """Execute a short Python snippet."""
        calls["run_python"].append(code)
        return "4"

    reg.register(search_documents)
    reg.register(run_python)
    return reg, calls


def act_reply(thought, tool_name, args):
    return f"Thought: {thought}\nAction: {tool_name}\nAction Input: {json.dumps(args)}"

def final_reply(thought, answer):
    return f"Thought: {thought}\nFinal Answer: {answer}"


APPROVAL = ["run_python", "web_search"]   # same list your config ships


# ── 1. Solves a 2–3 step problem using tools ────────────────────────────────

def test_agent_solves_multi_step_with_tools():
    reg, calls = build_fake_registry()
    llm = ScriptedLLM([
        act_reply("find X", "search_documents", {"query": "X"}),
        act_reply("find Y", "search_documents", {"query": "Y"}),
        act_reply("compare them", "run_python", {"code": "print(12-10)"}),
        final_reply("done", "X is 12, Y is 10, diff 2."),
    ])
    ex = AgentExecutor(llm, reg, max_iterations=8, require_approval=APPROVAL)

    result = ex.run("Compare X and Y", approver=lambda req: True)   # approve all

    assert result["status"] == "done"
    assert "diff 2" in result["answer"]
    assert calls["search_documents"] == ["X", "Y"]     # searched X, then Y
    assert calls["run_python"] == ["print(12-10)"]     # then computed


# ── 2. Trace shows Thought → Action → Observation → … → Answer ───────────────

def test_trace_thought_action_observation_order():
    reg, _ = build_fake_registry()
    llm = ScriptedLLM([
        act_reply("look it up", "search_documents", {"query": "capital"}),
        final_reply("got it", "Paris."),
    ])
    ex = AgentExecutor(llm, reg, max_iterations=5, require_approval=APPROVAL)

    result = ex.run("What is the capital?")
    types = [e["type"] for e in result["trace"]]

    assert types == ["thought", "action", "observation", "thought"]
    assert result["answer"] == "Paris."


# ── 3. Stops after max_iterations (no infinite loop) ────────────────────────

def test_agent_stops_at_max_iterations():
    reg, calls = build_fake_registry()
    # Never emits a Final Answer — always proposes another (no-approval) search.
    llm = ScriptedLLM([act_reply("keep going", "search_documents", {"query": "loop"})])
    ex = AgentExecutor(llm, reg, max_iterations=3, require_approval=APPROVAL)

    result = ex.run("loop forever")

    assert result["status"] == "done"
    assert "Stopped after 3 steps" in result["answer"]
    assert len(calls["search_documents"]) == 3         # exactly the budget, then it halts


# ── 4a. Approval gate: pauses before run_python, approving resumes ──────────

def test_approval_gate_pause_and_approve():
    reg, calls = build_fake_registry()
    llm = ScriptedLLM([
        act_reply("compute via code", "run_python", {"code": "2+2"}),
        final_reply("done", "The answer is 4."),
    ])
    ex = AgentExecutor(llm, reg, max_iterations=5, require_approval=APPROVAL)

    started = ex.start("what is 2+2 via code")
    assert started["status"] == "awaiting_approval"
    assert started["request"]["tool"] == "run_python"
    assert calls["run_python"] == []                   # frozen BEFORE the tool ran

    finished = ex.resume(started["thread_id"], True)   # sign the warrant
    assert finished["status"] == "done"
    assert calls["run_python"] == ["2+2"]              # approved → executed
    assert "4" in finished["answer"]


# ── 4b. Approval gate: rejecting makes the agent replan without the tool ────

def test_approval_gate_reject_triggers_replan_without_tool():
    reg, calls = build_fake_registry()
    llm = ScriptedLLM([
        act_reply("compute via code", "run_python", {"code": "2+2"}),
        final_reply("answer directly instead", "It is 4."),
    ])
    ex = AgentExecutor(llm, reg, max_iterations=5, require_approval=APPROVAL)

    started = ex.start("what is 2+2")
    assert started["status"] == "awaiting_approval"

    finished = ex.resume(started["thread_id"], False)  # deny the warrant
    assert finished["status"] == "done"
    assert calls["run_python"] == []                   # rejected → NEVER executed
    assert any(e["type"] == "observation" and "rejected" in e["content"].lower()
               for e in finished["trace"])             # rejection noted on the trail
    assert "4" in finished["answer"]                   # replanned to a direct answer