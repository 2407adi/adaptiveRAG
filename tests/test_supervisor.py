"""Block 3.4 — Multi-Agent Supervisor. Offline: one scripted LLM plays the
Chief AND every junior, in call order. No network, no real tools.
Run: pytest tests/test_supervisor.py -q
"""

import json

from adaptiverag.agents.supervisor import SupervisorAgent
from adaptiverag.agents.tools import tool, ToolRegistry


# ── Test doubles (same pattern as test_executor.py) ─────────────────────────

class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        if len(self._replies) > 1:
            return self._replies.pop(0)
        return self._replies[0]


def build_fake_registry():
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
        return "2"

    reg.register(search_documents)
    reg.register(run_python)
    return reg, calls


def act_reply(thought, tool_name, args):
    return f"Thought: {thought}\nAction: {tool_name}\nAction Input: {json.dumps(args)}"

def final_reply(thought, answer):
    return f"Thought: {thought}\nFinal Answer: {answer}"

def chief_dispatch(name):
    return f"Decision: {name}"

def chief_finish(answer):
    return f"Decision: FINISH\nFinal Answer: {answer}"


# ── 1. Full delegation chain: retriever → reasoner → validator → FINISH ─────

def test_supervisor_delegates_full_chain():
    reg, calls = build_fake_registry()
    llm = ScriptedLLM([
        chief_dispatch("retriever"),
        act_reply("gather evidence", "search_documents", {"query": "X vs Y"}),
        final_reply("collected", "Evidence: X=12, Y=10 (doc.txt)."),
        chief_dispatch("reasoner"),
        act_reply("compute the gap", "run_python", {"code": "12-10"}),
        final_reply("analyzed", "X exceeds Y by 2."),
        chief_dispatch("validator"),
        act_reply("verify X", "search_documents", {"query": "X value"}),
        final_reply("verified", "SUPPORTED: X=12. SUPPORTED: Y=10."),
        chief_finish("X is greater than Y by 2 (verified)."),
    ])
    sup = SupervisorAgent(llm, reg, max_handoffs=6, require_approval=[])

    result = sup.run("Compare X and Y")

    assert result["status"] == "done"
    assert "greater than Y by 2" in result["answer"]
    # The Chief delegated in the expected order — one report per junior:
    assert [r["agent"] for r in result["reports"]] == ["retriever", "reasoner", "validator"]
    assert calls["search_documents"] == ["X vs Y", "X value"]
    assert calls["run_python"] == ["12-10"]
    # Each sub-agent's work is visible in the trace (the block's verification):
    for r in result["reports"]:
        assert r["report"]                                   # a non-empty polished report
        assert any(e["type"] == "thought" for e in r["trace"])   # + the raw notepad stapled on


# ── 2. Sub-agents are different system prompts, not different codebases ─────

def test_workers_get_their_role_briefing():
    reg, _ = build_fake_registry()
    llm = ScriptedLLM([
        chief_dispatch("retriever"),
        final_reply("nothing to do", "No evidence needed."),
        chief_finish("Done."),
    ])
    SupervisorAgent(llm, reg, require_approval=[]).run("trivial question")

    assert any("You are the RETRIEVER" in p for p in llm.calls)     # badge reached the prompt
    assert any("You are the supervisor" in p for p in llm.calls)    # so did the Chief's


# ── 3. Approval freeze inside a junior's office bubbles up to the caller ────

def test_approval_bubbles_up_through_subgraph():
    reg, calls = build_fake_registry()
    llm = ScriptedLLM([
        chief_dispatch("reasoner"),
        act_reply("need code", "run_python", {"code": "1+1"}),
        final_reply("computed", "It is 2."),
        chief_finish("The answer is 2."),
    ])
    sup = SupervisorAgent(llm, reg, require_approval=["run_python"])

    started = sup.start("what is 1+1 via code")
    assert started["status"] == "awaiting_approval"      # froze INSIDE the subgraph…
    assert started["request"]["tool"] == "run_python"    # …and surfaced the request
    assert calls["run_python"] == []                     # frozen BEFORE execution

    finished = sup.resume(started["thread_id"], True)    # sign the warrant
    assert finished["status"] == "done"
    assert calls["run_python"] == ["1+1"]                # ran exactly once (no replay double-fire)
    assert "2" in finished["answer"]


# ── 4. Handoff budget: a Chief who never finishes gets cut off ──────────────

def test_handoff_budget_forces_finish():
    reg, _ = build_fake_registry()
    llm = ScriptedLLM([
        chief_dispatch("retriever"),
        final_reply("round 1", "Evidence A."),
        chief_dispatch("retriever"),
        final_reply("round 2", "Evidence B."),
        "Best-effort answer from the reports.",          # the forced-finish composition call
    ])
    sup = SupervisorAgent(llm, reg, max_handoffs=2, require_approval=[])

    result = sup.run("loop forever")

    assert result["status"] == "done"
    assert len(result["reports"]) == 2                   # exactly the budget, then cut off
    assert "Best-effort" in result["answer"]


# ── 5. Garbled Chief reply degrades to FINISH (no crash, no dispatch) ───────

def test_garbled_chief_reply_degrades_to_finish():
    reg, calls = build_fake_registry()
    llm = ScriptedLLM(["Hmm, tough one. Perhaps we should ponder."])
    result = SupervisorAgent(llm, reg, require_approval=[]).run("anything")

    assert result["status"] == "done"
    assert "ponder" in result["answer"]                  # whole reply became the answer
    assert result["reports"] == []                       # nobody was dispatched
    assert calls["search_documents"] == calls["run_python"] == []