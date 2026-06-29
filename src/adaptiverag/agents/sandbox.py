"""Sandboxed execution for the run_python tool.

The code we run here is written by the LLM — untrusted by definition. Defense
is layered: a restricted-builtins namespace (no open/import/eval), a small
calc-only module whitelist, execution in a separate process, and kernel
resource limits (CPU + memory) plus a wall-clock timeout.

NOT bulletproof: a determined escape via Python introspection is possible, so
the process isolation + rlimits are the real backstop. Production code that runs
truly untrusted input should use OS-level isolation (containers/gVisor/microVM).
See CLAUDE.md Block 3.1.
"""

from __future__ import annotations

import builtins as _builtins
from dataclasses import dataclass

import ast              # the tool that reads code-text and draws its "sentence-diagram" tree
import io               # gives us an in-memory text box we can use as our own out-tray
import contextlib       # has the helper that swaps the out-tray for ours

import multiprocessing          # the toolkit for creating and supervising separate "rooms" (processes)
from queue import Empty         # the specific signal we get when the slot under the door is empty

# ── What the sandboxed code is allowed to see ──────────────────────────

# A whitelist of safe builtins. Notably ABSENT: open, __import__, eval, exec,
# compile, input, globals, locals, vars, getattr, setattr, delattr.
_ALLOWED_BUILTINS = {
    name: getattr(_builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "int", "len", "list", "map", "max",
        "min", "pow", "print", "range", "repr", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "zip",
        "True", "False", "None",  # harmless, but explicit
    )
    if hasattr(_builtins, name)
}

# Calculation-only modules we hand to the code directly (no import needed,
# because __import__ is removed). All pure-compute, no filesystem/network.
import math, statistics, random, json as _json, re as _re, datetime as _datetime
import itertools, functools, collections, decimal, fractions

_ALLOWED_MODULES = {
    "math": math, "statistics": statistics, "random": random,
    "json": _json, "re": _re, "datetime": _datetime,
    "itertools": itertools, "functools": functools,
    "collections": collections, "decimal": decimal, "fractions": fractions,
}


@dataclass
class SandboxResult:
    """Outcome of one run_python execution."""
    ok: bool
    value: str | None = None      # repr of the last expression, e.g. "4"
    stdout: str = ""              # anything the code print()ed
    error: str | None = None      # message if it failed / was blocked / timed out



def _apply_limits(cpu_seconds: int, mem_bytes: int | None) -> None:
    """Ask the building manager (the OS) to cap this room's time and memory."""

    try:
        import resource                      # the control panel for talking to the building manager (Unix/Mac only)
    except ImportError:
        return                               # some buildings (e.g. Windows) lack this panel → skip; the wall-clock timer still guards us

    try:
        # Tell the manager: "evict this room once it has used cpu_seconds of the CPU's hands-time."
        # This is what stops an infinite loop — the OS kills the room for us.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass                                 # if this particular building won't honor the request, don't crash — just move on

    if mem_bytes:
        try:
            # Tell the manager: "this room's whiteboard may be at most mem_bytes big."
            # This is what stops a memory bomb like [0] * 10**12.
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass                             # macOS often ignores this one — fine, the time limit + timeout still apply


def _worker(code: str, cpu_seconds: int, mem_bytes: int | None, queue) -> None:
    """This whole function runs INSIDE the locked room (the child process).
    It runs the stranger's code under the rules, then slides the result
    back to the parent through `queue` (the slot under the door)."""

    _apply_limits(cpu_seconds, mem_bytes)    # first thing inside the room: set the time + memory caps on ourselves

    # Build the room's rulebook. "__builtins__" is the ONLY set of basic tools the
    # stranger is allowed to touch — our whitelist, so no open()/import/eval exist for them.
    # The ** spreads in the safe calculators (math, statistics, …) already sitting on the desk,
    # so their code can use math.sqrt without ever needing to "import" anything.
    namespace = {"__builtins__": _ALLOWED_BUILTINS, **_ALLOWED_MODULES}

    stdout = io.StringIO()                   # our own empty out-tray: anything the code print()s will pile up here, not on the screen

    try:
        tree = ast.parse(code, mode="exec")  # read the recipe text and draw its structure-tree — this does NOT cook/run anything yet

        # Look at the very last line of the tree and ask: is it a value (an answer) or just an action?
        last_expr: ast.Expr | None = None        # declare the type up front: either an expression node, or nothing
        if tree.body:                            # only if the code has at least one line
            last_node = tree.body[-1]            # grab the last line into a plain variable
            if isinstance(last_node, ast.Expr):  # is it a bare value (an "answer" line)?
                last_expr = last_node            # yes → keep it; Pylance now KNOWS it's an ast.Expr, so .value is valid
                tree.body.pop()                  # and remove it from the list so exec() runs only the action lines

        with contextlib.redirect_stdout(stdout):   # for everything inside this block, swap the real out-tray for OUR out-tray
            exec(compile(tree, "<sandbox>", "exec"), namespace)   # run all the ACTION lines, top to bottom, under the room's rulebook
            value_repr = None                 # default: no answer to report
            if last_expr is not None:         # if we set aside a final answer line earlier…
                result = eval(                # …now compute just that one line and grab the value it produces
                    compile(ast.Expression(last_expr.value), "<sandbox>", "eval"),  # compile the final line in "give me a value" mode
                    namespace,                # …still under the same restricted rulebook
                )
                value_repr = repr(result)     # turn that answer into text, e.g. the number 4 becomes the text "4"

        # Success: slide a result under the door — the answer, plus everything that piled up in our out-tray.
        queue.put(SandboxResult(ok=True, value=value_repr, stdout=stdout.getvalue()))

    except Exception as e:
        # Something went wrong: the code tried open() (→ NameError, because it's not in the rulebook),
        # ran out of CPU/memory (the manager hit it), or had bad syntax. Report failure,
        # still handing back whatever they managed to print before failing.
        queue.put(SandboxResult(ok=False, error=f"{type(e).__name__}: {e}",
                                stdout=stdout.getvalue()))
        


def run_python(
    code: str,
    timeout: float = 5.0,                      # how many REAL seconds (stopwatch) we'll wait before evicting
    cpu_seconds: int = 2,                      # how much CPU "hands-time" the building manager allows the room
    mem_bytes: int | None = 256 * 1024 * 1024, # how big the room's whiteboard may get (256 MB)
) -> SandboxResult:
    """Supervisor: build a fresh locked room, send the stranger's `code` in,
    stand guard with a stopwatch, and return whatever comes back."""

    ctx = multiprocessing.get_context("spawn")  # insist on the FRESH-ROOM policy — identical on Mac and Linux
    result_q = ctx.Queue()                       # the slot under the door the worker will pass its result through

    # Describe the room: run _worker (everything from Snippet 2), handing it the code,
    # the two caps, and the slot. Nothing has started yet — this is just the blueprint.
    proc = ctx.Process(target=_worker, args=(code, cpu_seconds, mem_bytes, result_q))

    proc.start()              # build the room and send the stranger in — they begin working now
    proc.join(timeout)        # lean on the wall with the stopwatch; return as soon as they finish OR `timeout` seconds pass

    # ── Situation A: stopwatch ran out and they're STILL in there ──
    if proc.is_alive():                          # still working after the clock expired → stuck/looping/stalling
        proc.terminate()                         # knock the door down: polite "please stop" (SIGTERM)
        proc.join(1)                             # give them up to 1 second to actually stop
        if proc.is_alive():                      # somehow still standing?
            proc.kill()                          # break the door, drag them out: forceful, cannot be ignored (SIGKILL)
            proc.join()                          # wait for the room to be fully cleared
        return SandboxResult(ok=False, error=f"Timed out after {timeout}s")  # report the eviction

    # ── Situation B: they finished on their own — check the slot under the door ──
    try:
        return result_q.get(timeout=1)           # there's a slip in the slot → read it; that's our answer (success OR clean error)
    except Empty:
        # Gone, but the slot is empty → the building manager evicted them mid-job
        # for blowing the CPU or memory cap, too abruptly to leave a note.
        return SandboxResult(ok=False, error="Killed (CPU or memory limit exceeded)")