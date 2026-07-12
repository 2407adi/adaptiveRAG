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
from dataclasses import dataclass, asdict

import ast              # the tool that reads code-text and draws its "sentence-diagram" tree
import io               # gives us an in-memory text box we can use as our own out-tray
import contextlib       # has the helper that swaps the out-tray for ours

import sys              # lets us find THIS interpreter (sys.executable) to launch a fresh, identical one
import json             # how the parent and the locked room pass notes (code in, result out)
import subprocess       # launches a brand-new EMPTY room (fresh interpreter) — not a clone of this crowded one
from pathlib import Path  # to locate src/ so the fresh room can import just this one light module

# Calculation-only modules we hand to the code directly (no import needed,
# because __import__ is removed). All pure-compute, no filesystem/network.
import math
import statistics
import random
import json as _json
import re as _re
import datetime as _datetime
import itertools
import functools
import collections
import decimal
import fractions

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


def _execute(code: str, cpu_seconds: int, mem_bytes: int | None) -> SandboxResult:
    """The work done INSIDE the locked room. Cap ourselves, run the stranger's
    code under the rulebook, and return a result slip. This is the SAME pure-compute
    logic as before — only the room it runs in has changed (a fresh, empty
    interpreter instead of a clone of this crowded parent process)."""

    _apply_limits(cpu_seconds, mem_bytes)    # first thing inside the room: set the time + memory caps on ourselves

    # The room's rulebook. "__builtins__" is the ONLY set of basic tools the stranger may
    # touch — our whitelist, so no open()/import/eval exist for them. The ** spreads in the
    # safe calculators (math, statistics, …) already on the desk, so code can use math.sqrt
    # without ever needing to "import".
    namespace = {"__builtins__": _ALLOWED_BUILTINS, **_ALLOWED_MODULES}

    stdout = io.StringIO()                   # our own empty out-tray: the code's print()s pile up here, not on the screen

    try:
        tree = ast.parse(code, mode="exec")  # read the recipe text and draw its structure-tree — runs nothing yet

        # Is the very last line a value (an "answer") or just an action?
        last_expr: ast.Expr | None = None        # declare the type up front: an expression node, or nothing
        if tree.body:                            # only if the code has at least one line
            last_node = tree.body[-1]            # grab the last line
            if isinstance(last_node, ast.Expr):  # is it a bare value (an "answer" line)?
                last_expr = last_node            # yes → keep it (Pylance now KNOWS it's an ast.Expr)
                tree.body.pop()                  # …and remove it so exec() runs only the action lines

        with contextlib.redirect_stdout(stdout):   # swap the real out-tray for OUR out-tray
            exec(compile(tree, "<sandbox>", "exec"), namespace)   # run the ACTION lines under the room's rulebook
            value_repr = None                 # default: no answer to report
            if last_expr is not None:         # if we set aside a final answer line…
                result = eval(                # …compute just that one line and grab its value
                    compile(ast.Expression(last_expr.value), "<sandbox>", "eval"),  # "give me a value" mode
                    namespace,                # …still under the same restricted rulebook
                )
                value_repr = repr(result)     # turn the answer into text, e.g. 4 → "4"

        return SandboxResult(ok=True, value=value_repr, stdout=stdout.getvalue())

    except Exception as e:
        # open() → NameError, import → ImportError, [0]*10**12 → MemoryError, bad syntax → SyntaxError.
        # Report failure, still handing back whatever they printed before failing.
        return SandboxResult(ok=False, error=f"{type(e).__name__}: {e}",
                             stdout=stdout.getvalue())


def _child_main() -> None:
    """The entry point that runs INSIDE the fresh room. Reads the job (code + caps)
    from the note slid under the door (stdin), runs it, and slides the result slip
    back out under the door (stdout) as JSON. This is what the parent launches."""
    payload = json.loads(sys.stdin.read())               # the note under the door: {"code", "cpu_seconds", "mem_bytes"}
    res = _execute(payload["code"], payload["cpu_seconds"], payload.get("mem_bytes"))
    sys.stdout.write(json.dumps(asdict(res)))            # the result slip, back under the door


# Where the `adaptiverag` package lives (src/), so the fresh room can import just THIS
# one light module and nothing heavy. sandbox.py is src/adaptiverag/agents/sandbox.py,
# so two levels up (parents[2]) is src/.
_SRC = str(Path(__file__).resolve().parents[2])


def run_python(
    code: str,
    timeout: float = 5.0,                      # REAL seconds (stopwatch) before we evict the room
    cpu_seconds: int = 2,                      # CPU "hands-time" the building manager allows the room
    mem_bytes: int | None = 256 * 1024 * 1024, # the room's whiteboard cap (256 MB)
) -> SandboxResult:
    """Supervisor: send the stranger's `code` down a CLEAN hallway into a fresh,
    EMPTY room (a brand-new minimal interpreter), stand guard with a stopwatch,
    and read back whatever slip comes out under the door.

    Why a fresh interpreter instead of multiprocessing-spawn: spawn CLONES the
    parent's module graph. Inside the full app the parent is huge (torch, chroma,
    langgraph…), so the clone was born oversized and our small caps killed it
    before it could answer — surfacing as a bogus "CPU/memory limit exceeded" on
    even 2+2. A fresh `python -E` imports only this one light module, so the caps
    apply to a genuinely tiny room, exactly as intended.
    """
    # The room's orders: put src/ on the path, import ONLY this light module, run the child entry.
    boot = (
        f"import sys; sys.path.insert(0, {_SRC!r}); "
        f"from adaptiverag.agents.sandbox import _child_main; _child_main()"
    )
    note = json.dumps({"code": code, "cpu_seconds": cpu_seconds, "mem_bytes": mem_bytes})

    try:
        # -E ignores PYTHON* env vars → a clean room. We slide the note in via stdin and
        # read the result slip from stdout. `timeout` is the wall-clock stopwatch.
        proc = subprocess.run(
            [sys.executable, "-E", "-c", boot],
            input=note, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(ok=False, error=f"Timed out after {timeout}s")   # stopwatch caught a stall/loop

    # No slip under the door (empty stdout) or a non-zero exit → the manager evicted the
    # room (CPU/memory cap) or the code crashed the interpreter. Now we can say why.
    if proc.returncode != 0 or not proc.stdout.strip():
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}"
        return SandboxResult(ok=False, error=f"Killed (resource limit or crash): {tail}")

    return SandboxResult(**json.loads(proc.stdout))   # read the result slip back into a SandboxResult