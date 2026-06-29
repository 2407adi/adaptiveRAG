"""The tool system: plain functions ("contractors") + a registry ("front desk").

Story: a talent agency. @tool is the badge-maker that prints a standardized
business card (name + job summary + what-it-needs) from a contractor's own
details. The ToolRegistry (next snippet) keeps the roster, dispatches jobs,
and stamps the logbook (audit log) on every call. See CLAUDE.md Block 3.1.
"""

from __future__ import annotations

import inspect                                  # lets us READ a function's name, docstring, and parameters
from dataclasses import dataclass
from typing import Callable, get_type_hints     # get_type_hints reads the type written next to each parameter

from .audit import AuditLog


# Translation table: Python type → the word the boss (LLM) expects on the card.
# JSON Schema is the boss's language; "str" on our side reads as "string" on the card.
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolSpec:
    """The contractor's printed business card."""
    name: str            # the contractor's name (e.g. "run_python")
    description: str     # the one-line "what I do", taken from the function's docstring
    parameters: dict     # the "what I need to start" section, as JSON Schema


def _build_schema(func: Callable) -> dict:
    """Write the 'what I need from you' section of the card by reading the args."""
    sig = inspect.signature(func)               # grab the contractor's list of parameters
    hints = get_type_hints(func)                # grab the type written next to each parameter
    properties: dict = {}                       # the per-field "name → its type" map for the card
    required: list[str] = []                    # which fields the boss MUST fill in

    for pname, param in sig.parameters.items():           # walk each parameter the contractor declared
        ptype = hints.get(pname, str)                     # its declared type (assume text if none given)
        json_type = _TYPE_MAP.get(ptype, "string")        # translate that type into the boss's language
        properties[pname] = {"type": json_type}           # record it on the card
        if param.default is inspect.Parameter.empty:      # no default value supplied?
            required.append(pname)                         # → then it's mandatory; the boss must provide it

    # The finished "what I need" section, in the standard JSON-Schema shape.
    return {"type": "object", "properties": properties, "required": required}


def tool(func: Callable) -> Callable:
    """The badge-maker: staple a business card (ToolSpec) onto a plain function."""
    spec = ToolSpec(                                       # print the card from the contractor's own details:
        name=func.__name__,                               #   name  = the function's name
        description=(inspect.getdoc(func) or "").strip(),  #   bio   = the function's docstring
        parameters=_build_schema(func),                    #   needs = built from the function's parameters
    )
    setattr(func, "tool_spec", spec)                      # pin the badge on (setattr keeps Pylance happy vs func.tool_spec = ...)
    return func                                            # hand back the SAME function — still plain and directly callable, just badged


# DEFINING THE TOOLSREGISTRY CLASS THAT WILL BE USED BY THE AGENT TO CALL THE TOOLS. THIS CLASS WILL BE USED TO REGISTER THE TOOLS AND CALL THEM WHEN NEEDED.

class ToolRegistry:
    """The front desk: keeps the roster, hands out business cards, dispatches
    jobs, and stamps the logbook (audit log) on every single call."""

    def __init__(self, audit_log: AuditLog | None = None):
        self._tools: dict[str, Callable] = {}    # the roster: contractor name → the function itself
        self._audit = audit_log                  # the logbook to stamp (None = don't log, handy in some tests)

    def register(self, func: Callable) -> None:
        """A contractor checks in — file them on the roster (if they're badged)."""
        spec = getattr(func, "tool_spec", None)              # glance at their badge
        if spec is None:                                     # no badge = never went through @tool
            raise ValueError(f"{getattr(func, '__name__', func)!r} has no @tool badge")
        self._tools[spec.name] = func                        # file them under their name

    def list_tools(self) -> list[dict]:
        """Hand the boss (LLM) the stack of business cards — one per contractor."""
        cards = []
        for func in self._tools.values():                    # walk the roster
            spec = getattr(func, "tool_spec")                # read each contractor's card
            cards.append({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            })
        return cards

    def call(self, name: str, args: dict):
        """Dispatch one job: find the contractor, send them in, stamp the logbook, return the result."""
        contractor = self._tools.get(name)                   # look up the named contractor on the roster

        if contractor is None:                               # boss asked for someone who isn't here…
            result, ok = f"Unknown tool: {name!r}", False    # …note it, don't crash the agency
        else:
            try:
                result = contractor(**args)                  # send them in with the job's arguments; bring back the result
                ok = True
            except Exception as e:                           # contractor flubbed it (bad args, internal error)…
                result = f"{type(e).__name__}: {e}"          # …write down what went wrong instead of melting down
                ok = False

        if self._audit is not None:                          # stamp the logbook for EVERY job — success or failure
            self._audit.append(tool=name, args=args, result=result, ok=ok)

        return result                                        # hand back a value on success, or an error note on failure
    

## WRITING THE CODE FOR THE THREE TOOLS THAT WE WILL REGISTER IN THE TOOL REGISTRY. THESE TOOLS ARE: 1) run_python, 2) search_documents, 3) web_search

# The secure ENGINE from Step 2. We alias it so the name doesn't collide with
# the TOOL named run_python defined just below (your earlier question — engine vs tool).
from .sandbox import run_python as _sandbox_run_python


def make_run_python(timeout: float = 5.0, cpu_seconds: int = 2,
                    max_memory_mb: int = 256) -> Callable:
    """Factory: hire the accountant with the agency's house rules (room limits)
    baked in, so the boss only ever supplies `code`."""
    mem_bytes = max_memory_mb * 1024 * 1024          # MB (human-friendly) → bytes (what the engine wants)

    @tool
    def run_python(code: str) -> str:
        """Execute a short Python snippet for calculations or data analysis and
        return its output. No file or network access is available."""
        outcome = _sandbox_run_python(               # walk the code to the locked room WITH the configured limits
            code, timeout=timeout, cpu_seconds=cpu_seconds, mem_bytes=mem_bytes,
        )                                            # timeout/cpu/mem are REMEMBERED from hire time (closure)
        if not outcome.ok:                           # blocked, errored, or evicted
            return f"Error: {outcome.error}"
        parts = []
        if outcome.stdout:
            parts.append(outcome.stdout.rstrip())
        if outcome.value is not None:
            parts.append(outcome.value)
        return "\n".join(parts) if parts else "(no output)"

    return run_python


from pathlib import Path        # lets us turn a long file path into just its filename

def make_search_documents(rag_chain) -> Callable:
    """Factory: hire a librarian who already knows where the archive is.

    `rag_chain` is any object with a `.retrieve(query) -> list` method (duck typing,
    so this stays domain-agnostic). We capture it once here, at hire time.
    """

    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded documents for passages relevant to the
        query. Returns the most relevant chunks, each labeled with its source."""
        # `rag_chain` below isn't a parameter — it's REMEMBERED from the factory
        # (the closure). That's why the boss only ever supplies `query`.
        results = rag_chain.retrieve(query)          # go to the known archive, pull the most relevant passages

        if not results:                             # archive had nothing on this topic
            return "No relevant passages found."

        blocks = []
        for i, r in enumerate(results, 1):          # number the passages 1, 2, 3… for easy citing
            source = Path(r.metadata.get("source", "unknown")).name   # just the filename, e.g. "report.pdf", not the full path
            blocks.append(f"[{i}] (source: {source})\n{r.text}")      # one labeled block per passage
        return "\n\n".join(blocks)                   # hand the boss a tidy, readable stack

    return search_documents                          # the hired, badged librarian — ready to put on the roster


def make_web_search(api_key: str | None = None, max_results: int = 3) -> Callable:
    """Factory: hire a web researcher with the agency's search account (Tavily).

    No account login (api_key) → we still hire them, but they'll report they
    can't work until configured, so the roster always has exactly 3 contractors.
    """
    client = None
    if api_key:                                       # only set up the account if we were given a login
        try:
            from tavily import TavilyClient           # lazy import: only needed when a key actually exists
            client = TavilyClient(api_key=api_key)     # log the researcher into the web-search account (baked in via closure)
        except ImportError:
            client = None                              # toolkit not installed → behave as "not configured"

    @tool
    def web_search(query: str) -> str:
        """Search the public web for up-to-date or supplementary information not
        found in the user's documents. Returns a few relevant snippets."""
        if client is None:                            # no working account was set up at hire time
            return "Web search is not configured (missing Tavily API key or client)."

        try:
            response = client.search(query=query, max_results=max_results)  # ask Tavily; `max_results` remembered from hire time
        except Exception as e:                        # network hiccup, rate limit, bad key…
            return f"Web search error: {type(e).__name__}: {e}"   # a clean note, not a crash

        results = response.get("results", [])         # Tavily returns a dict; the hits live under "results"
        if not results:
            return "No web results found."

        blocks = []
        for i, item in enumerate(results, 1):         # number the hits for easy citing
            title = item.get("title", "untitled")
            url = item.get("url", "")
            content = item.get("content", "")         # Tavily's short relevant snippet for that page
            blocks.append(f"[{i}] {title}\n{url}\n{content}")
        return "\n\n".join(blocks)                    # hand the boss a tidy, labeled stack of web snippets

    return web_search                                 # the hired, badged researcher

## Building the final tools registery usng the tools we wrote above. This is the final registry that will be used by the agent to call the tools.

def build_default_registry(rag_chain, tools_cfg, *,
                           hmac_key: str | None = None,
                           tavily_api_key: str | None = None) -> ToolRegistry:
    """Open the agency using the policy binder (tools_cfg) + secrets from the safe."""
    audit = AuditLog(tools_cfg.audit.path, key=hmac_key)   # logbook at the configured path, stamped with the secret key
    registry = ToolRegistry(audit_log=audit)               # seat the receptionist with that logbook

    # Accountant — hired with the room's house rules from the binder.
    registry.register(make_run_python(
        timeout=tools_cfg.sandbox.timeout,
        cpu_seconds=tools_cfg.sandbox.cpu_seconds,
        max_memory_mb=tools_cfg.sandbox.max_memory_mb,
    ))
    # Librarian — bound to this archive.
    registry.register(make_search_documents(rag_chain))
    # Researcher — given the web account only if web search is enabled in the binder.
    registry.register(make_web_search(
        tavily_api_key if tools_cfg.tavily.enabled else None,
        tools_cfg.tavily.max_results,
    ))
    return registry