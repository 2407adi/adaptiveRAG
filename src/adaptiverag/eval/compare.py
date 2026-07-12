"""Regression gate: compare an eval run against the stored baseline.

The EvalSuite produces the scorecards; this module only READS two of them
(baseline vs. current) and answers one question — did any metric drop more
than the threshold, proportionally? Exit code carries the verdict to CI.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys

DEFAULT_THRESHOLD = 0.05  # the noise-vs-news line: >5% relative drop = regression

# Per-metric overrides. router_accuracy moves in coarse steps — 25 samples means
# each flipped routing decision is a full 4-point step, so a 5% line lets ONE
# coin-flip sample nearly trip the alarm. It gets a two-flip allowance instead.
# The averaged LLM-judged metrics vary smoothly and keep the tight default.
THRESHOLDS = {
    "router_accuracy": 0.10,
}


def load_per_metric(path: str | Path) -> dict:
    """Pull just the per-metric means out of a saved EvalSuite result file."""
    summary = json.loads(Path(path).read_text())["summary"]
    return summary["per_metric"]           # {metric: {"n":…, "mean":…, "min":…, "max":…}}


def compare(baseline: dict, current: dict,
            thresholds: dict[str, float] | None = None) -> list[dict]:
    """One row of verdict per metric the two scorecards share.

    Each metric is judged against its own tolerance: THRESHOLDS override
    when present, DEFAULT_THRESHOLD otherwise."""
    thresholds = thresholds if thresholds is not None else THRESHOLDS
    rows = []
    for metric, base in baseline.items():
        threshold = thresholds.get(metric, DEFAULT_THRESHOLD)
        cur = current.get(metric)
        if cur is None:                    # metric missing today (e.g. errored samples)
            rows.append({"metric": metric, "baseline": base["mean"], "current": None,
                         "delta": None, "threshold": threshold, "regressed": True})
            continue                       # missing = treated as a failure, not ignored
        base_mean, cur_mean = base["mean"], cur["mean"]
        if base_mean == 0:                 # can't divide by zero — fall back to absolute drop
            delta = cur_mean - base_mean
        else:
            delta = (cur_mean - base_mean) / base_mean       # RELATIVE change (the key idea)
        rows.append({"metric": metric, "baseline": base_mean, "current": cur_mean,
                     "delta": delta, "threshold": threshold,
                     "regressed": delta < -threshold})
    return rows



def to_markdown(rows: list[dict]) -> str:
    """The human-facing scorecard — a markdown table for GitHub's bulletin board."""
    lines = ["## Eval regression report",
             "",
             "| Metric | Baseline | Current | Change | Tolerance | Verdict |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        tol = f"-{r['threshold']:.0%}"                         # e.g. '-5%' / '-10%'
        if r["current"] is None:
            lines.append(f"| {r['metric']} | {r['baseline']:.3f} | — | — | {tol} "
                         f"| REGRESSED (metric missing) |")
            continue
        verdict = "REGRESSED" if r["regressed"] else "OK"
        lines.append(f"| {r['metric']} | {r['baseline']:.3f} | {r['current']:.3f} "
                     f"| {r['delta']:+.1%} | {tol} | {verdict} |")   # +.1% → e.g. '-7.4%'
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    # Arg 1 = baseline file; arg 2 = current file. Both optional:
    baseline_path = Path(argv[1]) if len(argv) > 1 else Path("eval_results/baseline.json")
    if len(argv) > 2:
        current_path = Path(argv[2])
    else:
        # suite.py names runs '2026-07-12T03-00-00Z.json' — for that format,
        # alphabetical sort IS chronological sort, so newest = last.
        runs = sorted(Path("eval_results").glob("2*.json"))
        if not runs:
            print("No eval run found in eval_results/ — run the suite first.")
            return 1
        current_path = runs[-1]

    rows = compare(load_per_metric(baseline_path), load_per_metric(current_path))
    report = to_markdown(rows)

    print(report)                                        # always: plain print for local runs
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY") # in CI: the bulletin board's file path
    if summary_file:
        with open(summary_file, "a") as f:               # 'a' = append — play nice with
            f.write(report + "\n")                       # anything already pinned there

    return 1 if any(r["regressed"] for r in rows) else 0 # the machine-facing verdict


if __name__ == "__main__":
    sys.exit(main(sys.argv))