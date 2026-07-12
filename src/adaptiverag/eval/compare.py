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

THRESHOLD = 0.05          # the noise-vs-news line: >5% relative drop = regression


def load_per_metric(path: str | Path) -> dict:
    """Pull just the per-metric means out of a saved EvalSuite result file."""
    summary = json.loads(Path(path).read_text())["summary"]
    return summary["per_metric"]           # {metric: {"n":…, "mean":…, "min":…, "max":…}}


def compare(baseline: dict, current: dict, threshold: float = THRESHOLD) -> list[dict]:
    """One row of verdict per metric the two scorecards share."""
    rows = []
    for metric, base in baseline.items():
        cur = current.get(metric)
        if cur is None:                    # metric missing today (e.g. errored samples)
            rows.append({"metric": metric, "baseline": base["mean"],
                         "current": None, "delta": None, "regressed": True})
            continue                       # missing = treated as a failure, not ignored
        base_mean, cur_mean = base["mean"], cur["mean"]
        if base_mean == 0:                 # can't divide by zero — fall back to absolute drop
            delta = cur_mean - base_mean
        else:
            delta = (cur_mean - base_mean) / base_mean       # RELATIVE change (the key idea)
        rows.append({"metric": metric, "baseline": base_mean, "current": cur_mean,
                     "delta": delta, "regressed": delta < -threshold})
    return rows



def to_markdown(rows: list[dict]) -> str:
    """The human-facing scorecard — a markdown table for GitHub's bulletin board."""
    lines = ["## Eval regression report",
             "",
             "| Metric | Baseline | Current | Change | Verdict |",
             "|---|---|---|---|---|"]
    for r in rows:
        if r["current"] is None:
            lines.append(f"| {r['metric']} | {r['baseline']:.3f} | — | — | REGRESSED (metric missing) |")
            continue
        verdict = "REGRESSED" if r["regressed"] else "OK"
        lines.append(f"| {r['metric']} | {r['baseline']:.3f} | {r['current']:.3f} "
                     f"| {r['delta']:+.1%} | {verdict} |")     # +.1% → e.g. '+3.2%' / '-7.4%'
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