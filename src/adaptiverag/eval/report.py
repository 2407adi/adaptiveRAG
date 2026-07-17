"""Block 5.3: Ablation report — metric × rung scorecard + delta charts.

Reads one benchmark run directory (manifest.json + rung_*.json written by
AblationRunner) and produces:
    - ablation_report.md   — metric × rung table, per-layer deltas, the
                             "which layer paid rent" story, repro command
    - ablation_scores.png  — every metric across the ladder (line chart)
    - ablation_deltas.png  — per-layer delta bars for the headline metrics

Charts need matplotlib (dev extra). If it's missing, the markdown is still
generated and the charts are skipped with a warning — the report never
fails just because a plotting library is absent (e.g. in CI).

Entry point:
    python -m adaptiverag.eval.report eval_results/ablation/<run_id>
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Charted on the delta figure — the metrics that tell the layer story.
# (EM excluded: near-zero for long-form answers; router_accuracy excluded:
# exists on one rung only.)
HEADLINE_METRICS = ("f1", "faithfulness", "context_recall", "context_precision")


# ────────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────────

def load_run(run_dir: str | Path) -> tuple[dict, list[dict]]:
    """Read manifest + all rung result files, ordered by rung name.

    Rung names are prefixed 0_…6_, so lexicographic sort IS ladder order.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rungs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(run_dir.glob("rung_*.json"))
    ]
    if not rungs:
        raise FileNotFoundError(f"No rung_*.json files in {run_dir}")
    return manifest, rungs


def score_table(rungs: list[dict]) -> tuple[list[str], list[str], dict]:
    """Pivot rung files → (rung_names, metric_names, table[metric][rung]=mean)."""
    rung_names = [r["rung"] for r in rungs]
    metric_names: list[str] = []
    for r in rungs:                                   # preserve first-seen order
        for m in r["aggregate"]:
            if m not in metric_names:
                metric_names.append(m)
    table: dict[str, dict[str, float | None]] = {m: {} for m in metric_names}
    for r in rungs:
        for m in metric_names:
            stats = r["aggregate"].get(m) or {}
            table[m][r["rung"]] = stats.get("mean")
    return rung_names, metric_names, table


# ────────────────────────────────────────────────────────────────────
# Markdown
# ────────────────────────────────────────────────────────────────────

def _fmt(v: float | None) -> str:
    return "–" if v is None else f"{v:.3f}"


def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:+.3f}"


def _deltas(table: dict, rung_names: list[str]) -> dict:
    """delta[metric][rung] = mean(rung) − mean(previous rung).
    None whenever either side is missing (metric not applicable there)."""
    deltas: dict[str, dict[str, float | None]] = {}
    for metric, by_rung in table.items():
        deltas[metric] = {}
        for prev, cur in zip(rung_names, rung_names[1:]):
            a, b = by_rung.get(prev), by_rung.get(cur)
            deltas[metric][cur] = (b - a) if (a is not None and b is not None) else None
    return deltas


def build_markdown(manifest: dict, rungs: list[dict],
                   chart_files: list[str]) -> str:
    """Assemble the full report as a markdown string."""
    rung_names, metric_names, table = score_table(rungs)
    deltas = _deltas(table, rung_names)
    adds = {r["rung"]: r.get("adds", "") for r in rungs}

    lines: list[str] = []
    lines.append("# AdaptiveRAG Ablation Benchmark — every layer pays rent")
    lines.append("")
    lines.append(f"- **Dataset:** {manifest['dataset']} "
                 f"({manifest['sample_size']} questions, seed {manifest['seed']})")
    lines.append(f"- **Run:** {manifest['timestamp']}")
    lines.append("- **Method:** additive ablation — each rung enables exactly one "
                 "capability layer on top of the previous rung; same questions, "
                 "same corpus, same constants (web search off, memory off, "
                 "temperature 0).")
    if manifest.get("concise"):
        lines.append("- **Answer style:** concise mode — the system is asked for "
                     "short answer spans, so EM/F1 measure correctness rather "
                     "than verbosity. answer_relevancy is less informative in "
                     "this mode (a 3-word answer gives the reverse-engineering "
                     "judge little to work with).")
    lines.append("")

    # ── Ladder legend ──
    lines.append("## The ladder")
    lines.append("")
    lines.append("| Rung | Adds | Dispatch |")
    lines.append("|---|---|---|")
    for r in rungs:
        lines.append(f"| {r['rung']} | {r.get('adds', '')} | {r['dispatch']} |")
    lines.append("")

    # ── Scorecard: metric × rung ──
    lines.append("## Scorecard (mean per metric × rung)")
    lines.append("")
    lines.append("| Metric | " + " | ".join(rung_names) + " |")
    lines.append("|---|" + "---|" * len(rung_names))
    for m in metric_names:
        row = " | ".join(_fmt(table[m].get(rn)) for rn in rung_names)
        lines.append(f"| {m} | {row} |")
    lines.append("")
    lines.append("_– = not applicable (e.g. router_accuracy only exists on the "
                 "routed rung; context metrics need retrieved contexts). "
                 "exact_match is near zero by construction — the system writes "
                 "paragraph answers, EM demands the exact gold span; f1 is the "
                 "fair long-form comparison._")
    lines.append("")

    # ── Deltas: each rung vs the previous ──
    lines.append("## Per-layer deltas (rung vs previous rung)")
    lines.append("")
    delta_cols = rung_names[1:]
    lines.append("| Metric | " + " | ".join(delta_cols) + " |")
    lines.append("|---|" + "---|" * len(delta_cols))
    for m in metric_names:
        row = " | ".join(_fmt_delta(deltas[m].get(rn)) for rn in delta_cols)
        lines.append(f"| {m} | {row} |")
    lines.append("")

    # ── The story: biggest mover per layer ──
    lines.append("## What each layer bought")
    lines.append("")
    for rn in delta_cols:
        movers = [(m, deltas[m][rn]) for m in metric_names
                  if deltas[m].get(rn) is not None]
        if not movers:
            lines.append(f"- **{rn}** ({adds.get(rn, '')}): no comparable metrics.")
            continue
        best = max(movers, key=lambda x: x[1])
        worst = min(movers, key=lambda x: x[1])
        summary = f"biggest gain {_fmt_delta(best[1])} on {best[0]}"
        if worst[1] < 0:
            summary += f"; biggest cost {_fmt_delta(worst[1])} on {worst[0]}"
        lines.append(f"- **{rn}** ({adds.get(rn, '')}): {summary}.")
    lines.append("")

    # ── Runtime ──
    lines.append("## Runtime & errors")
    lines.append("")
    lines.append("| Rung | Elapsed (s) | Errors |")
    lines.append("|---|---|---|")
    for r in rungs:
        lines.append(f"| {r['rung']} | {r.get('elapsed_seconds', '–')} "
                     f"| {r.get('error_count', 0)} |")
    lines.append("")

    # ── Charts ──
    if chart_files:
        lines.append("## Charts")
        lines.append("")
        for name in chart_files:
            lines.append(f"![{name}]({name})")
            lines.append("")

    # ── Reproducibility ──
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python -m adaptiverag.eval.benchmark "
                 f"--dataset {manifest['dataset']} "
                 f"--sample {manifest['sample_size']} --seed {manifest['seed']}")
    lines.append("```")
    lines.append("")
    lines.append(f"Sample ids are pinned in `manifest.json` "
                 f"({len(manifest.get('sample_ids', []))} ids). LLM scoring is "
                 "nondeterministic at the margins — treat single-run deltas "
                 "under ~0.02 as noise.")
    lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Charts (matplotlib, optional)
# ────────────────────────────────────────────────────────────────────

def build_charts(rungs: list[dict], out_dir: str | Path) -> list[str]:
    """Write the two PNGs into out_dir; returns the filenames created.
    Silently skips (with a log warning) when matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")               # headless backend — no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping charts "
                       "(pip install -e '.[dev]')")
        return []

    out_dir = Path(out_dir)
    rung_names, metric_names, table = score_table(rungs)
    deltas = _deltas(table, rung_names)
    created: list[str] = []

    # ── 1. Scores across the ladder (line chart) ──
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(rung_names))
    for m in metric_names:
        ys = [table[m].get(rn) for rn in rung_names]
        if all(y is None for y in ys):
            continue
        ax.plot(x, [y if y is not None else float("nan") for y in ys],
                marker="o", label=m)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rung_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("AdaptiveRAG ablation — every capability layer, one rung at a time")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_scores.png", dpi=150)
    plt.close(fig)
    created.append("ablation_scores.png")

    # ── 2. Per-layer deltas (grouped bars, headline metrics) ──
    delta_cols = rung_names[1:]
    metrics = [m for m in HEADLINE_METRICS if m in table]
    if delta_cols and metrics:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        width = 0.8 / len(metrics)
        for i, m in enumerate(metrics):
            offsets = [j + i * width for j in range(len(delta_cols))]
            vals = [deltas[m].get(rn) or 0.0 for rn in delta_cols]
            ax.bar(offsets, vals, width=width, label=m)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([j + width * (len(metrics) - 1) / 2
                       for j in range(len(delta_cols))])
        ax.set_xticklabels(delta_cols, rotation=20, ha="right")
        ax.set_ylabel("Δ vs previous rung")
        ax.set_title("What each layer bought (delta per metric)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_deltas.png", dpi=150)
        plt.close(fig)
        created.append("ablation_deltas.png")

    return created


# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────

def generate_report(run_dir: str | Path,
                    copy_latest_to: str | Path | None = None) -> Path:
    """Produce ablation_report.md (+ charts) inside run_dir.

    copy_latest_to: optionally mirror the report + charts into a stable
    location (eval_results/ablation_report.md) so the README/CI always
    has a fixed path to the most recent report.
    """
    run_dir = Path(run_dir)
    manifest, rungs = load_run(run_dir)

    chart_files = build_charts(rungs, run_dir)
    markdown = build_markdown(manifest, rungs, chart_files)

    report_path = run_dir / "ablation_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    logger.info("Report written → %s", report_path)

    if copy_latest_to is not None:
        latest_dir = Path(copy_latest_to)
        latest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_path, latest_dir / "ablation_report.md")
        for name in chart_files:
            shutil.copyfile(run_dir / name, latest_dir / name)

    return report_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python -m adaptiverag.eval.report <run_dir>")
        raise SystemExit(2)
    path = generate_report(sys.argv[1], copy_latest_to=Path("eval_results"))
    print(f"Report → {path}")
