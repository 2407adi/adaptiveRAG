"""Dataset loader for the eval suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class EvalSample:
    """One labeled question from the eval dataset.

    Attributes:
        id: Stable identifier (e.g. "q014") — used as the key in result files.
        question: The user-facing query.
        ground_truth: Canonical answer. Used by context_recall.
        expected_route: What the router *should* output ("DIRECT" | "RAG" | "MULTI_STEP").
                        Used to score router accuracy as a free 5th metric.
        relevant_sources: Filenames of corpus docs that contain the answer.
                          Used by context_precision to label retrieved chunks
                          as relevant/irrelevant. Empty list = no doc needed
                          (DIRECT questions, or trick gap questions).
        diagnostic_intent: Human-readable note on what this question is probing.
                           Surfaced in result files when scores drop.
        skip_metrics: Names of metrics that should be skipped for this sample.
                      Use when a metric is not meaningful for the sample —
                      e.g. answer_relevancy on a "documents do not address
                      this" trick-gap answer (q012), where reverse-engineering
                      the original question from a noncommittal answer is
                      unreliable. The suite checks this list before calling
                      each metric.
    """
    id: str
    question: str
    ground_truth: str
    expected_route: str
    relevant_sources: list[str] = field(default_factory=list)
    diagnostic_intent: str = ""
    skip_metrics: list[str] = field(default_factory=list)


def load_dataset(path: str | Path) -> list[EvalSample]:
    """Load qa_pairs.json into a list of EvalSample records.

    Args:
        path: Path to the JSON file.

    Returns:
        List of EvalSample, in the order they appear in the file.

    Raises:
        FileNotFoundError: if the path does not exist.
        KeyError: if a sample is missing a required field.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        samples.append(EvalSample(
            id=item["id"],
            question=item["question"],
            ground_truth=item["ground_truth"],
            expected_route=item["expected_route"],
            relevant_sources=item.get("relevant_sources", []),
            diagnostic_intent=item.get("diagnostic_intent", ""),
            skip_metrics=item.get("skip_metrics", []),
        ))
    return samples