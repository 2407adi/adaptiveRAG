"""RAGAS-style evaluation metrics.

Each metric is a pure function: takes the data it needs plus its
dependencies, returns a score in [0, 1] or None if the metric is
not applicable to the sample (e.g. faithfulness on a DIRECT answer
that has no retrieved contexts).

Metrics:
    - faithfulness:      answer ↔ contexts
    - answer_relevancy:  answer ↔ question
    - context_recall:    contexts ↔ ground_truth
    - context_precision: contexts ↔ question (rank-aware)
"""

from __future__ import annotations

from ..reason.grounding import GroundingValidator
import json
import numpy as np


def faithfulness(
    answer: str,
    contexts: list[dict],
    validator: GroundingValidator,
) -> float | None:
    """Fraction of the answer's claims that are supported by the contexts.

    Thin wrapper around GroundingValidator.validate — that method already
    decomposes the answer into atomic claims and checks each one against
    the source chunks via LLM-as-judge entailment. Its .score field is
    grounded_claims / total_claims, which is exactly faithfulness.

    Args:
        answer:   The system's response text.
        contexts: Sources returned by RAGChain/MultiStepChain. Each dict
                  must have either 'full_text' or 'text_preview'
                  (matches the shape RAGChain produces).
        validator: A configured GroundingValidator. The suite owns its
                  lifecycle so the same instance is reused across samples.

    Returns:
        Score in [0, 1], or None when the metric is not meaningful:
        - No contexts: typically a DIRECT question that bypassed retrieval.
          Returning 0.0 here would punish correct routing, so we return None
          and let the suite exclude this sample from the faithfulness mean.

    Edge case:
        If the answer contains no extractable claims (e.g. a greeting or
        refusal), GroundingValidator returns 1.0 vacuously. We propagate
        that — it's not wrong (no false claims means nothing unsupported).
    """
    if not contexts:
        return None

    result = validator.validate(answer, contexts)
    return result.score

def answer_relevancy(
    question: str,
    answer: str,
    llm_client,
    embedder,
    n_generated: int = 3,
) -> float | None:
    """Mean cosine similarity between the original question and N questions
    that an LLM reverse-engineers from the answer alone.

    Intuition: an on-topic answer lets the LLM guess questions very close
    to the original. A rambling or off-topic answer produces guesses that
    diverge from the original, dragging the cosine similarity down.

    Args:
        question:    The original user question.
        answer:      The system's response.
        llm_client:  Has .generate(prompt) -> str.
        embedder:    Has .embed(text) -> list[float]. Same instance the
                     RAGChain uses — passed in by the suite.
        n_generated: Candidate questions to generate per sample. More
                     candidates = lower variance, higher LLM cost.
                     Default 3.

    Returns:
        Score in [0, 1], or None if:
        - the answer is empty (nothing to score), or
        - the LLM's question-generation output failed to parse.
    """
    if not answer.strip():
        return None

    candidates = _generate_questions(answer, llm_client, n_generated)
    if not candidates:
        return None

    q_vec = embedder.embed(question)
    sims = [_cosine(q_vec, embedder.embed(c)) for c in candidates]
    return float(np.mean(sims))


def _generate_questions(answer: str, llm_client, n: int) -> list[str]:
    """Ask the LLM to reverse-engineer N questions from the answer."""
    prompt = (
        f"You are reverse-engineering a question. Given an ANSWER, "
        f"generate {n} distinct questions that this answer could be "
        f"a direct response to.\n\n"
        f"Rules:\n"
        f"- Each question must be self-contained and end with '?'.\n"
        f"- Questions should reflect the SCOPE of the answer — do not "
        f"generalize or narrow.\n"
        f"- Output ONLY a JSON array of strings. No commentary.\n\n"
        f"Answer:\n{answer}\n\n"
        f"Output:"
    )
    raw = llm_client.generate(prompt)
    return _parse_questions(raw)


def _parse_questions(raw: str) -> list[str]:
    """Parse a JSON array of question strings, with a line-split fallback.

    Mirrors the parse-with-fallback pattern used in
    GroundingValidator._decompose_claims and MultiStepChain._parse_sub_questions.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        questions = json.loads(cleaned)
        if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
            return [q.strip() for q in questions if q.strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: line-by-line, keep things that look like questions
    out = []
    for line in raw.splitlines():
        line = line.strip().lstrip("0123456789.-) ").strip('" ')
        if line.endswith("?"):
            out.append(line)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)

def context_recall(
    ground_truth: str,
    contexts: list[dict],
    validator: GroundingValidator,
) -> float | None:
    """Fraction of ground-truth claims that are supported by the contexts.

    Symmetric counterpart to faithfulness:
        - faithfulness:    decompose the *answer*, check against contexts
        - context_recall:  decompose the *ground truth*, check against contexts

    Faithfulness alone cannot catch missing retrieval. If retrieval returns
    nothing relevant and the LLM says "I don't know", the answer is
    faithfully grounded in (irrelevant) retrieval — but the system is
    still broken. context_recall catches that case: if the GT says "the
    founders are X and Y" but retrieved chunks don't mention them, recall
    is low.

    Reuses GroundingValidator unchanged — same claim decomposition + LLM-
    as-judge entailment, just fed the GT instead of the answer.

    Args:
        ground_truth: The canonical answer for this question.
        contexts:     The retrieved sources.
        validator:    A configured GroundingValidator. Suite reuses the
                      same instance across faithfulness and context_recall.

    Returns:
        Score in [0, 1], or None when:
        - No contexts (DIRECT routing — recall is undefined).
        - Empty ground truth (nothing to recall against).
    """
    if not contexts or not ground_truth.strip():
        return None

    result = validator.validate(ground_truth, contexts)
    return result.score

def context_precision(
    contexts: list[dict],
    relevant_sources: list[str],
) -> float | None:
    """Mean Average Precision over retrieved contexts (rank-aware).

    Asks: are the relevant chunks ranked at the top? A retriever that
    puts the right chunk at position 1 scores higher than one that puts
    it at position 5 — even if both eventually surface it.

    Relevance labeling: a chunk is relevant iff its 'source' filename
    appears in `relevant_sources`. This is cheaper than RAGAS's LLM-as-
    judge approach but coarser — a chunk from the right file but with
    the wrong info gets over-counted as relevant. Adequate for catching
    file-level regressions; swap in LLM-as-judge later if needed.

    Args:
        contexts: Sources returned by RAGChain. Each dict must have a
                  'source' key (filename). RAGChain populates this via
                  Path(metadata['source']).name.
        relevant_sources: Filenames of corpus docs known to contain the
                          answer (from EvalSample).

    Returns:
        Score in [0, 1], where 1.0 = all relevant chunks ranked first,
        or None when:
        - No contexts (DIRECT routing — nothing to rank).
        - No relevant_sources labeled (can't compute without labels —
          this auto-skips q012 and the DIRECT questions).

        Returns 0.0 — not None — when contexts exist but none are
        relevant. That's a complete retrieval miss and IS the signal.
    """
    if not contexts:
        return None
    if not relevant_sources:
        return None

    relevant_set = set(relevant_sources)
    labels = [1 if c.get("source", "") in relevant_set else 0
              for c in contexts]

    total_relevant = sum(labels)
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_so_far = 0
    for i, label in enumerate(labels):
        if label == 1:
            relevant_so_far += 1
            precision_sum += relevant_so_far / (i + 1)

    return precision_sum / total_relevant