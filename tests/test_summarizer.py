"""Tests for CorpusSummarizer — corpus topic summary generation."""

from unittest.mock import MagicMock

from adaptiverag.ingest.summarizer import CorpusSummarizer
from adaptiverag.ingest.models import Document


# ── Helpers ────────────────────────────────────────────────

def _make_mock_llm(*responses: str) -> MagicMock:
    """Mock LLM client. One arg → fixed response; multiple → scripted sequence."""
    mock = MagicMock()
    if len(responses) == 1:
        mock.generate.return_value = responses[0]
    else:
        mock.generate.side_effect = list(responses)
    return mock


def _doc(text: str, source: str = "doc.txt") -> Document:
    """Construct a Document with source set in metadata."""
    return Document(text=text, metadata={"source": source})


# ── _build_stuff_prompt ────────────────────────────────────

class TestBuildStuffPrompt:

    def test_includes_document_content(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt")
        prompt = s._build_stuff_prompt([_doc("Solstice was founded April 2021", "a.md")])
        assert "Solstice was founded April 2021" in prompt

    def test_includes_source_header(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt")
        prompt = s._build_stuff_prompt([_doc("content", "funding.md")])
        assert "[document: funding.md]" in prompt

    def test_handles_missing_source_metadata(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt")
        prompt = s._build_stuff_prompt([Document(text="content", metadata={})])
        assert "[document: unknown]" in prompt


# ── _build_reduce_prompt ───────────────────────────────────

class TestBuildReducePrompt:

    def test_includes_all_batch_summaries(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt")
        prompt = s._build_reduce_prompt(["batch one paragraph", "batch two paragraph"])
        assert "batch one paragraph" in prompt
        assert "batch two paragraph" in prompt
        assert "[batch 1 summary]" in prompt
        assert "[batch 2 summary]" in prompt


# ── _split_oversize_doc ────────────────────────────────────

class TestSplitOversizeDoc:

    def test_splits_into_correct_number_of_pieces(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt", char_budget=100)
        pieces = s._split_oversize_doc(_doc("a" * 250, "huge.txt"))
        assert len(pieces) == 3  # ceil(250/100)

    def test_pieces_each_fit_within_budget(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt", char_budget=100)
        pieces = s._split_oversize_doc(_doc("a" * 250, "huge.txt"))
        for p in pieces:
            assert len(p.text) <= 100

    def test_pieces_labeled_with_part_metadata(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt", char_budget=100)
        pieces = s._split_oversize_doc(_doc("a" * 250, "huge.txt"))
        sources = [p.metadata["source"] for p in pieces]
        assert sources == [
            "huge.txt (part 1/3)",
            "huge.txt (part 2/3)",
            "huge.txt (part 3/3)",
        ]

    def test_concatenation_preserves_original_text(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt", char_budget=100)
        original = "a" * 250
        pieces = s._split_oversize_doc(_doc(original, "huge.txt"))
        assert "".join(p.text for p in pieces) == original


# ── _batch_documents ───────────────────────────────────────

class TestBatchDocuments:

    def test_small_corpus_fits_in_one_batch(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt", char_budget=1000)
        batches = s._batch_documents([_doc("a" * 100, "a"), _doc("b" * 100, "b")])
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_overflow_splits_into_multiple_batches(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt", char_budget=100)
        # 3 docs × 60 chars: first fits, second forces flush, etc. → 3 batches.
        docs = [_doc("a" * 60, f"{i}") for i in range(3)]
        assert len(s._batch_documents(docs)) == 3

    def test_oversize_doc_is_pre_split_then_batched(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt", char_budget=100)
        batches = s._batch_documents([_doc("a" * 250, "huge.txt")])
        # Every batch must respect the budget
        for batch in batches:
            assert sum(len(d.text) for d in batch) <= 100

    def test_empty_input_returns_empty_list(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt")
        assert s._batch_documents([]) == []


# ── generate ───────────────────────────────────────────────

class TestGenerate:

    def test_empty_documents_returns_none(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt")
        assert s.generate([]) is None

    def test_stuff_mode_single_llm_call(self, tmp_path):
        llm = _make_mock_llm("This is the corpus summary.")
        s = CorpusSummarizer(llm, tmp_path / "s.txt", char_budget=10_000)
        result = s.generate([_doc("Small content", "a.txt")])
        assert result == "This is the corpus summary."
        assert llm.generate.call_count == 1

    def test_map_reduce_mode_calls_llm_multiple_times(self, tmp_path):
        # 2 docs × 40 chars, budget 50 → 2 batches → 2 map + 1 reduce = 3 calls
        llm = _make_mock_llm("batch 1", "batch 2", "unified summary")
        s = CorpusSummarizer(llm, tmp_path / "s.txt", char_budget=50)
        docs = [_doc("a" * 40, f"{i}") for i in range(2)]
        result = s.generate(docs)
        assert result == "unified summary"
        assert llm.generate.call_count == 3

    def test_llm_exception_returns_none(self, tmp_path):
        llm = MagicMock()
        llm.generate.side_effect = Exception("API timeout")
        s = CorpusSummarizer(llm, tmp_path / "s.txt")
        assert s.generate([_doc("content", "a.txt")]) is None

    def test_empty_llm_response_returns_none(self, tmp_path):
        llm = _make_mock_llm("   \n   ")  # whitespace only
        s = CorpusSummarizer(llm, tmp_path / "s.txt")
        assert s.generate([_doc("content", "a.txt")]) is None


# ── save / load ────────────────────────────────────────────

class TestPersistence:

    def test_save_creates_parent_dir(self, tmp_path):
        persist_path = tmp_path / "nested" / "deeper" / "summary.txt"
        s = CorpusSummarizer(_make_mock_llm("x"), persist_path)
        s.save("a summary")
        assert persist_path.exists()
        assert persist_path.read_text(encoding="utf-8") == "a summary"

    def test_load_returns_none_when_file_missing(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "missing.txt")
        assert s.load() is None

    def test_load_returns_none_for_empty_file(self, tmp_path):
        persist_path = tmp_path / "empty.txt"
        persist_path.write_text("   \n  \n", encoding="utf-8")
        s = CorpusSummarizer(_make_mock_llm("x"), persist_path)
        assert s.load() is None

    def test_save_then_load_roundtrip(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt")
        s.save("Solstice Robotics, founded April 2021, raised $40M.")
        assert s.load() == "Solstice Robotics, founded April 2021, raised $40M."

    def test_save_uses_utf8_for_non_ascii(self, tmp_path):
        s = CorpusSummarizer(_make_mock_llm("x"), tmp_path / "s.txt")
        s.save("Acmé Corp raised €50M — successful round.")
        assert s.load() == "Acmé Corp raised €50M — successful round."


# ── generate_and_save ──────────────────────────────────────

class TestGenerateAndSave:

    def test_saves_when_generate_returns_summary(self, tmp_path):
        persist_path = tmp_path / "s.txt"
        s = CorpusSummarizer(_make_mock_llm("A real summary."), persist_path)
        result = s.generate_and_save([_doc("content", "a.txt")])
        assert result == "A real summary."
        assert persist_path.read_text(encoding="utf-8") == "A real summary."

    def test_skips_save_when_generate_returns_none(self, tmp_path):
        # Empty docs → generate returns None → no file should be written
        persist_path = tmp_path / "s.txt"
        s = CorpusSummarizer(_make_mock_llm("x"), persist_path)
        assert s.generate_and_save([]) is None
        assert not persist_path.exists()

    def test_skips_save_when_llm_fails(self, tmp_path):
        llm = MagicMock()
        llm.generate.side_effect = Exception("boom")
        persist_path = tmp_path / "s.txt"
        s = CorpusSummarizer(llm, persist_path)
        assert s.generate_and_save([_doc("content", "a.txt")]) is None
        assert not persist_path.exists()