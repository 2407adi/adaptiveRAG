class UnsupportedFileType(Exception):
    """Raised when a file's type has no registered extractor."""

    def __init__(self, filepath: str, extension: str):
        self.filepath = filepath
        self.extension = extension
        super().__init__(
            f"No extractor registered for '{extension}' files: {filepath}"
        )


class IngestTooLarge(Exception):
    """Raised when an upload would produce more chunks than the per-upload cap.

    Caught by the ingest job runner and surfaced to the user as a friendly
    'document too large' message — BEFORE any embedding compute is spent."""

    def __init__(self, total_chunks: int, max_chunks: int):
        self.total_chunks = total_chunks
        self.max_chunks = max_chunks
        super().__init__(
            f"upload would produce {total_chunks} chunks, over the "
            f"{max_chunks}-chunk limit — try uploading a smaller document "
            f"or splitting it into parts"
        )