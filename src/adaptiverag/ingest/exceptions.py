class UnsupportedFileType(Exception):
    """Raised when a file's type has no registered extractor."""

    def __init__(self, filepath: str, extension: str):
        self.filepath = filepath
        self.extension = extension
        super().__init__(
            f"No extractor registered for '{extension}' files: {filepath}"
        )