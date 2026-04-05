from dataclasses import dataclass, field


@dataclass
class Document:
    """A single unit of ingested content."""
    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata).__name__}")