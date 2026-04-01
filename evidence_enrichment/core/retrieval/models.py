"""Retrieval data models."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel


class Chunk(BaseModel):
    """A chunk of text extracted from a parsed document."""

    chunk_id: str  # sha256(url + str(index) + content_hash)[:16]
    document_url: str
    content_hash: str
    index: int
    content: str
    chunk_type: str = "text"  # "text" | "table"
    char_count: int = 0

    def model_post_init(self, __context: object) -> None:
        if not self.char_count:
            self.char_count = len(self.content)

    @staticmethod
    def make_id(document_url: str, index: int, content_hash: str) -> str:
        """Generate a stable, deterministic chunk ID."""
        raw = f"{document_url}|{index}|{content_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class RetrievalResult(BaseModel):
    """A single retrieval hit with hybrid scoring metadata."""

    chunk: Chunk
    score: float  # final hybrid score
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rank: int = 0
    is_table_boost: bool = False
    document_url: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.document_url:
            self.document_url = self.chunk.document_url
