"""OpenAI text embedder.

Uses ``text-embedding-3-small`` by default (1536 dims).  The OpenAI client is
imported lazily so the module can be imported without the ``openai`` package
installed (i.e., in test environments that use fake embeddings).
"""

from __future__ import annotations

import os


class OpenAIEmbedder:
    """Batch-aware OpenAI embedder.

    Parameters
    ----------
    model:
        OpenAI embedding model name (default ``text-embedding-3-small``).
    batch_size:
        Maximum texts per API call (max 100 per OpenAI limits).
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = min(batch_size, 100)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client: object | None = None  # lazy

    @property
    def dimensions(self) -> int:
        """Return expected embedding dimensions for the configured model."""
        _dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return _dims.get(self.model, 1536)

    def _get_client(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for embeddings. "
                    "Install with: pip install 'evidence_enrichment[retrieval]'"
                ) from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching as needed."""
        if not texts:
            return []
        client = self._get_client()
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = client.embeddings.create(model=self.model, input=batch)  # type: ignore[union-attr]
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = self.embed_texts([text])
        return results[0] if results else []
