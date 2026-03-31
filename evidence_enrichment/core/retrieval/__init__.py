"""Core retrieval subpackage."""

from evidence_enrichment.core.retrieval.chunker import TableAwareChunker
from evidence_enrichment.core.retrieval.embedder import OpenAIEmbedder
from evidence_enrichment.core.retrieval.models import Chunk, RetrievalResult
from evidence_enrichment.core.retrieval.retriever import HybridRetriever
from evidence_enrichment.core.retrieval.store import ChromaVectorStore

__all__ = [
    "Chunk",
    "RetrievalResult",
    "TableAwareChunker",
    "OpenAIEmbedder",
    "ChromaVectorStore",
    "HybridRetriever",
]
