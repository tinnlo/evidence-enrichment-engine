"""Tests for ChromaVectorStore using EphemeralClient (no disk I/O)."""

from __future__ import annotations

import pytest

from evidence_enrichment.core.retrieval.models import Chunk
from evidence_enrichment.core.retrieval.store import ChromaVectorStore, _collection_name


# ---------------------------------------------------------------------------
# Fake embeddings — 8-dimensional unit vectors, no API calls
# ---------------------------------------------------------------------------

def _fake_embed(n: int) -> list[list[float]]:
    """Return n distinct 8-dim float lists."""
    vecs = []
    for i in range(n):
        vec = [0.0] * 8
        vec[i % 8] = 1.0
        vecs.append(vec)
    return vecs


def _make_chunks(n: int, document_url: str = "https://example.com/doc") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"chunk{i:04d}",
            document_url=document_url,
            content_hash="hash_abc",
            index=i,
            content=f"This is chunk number {i} with some meaningful content.",
            chunk_type="text" if i % 2 == 0 else "table",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Ephemeral store fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    """ChromaVectorStore backed by an in-memory EphemeralClient via tmp_path."""
    import chromadb

    s = ChromaVectorStore(persist_path=str(tmp_path), embedding_model="text-embedding-3-small")
    # Monkey-patch to use ephemeral client so tests don't write to disk
    s._client = chromadb.EphemeralClient()
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_upsert_and_query(self, store):
        """Chunks upserted can be retrieved by query."""
        chunks = _make_chunks(3)
        embeddings = _fake_embed(3)
        store.upsert("ent_001", chunks, embeddings)

        query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # closest to chunk0
        results = store.query("ent_001", query_vec, top_k=3)
        assert len(results) >= 1
        returned_ids = [r.chunk.chunk_id for r in results]
        assert "chunk0000" in returned_ids  # chunk0 should rank highest

    def test_empty_upsert_no_error(self, store):
        """Upserting empty list is a no-op."""
        store.upsert("ent_002", [], [])  # should not raise

    def test_query_empty_collection_returns_empty(self, store):
        """Querying a collection with no data returns empty list."""
        query_vec = [1.0] + [0.0] * 7
        results = store.query("ent_empty", query_vec, top_k=5)
        assert results == []


class TestIdempotentUpsert:
    def test_upsert_same_id_overwrites(self, store):
        """Re-upserting a chunk with the same ID replaces it."""
        chunk = Chunk(
            chunk_id="stable_id",
            document_url="https://example.com",
            content_hash="h1",
            index=0,
            content="original content",
        )
        store.upsert("ent_003", [chunk], [_fake_embed(1)[0]])

        updated_chunk = chunk.model_copy(update={"content": "updated content"})
        store.upsert("ent_003", [updated_chunk], [_fake_embed(1)[0]])

        # Should have exactly 1 result, not 2
        results = store.query("ent_003", _fake_embed(1)[0], top_k=10)
        assert len(results) == 1
        assert results[0].chunk.content == "updated content"


class TestEntityIsolation:
    def test_separate_entities_isolated(self, store):
        """Chunks from different entities do not appear in each other's results."""
        # Use different document URLs so we can distinguish results by URL
        chunks_a = _make_chunks(2, document_url="https://alpha.com/doc")
        # Give entity_beta different chunk IDs by building manually
        chunks_b = [
            Chunk(
                chunk_id=f"beta_{i:04d}",
                document_url="https://beta.com/doc",
                content_hash="hash_beta",
                index=i,
                content=f"Beta chunk {i} content goes here.",
                chunk_type="text",
            )
            for i in range(2)
        ]
        embs = _fake_embed(2)

        store.upsert("entity_alpha", chunks_a, embs)
        store.upsert("entity_beta", chunks_b, embs)

        query_vec = [1.0] + [0.0] * 7
        results_a = store.query("entity_alpha", query_vec, top_k=10)
        results_b = store.query("entity_beta", query_vec, top_k=10)

        urls_a = {r.chunk.document_url for r in results_a}
        urls_b = {r.chunk.document_url for r in results_b}
        assert urls_a == {"https://alpha.com/doc"}
        assert urls_b == {"https://beta.com/doc"}
        assert urls_a.isdisjoint(urls_b), "Entity collections should not share document URLs"


class TestMetadataFiltering:
    def test_document_url_filter(self, store):
        """Querying with document_url where-filter returns only matching chunks."""
        chunks_doc1 = _make_chunks(2, document_url="https://example.com/doc1")
        # Build doc2 chunks with distinct IDs to avoid Chroma duplicate-ID rejection
        chunks_doc2 = [
            Chunk(
                chunk_id=f"doc2_{i:04d}",
                document_url="https://example.com/doc2",
                content_hash="hash_doc2",
                index=i,
                content=f"Doc2 chunk {i} with some content here.",
                chunk_type="text",
            )
            for i in range(2)
        ]
        all_chunks = chunks_doc1 + chunks_doc2
        all_embs = _fake_embed(4)

        store.upsert("ent_filter", all_chunks, all_embs)

        query_vec = [1.0] + [0.0] * 7
        results = store.query(
            "ent_filter",
            query_vec,
            top_k=10,
            where={"document_url": "https://example.com/doc1"},
        )
        returned_urls = {r.chunk.document_url for r in results}
        assert returned_urls == {"https://example.com/doc1"}

    def test_chunk_type_in_metadata(self, store):
        """chunk_type is preserved in metadata and available on returned chunks."""
        table_chunk = Chunk(
            chunk_id="table_01",
            document_url="https://example.com",
            content_hash="h1",
            index=0,
            content="Revenue Q1: $1.2B | Q2: $1.4B | Q3: $1.5B",
            chunk_type="table",
        )
        text_chunk = Chunk(
            chunk_id="text_01",
            document_url="https://example.com",
            content_hash="h1",
            index=1,
            content="The company was founded in 1990 in California.",
            chunk_type="text",
        )
        store.upsert("ent_types", [table_chunk, text_chunk], _fake_embed(2))

        results = store.query("ent_types", [1.0] + [0.0] * 7, top_k=10)
        type_map = {r.chunk.chunk_id: r.chunk.chunk_type for r in results}
        assert type_map.get("table_01") == "table"
        assert type_map.get("text_01") == "text"


class TestCollectionNaming:
    def test_collection_name_format(self):
        """Collection name matches the expected pattern."""
        name = _collection_name("entity_001", "text-embedding-3-small")
        assert name.startswith("entity_entity_001__")
        assert "text_embedding_3_small" in name
        assert name.endswith("_v1")

    def test_special_chars_sanitized(self):
        """Non-alphanumeric chars in entity_id are replaced by underscores."""
        name = _collection_name("acme corp/uk", "text-embedding-3-small")
        assert "/" not in name
        assert " " not in name

    def test_name_min_length(self):
        """Collection name is always at least 3 chars (Chroma minimum)."""
        name = _collection_name("x", "m")
        assert len(name) >= 3
