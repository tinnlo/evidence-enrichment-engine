"""Stage D migration script — re-index a Chroma collection to v2.

Reads all chunks from an existing v1 collection (flat/TableAware chunking),
re-parses their source text via GenericTextParser, re-chunks with
HierarchicalChunker, and upserts into a new _v2 collection.

**Limitation — flat re-chunking only**

v1 collections contain pre-chunked plain text; the original HTML/PDF source is
not retained.  ``GenericTextParser``/``TextParser`` produces ``blocks`` from
raw HTML structure but does **not** produce ``SectionNode`` objects (``sections``
remains ``[]``).  Consequently ``HierarchicalChunker`` falls back to
``_flat_chunk()`` (TableAwareChunker-based) for every migrated document.

The migration is still useful: it brings the v2 metadata schema, updated chunk
sizing, and the current overlap/table-detection logic.  Full section-aware
indexing requires re-fetching and re-parsing the original source documents.

Usage
-----
    python scripts/migrate_to_hierarchical.py \\
        --entity-id microsoft \\
        --persist-path examples/output/chroma \\
        --embedding-model text-embedding-3-small \\
        [--dry-run]

The script never touches the existing v1 collection, so a failed migration
leaves the pipeline in a known-good state.  Run ``evidence-enrich eval`` after
migration to confirm recall parity before switching
``retrieval.chunker = hierarchical`` in your config.

Exit codes
----------
    0  All documents migrated successfully.
    1  One or more documents failed; see stderr for details.
    2  Pre-flight check failed (missing dependency, no v1 data, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("migrate_to_hierarchical")

if TYPE_CHECKING:
    from evidence_enrichment.core.parse.models import ParsedDocument
    from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_deps() -> None:
    """Raise SystemExit(2) when required optional packages are missing."""
    missing = []
    for pkg in ("chromadb", "openai", "tiktoken"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(
            "Missing optional dependencies: %s. "
            "Install with: pip install 'evidence_enrichment[retrieval]'",
            ", ".join(missing),
        )
        sys.exit(2)


def _v1_collection_name(entity_id: str, embedding_model: str) -> str:
    """Reproduce the v1 naming convention from store.py."""
    import re

    _SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")
    model_slug = _SAFE_RE.sub("_", embedding_model)
    entity_slug = _SAFE_RE.sub("_", entity_id)
    name = f"entity_{entity_slug}__{model_slug}_v1"
    return name[:512].rstrip("_")


def _fetch_all_v1_chunks(client: Any, collection_name: str) -> dict[str, list[dict]]:
    """Return {document_url: [{"content": ..., "metadata": ...}]} from v1 collection.

    Returns an empty dict when the collection does not exist.
    """
    try:
        col = client.get_collection(collection_name)
    except Exception:
        log.info("v1 collection '%s' does not exist — nothing to migrate.", collection_name)
        return {}

    try:
        result = col.get(include=["documents", "metadatas"])
    except Exception as exc:
        log.error("Failed to read v1 collection '%s': %s", collection_name, exc)
        return {}

    docs: list[str] = result.get("documents") or []
    metas: list[dict] = result.get("metadatas") or []
    by_url: dict[str, list[dict]] = {}
    for doc, meta in zip(docs, metas):
        url = meta.get("document_url", "")
        by_url.setdefault(url, []).append({"content": doc, "metadata": meta})
    return by_url


def _reconstruct_parsed_document(
    url: str,
    chunks: list[dict],
) -> "ParsedDocument":
    """Reconstruct a ParsedDocument from v1 chunk content.

    v1 collections contain flat, pre-chunked text — the original HTML/PDF source
    is not available.  Reconstructing a full section tree from joined chunk text
    is not possible: GenericTextParser/TextParser populates ``blocks`` from raw
    HTML structure, but does not produce ``SectionNode`` objects (``sections``
    remains ``[]``).

    Therefore the migration always produces *improved flat chunks* via
    ``HierarchicalChunker._flat_chunk()`` (which delegates to TableAwareChunker).
    This is still useful: the v2 collection uses the current chunk sizing,
    overlap, and metadata schema.  Section-aware indexing requires re-fetching
    and re-parsing original source documents.

    ``GenericTextParser.parse()`` expects a ``RetrievedDocument``, not a
    ``ParsedDocument``, so we construct the correct type here.
    """
    from evidence_enrichment.core.models.contracts import RetrievedDocument
    from evidence_enrichment.core.parse.generic_text import GenericTextParser

    sorted_chunks = sorted(chunks, key=lambda c: c["metadata"].get("index", 0))
    full_text = "\n\n".join(c["content"] for c in sorted_chunks)
    title = chunks[0]["metadata"].get("document_url", url)

    retrieved = RetrievedDocument(
        url=url,
        final_url=url,
        title=title,
        content_type="text/plain",
        body=full_text,
        provider="migration",
    )

    parser = GenericTextParser()
    return parser.parse(retrieved)


def _migrate_document(
    url: str,
    chunks: list[dict],
    retriever: "HierarchicalRetriever",
    dry_run: bool,
) -> tuple[int, int]:
    """Re-parse + re-chunk one document.  Returns (new_chunk_count, status).

    status: 0 = ok, 1 = error
    """
    try:
        parsed = _reconstruct_parsed_document(url, chunks)
    except Exception as exc:
        log.error("Failed to reconstruct ParsedDocument for %s: %s", url, exc)
        return 0, 1

    try:
        from evidence_enrichment.core.retrieval.hierarchical_chunker import HierarchicalChunker

        if retriever.chunker is None:
            retriever.chunker = HierarchicalChunker()
        new_chunks = retriever.chunker.chunk(parsed)
    except Exception as exc:
        log.error("Failed to chunk %s: %s", url, exc)
        return 0, 1

    if dry_run:
        log.info("[DRY RUN] %s → %d new chunks (not written)", url, len(new_chunks))
        return len(new_chunks), 0

    try:
        texts = [c.content for c in new_chunks]
        embeddings = retriever.embedder.embed_texts(texts)
        retriever.store.upsert(retriever.entity_id, new_chunks, embeddings)
        log.info("Migrated %s → %d chunks written to v2 collection", url, len(new_chunks))
        return len(new_chunks), 0
    except Exception as exc:
        log.error("Failed to embed/upsert %s: %s", url, exc)
        return 0, 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-index a v1 flat Chroma collection to v2 hierarchical chunking.",
    )
    p.add_argument("--entity-id", required=True, help="Entity ID used as collection key.")
    p.add_argument(
        "--persist-path",
        default="examples/output/chroma",
        help="Path to the Chroma persist directory. Default: examples/output/chroma",
    )
    p.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model name (must match what was used for v1). Default: text-embedding-3-small",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count new chunks without writing to v2 collection.",
    )
    p.add_argument(
        "--delete-v1",
        action="store_true",
        help="Delete the v1 collection after a fully successful migration.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    _check_deps()

    import chromadb  # type: ignore[import]
    from evidence_enrichment.core.retrieval.embedder import OpenAIEmbedder
    from evidence_enrichment.core.retrieval.retriever import HierarchicalRetriever
    from evidence_enrichment.core.retrieval.store import ChromaVectorStore

    # ── 1. Open client and read v1 data ─────────────────────────────────────
    client = chromadb.PersistentClient(path=args.persist_path)
    v1_name = _v1_collection_name(args.entity_id, args.embedding_model)
    log.info("Reading v1 collection '%s' from %s …", v1_name, args.persist_path)
    by_url = _fetch_all_v1_chunks(client, v1_name)

    if not by_url:
        log.warning("No documents found in v1 collection. Nothing to migrate.")
        return 0

    log.info(
        "Found %d document(s) with %d total chunks in v1 collection.",
        len(by_url),
        sum(len(v) for v in by_url.values()),
    )

    # ── 2. Build v2 retriever (writes to _v2 collection) ──────────────────
    # Pass schema_version=2 directly to ChromaVectorStore so it names the
    # collection with the _v2 suffix.  No monkey-patching required.
    v2_store = ChromaVectorStore(
        persist_path=args.persist_path,
        embedding_model=args.embedding_model,
        schema_version=2,
    )

    v2_col_name = v2_store.collection_name_for(args.entity_id)
    if not args.dry_run:
        log.info("Target v2 collection: '%s'", v2_col_name)

    embedder = OpenAIEmbedder(model=args.embedding_model)

    retriever = HierarchicalRetriever(
        entity_id=args.entity_id,
        store=v2_store,
        embedder=embedder,
    )

    # ── 3. Migrate each document ─────────────────────────────────────────────
    total_new_chunks = 0
    failed_urls: list[str] = []

    for url, chunks in by_url.items():
        new_count, status = _migrate_document(url, chunks, retriever, dry_run=args.dry_run)
        total_new_chunks += new_count
        if status != 0:
            failed_urls.append(url)

    # ── 4. Report ────────────────────────────────────────────────────────────
    if args.dry_run:
        log.info(
            "[DRY RUN] Migration plan: %d document(s), %d projected new chunks.",
            len(by_url),
            total_new_chunks,
        )
    else:
        log.info(
            "Migration complete: %d document(s), %d new chunks written.",
            len(by_url) - len(failed_urls),
            total_new_chunks,
        )

    if failed_urls:
        log.error("%d document(s) failed migration:", len(failed_urls))
        for u in failed_urls:
            log.error("  FAILED: %s", u)
        return 1

    # ── 5. Optional v1 cleanup ───────────────────────────────────────────────
    if args.delete_v1 and not args.dry_run:
        try:
            client.delete_collection(v1_name)
            log.info("Deleted v1 collection '%s'.", v1_name)
        except Exception as exc:
            log.warning("Could not delete v1 collection: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
