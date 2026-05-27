"""Schema-driven typed extraction orchestrator.

``SchemaExtractor`` is called from the coordinator *after* the existing
analysis step (not instead of it) when ``schema_validation=True`` and the
target field has a registered schema.  It:

1. Builds a retrieval-grounded extraction prompt from the chunk context.
2. Sends the prompt to the provider LLM.
3. Validates the JSON response against the registered Pydantic schema.
4. Retries up to ``max_repair_attempts`` times using ``SchemaRepairHelper``
   when the initial response fails validation.
5. Returns an ``ExtractionResult`` regardless of whether validation passed —
   callers can inspect ``validation_passed`` and ``validation_errors``.

Design constraints
------------------
* Never imports from ``contracts.py`` (circular import risk via model_rebuild).
* Only instantiated when ``schema_validation=True`` — zero cost at import time
  for the default pipeline.
* The provider callable signature is either sync ``(prompt: str) -> str`` or
  async ``async (prompt: str) -> str``.  ``SchemaExtractor.extract()`` is
  async and must be awaited.  The repair helper is also async when
  ``async_provider_fn`` is provided.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Awaitable, Callable, Union

from pydantic import BaseModel, ValidationError

from evidence_enrichment.core.extraction.models import ExtractionResult, _coerce_nested_fields
from evidence_enrichment.core.extraction.repair import SchemaRepairHelper
from evidence_enrichment.core.extraction.schemas import SCHEMA_REGISTRY

ProviderFn = Union[Callable[[str], str], Callable[[str], Awaitable[str]]]

logger = logging.getLogger(__name__)


async def _call_provider(fn: ProviderFn, prompt: str) -> str:
    """Call a provider function that may be sync or async."""
    result = fn(prompt)
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


# Maximum number of content chunks included in the extraction prompt.
_MAX_PROMPT_CHUNKS = 12
# Characters of each chunk included in the prompt.
_MAX_CHUNK_CHARS = 1_500


def _build_extraction_prompt(
    field_name: str,
    schema_cls: type[BaseModel],
    chunks_text: str,
) -> str:
    """Build the initial extraction prompt.

    Parameters
    ----------
    field_name:
        Pipeline field being extracted.
    schema_cls:
        Pydantic class the response must conform to.
    chunks_text:
        Pre-formatted document chunks (one per line with a chunk-id header).

    Returns
    -------
    str
        Ready-to-send prompt.
    """
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    return (
        f"You are a structured data extraction assistant.\n\n"
        f"Extract the field \"{field_name}\" from the document chunks below.\n\n"
        f"TARGET JSON SCHEMA:\n```json\n{schema_json}\n```\n\n"
        f"DOCUMENT CHUNKS:\n{chunks_text}\n\n"
        f"Instructions:\n"
        f"- Return ONLY a valid JSON object that conforms to the schema above.\n"
        f"- Populate ``source_chunk_ids`` on every row with the chunk IDs "
        f"  listed in the chunk headers above.\n"
        f"- Set ``extraction_confidence`` to a float in [0.0, 1.0] reflecting "
        f"  how complete and unambiguous the evidence is.\n"
        f"- Do NOT include any explanation or markdown fencing — output raw JSON only."
    )


def _format_chunks(chunk_context: list[tuple[str, str]]) -> str:
    """Format (chunk_id, text) pairs for the prompt.

    Parameters
    ----------
    chunk_context:
        List of ``(chunk_id, chunk_text)`` tuples.

    Returns
    -------
    str
        Multi-line string with one block per chunk.
    """
    parts: list[str] = []
    for chunk_id, text in chunk_context[:_MAX_PROMPT_CHUNKS]:
        truncated = text[:_MAX_CHUNK_CHARS]
        if len(text) > _MAX_CHUNK_CHARS:
            truncated += " [...]"
        parts.append(f"[chunk_id={chunk_id}]\n{truncated}")
    return "\n\n---\n\n".join(parts)


def _safe_row_validate(row_cls: type[BaseModel], item: object) -> object:
    """Attempt ``row_cls.model_validate(item)``; return ``item`` unchanged on failure.

    Used inside ``_failed_result`` to coerce nested list elements to typed row
    instances without aborting when a row itself is invalid (e.g. empty
    ``source_chunk_ids``).
    """
    if not isinstance(item, dict):
        return item
    try:
        return row_cls.model_validate(item)
    except Exception:  # noqa: BLE001
        return item


class SchemaExtractor:
    """Orchestrates typed extraction for one pipeline field.

    Parameters
    ----------
    provider_fn:
        Callable ``(prompt: str) -> str`` or async variant that sends a
        prompt to an LLM and returns the raw text response.
    max_repair_attempts:
        How many times to ask the LLM to fix a ValidationError before giving
        up and returning a failed ``ExtractionResult``.  0 means attempt once
        with no repair retries.
    """

    def __init__(
        self,
        provider_fn: ProviderFn,
        max_repair_attempts: int = 2,
    ) -> None:
        self._provider_fn = provider_fn
        self._max_repair_attempts = max_repair_attempts
        self._repair_helper = SchemaRepairHelper(provider_fn)

    async def extract(
        self,
        field_name: str,
        chunk_context: list[tuple[str, str]],
    ) -> ExtractionResult | None:
        """Run typed extraction for ``field_name``.

        Parameters
        ----------
        field_name:
            The pipeline field to extract.  Must be present in
            ``SCHEMA_REGISTRY``; returns ``None`` if not registered.
        chunk_context:
            List of ``(chunk_id, chunk_text)`` tuples.  An empty list is
            valid — the LLM will see no document chunks and will likely return
            a low-confidence extraction or fail validation.

        Returns
        -------
        ExtractionResult | None
            ``None`` when ``field_name`` is not in ``SCHEMA_REGISTRY``.
            Otherwise always returns an ``ExtractionResult`` — callers must
            inspect ``validation_passed`` to determine whether the result is
            usable.
        """
        if field_name not in SCHEMA_REGISTRY:
            logger.debug(
                "SchemaExtractor: no schema registered for field %r — skipping",
                field_name,
            )
            return None

        schema_cls, schema_version = SCHEMA_REGISTRY[field_name]
        chunk_ids = [cid for cid, _ in chunk_context]
        chunks_text = _format_chunks(chunk_context)
        prompt = _build_extraction_prompt(field_name, schema_cls, chunks_text)

        # ── Initial attempt ──────────────────────────────────────────────────
        try:
            raw_response = await _call_provider(self._provider_fn, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SchemaExtractor: provider call failed for field %r: %s",
                field_name,
                exc,
            )
            return self._failed_result(
                field_name=field_name,
                schema_cls=schema_cls,
                schema_version=schema_version,
                chunk_ids=chunk_ids,
                errors=[f"provider error: {exc}"],
                repair_count=0,
            )

        raw_json = raw_response.strip()
        # Strip markdown fencing if present.
        if raw_json.startswith("```"):
            lines = raw_json.splitlines()
            raw_json = "\n".join(
                ln for ln in lines[1:] if not ln.strip().startswith("```")
            ).strip()

        parsed_dict, validation_errors = self._parse_and_validate(
            schema_cls, raw_json
        )

        if not validation_errors and parsed_dict is not None:
            return self._success_result(
                field_name=field_name,
                schema_cls=schema_cls,
                schema_version=schema_version,
                chunk_ids=chunk_ids,
                parsed_dict=parsed_dict,
                repair_count=0,
            )

        # ── Repair loop ──────────────────────────────────────────────────────
        repair_count = 0
        current_raw = raw_json
        current_errors = validation_errors
        current_dict = parsed_dict

        for attempt in range(self._max_repair_attempts):
            repair_count += 1
            logger.debug(
                "SchemaExtractor: repair attempt %d/%d for field %r",
                attempt + 1,
                self._max_repair_attempts,
                field_name,
            )
            repaired_dict, repaired_errors = await self._repair_helper.attempt_repair(
                field_name=field_name,
                schema_cls=schema_cls,
                raw_json=current_raw,
                validation_errors=current_errors,
            )
            if not repaired_errors and repaired_dict is not None:
                return self._success_result(
                    field_name=field_name,
                    schema_cls=schema_cls,
                    schema_version=schema_version,
                    chunk_ids=chunk_ids,
                    parsed_dict=repaired_dict,
                    repair_count=repair_count,
                )
            # Keep the best partial parse for the next attempt.
            if repaired_dict is not None:
                current_dict = repaired_dict
            current_errors = repaired_errors or current_errors

        # All attempts exhausted — surface a failed result.
        return self._failed_result(
            field_name=field_name,
            schema_cls=schema_cls,
            schema_version=schema_version,
            chunk_ids=chunk_ids,
            errors=current_errors,
            repair_count=repair_count,
            partial_dict=current_dict,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _parse_and_validate(
        self,
        schema_cls: type[BaseModel],
        raw_json: str,
    ) -> tuple[dict | None, list[str]]:
        """Parse ``raw_json`` and validate against ``schema_cls``.

        Returns ``(dict_or_none, error_messages)``.
        """
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            return None, [f"json_parse_error: {exc}"]
        try:
            schema_cls.model_validate(parsed)
            return parsed, []
        except ValidationError as exc:
            errors = [
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                if e.get("loc")
                else str(e["msg"])
                for e in exc.errors()
            ]
            return parsed, errors

    def _success_result(
        self,
        *,
        field_name: str,
        schema_cls: type[BaseModel],
        schema_version: int,
        chunk_ids: list[str],
        parsed_dict: dict,
        repair_count: int,
    ) -> ExtractionResult:
        validated = schema_cls.model_validate(parsed_dict)
        confidence = float(getattr(validated, "extraction_confidence", 0.0))
        return ExtractionResult(
            field_name=field_name,
            schema_cls_name=schema_cls.__name__,
            schema_version=schema_version,
            value=validated,
            chunks_used=chunk_ids,
            repair_count=repair_count,
            validation_passed=True,
            validation_errors=[],
            extraction_confidence=confidence,
        )

    def _failed_result(
        self,
        *,
        field_name: str,
        schema_cls: type[BaseModel],
        schema_version: int,
        chunk_ids: list[str],
        errors: list[str],
        repair_count: int,
        partial_dict: dict | None = None,
    ) -> ExtractionResult:
        """Return a failed ExtractionResult with the best partial value available.

        Attempts three strategies in order, stopping at the first that succeeds:

        1. ``model_validate`` — works when the only failures are cross-field
           validators (row-sum, pct-sum, headcount-total) because those run after
           field-level coercion.  This gives fully-typed nested rows.
        2. ``model_construct`` with individually coerced nested rows — used when
           ``model_validate`` fails with a ``ValidationError`` (the cross-field
           path).  We coerce each element of list fields against any registered
           row-class so nested items are typed rather than raw dicts.
        3. Bare ``model_construct`` — last resort when the partial dict is
           structurally broken (missing required keys, unexpected types).
        """
        if partial_dict is None:
            value: BaseModel = schema_cls.model_construct()
            confidence = float(getattr(value, "extraction_confidence", 0.0))
            return ExtractionResult(
                field_name=field_name,
                schema_cls_name=schema_cls.__name__,
                schema_version=schema_version,
                value=value,
                chunks_used=chunk_ids,
                repair_count=repair_count,
                validation_passed=False,
                validation_errors=errors,
                extraction_confidence=0.0,
            )

        try:
            value = schema_cls.model_validate(partial_dict)
        except ValidationError:
            # Cross-field validators failed.  Coerce nested list and scalar
            # BaseModel fields so downstream code gets typed instances, not
            # raw dicts.
            coerced = _coerce_nested_fields(schema_cls, partial_dict)
            try:
                value = schema_cls.model_construct(**coerced)
            except Exception:  # noqa: BLE001
                value = schema_cls.model_construct()
        except Exception:  # noqa: BLE001  — unexpected (not ValidationError)
            try:
                value = schema_cls.model_construct(**partial_dict)
            except Exception:  # noqa: BLE001
                value = schema_cls.model_construct()
        confidence = float(getattr(value, "extraction_confidence", 0.0))
        return ExtractionResult(
            field_name=field_name,
            schema_cls_name=schema_cls.__name__,
            schema_version=schema_version,
            value=value,
            chunks_used=chunk_ids,
            repair_count=repair_count,
            validation_passed=False,
            validation_errors=errors,
            extraction_confidence=confidence,
        )
