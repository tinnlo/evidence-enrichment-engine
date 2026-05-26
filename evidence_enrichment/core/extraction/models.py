"""Typed extraction result model.

``ExtractionResult`` is a parallel artifact that sits alongside the existing
``FactClaim`` pipeline.  ``PipelineRunResult.extraction_results`` collects
these when ``schema_validation=True`` and the field has a registered schema.

Design constraints
------------------
* This module must NOT import anything from
  ``evidence_enrichment.core.models.contracts`` — doing so would create a
  circular import because ``contracts.py`` imports this module at the bottom
  of its file to resolve the ``ExtractionResult`` forward reference.
* ``value`` round-trips through ``model_dump`` / ``model_validate`` using a
  custom serialiser that embeds ``__schema_cls__`` in the dict, and a
  ``model_validator(mode="before")`` that dispatches on that tag via
  ``SCHEMA_REGISTRY``.  This avoids a discriminated-union annotation (which
  would require listing all concrete types here) while remaining fully typed.
"""

from __future__ import annotations

import typing
import types
from typing import Any, get_args, get_origin

from pydantic import BaseModel, Field, model_serializer, model_validator


def _safe_coerce_row(row_cls: type[BaseModel], item: object) -> object:
    """Coerce ``item`` to ``row_cls`` via ``model_validate``; return ``item`` on failure."""
    if not isinstance(item, dict):
        return item
    try:
        return row_cls.model_validate(item)
    except Exception:  # noqa: BLE001
        return item


def _unwrap_model_type(annotation: Any) -> type[BaseModel] | None:
    """Return the ``BaseModel`` subclass buried in ``annotation``, or ``None``.

    Handles plain ``MyModel``, ``MyModel | None`` (PEP 604 union),
    ``Optional[MyModel]`` (``typing.Union[MyModel, None]``), and nested forms.
    """
    if annotation is None:
        return None
    # Direct subclass — most common case.
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    # Union / Optional — inspect each arg.
    # get_origin returns typing.Union for Optional[X] and types.UnionType for X|Y.
    origin = get_origin(annotation)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        for arg in get_args(annotation):
            result = _unwrap_model_type(arg)
            if result is not None:
                return result
    return None


def _coerce_nested_fields(schema_cls: type[BaseModel], raw: dict) -> dict:
    """Return a copy of *raw* with nested ``BaseModel`` fields coerced.

    For every field on *schema_cls* whose annotation resolves to a
    ``BaseModel`` subclass (direct, Optional, or list-of-model) we attempt
    ``model_validate`` on the raw value.  On failure the original raw value
    is kept so ``model_construct`` can still proceed.

    This function does **not** run any cross-field validators — it is
    intentionally used on the failure path where we want typed instances
    without triggering validators that would raise again.
    """
    coerced = dict(raw)
    for field_name_inner, field_info in schema_cls.model_fields.items():
        raw_val = coerced.get(field_name_inner)
        annotation = field_info.annotation

        # --- list[SomeModel] -------------------------------------------------
        origin = get_origin(annotation)
        if origin is list:
            args = get_args(annotation)
            if args:
                inner = _unwrap_model_type(args[0])
                if inner is not None and isinstance(raw_val, list):
                    coerced[field_name_inner] = [
                        _safe_coerce_row(inner, item) for item in raw_val
                    ]
            continue

        # --- SomeModel | None  (scalar) --------------------------------------
        inner = _unwrap_model_type(annotation)
        if inner is not None and isinstance(raw_val, dict):
            try:
                coerced[field_name_inner] = inner.model_validate(raw_val)
            except Exception:  # noqa: BLE001
                pass  # keep raw dict — model_construct will store it as-is

    return coerced


class ExtractionResult(BaseModel):
    """Typed, schema-validated extraction artifact for one pipeline field.

    Attributes
    ----------
    field_name:
        The pipeline field this extraction targets (e.g. ``"geographic_revenue"``).
    schema_cls_name:
        Name of the Pydantic model class used for validation
        (e.g. ``"GeographicRevenueExtraction"``).
    schema_version:
        Integer version of the schema, matching the ``SCHEMA_VERSION`` on the
        schema class (default 1).
    value:
        The validated extraction.  The concrete type is a Pydantic BaseModel
        subclass registered in ``SCHEMA_REGISTRY``.  Serialises to a dict that
        includes a ``__schema_cls__`` key so ``model_validate`` can reconstruct
        the correct concrete type.
    chunks_used:
        ``chunk_id`` values of the retrieval chunks that fed the extraction
        prompt.  Non-empty when retrieved via ``HierarchicalRetriever``; may be
        empty when the extraction was driven by the plain document text.
    repair_count:
        Number of LLM repair attempts that were needed after the initial
        ``ValidationError``.  0 means the first response validated cleanly.
    validation_passed:
        ``True`` when the extracted value passed all Pydantic validators
        (including cross-field validators such as row-sum checks).
        ``False`` when all repair attempts were exhausted without a valid
        response — the ``value`` will then be the best partial parse available,
        and ``validation_errors`` will list the remaining failures.
    validation_errors:
        Human-readable error messages from the last ``ValidationError``, if any.
    extraction_confidence:
        Float in ``[0.0, 1.0]`` copied from the extracted model's own
        ``extraction_confidence`` field (if present), otherwise ``0.0``.
    """

    field_name: str
    schema_cls_name: str
    schema_version: int = 1
    value: BaseModel
    chunks_used: list[str] = Field(default_factory=list)
    repair_count: int = 0
    validation_passed: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    extraction_confidence: float = 0.0

    # ── Round-trip serialisation ─────────────────────────────────────────────

    @model_serializer(mode="wrap")
    def _serialise(self, handler: Any) -> dict[str, Any]:
        """Dump the model and embed a ``__schema_cls__`` tag inside ``value``."""
        data = handler(self)
        # The default handler serialises ``value: BaseModel`` as ``{}`` because
        # bare BaseModel has no declared fields.  Replace with a proper dump of
        # the concrete instance, then inject the type tag for round-trip fidelity.
        if isinstance(self.value, BaseModel):
            value_dict = self.value.model_dump()
            value_dict["__schema_cls__"] = type(self.value).__name__
            data["value"] = value_dict
        return data

    @model_validator(mode="before")
    @classmethod
    def _deserialise_value(cls, data: Any) -> Any:
        """Reconstruct the concrete ``value`` type from ``__schema_cls__`` tag.

        When ``validation_passed`` is ``False`` the stored ``value`` dict is a
        partial/invalid payload — re-validating it against the full schema would
        raise ``ValidationError``.  In that case we use ``model_construct`` with
        per-element coercion for nested list fields so that downstream code still
        gets typed row instances, not raw dicts.
        """
        if not isinstance(data, dict):
            return data
        value = data.get("value")
        if not isinstance(value, dict) or "__schema_cls__" not in value:
            return data
        cls_name = value.pop("__schema_cls__")
        # Lazy import avoids any circular-import risk at module load time.
        from evidence_enrichment.core.extraction.schemas import SCHEMA_REGISTRY  # noqa: PLC0415

        matching = [
            schema_cls
            for schema_cls, _ver in SCHEMA_REGISTRY.values()
            if schema_cls.__name__ == cls_name
        ]
        if not matching:
            return data
        schema_cls = matching[0]
        data = dict(data)
        validation_passed = data.get("validation_passed", True)
        if validation_passed:
            # Happy path: full validation — all cross-field checks run.
            data["value"] = schema_cls.model_validate(value)
        else:
            # Failure path: partial payload — skip cross-field validators.
            # Coerce nested list and scalar BaseModel fields so downstream code
            # gets typed instances, not raw dicts.
            from evidence_enrichment.core.extraction.schemas import SCHEMA_REGISTRY  # noqa: PLC0415 (already imported above)

            coerced = _coerce_nested_fields(schema_cls, value)
            try:
                data["value"] = schema_cls.model_construct(**coerced)
            except Exception:  # noqa: BLE001
                data["value"] = schema_cls.model_construct()
        return data
