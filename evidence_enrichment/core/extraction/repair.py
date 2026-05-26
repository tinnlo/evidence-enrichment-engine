"""LLM repair helper for failed schema validation.

``SchemaRepairHelper`` wraps a provider callable and asks the LLM to fix a
ValidationError-rejected JSON response.  It is intentionally thin — the
caller (``SchemaExtractor``) owns the retry loop and the final decision about
whether to surface a failed extraction.

Design notes
------------
* The repair prompt embeds the original raw JSON, the validation error
  messages, and the schema field descriptions so the LLM has all the context
  it needs without an additional retrieval round-trip.
* The helper never touches ``FactClaim`` or the main pipeline — it is only
  called when ``schema_validation=True`` and a ``ValidationError`` is raised.
* ``_build_repair_prompt`` is a pure function so it can be tested without a
  live LLM.
* The provider callable may be sync or async; ``attempt_repair`` is async
  and handles both.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Awaitable, Callable, Union

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ProviderFn = Union[Callable[[str], str], Callable[[str], Awaitable[str]]]

# Maximum characters of the raw LLM response included in the repair prompt.
# Long responses are truncated to avoid blowing the context window on repair.
_MAX_RAW_JSON_CHARS = 4_000


def _build_repair_prompt(
    field_name: str,
    schema_cls: type[BaseModel],
    raw_json: str,
    validation_errors: list[str],
) -> str:
    """Build a repair prompt that asks the LLM to fix a bad JSON response.

    Parameters
    ----------
    field_name:
        The pipeline field being extracted (used for framing the task).
    schema_cls:
        The Pydantic class the response must conform to.
    raw_json:
        The original LLM response that failed validation.
    validation_errors:
        Human-readable error messages from the ``ValidationError``.

    Returns
    -------
    str
        A ready-to-send prompt string.
    """
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    truncated_raw = raw_json[:_MAX_RAW_JSON_CHARS]
    if len(raw_json) > _MAX_RAW_JSON_CHARS:
        truncated_raw += f"\n... [truncated — original was {len(raw_json)} chars]"

    errors_block = "\n".join(f"  - {e}" for e in validation_errors)

    return (
        f"You previously attempted to extract structured data for the field "
        f'"{field_name}" but the response failed JSON schema validation.\n\n'
        f"VALIDATION ERRORS:\n{errors_block}\n\n"
        f"YOUR PREVIOUS RESPONSE (that failed):\n```json\n{truncated_raw}\n```\n\n"
        f"TARGET JSON SCHEMA:\n```json\n{schema_json}\n```\n\n"
        f"Please return ONLY a corrected JSON object that strictly conforms to "
        f"the schema above and fixes all listed validation errors. "
        f"Do not include any explanation or markdown fencing — output raw JSON only."
    )


class SchemaRepairHelper:
    """Asks a provider LLM to fix a ValidationError-rejected JSON extraction.

    Parameters
    ----------
    provider_fn:
        A callable ``(prompt: str) -> str`` (sync or async) that sends the
        repair prompt to an LLM and returns the raw text response.
    """

    def __init__(self, provider_fn: ProviderFn) -> None:
        self._provider_fn = provider_fn

    async def attempt_repair(
        self,
        field_name: str,
        schema_cls: type[BaseModel],
        raw_json: str,
        validation_errors: list[str],
    ) -> tuple[Any, list[str]]:
        """Send a repair prompt and return the parsed JSON dict (or None on failure).

        Returns
        -------
        tuple[dict | None, list[str]]
            ``(parsed_dict_or_none, remaining_error_messages)``
            ``parsed_dict_or_none`` is ``None`` when the LLM response could not
            be parsed as JSON (i.e. a malformed response, not a ValidationError).
            ``remaining_error_messages`` is empty on a clean parse.
        """
        prompt = _build_repair_prompt(field_name, schema_cls, raw_json, validation_errors)
        try:
            raw = self._provider_fn(prompt)
            response = await raw if inspect.isawaitable(raw) else raw
        except Exception as exc:  # noqa: BLE001
            logger.warning("SchemaRepairHelper: provider call failed: %s", exc)
            return None, [f"provider error: {exc}"]

        # Strip optional markdown fencing the LLM might add despite instructions.
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Drop opening ``` line and closing ``` line.
            inner = [
                ln for ln in lines[1:]
                if not ln.strip().startswith("```")
            ]
            cleaned = "\n".join(inner).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("SchemaRepairHelper: repair response is not valid JSON: %s", exc)
            return None, [f"json_parse_error: {exc}"]

        # Validate against the schema here to surface remaining errors to the
        # caller; the caller decides whether to retry or surface a failure.
        try:
            schema_cls.model_validate(parsed)
            return parsed, []
        except ValidationError as exc:
            errors = [str(e["msg"]) for e in exc.errors()]
            return parsed, errors
