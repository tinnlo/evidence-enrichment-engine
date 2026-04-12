"""PII detection check using presidio-analyzer with regex fallback."""

from __future__ import annotations

import logging
import re

from evidence_enrichment.guardrails.models import CheckResult

logger = logging.getLogger(__name__)

# Regex fallback patterns — deterministic, no NLP models required
_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # IBAN: starts with 2 alpha, 2 digits, then 11-30 alphanumeric
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    # UK NIN: e.g. AB 12 34 56 C  (with optional spaces)
    (
        "UK_NIN",
        re.compile(
            r"\b[A-CEGHJ-PR-TW-Z]{1}[A-CEGHJ-NPR-TW-Z]{1}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"
        ),
    ),
]


def _regex_check(texts: list[str]) -> tuple[bool, int]:
    """Return (found_pii, match_count) using regex patterns."""
    count = 0
    for text in texts:
        for _, pattern in _PATTERNS:
            count += len(pattern.findall(text))
    return count > 0, count


def check_pii(
    texts: list[str],
    *,
    entities: list[str] | None = None,
) -> CheckResult:
    """Check for PII in the given texts.

    Tries presidio-analyzer first; falls back to regex on ImportError,
    AnalyzerEngine init failure, or any runtime analyze failure.
    """
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]

        analyzer = AnalyzerEngine()
        requested_entities = entities or [
            "EMAIL_ADDRESS",
            "IBAN_CODE",
            "UK_NHS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "CRYPTO",
            "US_SSN",
        ]
        total_results = 0
        for text in texts:
            if not text:
                continue
            results = analyzer.analyze(
                text=text, entities=requested_entities, language="en"
            )
            total_results += len(results)
        if total_results > 0:
            return CheckResult(
                name="pii",
                passed=False,
                reason=f"{total_results} PII entity match(es) found",
            )
        return CheckResult(name="pii", passed=True)
    except Exception as exc:
        logger.debug("presidio check failed (%s), using regex fallback", exc)
        found, count = _regex_check(texts)
        if found:
            return CheckResult(
                name="pii",
                passed=False,
                reason=f"{count} PII pattern match(es) found (regex fallback)",
            )
        return CheckResult(name="pii", passed=True)
