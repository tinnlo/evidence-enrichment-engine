"""Cache key generation with mode isolation for replay independence."""

import hashlib


def generate_fetch_key(url: str, mode: str) -> str:
    """Generate cache key for fetch results.

    Format: fetch:v1:{mode}:{url_hash}

    Mode isolation ensures replay and live caches don't cross-contaminate:
    - live: production fetches
    - replay: deterministic replay from artifacts
    - auto: mode determined by runtime context

    Args:
        url: The URL being fetched
        mode: Execution mode (live/replay/auto)

    Returns:
        Cache key string
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"fetch:v1:{mode}:{url_hash}"


def generate_assessment_key(
    content_hash: str, company_name: str, mode: str, url: str
) -> str:
    """Generate cache key for assessment results.

    Format: assess:v2:{mode}:{content_hash}:{company_hash}:{url_hash}

    Mode isolation prevents replay/live cross-contamination.
    URL hash ensures same content at different URLs gets separate assessments
    since document_type and source_authority_score depend on URL.

    Args:
        content_hash: Hash of the content being assessed
        company_name: Entity name for context-specific assessment
        mode: Execution mode (live/replay/auto)
        url: Document URL (affects document_type and authority score)

    Returns:
        Cache key string
    """
    company_hash = hashlib.sha256(company_name.encode()).hexdigest()[:16]
    content_short = content_hash[:16] if len(content_hash) >= 16 else content_hash
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"assess:v2:{mode}:{content_short}:{company_hash}:{url_hash}"
