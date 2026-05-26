"""Safe document fetching."""

from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

import httpx

from evidence_enrichment.core.models.contracts import RetrievedDocument, SearchResult

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style.*?</style>", re.IGNORECASE | re.DOTALL)

_MAX_REDIRECTS = 10
_ALLOWED_SCHEMES = {"http", "https"}
_MAX_BODY_BYTES = 5 * 1024 * 1024   # 5 MB — text/HTML responses
_MAX_PDF_BYTES = 50 * 1024 * 1024   # 50 MB — PDF responses
_ALLOWED_CONTENT_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
)

# Populated only when PDFStructuredParser (and its deps) can be imported.
# Probing the parser module — not individual packages — keeps the fetch gate
# and parser registration gate in sync: if either pdfplumber or pymupdf is
# absent the import fails and PDFs remain blocked exactly as today.
_ALLOWED_BINARY_PREFIXES: tuple[str, ...] = ()
try:
    from evidence_enrichment.core.parse.pdf_structured import PDFStructuredParser as _  # noqa: F401
    _ALLOWED_BINARY_PREFIXES = ("application/pdf",)
except ImportError:
    pass


async def _validate_host(hostname: str) -> None:
    """Raise ValueError if *hostname* resolves to any private/local IP.

    Fails closed on DNS resolution errors — an unresolvable hostname is
    treated as unsafe rather than bypassed, preventing DNS-rebinding attacks
    that exploit transient resolution failures.

    Note: this pre-flight DNS check is a best-effort SSRF guard.  Full
    protection against DNS-rebinding / TOCTOU would require pinning the
    resolved IP inside a custom transport so that the TCP connection is
    forced to the pre-checked address.  That level of hardening is deferred
    as out-of-scope for this demo; the check is retained because it rejects
    the majority of accidental or unsophisticated SSRF attempts.

    The DNS lookup is dispatched to a thread-pool worker via
    ``asyncio.to_thread`` so that a slow or misconfigured hostname does not
    block the async event loop.
    """
    if not hostname:
        raise ValueError("Blocked empty hostname")
    try:
        addr_info = await asyncio.to_thread(
            socket.getaddrinfo, hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError(f"Blocked unresolvable host {hostname}: {exc}") from exc
    for _family, _type, _proto, _canonname, sockaddr in addr_info:
        ip = ipaddress.ip_address(sockaddr[0])
        # Reject any address that is not a globally-routable unicast address.
        # This covers: loopback, private, link-local, reserved, unspecified
        # (0.0.0.0 / ::), and multicast ranges — all of which are invalid
        # Internet destinations and potential SSRF vectors.
        if not ip.is_global or ip.is_multicast:
            raise ValueError(f"Blocked non-global/multicast host {hostname} ({ip})")


async def _validate_url(url: str) -> None:
    """Raise ValueError if *url* has a disallowed scheme or private hostname."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Blocked disallowed scheme: {parsed.scheme}:// in {url}")
    await _validate_host(parsed.hostname or "")


def _check_content_type(content_type: str, url: str) -> None:
    ct = content_type.split(";")[0].strip().lower()
    if ct and not (
        ct.startswith(_ALLOWED_CONTENT_PREFIXES)
        or ct.startswith(_ALLOWED_BINARY_PREFIXES)
    ):
        logging.warning("Skipping non-text content type %s for %s", ct, url)
        raise ValueError(f"Blocked non-text content type: {ct} for {url}")


def _is_binary_content_type(content_type: str) -> bool:
    ct = content_type.split(";")[0].strip().lower()
    return bool(_ALLOWED_BINARY_PREFIXES) and ct.startswith(_ALLOWED_BINARY_PREFIXES)


class DocumentFetcher:
    async def fetch(self, result: SearchResult) -> RetrievedDocument:
        await _validate_url(result.url)
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            url = result.url
            for hop in range(_MAX_REDIRECTS + 1):
                async with client.stream(
                    "GET",
                    url,
                    headers={"User-Agent": "evidence-enrichment-engine/0.1"},
                ) as response:
                    if not response.is_redirect:
                        # Check Content-Length before reading any bytes
                        cl_header = response.headers.get("content-length")
                        content_type_early = response.headers.get("content-type", "text/html")
                        early_size_limit = (
                            _MAX_PDF_BYTES
                            if _is_binary_content_type(content_type_early)
                            else _MAX_BODY_BYTES
                        )
                        if cl_header is not None:
                            try:
                                if int(cl_header) > early_size_limit:
                                    raise ValueError(
                                        f"Content-Length {cl_header} exceeds {early_size_limit} bytes for {url}"
                                    )
                            except (ValueError, TypeError) as exc:
                                # Re-raise our own ValueError; ignore non-integer header
                                if "Content-Length" in str(exc):
                                    raise

                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "text/html")
                        final_url = str(response.url)
                        await _validate_url(final_url)
                        _check_content_type(content_type, final_url)

                        is_binary = _is_binary_content_type(content_type)
                        size_limit = _MAX_PDF_BYTES if is_binary else _MAX_BODY_BYTES

                        # Stream body with hard cutoff at size_limit
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            total += len(chunk)
                            if total > size_limit:
                                raise ValueError(
                                    f"Response body exceeds {size_limit} bytes for {final_url}"
                                )
                            chunks.append(chunk)

                        raw_bytes = b"".join(chunks)

                        if is_binary:
                            return RetrievedDocument(
                                url=result.url,
                                final_url=final_url,
                                title=result.title,
                                content_type=content_type,
                                body="",
                                body_bytes=raw_bytes,
                                provider=result.provider.value,
                            )

                        encoding = response.encoding or "utf-8"
                        try:
                            body = raw_bytes.decode(encoding, errors="replace")
                        except LookupError:
                            # Unknown/invalid charset label from server — fall back to UTF-8.
                            body = raw_bytes.decode("utf-8", errors="replace")
                        return RetrievedDocument(
                            url=result.url,
                            final_url=final_url,
                            title=result.title,
                            content_type=content_type,
                            body=body,
                            provider=result.provider.value,
                        )

                    # Redirect path
                    if hop == _MAX_REDIRECTS:
                        raise ValueError(f"Too many redirects fetching {result.url}")
                    next_request = response.next_request
                    if next_request is None:
                        break
                    next_url = str(next_request.url)
                    await _validate_url(next_url)
                    url = next_url

        raise ValueError(f"Redirect loop resolved without response for {result.url}")


def html_to_text(body: str) -> str:
    cleaned = _SCRIPT_RE.sub(" ", body)
    cleaned = _STYLE_RE.sub(" ", cleaned)
    cleaned = _TAG_RE.sub(" ", cleaned)
    return " ".join(html.unescape(cleaned).split()).strip()


def normalized_hostname(url: str) -> str:
    """Return the bare hostname (no port, no userinfo) from *url*, lower-cased.

    Uses ``urlparse().hostname`` so that ``https://sec.gov:443/...`` correctly
    returns ``sec.gov`` rather than ``sec.gov:443``.

    Note: this returns the full hostname, not a registrable/eTLD+1 domain.
    For example ``data.sec.gov`` → ``data.sec.gov``, not ``sec.gov``.
    Use a PSL-based library such as ``tldextract`` if registrable-domain
    grouping is needed.
    """
    hostname = urlparse(url).hostname or ""
    hostname = hostname.lower()
    # Only strip a leading "www." label, not occurrences elsewhere in the hostname.
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname
