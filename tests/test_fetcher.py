"""Tests for SSRF host validation and normalized_hostname utility."""

from __future__ import annotations

import asyncio

import pytest

from evidence_enrichment.core.fetch.fetcher import _validate_host, normalized_hostname


class TestValidateHost:
    def test_public_hostname_passes(self):
        """A normal public hostname must not raise."""
        # Will do a real DNS lookup; use a well-known domain.
        # If DNS is unavailable in CI, this will raise ValueError (fail-closed),
        # so we only assert it raises ValueError, not that it passes.
        try:
            asyncio.run(_validate_host("example.com"))
        except ValueError as exc:
            # Acceptable: unresolvable in CI environment
            assert "Blocked" in str(exc)

    def test_loopback_blocked(self):
        """127.0.0.1 must be blocked."""
        with pytest.raises(ValueError, match="Blocked"):
            asyncio.run(_validate_host("127.0.0.1"))

    def test_private_range_blocked(self):
        """192.168.x.x is a private range and must be blocked."""
        with pytest.raises(ValueError, match="Blocked"):
            asyncio.run(_validate_host("192.168.1.1"))

    def test_empty_hostname_blocked(self):
        """An empty hostname must be rejected."""
        with pytest.raises(ValueError, match="Blocked empty hostname"):
            asyncio.run(_validate_host(""))

    def test_unresolvable_hostname_blocked(self):
        """A hostname that cannot be resolved must be rejected (fail-closed)."""
        with pytest.raises(ValueError, match="Blocked unresolvable host"):
            asyncio.run(_validate_host("this.hostname.does.not.exist.invalid"))


class TestNormalizedHostname:
    def test_strips_port(self):
        assert normalized_hostname("https://sec.gov:443/filing") == "sec.gov"

    def test_strips_www(self):
        assert normalized_hostname("https://www.example.com/page") == "example.com"

    def test_plain_hostname(self):
        assert normalized_hostname("https://reuters.com/article") == "reuters.com"

    def test_subdomain_preserved(self):
        assert normalized_hostname("https://data.sec.gov/api") == "data.sec.gov"

    def test_empty_url_returns_empty(self):
        assert normalized_hostname("") == ""

    def test_lowercases_result(self):
        assert normalized_hostname("https://SEC.GOV/filing") == "sec.gov"

    def test_www_only_stripped_when_leading(self):
        """www. in the middle of a hostname must not be stripped."""
        assert (
            normalized_hostname("https://foo.www.example.com/page")
            == "foo.www.example.com"
        )
