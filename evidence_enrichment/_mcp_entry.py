"""Guarded entry point for the ``evidence-enrich-mcp`` console script.

This thin wrapper defers the import of ``mcp_server`` until the entry point
function is actually called.  That means a base install (without the ``[mcp]``
optional extra) can resolve the script path without raising an ImportError at
import time; users receive a clear installation hint instead of a traceback.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Launch the MCP server, or print installation guidance if mcp is absent."""
    try:
        from evidence_enrichment.mcp_server import main as _main
    except ImportError:
        print(
            "The MCP server requires the 'mcp' package, which is not installed.\n"
            "\n"
            "Install it with:\n"
            "\n"
            "    pip install 'evidence_enrichment[mcp]'\n"
            "\n"
            "or:\n"
            "\n"
            "    pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    _main()
