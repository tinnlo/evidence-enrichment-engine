"""Replay bundle loading."""

from __future__ import annotations

import json
from pathlib import Path


def load_replay_bundle(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

