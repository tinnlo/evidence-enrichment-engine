"""Context pack resolution."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from evidence_enrichment.core.models.contracts import ContextEntry, ResolvedContextBundle, StageContextBundle


PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


class ContextResolver:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.manifest = _load_yaml(manifest_path)
        self.root_dir = manifest_path.parent.parent

    def resolve(self, *, entity_id: str, field_name: str) -> ResolvedContextBundle:
        task_name = str(self.manifest.get("task_name") or "unknown_task")
        load_order = list(self.manifest.get("load_order") or [])
        files = dict(self.manifest.get("files") or {})
        stage_defs = dict(self.manifest.get("stages") or {})
        stages: dict[str, StageContextBundle] = {}

        for stage_name, stage_def in stage_defs.items():
            entry_ids = [entry_id for entry_id in load_order if entry_id in set(stage_def.get("uses") or [])]
            budget_chars = int(stage_def.get("max_chars") or 0)
            used_chars = 0
            loaded_entries: list[tuple[int, str, dict[str, Any], str]] = []
            for entry_id in entry_ids:
                file_def = files.get(entry_id, {})
                raw_path = Path(str(file_def.get("path") or ""))
                path = raw_path if raw_path.is_absolute() else self.root_dir / raw_path
                content = path.read_text(encoding="utf-8").strip()
                loaded_entries.append((PRIORITY_ORDER.get(str(file_def.get("priority") or "medium"), 2), entry_id, file_def, content))

            entries_by_id: dict[str, ContextEntry] = {}
            for _, entry_id, file_def, content in sorted(loaded_entries, key=lambda row: (row[0], entry_ids.index(row[1]))):
                original_chars = len(content)
                remaining = max(budget_chars - used_chars, 0)
                included_content = content[:remaining].rstrip() if remaining < original_chars else content
                included_chars = len(included_content)
                included = included_chars > 0
                truncated = included and included_chars < original_chars
                if included:
                    used_chars += included_chars
                path = Path(str(file_def.get("path") or ""))
                entries_by_id[entry_id] = ContextEntry(
                    entry_id=entry_id,
                    path=str(path),
                    priority=str(file_def.get("priority") or "medium"),
                    required=bool(file_def.get("required", True)),
                    original_chars=original_chars,
                    included_chars=included_chars,
                    approx_tokens=math.ceil(included_chars / 4) if included_chars else 0,
                    truncated=truncated,
                    included=included,
                    content=included_content,
                )

            entries = [entries_by_id[entry_id] for entry_id in entry_ids]
            excluded = [entry.entry_id for entry in entries if not entry.included]

            stages[stage_name] = StageContextBundle(
                stage=stage_name,
                budget_chars=budget_chars,
                used_chars=used_chars,
                approx_tokens=math.ceil(used_chars / 4) if used_chars else 0,
                entry_ids=entry_ids,
                entries=entries,
                excluded_entry_ids=excluded,
            )

        return ResolvedContextBundle(
            task_name=task_name,
            manifest_path=str(self.manifest_path),
            entity_id=entity_id,
            field_name=field_name,
            load_order=load_order,
            stages=stages,
        )
