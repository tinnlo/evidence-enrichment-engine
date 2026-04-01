"""Local tracer and artifact writer."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field


class SpanRecord(BaseModel):
    trace_id: str
    stage: str
    provider: str
    mode: str
    latency_ms: float
    entity_id: str
    field: str
    input_count: int = 0
    output_count: int = 0
    decision: str | None = None
    overall_confidence: float | None = None
    chunk_count: int | None = None
    top_chunk_score: float | None = None


class TraceSummary(BaseModel):
    trace_id: str
    total_spans: int
    total_latency_ms: float
    stages: list[str] = Field(default_factory=list)
    decision: str | None = None
    overall_confidence: float | None = None


@dataclass
class TraceArtifacts:
    trace_dir: Path
    spans_path: Path
    summary_path: Path
    timeline_path: Path
    openinference_path: Path

    def as_refs(self) -> dict[str, str]:
        return {
            "trace_dir": str(self.trace_dir),
            "spans": str(self.spans_path),
            "trace_summary": str(self.summary_path),
            "trace_timeline": str(self.timeline_path),
            "openinference_trace": str(self.openinference_path),
        }


class LocalTracer:
    def __init__(self, *, mode: str, entity_id: str, field_name: str):
        self.trace_id = str(uuid.uuid4())
        self.mode = mode
        self.entity_id = entity_id
        self.field_name = field_name
        self.spans: list[SpanRecord] = []

    @contextmanager
    def span(
        self,
        stage: str,
        *,
        provider: str,
        input_count: int = 0,
    ) -> Iterator[dict]:
        started = time.perf_counter()
        payload: dict = {"output_count": 0, "decision": None, "overall_confidence": None}
        try:
            yield payload
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            self.spans.append(
                SpanRecord(
                    trace_id=self.trace_id,
                    stage=stage,
                    provider=provider,
                    mode=self.mode,
                    latency_ms=round(latency_ms, 3),
                    entity_id=self.entity_id,
                    field=self.field_name,
                    input_count=input_count,
                    output_count=int(payload.get("output_count", 0)),
                    decision=payload.get("decision"),
                    overall_confidence=payload.get("overall_confidence"),
                )
            )

    def write(self, root: Path) -> TraceArtifacts:
        trace_dir = root / self.trace_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        spans_path = trace_dir / "spans.jsonl"
        summary_path = trace_dir / "trace_summary.json"
        timeline_path = trace_dir / "trace_timeline.md"
        openinference_path = trace_dir / "openinference_trace.json"

        spans_path.write_text(
            "\n".join(span.model_dump_json() for span in self.spans) + ("\n" if self.spans else ""),
            encoding="utf-8",
        )

        summary = TraceSummary(
            trace_id=self.trace_id,
            total_spans=len(self.spans),
            total_latency_ms=round(sum(span.latency_ms for span in self.spans), 3),
            stages=[span.stage for span in self.spans],
        )
        # Populate decision and confidence from the review_gate span if present
        gate_spans = [s for s in self.spans if s.stage == "review_gate"]
        if gate_spans:
            last_gate = gate_spans[-1]
            summary.decision = last_gate.decision
            summary.overall_confidence = last_gate.overall_confidence
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        timeline_lines = [f"# Trace Timeline: {self.trace_id}", ""]
        for index, span in enumerate(self.spans, start=1):
            timeline_lines.append(
                f"{index}. `{span.stage}` | provider=`{span.provider}` | latency_ms={span.latency_ms} | input={span.input_count} | output={span.output_count}"
            )
        timeline_path.write_text("\n".join(timeline_lines) + "\n", encoding="utf-8")

        openinference_payload = {
            "trace_id": self.trace_id,
            "spans": [
                {
                    "name": span.stage,
                    "attributes": span.model_dump(),
                }
                for span in self.spans
            ],
        }
        openinference_path.write_text(json.dumps(openinference_payload, indent=2), encoding="utf-8")
        return TraceArtifacts(
            trace_dir=trace_dir,
            spans_path=spans_path,
            summary_path=summary_path,
            timeline_path=timeline_path,
            openinference_path=openinference_path,
        )
