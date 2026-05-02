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
    agent_iterations: int | None = None
    model_name: str | None = None
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None
    estimated_total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    budget_status: str | None = None
    downgrade_applied: str | None = None


class TraceSummary(BaseModel):
    trace_id: str
    total_spans: int
    total_latency_ms: float
    stages: list[str] = Field(default_factory=list)
    decision: str | None = None
    overall_confidence: float | None = None
    total_estimated_cost_usd: float | None = None
    cost_by_stage: dict[str, float] | None = None
    cost_by_model: dict[str, float] | None = None
    budget_status: str | None = None
    budget_limit_usd: float | None = None


@dataclass
class TraceArtifacts:
    trace_dir: Path
    spans_path: Path
    summary_path: Path
    timeline_path: Path
    openinference_path: Path
    finops_summary_path: Path | None = None

    def as_refs(self) -> dict[str, str]:
        refs = {
            "trace_dir": str(self.trace_dir),
            "spans": str(self.spans_path),
            "trace_summary": str(self.summary_path),
            "trace_timeline": str(self.timeline_path),
            "openinference_trace": str(self.openinference_path),
        }
        if self.finops_summary_path is not None:
            refs["finops_summary"] = str(self.finops_summary_path)
        return refs


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
        payload: dict = {
            "output_count": 0,
            "decision": None,
            "overall_confidence": None,
            "model_name": None,
            "estimated_input_tokens": None,
            "estimated_output_tokens": None,
            "estimated_total_tokens": None,
            "estimated_cost_usd": None,
            "budget_status": None,
            "downgrade_applied": None,
        }
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
                    agent_iterations=payload.get("agent_iterations"),
                    model_name=payload.get("model_name"),
                    estimated_input_tokens=payload.get("estimated_input_tokens"),
                    estimated_output_tokens=payload.get("estimated_output_tokens"),
                    estimated_total_tokens=payload.get("estimated_total_tokens"),
                    estimated_cost_usd=payload.get("estimated_cost_usd"),
                    budget_status=payload.get("budget_status"),
                    downgrade_applied=payload.get("downgrade_applied"),
                )
            )

    def write(self, root: Path, finops_data: dict | None = None) -> TraceArtifacts:
        trace_dir = root / self.trace_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        spans_path = trace_dir / "spans.jsonl"
        summary_path = trace_dir / "trace_summary.json"
        timeline_path = trace_dir / "trace_timeline.md"
        openinference_path = trace_dir / "openinference_trace.json"
        finops_summary_path: Path | None = None

        spans_path.write_text(
            "\n".join(span.model_dump_json() for span in self.spans)
            + ("\n" if self.spans else ""),
            encoding="utf-8",
        )

        finops_spans = [s for s in self.spans if s.estimated_cost_usd is not None]
        total_cost = round(sum(s.estimated_cost_usd or 0.0 for s in finops_spans), 8)
        cost_by_stage: dict[str, float] = {}
        cost_by_model: dict[str, float] = {}
        for s in finops_spans:
            if s.estimated_cost_usd is not None:
                cost_by_stage[s.stage] = round(
                    cost_by_stage.get(s.stage, 0.0) + s.estimated_cost_usd, 8
                )
            if s.estimated_cost_usd is not None and s.model_name:
                cost_by_model[s.model_name] = round(
                    cost_by_model.get(s.model_name, 0.0) + s.estimated_cost_usd, 8
                )

        summary = TraceSummary(
            trace_id=self.trace_id,
            total_spans=len(self.spans),
            total_latency_ms=round(sum(span.latency_ms for span in self.spans), 3),
            stages=[span.stage for span in self.spans],
            total_estimated_cost_usd=total_cost if finops_spans else None,
            cost_by_stage=cost_by_stage if finops_spans else None,
            cost_by_model=cost_by_model if finops_spans else None,
            budget_status=finops_data.get("budget_decision", {}).get("status") if finops_data else None,
            budget_limit_usd=finops_data.get("budget_decision", {}).get("budget_limit_usd") if finops_data else None,
        )
        gate_spans = [s for s in self.spans if s.stage == "review_gate"]
        if gate_spans:
            last_gate = gate_spans[-1]
            summary.decision = last_gate.decision
            summary.overall_confidence = last_gate.overall_confidence
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        timeline_lines = [f"# Trace Timeline: {self.trace_id}", ""]
        for index, span in enumerate(self.spans, start=1):
            cost_str = ""
            if span.estimated_cost_usd is not None:
                cost_str = f" | cost=${span.estimated_cost_usd:.6f}"
            model_str = ""
            if span.model_name:
                model_str = f" | model={span.model_name}"
            downgrade_str = ""
            if span.downgrade_applied and span.downgrade_applied != "none":
                downgrade_str = f" | downgrade={span.downgrade_applied}"
            timeline_lines.append(
                f"{index}. `{span.stage}` | provider=`{span.provider}` | "
                f"latency_ms={span.latency_ms} | input={span.input_count} | "
                f"output={span.output_count}{model_str}{cost_str}{downgrade_str}"
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
        openinference_path.write_text(
            json.dumps(openinference_payload, indent=2, default=str),
            encoding="utf-8",
        )

        if finops_data is not None:
            finops_summary_path = trace_dir / "finops_summary.json"
            finops_summary_path.write_text(
                json.dumps(finops_data, indent=2, default=str),
                encoding="utf-8",
            )

        return TraceArtifacts(
            trace_dir=trace_dir,
            spans_path=spans_path,
            summary_path=summary_path,
            timeline_path=timeline_path,
            openinference_path=openinference_path,
            finops_summary_path=finops_summary_path,
        )
