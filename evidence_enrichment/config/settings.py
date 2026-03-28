"""Settings and repo-local config loading."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from evidence_enrichment.observability.langsmith import apply_langsmith_env


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


class StageProviderConfig(BaseModel):
    provider_order: list[str] = Field(default_factory=list)


class HQCountryThresholds(BaseModel):
    auto_approve_min_confidence: float = 0.85
    review_min_confidence: float = 0.50


class Settings(BaseModel):
    default_mode: str = "auto"
    replay_dir: str = "examples/replay"
    prompt_dir: str = "prompts"
    context_dir: str = "context"
    trace_output_dir: str = "examples/output/traces"
    eval_output_dir: str = "evals/output"
    search: StageProviderConfig = Field(default_factory=StageProviderConfig)
    analysis: StageProviderConfig = Field(default_factory=StageProviderConfig)
    synthesis: StageProviderConfig = Field(default_factory=StageProviderConfig)
    thresholds: dict[str, HQCountryThresholds] = Field(default_factory=dict)
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"
    serper_api_key: str | None = None
    tavily_api_key: str | None = None

    @classmethod
    def load(
        cls,
        *,
        config_file: str = "evidence_enrichment.yaml",
        env_file: str = ".env",
    ) -> "Settings":
        config = _load_yaml(Path(config_file))
        env_values = _parse_env_file(Path(env_file))
        data: dict[str, Any] = dict(config)
        data["openai_api_key"] = os.getenv("OPENAI_API_KEY") or env_values.get("OPENAI_API_KEY")
        data["openai_model"] = os.getenv("OPENAI_MODEL") or env_values.get("OPENAI_MODEL") or data.get("openai_model", "gpt-4.1-mini")
        data["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY") or env_values.get("ANTHROPIC_API_KEY")
        data["anthropic_model"] = os.getenv("ANTHROPIC_MODEL") or env_values.get("ANTHROPIC_MODEL") or data.get("anthropic_model", "claude-3-5-sonnet-latest")
        data["serper_api_key"] = os.getenv("SERPER_API_KEY") or env_values.get("SERPER_API_KEY")
        data["tavily_api_key"] = os.getenv("TAVILY_API_KEY") or env_values.get("TAVILY_API_KEY")
        apply_langsmith_env(env_values)
        return cls(**data)

    @property
    def replay_path(self) -> Path:
        return Path(self.replay_dir)

    @property
    def context_path(self) -> Path:
        return Path(self.context_dir)

    @property
    def trace_output_path(self) -> Path:
        return Path(self.trace_output_dir)

    @property
    def eval_output_path(self) -> Path:
        return Path(self.eval_output_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()
