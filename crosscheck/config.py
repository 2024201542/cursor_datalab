from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

PROVIDERS = ("openai", "anthropic", "mock")


@dataclass
class LabelDef:
    name: str
    definition: str = ""
    examples: list[str] = field(default_factory=list)
    counter_examples: list[str] = field(default_factory=list)


@dataclass
class TaskConfig:
    name: str
    description: str
    labels: list[LabelDef]
    rules: list[str] = field(default_factory=list)

    @property
    def label_names(self) -> list[str]:
        return [lab.name for lab in self.labels]


@dataclass
class ModelConfig:
    name: str
    provider: str = "openai"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    api_key: str = ""
    weight: float = 1.0
    temperature: float = 0.0
    max_tokens: int = 512
    json_mode: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    mock_accuracy: float = 0.8
    enabled: bool = True

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


@dataclass
class PipelineConfig:
    concurrency: int = 8
    min_votes: int = 2
    cross_review: bool = True
    accept_threshold: float = 0.6
    use_arbiter: bool = True
    arbiter_threshold: float = 0.7
    timeout: float = 60
    max_retries: int = 3


@dataclass
class CacheConfig:
    enabled: bool = True
    path: str = ".cache/llm_cache.jsonl"


@dataclass
class Config:
    task: TaskConfig
    models: list[ModelConfig]
    arbiter: ModelConfig | None
    pipeline: PipelineConfig
    cache: CacheConfig
    mock_keywords: dict[str, list[str]]


def _build(cls, data: dict, where: str):
    if not isinstance(data, dict):
        raise ValueError(f"{where} 应该是一个字典，实际是: {data!r}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{where} 存在未知配置项: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path, mock: bool = False) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    t = raw.get("task") or {}
    task = TaskConfig(
        name=t.get("name", ""),
        description=t.get("description", ""),
        labels=[_build(LabelDef, lab, f"task.labels[{i}]") for i, lab in enumerate(t.get("labels") or [])],
        rules=list(t.get("rules") or []),
    )

    models = [_build(ModelConfig, m, f"models[{i}]") for i, m in enumerate(raw.get("models") or [])]
    models = [m for m in models if m.enabled]

    arbiter = None
    if raw.get("arbiter"):
        arbiter = _build(ModelConfig, raw["arbiter"], "arbiter")
        if not arbiter.enabled:
            arbiter = None

    config = Config(
        task=task,
        models=models,
        arbiter=arbiter,
        pipeline=_build(PipelineConfig, raw.get("pipeline") or {}, "pipeline"),
        cache=_build(CacheConfig, raw.get("cache") or {}, "cache"),
        mock_keywords=(raw.get("mock") or {}).get("keywords") or {},
    )

    if mock:
        for m in config.models + ([config.arbiter] if config.arbiter else []):
            m.provider = "mock"

    _validate(config)
    return config


def _validate(config: Config) -> None:
    names = config.task.label_names
    if len(names) < 2:
        raise ValueError("task.labels 至少需要 2 个类别")
    if len(set(names)) != len(names):
        raise ValueError(f"类别名称重复: {names}")

    if len(config.models) < 2:
        raise ValueError("至少需要启用 2 个模型才能互检")
    model_names = [m.name for m in config.models]
    if len(set(model_names)) != len(model_names):
        raise ValueError(f"模型 name 重复: {model_names}")

    for m in config.models + ([config.arbiter] if config.arbiter else []):
        if m.provider not in PROVIDERS:
            raise ValueError(f"模型 {m.name} 的 provider 必须是 {PROVIDERS} 之一，实际是 {m.provider!r}")
        if m.weight <= 0:
            raise ValueError(f"模型 {m.name} 的 weight 必须大于 0")

    unknown = set(config.mock_keywords) - set(names)
    if unknown:
        raise ValueError(f"mock.keywords 中存在未定义的类别: {sorted(unknown)}")

    pc = config.pipeline
    if pc.min_votes < 1 or pc.concurrency < 1:
        raise ValueError("pipeline.min_votes 和 pipeline.concurrency 必须 >= 1")
