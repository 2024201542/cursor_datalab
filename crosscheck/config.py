from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

PROVIDERS = ("openai", "anthropic", "local", "mock")
DISAGREEMENT_ACTIONS = {
    "review": "交叉复核后投票，仍不通过再仲裁",
    "arbiter": "跳过复核，直接交给仲裁模型",
    "human": "直接转人工审核",
}


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
    extra_body: dict = field(default_factory=dict)
    max_concurrency: int = 0
    rpm: int = 0
    mock_accuracy: float = 0.8
    train_path: str = ""  # provider=local 时的训练集；留空则用 fewshot.path
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
    disagreement_action: str = "review"
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
class FewShotConfig:
    """动态示例：为每条待分类文本从训练集中检索最相似的 k 条已标注样本放进提示词。"""
    path: str = ""
    k: int = 6
    text_col: str = "text"
    label_col: str = "label"
    max_chars: int = 300

    @property
    def enabled(self) -> bool:
        return bool(self.path) and self.k > 0


@dataclass
class Config:
    task: TaskConfig
    models: list[ModelConfig]
    arbiter: ModelConfig | None
    pipeline: PipelineConfig
    cache: CacheConfig
    mock_keywords: dict[str, list[str]]
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)


def load_dotenv(path: str | Path = ".env") -> None:
    """把 .env 中的 KEY=VALUE 读入环境变量（已存在的环境变量优先）。"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("\"'"))


def _build(cls, data: dict, where: str):
    if not isinstance(data, dict):
        raise ValueError(f"{where} 应该是一个字典，实际是: {data!r}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{where} 存在未知配置项: {sorted(unknown)}")
    return cls(**data)


def save_dotenv(updates: dict[str, str], path: str | Path = ".env") -> None:
    """更新或追加 .env 中的键值，同时写入当前进程的环境变量。"""
    p = Path(path)
    lines = p.read_text(encoding="utf-8-sig").splitlines() if p.exists() else []
    remaining = dict(updates)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"
    lines += [f"{k}={v}" for k, v in remaining.items()]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ.update(updates)


def read_raw_config(path: str | Path) -> dict:
    """读取 yaml；若含 base 字段，则以 base 指向的配置为底，当前文件的顶层字段覆盖之。"""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("base", None)
    if base:
        return {**read_raw_config((path.parent / base).resolve()), **raw}
    return raw


def save_config(raw: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_config(path: str | Path, mock: bool = False) -> Config:
    return config_from_dict(read_raw_config(path), mock=mock)


def config_from_dict(raw: dict, mock: bool = False) -> Config:
    raw = copy.deepcopy(raw)

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
        fewshot=_build(FewShotConfig, raw.get("fewshot") or {}, "fewshot"),
    )

    if mock:
        for m in config.models + ([config.arbiter] if config.arbiter else []):
            if m.provider != "local":
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
        if m.provider == "local" and not (m.train_path or config.fewshot.path):
            raise ValueError(f"本地模型 {m.name} 需要 train_path（或配置 fewshot.path）作为训练集")
    if config.arbiter and config.arbiter.provider == "local":
        raise ValueError("仲裁模型不能是本地模型（它无法阅读其他评审员的意见）")

    unknown = set(config.mock_keywords) - set(names)
    if unknown:
        raise ValueError(f"mock.keywords 中存在未定义的类别: {sorted(unknown)}")

    pc = config.pipeline
    if pc.min_votes < 1 or pc.concurrency < 1:
        raise ValueError("pipeline.min_votes 和 pipeline.concurrency 必须 >= 1")
    if pc.disagreement_action not in DISAGREEMENT_ACTIONS:
        raise ValueError(f"pipeline.disagreement_action 必须是 {list(DISAGREEMENT_ACTIONS)} 之一")
