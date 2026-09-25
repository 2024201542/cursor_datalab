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
    # 选项顺序随机化：每个模型、每条文本看到的类别顺序不同（按哈希固定，缓存仍可复用），消除位置偏好
    shuffle_labels: bool = False

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
    # 读取标签的输出概率作为置信度（仅 OpenAI 兼容接口）。DeepSeek 在温度 0 时只返回 0 / 1，没有区分度
    logprobs: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)
    max_concurrency: int = 0
    rpm: int = 0
    mock_accuracy: float = 0.8
    train_path: str = ""  # provider=local 时的训练集；留空则用 fewshot.path
    price_in: float = 0.0   # 输入单价，元 / 百万 token；0 表示未设置，只统计 token 不算费用
    price_out: float = 0.0  # 输出单价，元 / 百万 token
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
    # 级联：首轮先只调用这些模型（按名称），它们全部给出有效且一致的结果就直接采纳，不再调用其余模型
    cascade: list[str] = field(default_factory=list)
    # 级联的置信度门槛：首批模型一致且每个模型的置信度都 ≥ 该值才直接采纳；0 表示只看是否一致。
    # 置信度优先用 logprobs 概率，其次是模型自报的置信度，本地小模型为预测概率。> 0 时首批可以只有 1 个模型
    cascade_min_confidence: float = 0.0
    disagreement_action: str = "human"
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
    """读取 yaml；若含 base 字段，则以 base 指向的配置为底，当前文件的顶层字段整体覆盖之。

    带点的键只改一个嵌套字段，例如 `task.shuffle_labels: true` 只修改 task 下的 shuffle_labels。
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("base", None)
    dotted = {k: raw.pop(k) for k in [k for k in raw if "." in k]}
    if base:
        raw = {**read_raw_config((path.parent / base).resolve()), **raw}
    for key, value in dotted.items():
        node = raw
        *parents, leaf = key.split(".")
        for part in parents:
            node[part] = copy.deepcopy(node.get(part) or {})
            node = node[part]
        node[leaf] = value
    return raw


def save_config(raw: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def base_of(path: str | Path) -> Path | None:
    """配置文件 base 字段指向的文件（绝对路径）；没有 base 返回 None。"""
    p = Path(path)
    if not p.exists():
        return None
    base = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("base")
    return (p.parent / base).resolve() if base else None


def save_config_diff(raw: dict, path: str | Path, base: str | Path | None) -> None:
    """有 base 时只写入与 base 不同的顶层字段，这样模型配置等公共部分仍随 base 更新。"""
    path = Path(path)
    if base is None or Path(base).resolve() == path.resolve():
        save_config(raw, path)
        return
    base = Path(base).resolve()
    base_raw = read_raw_config(base)
    rel = Path(os.path.relpath(base, path.resolve().parent)).as_posix()
    save_config({"base": rel, **{k: v for k, v in raw.items() if base_raw.get(k) != v}}, path)


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
        shuffle_labels=bool(t.get("shuffle_labels", False)),
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
    if pc.cascade:
        missing = [n for n in pc.cascade if n not in model_names]
        if missing:
            raise ValueError(f"pipeline.cascade 中的模型不存在或未启用: {missing}")
        need = 1 if pc.cascade_min_confidence > 0 else 2
        if len(set(pc.cascade)) < need or len(set(pc.cascade)) >= len(model_names):
            raise ValueError("pipeline.cascade 要少于全部投票模型，且至少 2 个（设置了 cascade_min_confidence 时可以只有 1 个），"
                             "否则级联没有意义")
    if not 0 <= pc.cascade_min_confidence <= 1:
        raise ValueError("pipeline.cascade_min_confidence 必须在 0 到 1 之间")
