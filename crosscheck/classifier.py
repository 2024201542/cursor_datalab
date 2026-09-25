from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ModelConfig, TaskConfig
from .llm import BaseLLM, LLMError
from .prompts import SYSTEM_PROMPT, build_arbiter_prompt, build_classify_prompt, build_review_prompt


@dataclass
class Prediction:
    model: str
    label: str | None
    confidence: float = 0.0
    reason: str = ""
    error: str | None = None
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and self.label is not None

    def to_dict(self) -> dict:
        return asdict(self)


class ParseError(ValueError):
    pass


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_JSON_RE = re.compile(r"\{.*\}", re.S)
_STRIP_CHARS = " \t\n\"'“”‘’「」『』[]【】"


def normalize_label(value: str, labels: list[str]) -> str | None:
    v = value.strip(_STRIP_CHARS)
    if v in labels:
        return v
    lowered = {lab.lower(): lab for lab in labels}
    if v.lower() in lowered:
        return lowered[v.lower()]
    hits = [lab for lab in labels if lab in v]
    return hits[0] if len(hits) == 1 else None


def parse_output(raw: str, labels: list[str]) -> tuple[str, float, str]:
    text = _THINK_RE.sub("", raw).strip()
    m = _JSON_RE.search(text)
    if not m:
        raise ParseError("回复中没有找到 JSON")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON 格式错误: {e}") from e
    if not isinstance(obj, dict):
        raise ParseError("JSON 不是对象")

    label = normalize_label(str(obj.get("label", "")), labels)
    if label is None:
        raise ParseError(f"标签不在候选集合中: {obj.get('label')!r}")

    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    if 1 < conf <= 100:
        conf /= 100
    conf = min(max(conf, 0.0), 1.0)
    return label, conf, str(obj.get("reason", "")).strip()


class LLMCache:
    """以 jsonl 追加写入的回复缓存：相同模型 + 相同提示词只调用一次。"""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.data: dict[str, str] = {}
        self._fh = None
        if enabled and self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["response"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue

    @staticmethod
    def make_key(cfg: ModelConfig, system: str, user: str) -> str:
        payload = json.dumps(
            [cfg.provider, cfg.base_url, cfg.model, cfg.temperature, system, user],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self.data.get(key) if self.enabled else None

    def put(self, key: str, response: str) -> None:
        if not self.enabled:
            return
        self.data[key] = response
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class RateLimiter:
    """单个模型的并发上限 + 每分钟请求数上限（0 表示不限制）。"""

    def __init__(self, max_concurrency: int = 0, rpm: int = 0):
        self.sem = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    @asynccontextmanager
    async def slot(self):
        if self.sem is not None:
            await self.sem.acquire()
        try:
            if self.interval:
                async with self._lock:
                    now = time.monotonic()
                    wait = self._next - now
                    self._next = max(now, self._next) + self.interval
                if wait > 0:
                    await asyncio.sleep(wait)
            yield
        finally:
            if self.sem is not None:
                self.sem.release()


class Classifier:
    def __init__(self, task: TaskConfig, llm: BaseLLM, cache: LLMCache, semaphore: asyncio.Semaphore, max_chars: int = 300):
        self.task = task
        self.max_chars = max_chars
        self.llm = llm
        self.cache = cache
        self.sem = semaphore
        self.limiter = RateLimiter(llm.cfg.max_concurrency, llm.cfg.rpm)
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0}

    @property
    def name(self) -> str:
        return self.llm.cfg.name

    async def classify(self, text: str, examples=None) -> Prediction:
        return await self._call(build_classify_prompt(self.task, text, examples, self.max_chars))

    async def review(self, text: str, own: Prediction | None, peers: list[Prediction], examples=None) -> Prediction:
        return await self._call(build_review_prompt(self.task, text, own, peers, examples, self.max_chars))

    async def arbitrate(self, text: str, opinions: list[Prediction], examples=None) -> Prediction:
        return await self._call(build_arbiter_prompt(self.task, text, opinions, examples, self.max_chars))

    async def _call(self, user: str) -> Prediction:
        key = LLMCache.make_key(self.llm.cfg, SYSTEM_PROMPT, user)
        raw = self.cache.get(key) if self.llm.cacheable else None
        from_cache = raw is not None
        if from_cache:
            self.stats["cache_hits"] += 1
        else:
            async with self.limiter.slot(), self.sem:
                self.stats["calls"] += 1
                try:
                    raw = await self.llm.chat(SYSTEM_PROMPT, user)
                except LLMError as e:
                    self.stats["errors"] += 1
                    return Prediction(self.name, None, error=str(e))

        try:
            label, conf, reason = parse_output(raw, self.task.label_names)
        except ParseError as e:
            self.stats["errors"] += 1
            return Prediction(self.name, None, error=f"解析失败: {e}", raw=raw)

        if self.llm.cacheable and not from_cache:
            self.cache.put(key, raw)
        return Prediction(self.name, label, conf, reason, raw=raw)
