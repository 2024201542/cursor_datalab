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
from .cost import cost_of, estimate_tokens
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


_LOOSE_LABEL = re.compile(r'"label"\s*:\s*"([^"]*)"')
_LOOSE_CONF = re.compile(r'"confidence"\s*:\s*"?([0-9.]+)')
_LOOSE_REASON = re.compile(r'"reason"\s*:\s*"(.*)"\s*}', re.S)


def _loose_fields(text: str) -> dict | None:
    """模型常在理由里写未转义的英文双引号（如 "封板"是积极信号），导致 JSON 不合法；此时逐字段提取。"""
    lab = _LOOSE_LABEL.search(text)
    if not lab:
        return None
    conf = _LOOSE_CONF.search(text)
    reason = _LOOSE_REASON.search(text)
    return {"label": lab.group(1), "confidence": conf.group(1) if conf else 0.5,
            "reason": reason.group(1) if reason else ""}


def parse_output(raw: str, labels: list[str]) -> tuple[str, float, str]:
    text = _THINK_RE.sub("", raw).strip()
    m = _JSON_RE.search(text)
    if not m:
        raise ParseError("回复中没有找到 JSON")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        obj = _loose_fields(m.group(0))
        if obj is None:
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
        self.usage: dict[str, tuple[int, int]] = {}
        self._fh = None
        if enabled and self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["response"]
                        if rec.get("usage"):
                            self.usage[rec["key"]] = tuple(rec["usage"])
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

    def put(self, key: str, response: str, usage: tuple[int, int] | None = None) -> None:
        if not self.enabled:
            return
        self.data[key] = response
        rec = {"key": key, "response": response}
        if usage:
            self.usage[key] = usage
            rec["usage"] = list(usage)
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
        # in/out_tokens 与 cost 只统计实际请求；saved 是缓存命中省下的费用
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0, "in_tokens": 0, "out_tokens": 0,
                      "estimated_calls": 0, "cost": 0.0, "saved": 0.0}

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
        cfg = self.llm.cfg
        key = LLMCache.make_key(cfg, SYSTEM_PROMPT, user)
        raw = self.cache.get(key) if self.llm.cacheable else None
        from_cache = raw is not None
        usage = None
        if from_cache:
            self.stats["cache_hits"] += 1
            if self.llm.billable:
                tin, tout = self.cache.usage.get(key) or (estimate_tokens(SYSTEM_PROMPT + user), estimate_tokens(raw))
                self.stats["saved"] += cost_of(cfg, tin, tout)
        else:
            async with self.limiter.slot(), self.sem:
                self.stats["calls"] += 1
                try:
                    reply = await self.llm.complete(SYSTEM_PROMPT, user)
                except LLMError as e:
                    self.stats["errors"] += 1
                    return Prediction(self.name, None, error=str(e))
            raw = reply.text
            if self.llm.billable:
                if reply.in_tokens is None or reply.out_tokens is None:
                    self.stats["estimated_calls"] += 1
                tin = reply.in_tokens if reply.in_tokens is not None else estimate_tokens(SYSTEM_PROMPT + user)
                tout = reply.out_tokens if reply.out_tokens is not None else estimate_tokens(raw)
                usage = (int(tin), int(tout))
                self.stats["in_tokens"] += usage[0]
                self.stats["out_tokens"] += usage[1]
                self.stats["cost"] += cost_of(cfg, *usage)

        try:
            label, conf, reason = parse_output(raw, self.task.label_names)
        except ParseError as e:
            self.stats["errors"] += 1
            return Prediction(self.name, None, error=f"解析失败: {e}", raw=raw)

        if self.llm.cacheable and not from_cache:
            self.cache.put(key, raw, usage)
        return Prediction(self.name, label, conf, reason, raw=raw)
