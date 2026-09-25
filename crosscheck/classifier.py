from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ModelConfig, TaskConfig, split_labels
from .cost import cost_of, estimate_tokens
from .llm import BaseLLM, LLMError
from .prompts import (
    SYSTEM_PROMPT,
    build_arbiter_prompt,
    build_classify_prompt,
    build_devil_prompt,
    build_review_prompt,
    prompt_seed,
)


@dataclass
class Prediction:
    model: str
    label: str | None
    confidence: float = 0.0
    reason: str = ""
    error: str | None = None
    raw: str = ""
    prob: float | None = None  # 开启 logprobs 时标签的输出概率
    samples: list[str] | None = None  # Self-Consistency 时每次采样的标签；confidence 为多数答案的占比
    evidence: str = ""  # 要求证据时模型引用的原文
    evidence_ok: bool | None = None  # 引用是否确实出现在原文中；None 表示没有要求证据
    calibrated: float | None = None  # 按校准曲线换算后的“真实正确率”

    @property
    def ok(self) -> bool:
        return self.error is None and self.label is not None

    @property
    def certainty(self) -> float:
        """级联判断用的置信度：优先用校准后的值，其次 logprobs 概率，最后是模型自报的置信度（采样时为答案稳定性）。"""
        if self.calibrated is not None:
            return self.calibrated
        return self.prob if self.prob is not None else self.confidence

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


_LABEL_VALUE_RE = re.compile(r'"label"\s*:\s*"')


def label_prob(tokens: list[tuple[str, float]] | None) -> float | None:
    """输出中 label 字段取值的概率：覆盖取值文字的所有 token 的概率之积。"""
    if not tokens:
        return None
    text = "".join(t for t, _ in tokens)
    m = None
    for m in _LABEL_VALUE_RE.finditer(_THINK_RE.sub(lambda x: " " * len(x.group(0)), text)):
        pass  # 取最后一个，跳过思考内容里可能出现的示例
    if m is None:
        return None
    start = m.end()
    end = text.find('"', start)
    if end <= start:
        return None
    total, pos = 0.0, 0
    for tok, lp in tokens:
        if pos < end and pos + len(tok) > start:
            total += lp
        pos += len(tok)
    return round(math.exp(max(total, -50.0)), 4)


def parse_output(raw: str, labels: list[str]) -> tuple[str, float, str]:
    label, conf, reason, _ = parse_reply(raw, labels)
    return label, conf, reason


_LOOSE_LABELS = re.compile(r'"labels"\s*:\s*\[([^\]]*)\]')
_LOOSE_EVIDENCE = re.compile(r'"evidence"\s*:\s*"([^"]*)"')


def parse_reply(raw: str, labels: list[str], task: TaskConfig | None = None) -> tuple[str, float, str, str]:
    """解析模型回复，返回 (标签, 置信度, 理由, 证据)。多标签任务的标签为规范写法“A|B”。"""
    text = _THINK_RE.sub("", raw).strip()
    m = _JSON_RE.search(text)
    if not m:
        raise ParseError("回复中没有找到 JSON")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        obj = _loose_fields(m.group(0))
        if obj is None:
            many = _LOOSE_LABELS.search(m.group(0))
            if many is None:
                raise ParseError(f"JSON 格式错误: {e}") from e
            obj = {"labels": re.findall(r'"([^"]*)"', many.group(1)), "confidence": 0.5}
        ev = _LOOSE_EVIDENCE.search(m.group(0))
        if ev:
            obj["evidence"] = ev.group(1)
    if not isinstance(obj, dict):
        raise ParseError("JSON 不是对象")

    if task is not None and task.multi_label:
        values = obj.get("labels")
        if values is None:
            values = obj.get("label", "")
        if isinstance(values, str):
            values = split_labels(values)
        if not isinstance(values, list):
            raise ParseError(f"labels 不是列表: {values!r}")
        picked = [normalize_label(str(v), labels) for v in values]
        if not picked or any(v is None for v in picked):
            raise ParseError(f"标签不在候选集合中: {values!r}")
        label = task.join(list(dict.fromkeys(picked))[: task.max_labels])
    else:
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
    return label, conf, str(obj.get("reason", "")).strip(), str(obj.get("evidence") or "").strip()


_QUOTE_STRIP = " \t\n\"'“”‘’「」『』《》…."
_SPACE_RE = re.compile(r"\s+")


def evidence_in_text(evidence: str, text: str) -> bool:
    """证据是否逐字出现在原文中（忽略空白和首尾引号；长引用允许用省略号拼接的多段，每段都要在原文中）。"""
    ev = _SPACE_RE.sub("", evidence).strip(_QUOTE_STRIP)
    if len(ev) < 2:
        return False
    body = _SPACE_RE.sub("", text)
    pieces = [p.strip(_QUOTE_STRIP) for p in re.split(r"…+|\.{3,}", ev)]
    pieces = [p for p in pieces if p]
    return bool(pieces) and all(p in body for p in pieces)


class LLMCache:
    """以 jsonl 追加写入的回复缓存：相同模型 + 相同提示词只调用一次。"""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.data: dict[str, str] = {}
        self.usage: dict[str, tuple[int, int]] = {}
        self.probs: dict[str, float] = {}
        self._fh = None
        if enabled and self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["response"]
                        if rec.get("usage"):
                            self.usage[rec["key"]] = tuple(rec["usage"])
                        if rec.get("prob") is not None:
                            self.probs[rec["key"]] = rec["prob"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue

    @staticmethod
    def make_key(cfg: ModelConfig, system: str, user: str, tag: str = "") -> str:
        parts = [cfg.provider, cfg.base_url, cfg.model, cfg.temperature, system, user]
        if cfg.logprobs:
            parts.append("logprobs")  # 开启前缓存的回复没有概率，不能复用
        if tag:
            parts.append(tag)  # 同一提示词的多次采样各占一条缓存
        payload = json.dumps(parts, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self.data.get(key) if self.enabled else None

    def put(self, key: str, response: str, usage: tuple[int, int] | None = None, prob: float | None = None) -> None:
        if not self.enabled:
            return
        self.data[key] = response
        rec = {"key": key, "response": response}
        if usage:
            self.usage[key] = usage
            rec["usage"] = list(usage)
        if prob is not None:
            self.probs[key] = prob
            rec["prob"] = prob
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
    def __init__(self, task: TaskConfig, llm: BaseLLM, cache: LLMCache, semaphore: asyncio.Semaphore, max_chars: int = 300,
                 *, evidence: bool = False, view: str = "full", sampler: BaseLLM | None = None, samples: int = 1):
        self.task = task
        self.max_chars = max_chars
        self.evidence = evidence
        self.view = view
        self.sampler = sampler  # Self-Consistency 采样用的同一模型（温度不同）
        self.samples = samples if sampler is not None else 1
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

    def _seed(self, text: str) -> str | None:
        return prompt_seed(self.task, self.name, text)

    async def classify(self, text: str, examples=None) -> Prediction:
        user = build_classify_prompt(self.task, text, examples, self.max_chars, self._seed(text), self.evidence)
        first = await self._call(user, text)
        if self.samples <= 1:
            return first
        rest = await asyncio.gather(*(self._call(user, text, self.sampler, f"sample{k}") for k in range(1, self.samples)))
        return self._self_consistency([first, *rest])

    def _self_consistency(self, runs: list[Prediction]) -> Prediction:
        valid = [p for p in runs if p.ok]
        if not valid:
            return runs[0]
        count: dict[str, int] = {}
        conf: dict[str, float] = {}
        for p in valid:
            count[p.label] = count.get(p.label, 0) + 1
            conf[p.label] = conf.get(p.label, 0.0) + p.confidence
        label = max(count, key=lambda lab: (count[lab], conf[lab]))
        rep = next(p for p in valid if p.label == label)
        return Prediction(self.name, label, round(count[label] / len(runs), 3), rep.reason, raw=rep.raw,
                          samples=[p.label or "失败" for p in runs], evidence=rep.evidence, evidence_ok=rep.evidence_ok)

    async def review(self, text: str, own: Prediction | None, peers: list[Prediction], examples=None,
                     round_no: int = 1, devil: Prediction | None = None) -> Prediction:
        return await self._call(build_review_prompt(self.task, text, own, peers, examples, self.max_chars, self._seed(text),
                                                    self.view, self.evidence, round_no, devil), text)

    async def arbitrate(self, text: str, opinions: list[Prediction], examples=None) -> Prediction:
        return await self._call(build_arbiter_prompt(self.task, text, opinions, examples, self.max_chars, self._seed(text),
                                                     self.view, self.evidence), text)

    async def devil(self, text: str, majority: str, supporters: list[Prediction], examples=None) -> Prediction:
        return await self._call(build_devil_prompt(self.task, text, majority, supporters, examples, self.max_chars,
                                                   self._seed(text)), None)

    async def _call(self, user: str, text: str | None = None, llm: BaseLLM | None = None, tag: str = "") -> Prediction:
        llm = llm or self.llm
        cfg = llm.cfg
        key = LLMCache.make_key(cfg, SYSTEM_PROMPT, user, tag)
        raw = self.cache.get(key) if llm.cacheable else None
        from_cache = raw is not None
        usage = prob = None
        if from_cache:
            prob = self.cache.probs.get(key)
            self.stats["cache_hits"] += 1
            if llm.billable:
                tin, tout = self.cache.usage.get(key) or (estimate_tokens(SYSTEM_PROMPT + user), estimate_tokens(raw))
                self.stats["saved"] += cost_of(cfg, tin, tout)
        else:
            async with self.limiter.slot(), self.sem:
                self.stats["calls"] += 1
                try:
                    reply = await llm.complete(SYSTEM_PROMPT, user)
                except LLMError as e:
                    self.stats["errors"] += 1
                    return Prediction(self.name, None, error=str(e))
            raw = reply.text
            prob = label_prob(reply.tokens)
            if llm.billable:
                if reply.in_tokens is None or reply.out_tokens is None:
                    self.stats["estimated_calls"] += 1
                tin = reply.in_tokens if reply.in_tokens is not None else estimate_tokens(SYSTEM_PROMPT + user)
                tout = reply.out_tokens if reply.out_tokens is not None else estimate_tokens(raw)
                usage = (int(tin), int(tout))
                self.stats["in_tokens"] += usage[0]
                self.stats["out_tokens"] += usage[1]
                self.stats["cost"] += cost_of(cfg, *usage)

        try:
            label, conf, reason, evidence = parse_reply(raw, self.task.label_names, self.task)
        except ParseError as e:
            self.stats["errors"] += 1
            return Prediction(self.name, None, error=f"解析失败: {e}", raw=raw)

        if llm.cacheable and not from_cache:
            self.cache.put(key, raw, usage, prob)
        ev_ok = evidence_in_text(evidence, text) if self.evidence and text is not None else None
        return Prediction(self.name, label, conf, reason, raw=raw, prob=prob, evidence=evidence, evidence_ok=ev_ok)
