from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from .aggregate import weighted_vote
from .classifier import Classifier, LLMCache, Prediction
from .config import Config
from .llm import create_llm

STATUS_CONSENSUS = "consensus"
STATUS_MAJORITY = "majority"
STATUS_ARBITRATED = "arbitrated"
STATUS_HUMAN = "need_human"

STATUS_TEXT = {
    STATUS_CONSENSUS: "首轮一致通过",
    STATUS_MAJORITY: "复核后多数通过",
    STATUS_ARBITRATED: "仲裁模型决定",
    STATUS_HUMAN: "需人工审核",
}


@dataclass
class ItemResult:
    id: str
    text: str
    label: str | None  # 最终标签；need_human 时为建议标签
    status: str
    confidence: float
    round1: list[Prediction]
    round2: list[Prediction] = field(default_factory=list)
    arbiter: Prediction | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "status": self.status,
            "confidence": self.confidence,
            "note": self.note,
            "round1": [p.to_dict() for p in self.round1],
            "round2": [p.to_dict() for p in self.round2],
            "arbiter": self.arbiter.to_dict() if self.arbiter else None,
        }


class CrossCheckPipeline:
    """流程：多个模型独立分类 -> 全部一致则采纳 -> 否则交叉复核并加权投票 -> 仍不通过则仲裁 -> 仍不通过则转人工。"""

    def __init__(self, config: Config, weights: dict[str, float] | None = None):
        self.config = config
        self.weights = {m.name: m.weight for m in config.models}
        if weights:
            self.weights.update({k: v for k, v in weights.items() if k in self.weights})
        self.classifiers: list[Classifier] = []
        self.arbiter: Classifier | None = None

    async def __aenter__(self) -> CrossCheckPipeline:
        pc = self.config.pipeline
        self._http = httpx.AsyncClient(
            timeout=pc.timeout,
            limits=httpx.Limits(max_connections=pc.concurrency * 2),
        )
        self.cache = LLMCache(self.config.cache.path, self.config.cache.enabled)
        try:
            sem = asyncio.Semaphore(pc.concurrency)
            self.classifiers = [
                Classifier(self.config.task, create_llm(m, self._http, self.config), self.cache, sem)
                for m in self.config.models
            ]
            if self.config.arbiter and pc.use_arbiter:
                llm = create_llm(self.config.arbiter, self._http, self.config)
                self.arbiter = Classifier(self.config.task, llm, self.cache, sem)
        except Exception:
            await self.__aexit__()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self._http.aclose()
        self.cache.close()

    async def process(self, item_id: str, text: str) -> ItemResult:
        pc = self.config.pipeline

        r1 = list(await asyncio.gather(*(c.classify(text) for c in self.classifiers)))
        valid1 = [p for p in r1 if p.ok]
        if len(valid1) >= pc.min_votes and len({p.label for p in valid1}) == 1:
            conf = sum(p.confidence for p in valid1) / len(valid1)
            return ItemResult(item_id, text, valid1[0].label, STATUS_CONSENSUS, round(conf, 3), r1)

        # 首轮出现分歧时的处理策略：review 复核投票 / arbiter 直接仲裁 / human 直接转人工
        action = pc.disagreement_action
        r2: list[Prediction] = []
        final = r1
        if action == "review" and pc.cross_review and valid1:
            r2 = list(await asyncio.gather(*(
                c.review(text, r1[i], [p for j, p in enumerate(r1) if j != i and p.ok])
                for i, c in enumerate(self.classifiers)
            )))
            final = [b if b.ok else a for a, b in zip(r1, r2)]

        vote = weighted_vote(final, self.weights)
        if (action == "review" and vote.total >= pc.min_votes
                and vote.agree * 2 > vote.total and vote.share >= pc.accept_threshold):
            return ItemResult(item_id, text, vote.label, STATUS_MAJORITY, round(vote.share, 3), r1, r2)

        arb = None
        if action != "human" and self.arbiter is not None and vote.total > 0:
            arb = await self.arbiter.arbitrate(text, [p for p in final if p.ok])
            if arb.ok and arb.confidence >= pc.arbiter_threshold:
                return ItemResult(item_id, text, arb.label, STATUS_ARBITRATED, round(arb.confidence, 3), r1, r2, arb)

        suggestion = arb.label if arb is not None and arb.ok else vote.label
        if vote.total == 0:
            note = "所有模型调用失败"
        elif action == "human":
            note = "首轮分歧，按策略直接转人工"
        else:
            note = "模型分歧未解决"
        return ItemResult(item_id, text, suggestion, STATUS_HUMAN, round(vote.share, 3), r1, r2, arb, note)

    async def run(
        self,
        items: list[dict],
        progress: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ItemResult]:
        item_sem = asyncio.Semaphore(self.config.pipeline.concurrency * 2)
        total, done, start = len(items), 0, time.monotonic()
        step = max(1, total // 20)

        async def one(item: dict) -> ItemResult:
            nonlocal done
            async with item_sem:
                res = await self.process(item["id"], item["text"])
            done += 1
            if on_progress is not None:
                on_progress(done, total)
            if progress and (done % step == 0 or done == total):
                print(f"\r进度 {done}/{total}  用时 {time.monotonic() - start:.1f}s", end="", file=sys.stderr, flush=True)
            return res

        results = await asyncio.gather(*(one(it) for it in items))
        if progress and total:
            print(file=sys.stderr)
        return list(results)

    def stats(self) -> dict[str, dict[str, int]]:
        out = {c.name: dict(c.stats) for c in self.classifiers}
        if self.arbiter is not None:
            out[f"{self.arbiter.name}(仲裁)"] = dict(self.arbiter.stats)
        return out
