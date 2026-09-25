from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import httpx

from .aggregate import VoteResult, multilabel_vote, weighted_vote
from .classifier import Classifier, LLMCache, Prediction
from .config import Config, parent_of
from .llm import LLMError, create_llm
from .local_model import get_bank
from .prompts import show_label

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
    debate: list[list[Prediction]] = field(default_factory=list)  # 多轮辩论时每一轮的复核结果（最后一轮即 round2）
    devil: Prediction | None = None  # 魔鬼代言人的反方意见
    jury: list[Prediction] = field(default_factory=list)  # 评审团成员各自的判断

    def to_dict(self) -> dict:
        d = {
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
        if self.debate:
            d["debate"] = [[p.to_dict() for p in rnd] for rnd in self.debate]
        if self.devil:
            d["devil"] = self.devil.to_dict()
        if self.jury:
            d["jury"] = [p.to_dict() for p in self.jury]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ItemResult:
        keep = set(Prediction.__dataclass_fields__)
        pred = lambda p: Prediction(**{k: v for k, v in p.items() if k in keep})
        return cls(d["id"], d["text"], d["label"], d["status"], d["confidence"],
                   [pred(p) for p in d["round1"]], [pred(p) for p in d.get("round2") or []],
                   pred(d["arbiter"]) if d.get("arbiter") else None, d.get("note", ""),
                   [[pred(p) for p in rnd] for rnd in d.get("debate") or []],
                   pred(d["devil"]) if d.get("devil") else None,
                   [pred(p) for p in d.get("jury") or []])


def load_calibration(config: Config):
    """读取 pipeline.calibration 指向的校准文件，并检查类别是否与当前任务一致。"""
    from .calibrate import Calibration

    path = config.pipeline.calibration
    if not path:
        return None
    try:
        cal = Calibration.load(path)
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise LLMError(f"无法读取校准文件 {path}：{e}") from e
    if set(cal.labels) != set(config.task.label_names):
        raise LLMError(f"校准文件 {path} 的类别 {cal.labels} 与当前任务的类别不一致，请用当前任务的金标准重新校准")
    return cal


class CrossCheckPipeline:
    """流程：多个模型独立分类 -> 全部一致则采纳 -> 否则交叉复核并加权投票 -> 仍不通过则仲裁 -> 仍不通过则转人工。"""

    def __init__(self, config: Config, weights: dict[str, float] | None = None):
        self.config = config
        self.weights = {m.name: m.weight for m in config.models}
        if weights:
            self.weights.update({k: v for k, v in weights.items() if k in self.weights})
        self.classifiers: list[Classifier] = []
        self.arbiter: Classifier | None = None
        self.jury: list[Classifier] = []
        self.bank = None
        self.calibration = None

    async def __aenter__(self) -> CrossCheckPipeline:
        pc = self.config.pipeline
        self._http = httpx.AsyncClient(
            timeout=pc.timeout,
            limits=httpx.Limits(max_connections=pc.concurrency * 2),
        )
        self.cache = LLMCache(self.config.cache.path, self.config.cache.enabled)
        try:
            fs = self.config.fewshot
            if fs.enabled:
                try:
                    if fs.retriever == "embedding":
                        from .embed import get_vector_bank
                        self.bank = get_vector_bank(fs, tuple(self.config.task.label_names))
                    else:
                        self.bank = get_bank(fs.path, tuple(self.config.task.label_names), fs.text_col, fs.label_col)
                except (OSError, ValueError, ImportError) as e:
                    raise LLMError(f"动态示例无法读取训练数据 {fs.path}：{e}") from e
            self.calibration = load_calibration(self.config)
            sem = asyncio.Semaphore(pc.concurrency)
            opts = {"evidence": pc.require_evidence, "view": pc.review_view}

            def make(m, sampling: bool = False) -> Classifier:
                sampler = None
                if (sampling and pc.samples > 1 and m.provider != "local"
                        and (not pc.sample_models or m.name in pc.sample_models)):
                    sampler = create_llm(replace(m, temperature=pc.sample_temperature), self._http, self.config)
                return Classifier(self.config.task, create_llm(m, self._http, self.config), self.cache, sem,
                                  fs.max_chars, sampler=sampler, samples=pc.samples, **opts)

            self.classifiers = [make(m, sampling=True) for m in self.config.models]
            if self.config.arbiter and pc.use_arbiter:
                self.arbiter = make(self.config.arbiter)
            if pc.use_arbiter:
                self.jury = [make(j) for j in self.config.jury]
        except Exception:
            await self.__aexit__()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self._http.aclose()
        self.cache.close()

    # ------------------------------------------------------------ 投票
    @property
    def _penalty(self) -> float:
        pc = self.config.pipeline
        return pc.evidence_penalty if pc.require_evidence else 1.0

    def vote(self, preds: list[Prediction]) -> VoteResult:
        task = self.config.task
        if task.multi_label:
            return multilabel_vote(preds, self.weights, task.join, self._penalty)
        if self.calibration is not None:
            return self.calibration.vote(preds, self._penalty)
        return weighted_vote(preds, self.weights, self._penalty)

    def _calibrate(self, preds: list[Prediction]) -> None:
        if self.calibration is None:
            return
        for p in preds:
            if p.ok and p.calibrated is None:
                p.calibrated = self.calibration.calibrate(p.model, p.certainty)

    def _devil_model(self) -> Classifier | None:
        if self.arbiter is not None:
            return self.arbiter
        return next((c for c in self.classifiers if c.llm.cfg.provider != "local"), None)

    # ------------------------------------------------------------ 单条样本
    async def process(self, item_id: str, text: str) -> ItemResult:
        pc = self.config.pipeline
        ex = self.bank.nearest(text, self.config.fewshot.k) if self.bank is not None else None

        first = [c for c in self.classifiers if c.name in pc.cascade] if pc.cascade else self.classifiers
        r1 = list(await asyncio.gather(*(c.classify(text, ex) for c in first)))
        self._calibrate(r1)
        if len(first) < len(self.classifiers):
            th = pc.cascade_min_confidence
            if (all(p.ok for p in r1) and len({p.label for p in r1}) == 1
                    and all(p.certainty >= th for p in r1)):
                conf = sum(p.confidence for p in r1) / len(r1)
                names = " / ".join(c.name for c in first)
                kind = "校准后置信度" if self.calibration is not None else "置信度"
                conds = (["一致"] if len(first) > 1 else []) + ([f"{kind} ≥ {th:g}"] if th > 0 else [])
                cond = "且".join(conds)
                note = f"级联：{names} {cond}，未调用其余模型"
                return ItemResult(item_id, text, r1[0].label, STATUS_CONSENSUS, round(conf, 3), r1, note=note)
            rest = [c for c in self.classifiers if c not in first]
            more = list(await asyncio.gather(*(c.classify(text, ex) for c in rest)))
            self._calibrate(more)
            done = {p.model: p for p in r1 + more}
            r1 = [done[c.name] for c in self.classifiers]
        valid1 = [p for p in r1 if p.ok]
        unanimous = len(valid1) >= pc.min_votes and len({p.label for p in valid1}) == 1
        notes: list[str] = []
        skip_review = False
        if self.calibration is not None and pc.min_posterior > 0 and len(valid1) >= pc.min_votes:
            v = self.vote(r1)
            if v.share >= pc.min_posterior:
                status = STATUS_CONSENSUS if unanimous else STATUS_MAJORITY
                note = f"后验概率 {v.share:.2f} ≥ {pc.min_posterior:g}" + ("" if unanimous else "，首轮有分歧也直接采纳")
                return ItemResult(item_id, text, v.label, status, round(v.share, 3), r1, note=note)
            if unanimous:
                notes.append(f"首轮一致但后验概率 {v.share:.2f} < {pc.min_posterior:g}")
                skip_review = True  # 大家意见相同，复核没有意义，直接交给仲裁 / 人工
        elif unanimous:
            conf = sum(p.confidence for p in valid1) / len(valid1)
            return ItemResult(item_id, text, valid1[0].label, STATUS_CONSENSUS, round(conf, 3), r1)

        if self.config.task.hierarchical and valid1 and not unanimous:
            parents = {parent_of(p.label) for p in valid1}
            if len(parents) == 1:
                notes.append(f"一级类一致（{parents.pop()}），二级类有分歧")

        # 首轮出现分歧时的处理策略：review 复核投票 / arbiter 直接仲裁 / human 直接转人工
        action = pc.disagreement_action
        r2: list[Prediction] = []
        debate: list[list[Prediction]] = []
        devil = None
        final = r1
        if action == "review" and pc.cross_review and valid1 and not skip_review:
            dm = self._devil_model() if pc.devil_advocate else None
            if dm is not None:
                maj = self.vote(r1).label
                devil = await dm.devil(text, maj, [p for p in valid1 if p.label == maj], ex)
            for rnd in range(1, pc.debate_rounds + 1):
                got = list(await asyncio.gather(*(
                    c.review(text, final[i], [p for j, p in enumerate(final) if j != i and p.ok], ex,
                             rnd, devil if rnd == 1 else None)
                    for i, c in enumerate(self.classifiers)
                )))
                debate.append(got)
                final = [b if b.ok else a for a, b in zip(final, got)]
                if len({p.label for p in final if p.ok}) == 1:
                    break
            r2 = debate[-1]
            if pc.debate_rounds > 1:
                notes.append(f"辩论 {len(debate)} 轮" + ("后达成一致" if len({p.label for p in final if p.ok}) == 1 else ""))
            else:
                debate = []

        vote = self.vote(final)
        note = "；".join(notes)
        if (action == "review" and not skip_review and vote.total >= pc.min_votes
                and vote.agree * 2 > vote.total and vote.share >= pc.accept_threshold):
            return ItemResult(item_id, text, vote.label, STATUS_MAJORITY, round(vote.share, 3), r1, r2,
                              note=note, debate=debate, devil=devil)

        arb, jury = None, []
        if action != "human" and vote.total > 0:
            opinions = [p for p in final if p.ok]
            if self.jury:
                jury = list(await asyncio.gather(*(j.arbitrate(text, opinions, ex) for j in self.jury)))
                arb = self._jury_verdict(jury)
                if arb.ok and arb.confidence >= pc.jury_threshold:
                    note = "；".join(notes + [f"评审团 {arb.reason.split('：')[0]}"])
                    return ItemResult(item_id, text, arb.label, STATUS_ARBITRATED, round(arb.confidence, 3),
                                      r1, r2, arb, note, debate, devil, jury)
            elif self.arbiter is not None:
                arb = await self.arbiter.arbitrate(text, opinions, ex)
                if arb.ok and arb.confidence >= pc.arbiter_threshold:
                    return ItemResult(item_id, text, arb.label, STATUS_ARBITRATED, round(arb.confidence, 3),
                                      r1, r2, arb, note, debate, devil)

        suggestion = arb.label if arb is not None and arb.ok else vote.label
        if vote.total == 0:
            reason = "所有模型调用失败"
        elif action == "human":
            reason = "首轮分歧，按策略直接转人工" if not skip_review else "按策略转人工"
        else:
            reason = "模型分歧未解决" if not skip_review else "仲裁未通过"
        note = "；".join(notes + [reason])
        return ItemResult(item_id, text, suggestion, STATUS_HUMAN, round(vote.share, 3), r1, r2, arb, note,
                          debate, devil, jury)

    def _jury_verdict(self, jury: list[Prediction]) -> Prediction:
        """评审团按人数投票：同意人数最多的标签胜出，置信度 = 同意人数 / 评审团人数（调用失败的成员算不同意）。"""
        valid = [p for p in jury if p.ok]
        if not valid:
            return Prediction("评审团", None, error="评审团成员全部调用失败")
        count = Counter(p.label for p in valid)
        conf = Counter()
        for p in valid:
            conf[p.label] += p.confidence
        label = max(count, key=lambda lab: (count[lab], conf[lab]))
        detail = "；".join(f"{p.model}:{show_label(p.label) if p.ok else '失败'}" for p in jury)
        return Prediction("评审团", label, round(count[label] / len(jury), 3),
                          f"{count[label]}/{len(jury)} 同意：{detail}")

    async def run(
        self,
        items: list[dict],
        progress: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        on_result: Callable[[ItemResult], None] | None = None,
    ) -> list[ItemResult]:
        """on_result 在每条样本处理完时调用（用于写断点文件）。"""
        prep = getattr(self.bank, "prepare", None)
        if prep is not None:
            prep([it["text"] for it in items])
        item_sem = asyncio.Semaphore(self.config.pipeline.concurrency * 2)
        total, done, start = len(items), 0, time.monotonic()
        step = max(1, total // 20)

        async def one(item: dict) -> ItemResult:
            nonlocal done
            async with item_sem:
                res = await self.process(item["id"], item["text"])
            if on_result is not None:
                on_result(res)
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
        for j in self.jury:
            out[f"{j.name}(评审团)"] = dict(j.stats)
        bank = self.bank
        tokens = int(getattr(bank, "tokens", 0) or 0)
        if tokens:
            price = float(getattr(getattr(bank, "fs", None), "embed_price", 0) or 0)
            out["向量检索"] = {"calls": int(getattr(bank, "new_texts", 0) or 0), "cache_hits": 0, "errors": 0,
                           "in_tokens": tokens, "out_tokens": 0, "estimated_calls": 0,
                           "cost": tokens * price / 1_000_000, "saved": 0.0}
        return out
