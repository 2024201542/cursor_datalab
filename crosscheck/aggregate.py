from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .classifier import Prediction


@dataclass
class VoteResult:
    label: str | None
    share: float  # 获胜标签的加权得票占比
    agree: int  # 投给获胜标签的模型数
    total: int  # 有效票数
    scores: dict[str, float] = field(default_factory=dict)


def weighted_vote(preds: list[Prediction], weights: dict[str, float]) -> VoteResult:
    valid = [p for p in preds if p.ok]
    if not valid:
        return VoteResult(None, 0.0, 0, 0)
    scores: dict[str, float] = defaultdict(float)
    for p in valid:
        scores[p.label] += weights.get(p.model, 1.0) * max(p.confidence, 0.05)
    label = max(scores, key=scores.get)
    agree = sum(p.label == label for p in valid)
    return VoteResult(label, scores[label] / sum(scores.values()), agree, len(valid), dict(scores))


@dataclass
class DSResult:
    labels: dict[str, str]  # item -> 聚合后的标签
    probs: dict[str, float]  # item -> 该标签的后验概率
    worker_accuracy: dict[str, float]  # 模型 -> 估计准确率
    confusion: dict[str, list[list[float]]]  # 模型 -> 混淆矩阵 [真实类][预测类]
    class_prior: list[float]


def dawid_skene(
    annotations: list[tuple[str, str, str]],
    labels: list[str],
    n_iter: int = 100,
    smoothing: float = 0.01,
    tol: float = 1e-6,
) -> DSResult:
    """Dawid-Skene EM：不需要标准答案，同时估计每条样本的真实标签和每个模型的混淆矩阵。

    annotations: [(样本id, 模型名, 标签), ...]
    """
    idx = {lab: k for k, lab in enumerate(labels)}
    K = len(labels)
    by_item: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for item, worker, label in annotations:
        if label in idx:
            by_item[item].append((worker, idx[label]))
    items = list(by_item)
    workers = sorted({w for anns in by_item.values() for w, _ in anns})
    if not items:
        return DSResult({}, {}, {}, {}, [])

    T: dict[str, list[float]] = {}
    for i in items:
        counts = [0.0] * K
        for _, lab in by_item[i]:
            counts[lab] += 1
        s = sum(counts)
        T[i] = [c / s for c in counts]

    prior = [1.0 / K] * K
    conf: dict[str, list[list[float]]] = {}
    for _ in range(n_iter):
        prior = [(sum(T[i][k] for i in items) + smoothing) / (len(items) + K * smoothing) for k in range(K)]
        conf = {w: [[smoothing] * K for _ in range(K)] for w in workers}
        for i in items:
            for w, lab in by_item[i]:
                for k in range(K):
                    conf[w][k][lab] += T[i][k]
        for w in workers:
            for k in range(K):
                s = sum(conf[w][k])
                conf[w][k] = [v / s for v in conf[w][k]]

        change = 0.0
        for i in items:
            logp = [math.log(prior[k]) for k in range(K)]
            for w, lab in by_item[i]:
                for k in range(K):
                    logp[k] += math.log(conf[w][k][lab])
            m = max(logp)
            p = [math.exp(x - m) for x in logp]
            s = sum(p)
            p = [x / s for x in p]
            change = max(change, max(abs(a - b) for a, b in zip(p, T[i])))
            T[i] = p
        if change < tol:
            break

    out_labels, out_probs = {}, {}
    for i in items:
        k = max(range(K), key=lambda j: T[i][j])
        out_labels[i] = labels[k]
        out_probs[i] = T[i][k]
    worker_acc = {w: sum(prior[k] * conf[w][k][k] for k in range(K)) for w in workers}
    return DSResult(out_labels, out_probs, worker_acc, conf, prior)
