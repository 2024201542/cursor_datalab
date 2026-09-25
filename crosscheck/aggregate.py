from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .classifier import Prediction
from .config import split_labels


@dataclass
class VoteResult:
    label: str | None
    share: float  # 获胜标签的加权得票占比
    agree: int  # 投给获胜标签的模型数
    total: int  # 有效票数
    scores: dict[str, float] = field(default_factory=dict)


def _vote_weight(p: Prediction, weights: dict[str, float], evidence_penalty: float) -> float:
    w = weights.get(p.model, 1.0) * max(p.confidence, 0.05)
    return w * evidence_penalty if p.evidence_ok is False else w


def weighted_vote(preds: list[Prediction], weights: dict[str, float], evidence_penalty: float = 1.0) -> VoteResult:
    """evidence_penalty：要求证据时，引用不在原文中的判断乘以该系数。"""
    valid = [p for p in preds if p.ok]
    if not valid:
        return VoteResult(None, 0.0, 0, 0)
    scores: dict[str, float] = defaultdict(float)
    for p in valid:
        scores[p.label] += _vote_weight(p, weights, evidence_penalty)
    label = max(scores, key=scores.get)
    agree = sum(p.label == label for p in valid)
    return VoteResult(label, scores[label] / sum(scores.values()), agree, len(valid), dict(scores))


def multilabel_vote(preds: list[Prediction], weights: dict[str, float], join, evidence_penalty: float = 1.0) -> VoteResult:
    """多标签投票：每个类别单独表决，加权过半的类别入选（一个都没过半时取得分最高的）。

    share 取各类别表决中最弱的那一票的占比，agree 为给出的标签集合与结果完全相同的模型数。
    """
    valid = [p for p in preds if p.ok]
    if not valid:
        return VoteResult(None, 0.0, 0, 0)
    total = 0.0
    scores: dict[str, float] = defaultdict(float)
    for p in valid:
        w = _vote_weight(p, weights, evidence_penalty)
        total += w
        for lab in split_labels(p.label):
            scores[lab] += w
    frac = {lab: s / total for lab, s in scores.items()}
    chosen = [lab for lab, f in frac.items() if f > 0.5] or [max(frac, key=frac.get)]
    label = join(chosen)
    share = min(max(f, 1 - f) for f in frac.values())
    return VoteResult(label, share, sum(p.label == label for p in valid), len(valid), frac)


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


@dataclass
class CrowdResult:
    labels: dict[str, str]  # item -> 聚合后的标签
    probs: dict[str, float]  # item -> 该标签的后验（或得票）概率
    skill: dict[str, float]  # 模型 -> 估计的能力（各算法含义不同，越大越可靠）


def _group(annotations: list[tuple[str, str, str]], labels: list[str]) -> tuple[dict[str, list[tuple[str, int]]], list[str]]:
    idx = {lab: k for k, lab in enumerate(labels)}
    by_item: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for item, worker, label in annotations:
        if label in idx:
            by_item[item].append((worker, idx[label]))
    return by_item, sorted({w for anns in by_item.values() for w, _ in anns})


def _finish(post: dict[str, list[float]], labels: list[str], skill: dict[str, float]) -> CrowdResult:
    out_l, out_p = {}, {}
    for i, p in post.items():
        k = max(range(len(labels)), key=lambda j: p[j])
        out_l[i], out_p[i] = labels[k], p[k]
    return CrowdResult(out_l, out_p, skill)


def _normalize(logp: list[float]) -> list[float]:
    m = max(logp)
    p = [math.exp(x - m) for x in logp]
    s = sum(p)
    return [x / s for x in p]


def wawa(annotations: list[tuple[str, str, str]], labels: list[str]) -> CrowdResult:
    """Wawa（Worker Agreement with Aggregate）：先多数投票，再以每个模型与多数结果的一致率为权重重新投票。"""
    by_item, workers = _group(annotations, labels)
    K = len(labels)

    def vote(weight: dict[str, float]) -> dict[str, list[float]]:
        out = {}
        for i, anns in by_item.items():
            s = [0.0] * K
            for w, lab in anns:
                s[lab] += weight[w]
            total = sum(s) or 1.0
            out[i] = [x / total for x in s]
        return out

    mv = vote({w: 1.0 for w in workers})
    skill = {}
    for w in workers:
        hits = [max(range(K), key=lambda j: mv[i][j]) == lab for i, anns in by_item.items() for ww, lab in anns if ww == w]
        skill[w] = sum(hits) / len(hits) if hits else 0.0
    return _finish(vote(skill), labels, skill)


def mace(annotations: list[tuple[str, str, str]], labels: list[str], n_iter: int = 50, smoothing: float = 0.5) -> CrowdResult:
    """MACE（Hovy et al. 2013）：每个模型以概率 θ 认真作答、否则按自己的偏好分布 ξ “乱答”。

    EM 同时估计 θ（能力）、ξ（偏好）和真实标签。适合识别总是偏向某个类别的模型。
    """
    by_item, workers = _group(annotations, labels)
    K = len(labels)
    if not by_item:
        return CrowdResult({}, {}, {})
    theta = {w: 0.8 for w in workers}
    xi = {w: [1.0 / K] * K for w in workers}
    post: dict[str, list[float]] = {}
    for _ in range(n_iter):
        for i, anns in by_item.items():
            logp = [0.0] * K
            for w, a in anns:
                for k in range(K):
                    logp[k] += math.log(theta[w] * (a == k) + (1 - theta[w]) * xi[w][a] + 1e-12)
            post[i] = _normalize(logp)
        honest = {w: smoothing for w in workers}
        n = {w: 2 * smoothing for w in workers}
        spam = {w: [smoothing / K] * K for w in workers}
        for i, anns in by_item.items():
            for w, a in anns:
                # 该条标注是“认真作答”的后验：只有真实标签恰为 a 时才可能
                good = post[i][a] * theta[w] / (theta[w] + (1 - theta[w]) * xi[w][a] + 1e-12)
                honest[w] += good
                n[w] += 1
                spam[w][a] += 1 - good
        theta = {w: min(max(honest[w] / n[w], 0.01), 0.99) for w in workers}
        xi = {w: [x / sum(spam[w]) for x in spam[w]] for w in workers}
    return _finish(post, labels, theta)


def glad(annotations: list[tuple[str, str, str]], labels: list[str], n_iter: int = 30, lr: float = 0.1) -> CrowdResult:
    """GLAD（Whitehill et al. 2009）：答对的概率 = sigmoid(模型能力 α × 样本容易度 β)，答错时均匀落在其余类别。

    同时估计每个模型的能力和每条样本的难度；每条样本只有 3~4 个模型作答时，难度的估计比较粗。
    """
    by_item, workers = _group(annotations, labels)
    K = len(labels)
    if not by_item or K < 2:
        return CrowdResult({}, {}, {})
    alpha = {w: 1.0 for w in workers}
    logb = {i: 0.0 for i in by_item}
    sig = lambda x: 1 / (1 + math.exp(-max(min(x, 30), -30)))
    post: dict[str, list[float]] = {}
    for _ in range(n_iter):
        for i, anns in by_item.items():
            b = math.exp(logb[i])
            logp = [0.0] * K
            for w, a in anns:
                s = sig(alpha[w] * b)
                for k in range(K):
                    logp[k] += math.log(s if a == k else (1 - s) / (K - 1) + 1e-12)
            post[i] = _normalize(logp)
        for _ in range(5):  # M 步：梯度上升，α ~ N(1, 1)、log β ~ N(0, 1) 作为先验防止发散
            g_a = {w: -(alpha[w] - 1) for w in workers}
            g_b = {i: -logb[i] for i in by_item}
            for i, anns in by_item.items():
                b = math.exp(logb[i])
                for w, a in anns:
                    d = post[i][a] - sig(alpha[w] * b)
                    g_a[w] += d * b
                    g_b[i] += d * alpha[w] * b
            alpha = {w: alpha[w] + lr * g_a[w] for w in workers}
            logb = {i: max(min(logb[i] + lr * g_b[i], 3.0), -3.0) for i in by_item}
    return _finish(post, labels, alpha)
