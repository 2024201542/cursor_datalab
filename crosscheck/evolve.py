"""闭环里不花钱的两件事：该先审哪几条（主动学习），以及金标准里哪些标签可能标错了。

标签噪声用的是 cleanlab 的 confident learning 思路：几个模型很有把握地一致同意另一个标签时，
金标准更值得复查。没有引入 cleanlab。
"""
from __future__ import annotations

import random
from collections import Counter


def _preds(rec: dict) -> list[dict]:
    return [p for p in rec.get("round1") or [] if p.get("label") and not p.get("error")]


def uncertainty(rec: dict) -> float:
    """越大越该先看。标签越分散、自报置信度越低，分数越高。"""
    preds = _preds(rec)
    if len(preds) < 2:
        return 1.0
    labels = [p["label"] for p in preds]
    top = Counter(labels).most_common(1)[0][1] / len(labels)
    conf = sum(float(p.get("confidence") or 0) for p in preds) / len(preds)
    return round((1 - top) * 0.65 + (1 - conf) * 0.35, 4)


def why(rec: dict) -> str:
    preds = _preds(rec)
    if len(preds) < 2:
        return "有效回答不足 2 个"
    n = len(set(p["label"] for p in preds))
    conf = sum(float(p.get("confidence") or 0) for p in preds) / len(preds)
    if n > 1:
        return f"{len(preds)} 个模型给出 {n} 种标签，平均置信度 {conf:.0%}"
    return f"模型一致，平均置信度 {conf:.0%}"


def prioritize(records: list[dict], k: int | None = None, sim_threshold: float = 0.9) -> list[dict]:
    """不确定度从高到低；和已选样本过于相似的先往后放，避免连着审十几条几乎一样的文本。"""
    ranked = sorted(records, key=lambda r: (-uncertainty(r), str(r.get("id"))))
    if len(ranked) < 3 or len(ranked) > 3000:
        return ranked[:k] if k else ranked
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return ranked[:k] if k else ranked
    mat = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=20_000, min_df=1).fit_transform(
        [r.get("text") or "" for r in ranked])
    chosen_at: list[int] = []
    blocked: set[int] = set()
    for i in range(len(ranked)):
        if i in blocked:
            continue
        chosen_at.append(i)
        if k and len(chosen_at) >= k:
            break
        sims = cosine_similarity(mat[i], mat).ravel()
        blocked.update(j for j, s in enumerate(sims) if s >= sim_threshold)
    chosen = {i for i in chosen_at}
    ordered = [ranked[i] for i in chosen_at] + [r for i, r in enumerate(ranked) if i not in chosen]
    return ordered[:k] if k else ordered


def _errors(records: list[dict]) -> tuple[list[dict], set]:
    labeled = [r for r in records if r.get("gold")]
    return labeled, {r.get("id") for r in labeled if r.get("label") != r.get("gold")}


def audit(records: list[dict], ks: tuple[int, ...] = (10, 20, 30, 50)) -> list[dict]:
    """前 k 条能抓住多少「系统标签 ≠ 标准答案」。随机抽 k 条的期望是 k / 总数 × 错误数。"""
    labeled, errors = _errors(records)
    n, n_err = len(labeled), len(errors)
    if n < 2 or not n_err:
        return []
    disagree_first = sorted(labeled, key=lambda r: (len({p["label"] for p in _preds(r)}) <= 1, -uncertainty(r), str(r.get("id"))))
    orders = {
        "不确定度（相似的往后放）": [r.get("id") for r in prioritize(labeled)],
        "先看有分歧的": [r.get("id") for r in disagree_first],
    }
    rows = []
    for name, ids in orders.items():
        for k in ks:
            if k > n:
                continue
            hit = sum(i in errors for i in ids[:k])
            rows.append({
                "排序": name,
                "审核条数": k,
                "抓住的错": hit,
                "错误总数": n_err,
                "召回": round(hit / n_err, 4),
                "随机期望": round(k * n_err / n, 2),
            })
    return rows


def noise_candidates(records: list[dict], min_agree: int = 2, min_conf: float = 0.75) -> list[dict]:
    """至少 min_agree 个模型以不低于 min_conf 的平均置信度投了同一个、且不同于标准答案的标签。"""
    rows = []
    for r in records:
        gold = r.get("gold")
        preds = _preds(r)
        if not gold or len(preds) < min_agree:
            continue
        groups: dict[str, list[float]] = {}
        for p in preds:
            if p["label"] != gold:
                groups.setdefault(p["label"], []).append(float(p.get("confidence") or 0))
        if not groups:
            continue
        lab, confs = max(groups.items(), key=lambda kv: (len(kv[1]), sum(kv[1]) / len(kv[1])))
        if len(confs) < min_agree:
            continue
        mean_c = sum(confs) / len(confs)
        if mean_c < min_conf:
            continue
        rows.append({
            "id": r.get("id"),
            "text": r.get("text") or "",
            "gold": gold,
            "suspect": lab,
            "同意该标签的模型数": len(confs),
            "模型数": len(preds),
            "平均置信度": round(mean_c, 3),
            "分数": round(len(confs) / len(preds) * mean_c, 3),
        })
    rows.sort(key=lambda x: (-x["分数"], -x["同意该标签的模型数"], str(x["id"])))
    return rows


def local_gain(train_texts: list[str], train_labels: list[str], pool: list[dict], k: int,
               order: str = "active", seed: int = 0) -> float | None:
    """把 pool 里 k 条的标准答案加进训练集，重新训练本地小模型，在剩下的 pool 上算准确率。

    active：按不确定度挑；random：随机挑。pool 每条需要 text、gold，以及 round1（用来算不确定度）。
    """
    from .local_model import TextClassifier

    usable = [r for r in pool if r.get("text") and r.get("gold")]
    if k < 1 or len(usable) <= k + 1:
        return None
    if order == "active":
        picked = prioritize(usable, k)
    else:
        rng = random.Random(seed)
        picked = rng.sample(usable, k)
    picked_ids = {r.get("id") for r in picked}
    rest = [r for r in usable if r.get("id") not in picked_ids]
    if len({*train_labels, *(r["gold"] for r in picked)}) < 2 or not rest:
        return None
    clf = TextClassifier(list(train_texts) + [r["text"] for r in picked],
                         list(train_labels) + [r["gold"] for r in picked])
    return sum(clf.predict(r["text"])[0] == r["gold"] for r in rest) / len(rest)
