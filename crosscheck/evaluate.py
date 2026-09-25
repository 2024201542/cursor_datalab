from __future__ import annotations

import math
from collections import Counter

from .aggregate import weighted_vote
from .classifier import Prediction
from .pipeline import STATUS_HUMAN, STATUS_TEXT, ItemResult


def fleiss_kappa(rows: list[list[str]], labels: list[str]) -> float | None:
    """rows: 每条样本各模型给出的标签（每行长度必须相同）。"""
    rows = [r for r in rows if r]
    if not rows:
        return None
    n = len(rows[0])
    if n < 2 or any(len(r) != n for r in rows):
        return None
    N = len(rows)
    counts = [Counter(r) for r in rows]
    p_bar = sum((sum(c[lab] ** 2 for lab in labels) - n) / (n * (n - 1)) for c in counts) / N
    p_e = sum((sum(c[lab] for c in counts) / (N * n)) ** 2 for lab in labels)
    if p_e >= 1:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def log_odds_weight(acc: float, n_labels: int) -> float:
    """独立分类器加权投票的最优权重 log(acc*(K-1)/(1-acc))，准确率不高于随机猜测时给一个很小的权重。"""
    acc = min(max(acc, 1.0 / n_labels + 0.01), 0.99)
    return round(max(math.log(acc * (n_labels - 1) / (1 - acc)), 0.05), 3)


def _acc(preds_by_item: list[Prediction | None], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(1 for p, g in zip(preds_by_item, gold) if p is not None and p.ok and p.label == g) / len(gold)


def _find(preds: list[Prediction], name: str) -> Prediction | None:
    return next((p for p in preds if p.model == name), None)


def majority_vote(preds: list[Prediction]) -> str | None:
    """简单多数投票，平票时取置信度之和更高的标签。"""
    valid = [p for p in preds if p.ok]
    if not valid:
        return None
    count, conf = Counter(), Counter()
    for p in valid:
        count[p.label] += 1
        conf[p.label] += p.confidence
    return max(count, key=lambda lab: (count[lab], conf[lab]))


def _per_class(pred_labels: list[str | None], gold_list: list[str], labels: list[str]) -> dict[str, float | None]:
    out = {}
    for lab in labels:
        idx = [i for i, g in enumerate(gold_list) if g == lab]
        out[lab] = sum(pred_labels[i] == lab for i in idx) / len(idx) if idx else None
    return out


def _label_acc(pred_labels: list[str | None], gold_list: list[str]) -> float:
    return sum(p == g for p, g in zip(pred_labels, gold_list)) / len(gold_list) if gold_list else 0.0


def evaluate(
    results: list[ItemResult],
    gold: dict[str, str],
    model_names: list[str],
    labels: list[str],
    weights: dict[str, float] | None = None,
) -> dict:
    results = [r for r in results if r.id in gold]
    gold_list = [gold[r.id] for r in results]
    N = len(results)

    per_model = {}
    strategy_preds: dict[str, list[str | None]] = {}
    for m in model_names:
        r1 = [_find(r.round1, m) for r in results]
        # 首轮就一致的样本没有复核轮次，复核后准确率按首轮结果计算
        r2 = [(_find(r.round2, m) if r.round2 else None) or _find(r.round1, m) for r in results]
        per_model[m] = {"round1_acc": _acc(r1, gold_list), "round2_acc": _acc(r2, gold_list)}
        strategy_preds[m] = [p.label if p is not None and p.ok else None for p in r1]

    strategy_preds["多数投票"] = [majority_vote(r.round1) for r in results]
    strategy_preds["加权投票"] = [weighted_vote(r.round1, weights or {}).label for r in results]
    strategy_preds["互检系统"] = [r.label for r in results]

    strategies = {
        name: {
            "acc": _label_acc(preds, gold_list),
            "per_class": _per_class(preds, gold_list, labels),
            "kind": "model" if name in model_names else "ensemble",
        }
        for name, preds in strategy_preds.items()
    }
    any_correct = sum(
        any(p.ok and p.label == g for p in r.round1) for r, g in zip(results, gold_list)
    ) / N if N else 0.0

    items = []
    for i, (r, g) in enumerate(zip(results, gold_list)):
        items.append({
            "id": r.id, "text": r.text, "gold": g, "status": r.status,
            **{name: preds[i] for name, preds in strategy_preds.items()},
        })

    auto = [(r, g) for r, g in zip(results, gold_list) if r.status != STATUS_HUMAN]
    by_status = {}
    for status in STATUS_TEXT:
        sub = [(r, g) for r, g in zip(results, gold_list) if r.status == status]
        if sub:
            by_status[status] = {"count": len(sub), "acc": sum(r.label == g for r, g in sub) / len(sub)}

    kappa_rows = []
    for r in results:
        row = [p.label for p in (_find(r.round1, m) for m in model_names) if p is not None and p.ok]
        if len(row) == len(model_names):
            kappa_rows.append(row)

    confusion = {g: Counter() for g in labels}
    for r, g in zip(results, gold_list):
        if g in confusion:
            confusion[g][r.label or "无"] += 1

    return {
        "n": N,
        "labels": labels,
        "model_names": model_names,
        "per_model": per_model,
        "strategies": strategies,
        "any_correct": any_correct,
        "items": items,
        "system_acc_all": sum(r.label == g for r, g in zip(results, gold_list)) / N if N else 0.0,
        "auto_coverage": len(auto) / N if N else 0.0,
        "auto_acc": sum(r.label == g for r, g in auto) / len(auto) if auto else 0.0,
        "by_status": by_status,
        "fleiss_kappa_round1": fleiss_kappa(kappa_rows, labels),
        "confusion": {g: dict(c) for g, c in confusion.items()},
        "suggested_weights": {m: log_odds_weight(v["round1_acc"], len(labels)) for m, v in per_model.items()},
        "errors": [
            {"id": r.id, "text": r.text, "gold": g, "pred": r.label, "status": r.status}
            for r, g in zip(results, gold_list) if r.label != g
        ],
    }


def format_report(rep: dict, labels: list[str]) -> str:
    pct = lambda x: f"{x * 100:.1f}%"
    lines = [f"评估样本数: {rep['n']}", "", "[准确率对比]"]
    for name, v in rep["strategies"].items():
        tag = "单模型" if v["kind"] == "model" else "组合"
        lines.append(f"  {tag:<4} {name:<10} {pct(v['acc']):>7}")
    lines.append(f"  参考   任一模型答对（理论上限） {pct(rep['any_correct'])}")

    lines += ["", "[单个模型：交叉复核前后]"]
    for m, v in rep["per_model"].items():
        lines.append(f"  {m:<12} 首轮 {pct(v['round1_acc']):>7}   复核后 {pct(v['round2_acc']):>7}")

    k = rep["fleiss_kappa_round1"]
    lines += [
        "",
        "[互检系统]",
        f"  整体准确率（含人工建议标签）: {pct(rep['system_acc_all'])}",
        f"  自动采纳比例: {pct(rep['auto_coverage'])}   自动采纳部分的准确率: {pct(rep['auto_acc'])}",
        f"  首轮模型间一致性 Fleiss' Kappa: {'无法计算' if k is None else f'{k:.3f}'}",
        "",
        "[按处理状态]",
    ]
    for s, v in rep["by_status"].items():
        lines.append(f"  {STATUS_TEXT[s]:<10} {v['count']:>4} 条   准确率 {pct(v['acc'])}")

    cols = labels + (["无"] if any("无" in row for row in rep["confusion"].values()) else [])
    lines += ["", "[混淆矩阵] 行=真实标签 列=系统标签", "  " + "真实\\预测".ljust(8) + "".join(c.ljust(6) for c in cols)]
    for g in labels:
        row = rep["confusion"].get(g, {})
        lines.append("  " + g.ljust(8) + "".join(str(row.get(c, 0)).ljust(6) for c in cols))

    lines += ["", "[建议权重]（已写入 weights.json，classify 时用 --weights 加载）"]
    lines += [f"  {m}: {w}" for m, w in rep["suggested_weights"].items()]

    if rep["errors"]:
        lines += ["", f"[判错样本] 共 {len(rep['errors'])} 条"]
        for e in rep["errors"][:30]:
            lines.append(f"  #{e['id']} 真实={e['gold']} 系统={e['pred']} ({STATUS_TEXT[e['status']]})  {e['text'][:40]}")
    return "\n".join(lines)
