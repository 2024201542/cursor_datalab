"""对比三个经济学数据集上 基线 / 动态示例 / 混合投票 三个版本的效果。

用法：python scripts/compare_econ.py        结果写入 output/econ_compare.txt 和 output/econ_compare.json
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {"climate": "年报气候段落", "fomc": "FOMC 鹰鸽", "finfe": "股吧情绪"}
VARIANTS = {"": "基线", "_fewshot": "动态示例", "_hybrid": "动态示例+本地小模型"}
LLMS = ("deepseek", "kimi", "qwen")


def load(name: str, variant: str):
    gold = {r["id"]: r["label"] for r in csv.DictReader(open(ROOT / f"data/econ_{name}_gold.csv", encoding="utf-8-sig"))}
    path = ROOT / f"output/econ_{name}{variant}/eval.jsonl"
    if not path.exists():
        return None, gold
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows, gold


def votes(row, models=None) -> dict[str, str]:
    return {p["model"]: p["label"] for p in row["round1"] if p.get("label") and (models is None or p["model"] in models)}


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = {g for g, _ in pairs}
    f1s = []
    for c in labels:
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def policy(rows, gold, need: int, models=None):
    """同意票数 >= need 的样本自动采纳，其余转人工。返回 覆盖率、自动采纳准确率、人工兜底后整体准确率。"""
    auto = ok = 0
    for r in rows:
        v = votes(r, models)
        if not v:
            continue
        lab, cnt = Counter(v.values()).most_common(1)[0]
        if cnt >= need:
            auto += 1
            ok += lab == gold[r["id"]]
    n = len(rows)
    return {"coverage": auto / n, "auto_acc": ok / auto if auto else 0.0, "with_human": (ok + n - auto) / n}


def summarize(name: str, variant: str):
    rows, gold = load(name, variant)
    if rows is None:
        return None
    n = len(rows)
    models = sorted({p["model"] for r in rows for p in r["round1"]})
    single = {m: sum(votes(r).get(m) == gold[r["id"]] for r in rows) / n for m in models}
    maj3 = []
    for r in rows:
        v = votes(r, LLMS)
        maj3.append(Counter(v.values()).most_common(1)[0][0] if v else "")
    system = [(gold[r["id"]], r["label"]) for r in rows]
    cons = [r for r in rows if r["status"] == "consensus"]
    out = {
        "n": n,
        "single": single,
        "llm_majority": sum(p == gold[r["id"]] for p, r in zip(maj3, rows)) / n,
        "any_llm_correct": sum(gold[r["id"]] in votes(r, LLMS).values() for r in rows) / n,
        "system_acc": sum(g == p for g, p in system) / n,
        "system_f1": macro_f1(system),
        "consensus_rate": len(cons) / n,
        "consensus_acc": sum(r["label"] == gold[r["id"]] for r in cons) / len(cons) if cons else 0.0,
        "policies": {
            "三个大模型全一致": policy(rows, gold, 3, LLMS),
            "三个大模型至少两票": policy(rows, gold, 2, LLMS),
        },
    }
    out["with_human"] = (sum(r["label"] == gold[r["id"]] for r in cons) + n - len(cons)) / n
    if "local" in models:
        out["policies"]["四票全一致"] = policy(rows, gold, 4)
        out["policies"]["四票至少三票"] = policy(rows, gold, 3)
    return out


def main():
    result, lines = {}, []
    for name, zh in DATASETS.items():
        lines.append(f"\n==================== {zh}（{name}） ====================")
        for v, vz in VARIANTS.items():
            s = summarize(name, v)
            if s is None:
                continue
            result[f"{name}{v}"] = s
            singles = "  ".join(f"{m} {a:.1%}" for m, a in s["single"].items())
            lines += [
                f"\n[{vz}]  单模型：{singles}",
                f"  三模型多数票 {s['llm_majority']:.1%} | 任一模型答对 {s['any_llm_correct']:.1%} | "
                f"系统全自动 {s['system_acc']:.1%}（宏 F1 {s['system_f1']:.3f}）",
                f"  首轮一致 {s['consensus_rate']:.1%} 的样本，准确率 {s['consensus_acc']:.1%}；"
                f"其余交人工后整体 {s['with_human']:.1%}",
            ]
            for pn, p in s["policies"].items():
                lines.append(f"    策略「{pn}」自动采纳 {p['coverage']:.1%}，准确率 {p['auto_acc']:.1%}，人工兜底后 {p['with_human']:.1%}")
    text = "\n".join(lines)
    (ROOT / "output/econ_compare.txt").write_text(text, encoding="utf-8-sig")
    (ROOT / "output/econ_compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
