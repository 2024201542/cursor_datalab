"""在一小批有标准答案的样本上搜索更好的边界规则。

对应 DSPy 的两步：先让模型看着判错的样本提出新规则（指令提案），再只在调参集上比较，
留出集只用来报告最后胜出的那一版，不参与挑选。不引入 dspy 库。
"""
from __future__ import annotations

import asyncio
import copy
import random
from collections.abc import Callable

from .config import Config, LabelDef, ModelConfig, TaskConfig, config_from_dict
from .drafting import ask_json, drafting_models
from .pipeline import CrossCheckPipeline
from .prompts import render_guide
from .validation import split_dev_test


def stratified_take(rows: list[dict], n: int, key: str = "label", seed: int = 0) -> list[dict]:
    """按类别比例抽 n 条。每个有样本的类别至少 1 条（n 够的时候）。"""
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get(key, "")), []).append(r)
    for v in by.values():
        rng.shuffle(v)
    alloc = {k: len(v) / len(rows) * n for k, v in by.items()}
    sizes = {k: int(v) for k, v in alloc.items()}
    left = n - sum(sizes.values())
    for k, _ in sorted(alloc.items(), key=lambda kv: kv[1] - sizes[kv[0]], reverse=True):
        if left <= 0:
            break
        if sizes[k] < len(by[k]):
            sizes[k] += 1
            left -= 1
    if n >= len(by):
        for k in list(sizes):
            if sizes[k] == 0 and by[k] and left >= 0:
                donor = max((c for c in sizes if sizes[c] > 1), default=None)
                if donor is None:
                    break
                sizes[donor] -= 1
                sizes[k] = 1
    out = []
    for k, v in by.items():
        out.extend(v[:sizes[k]])
    return out


def _guide(task: dict) -> str:
    return render_guide(TaskConfig(
        name=task.get("name", ""), description=task.get("description", ""),
        labels=[LabelDef(lab["name"], lab.get("definition", ""), list(lab.get("examples") or []),
                         list(lab.get("counter_examples") or [])) for lab in task.get("labels") or []],
        rules=list(task.get("rules") or []),
    ))


def prompt_propose(task: dict, records: list[dict]) -> str:
    lines = []
    for i, r in enumerate(records[:12], 1):
        votes = "；".join(f"{p.get('model')}={p.get('label') or '失败'}" for p in r.get("round1") or [])
        text = str(r.get("text") or "").replace("\n", " ")
        if len(text) > 160:
            text = text[:160] + "…"
        lines.append(f"{i}. 标准答案：{r.get('gold')}；系统给出：{r.get('label')}；各模型：{votes}\n   文本：{text}")
    return (
        "下面这些样本的分类结果和标准答案不一致。请改边界规则，让人更容易判对，同时不要推翻已经判对的情况。\n"
        "要求：\n"
        "1. 给出 2 套彼此不同的完整规则，用来替换现有规则，不要只写“删掉第 3 条”这种差异说明；\n"
        "2. 只用现有的类别名称，不要新增类别，也不要改类别定义；\n"
        "3. 每条规则一句话，能直接执行。\n\n"
        '只输出这个 JSON：\n{"variants": [{"name": "不超过10个字", "rules": ["规则"]}]}\n\n'
        f"# 当前分类标准\n{_guide(task)}\n\n# 判错样本\n" + "\n".join(lines)
    )


def variants_from_obj(obj: dict, current: list[str], limit: int = 2) -> list[dict]:
    """丢掉和现有规则相同、或空的提案。"""
    out = []
    seen = {tuple(current)}
    for v in obj.get("variants") or []:
        if not isinstance(v, dict):
            continue
        rules = []
        for x in v.get("rules") or []:
            s = str(x).strip()
            if s and s not in rules and len(s) <= 200:
                rules.append(s)
        key = tuple(rules)
        if not rules or len(rules) > 20 or key in seen:
            continue
        seen.add(key)
        out.append({"name": str(v.get("name") or "提案").strip()[:20] or "提案", "rules": rules})
        if len(out) >= limit:
            break
    return out


def propose_variants(config: Config, model: ModelConfig, task: dict, records: list[dict]) -> list[dict]:
    errors = [r for r in records if r.get("gold") and r.get("label") != r.get("gold")]
    if len(errors) < 3:
        return []
    obj = ask_json(config, model, prompt_propose(task, errors), max_tokens=3000)
    return variants_from_obj(obj, list(task.get("rules") or []))


def pick_winner(rows: list[dict]) -> dict:
    """调参集准确率最高的一版；打平时留在前面的（调用方应把当前规则放第一个）。"""
    return max(rows, key=lambda r: (r["dev_acc"], -rows.index(r)))


async def _accuracy(raw: dict, items: list[dict]) -> tuple[float, float, list]:
    config = config_from_dict(copy.deepcopy(raw))
    async with CrossCheckPipeline(config) as pipe:
        results = await pipe.run([{"id": it["id"], "text": it["text"]} for it in items], progress=False)
    gold = {it["id"]: it["label"] for it in items}
    acc = sum(r.label == gold.get(r.id) for r in results) / len(results) if results else 0.0
    cost = sum(s.get("cost", 0.0) for s in pipe.stats().values())
    records = []
    for r in results:
        d = r.to_dict()
        d["gold"] = gold.get(r.id)
        records.append(d)
    return acc, cost, records


def _with_rules(raw: dict, rules: list[str]) -> dict:
    out = copy.deepcopy(raw)
    out.setdefault("task", {})["rules"] = list(rules)
    return out


def optimize_rules(raw: dict, items: list[dict], limit: int = 36, holdout_ratio: float = 0.34,
                   seed: int = 0, log: Callable[[str], None] | None = None) -> dict:
    """items 需要 id、text、label（标准答案）。只改规则，不改类别定义。"""
    say = log or (lambda _m: None)
    rows = stratified_take(items, limit, seed=seed)
    dev, hold = split_dev_test(rows, label_key="label", test_ratio=holdout_ratio, seed=seed)
    if len(dev) < 6 or len(hold) < 3:
        raise ValueError(f"样本太少（调参 {len(dev)} 条、留出 {len(hold)} 条），至少需要大约 15 条、且每个类别都有样本")

    say(f"调参集 {len(dev)} 条，留出集 {len(hold)} 条。先跑当前规则…")
    base_acc, base_cost, dev_records = asyncio.run(_accuracy(raw, dev))
    task = (raw.get("task") or {})
    tried = [{"name": "当前规则", "rules": list(task.get("rules") or []), "dev_acc": base_acc, "dev_cost": base_cost}]

    config = config_from_dict(copy.deepcopy(raw))
    models = drafting_models(config)
    model = next((m for m in models if m.name == "qwen"), models[0] if models else None)
    variants: list[dict] = []
    note = ""
    if model is None:
        note = "没有可用于提案的大模型，只报告了当前规则。"
    else:
        say(f"用 {model.name} 根据调参集上的错例提出新规则…")
        try:
            variants = propose_variants(config, model, task, dev_records)
        except Exception as e:
            note = f"提案失败：{e}"
        if not variants and not note:
            note = "模型没有给出和现有规则不同的提案。"
    for v in variants:
        say(f"在调参集上试「{v['name']}」…")
        acc, cost, _ = asyncio.run(_accuracy(_with_rules(raw, v["rules"]), dev))
        tried.append({**v, "dev_acc": acc, "dev_cost": cost})

    winner = pick_winner(tried)
    say(f"调参集上最好的是「{winner['name']}」（{winner['dev_acc']:.1%}）。在留出集上各跑一次…")
    hold_rows = []
    seen = set()
    for row in (tried[0], winner):
        if row["name"] in seen:
            continue
        seen.add(row["name"])
        acc, cost, _ = asyncio.run(_accuracy(_with_rules(raw, row["rules"]), hold))
        hold_rows.append({"name": row["name"], "acc": acc, "cost": cost})

    return {
        "dev_n": len(dev),
        "hold_n": len(hold),
        "rows": tried,
        "winner": winner["name"],
        "winner_rules": winner["rules"],
        "holdout": hold_rows,
        "note": note,
        "model": model.name if model else "",
    }


def format_optimize(rep: dict) -> str:
    pct = lambda x: f"{x * 100:.1f}%"
    lines = [f"调参集 {rep['dev_n']} 条（用来挑选）　留出集 {rep['hold_n']} 条（只报告一次）"]
    if rep.get("note"):
        lines.append(rep["note"])
    lines.append("调参集：")
    for r in rep["rows"]:
        mark = "  ← 选中" if r["name"] == rep["winner"] else ""
        lines.append(f"  {r['name']}  {pct(r['dev_acc'])}  本次费用 {r['dev_cost']:.4f} 元{mark}")
    lines.append("留出集：")
    for r in rep["holdout"]:
        lines.append(f"  {r['name']}  {pct(r['acc'])}  本次费用 {r['cost']:.4f} 元")
    return "\n".join(lines)
