"""分歧样本上的规则修改建议，以及多次运行之间的漂移检测。"""
from __future__ import annotations

from .drafting import ask_json
from .prompts import render_guide
from .config import Config, LabelDef, ModelConfig, TaskConfig

HUMAN_UP = 0.08
ACC_DOWN = 0.05
AUTO_DOWN = 0.10
TV_DISTANCE = 0.15


def disagreement_records(records: list[dict], limit: int = 20) -> list[dict]:
    """首轮至少两个模型给出了不同标签的样本，不同的分歧组合尽量都抽到。"""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        labels = [p.get("label") for p in r.get("round1") or [] if p.get("label") and not p.get("error")]
        pair = tuple(sorted(set(labels)))
        if len(pair) >= 2:
            groups.setdefault(pair, []).append(r)
    picked: list[dict] = []
    while len(picked) < limit and any(groups.values()):
        for pair in list(groups):
            if len(picked) >= limit:
                break
            if groups[pair]:
                picked.append(groups[pair].pop(0))
    return picked


def _guide(task: dict) -> str:
    return render_guide(TaskConfig(
        name=task.get("name", ""), description=task.get("description", ""),
        labels=[LabelDef(lab["name"], lab.get("definition", ""), list(lab.get("examples") or []),
                         list(lab.get("counter_examples") or [])) for lab in task.get("labels") or []],
        rules=list(task.get("rules") or []),
    ))


def prompt_suggest(task: dict, records: list[dict]) -> str:
    lines = []
    for i, r in enumerate(records, 1):
        votes = "；".join(f"{p.get('model')}={p.get('label') or '失败'}" for p in r.get("round1") or [])
        text = str(r.get("text") or "").replace("\n", " ")
        if len(text) > 180:
            text = text[:180] + "…"
        lines.append(f"{i}. 模型意见：{votes}\n   文本：{text}")
    return (
        "下面是一次分类运行中模型之间存在分歧的样本。请对照分类标准，归纳分歧集中在哪些边界上，"
        "并给出可以直接追加到标准里的边界规则。\n\n"
        "要求：\n"
        "1. 只根据这些样本说话，不要编造样本里没有的现象；\n"
        "2. rule_edits 每条都是一句完整的判定规则（什么情况归入哪一类），可以原样追加，不要写“建议修改第 3 条”这种无法直接使用的话；\n"
        "3. 样本太少或看不出规律时，patterns 和 rule_edits 留空，并在 summary 里说明。\n\n"
        "只输出这个 JSON：\n"
        '{"summary": "一两句话", "patterns": [{"pair": "甲 vs 乙", "problem": "分歧出在哪里", "suggestion": "怎么改标准"}],\n'
        ' "rule_edits": ["可直接追加的规则"]}\n\n'
        f"# 当前分类标准\n{_guide(task)}\n\n# 分歧样本\n" + "\n".join(lines)
    )


def suggest_rules(config: Config, model: ModelConfig, task: dict, records: list[dict], limit: int = 20) -> dict:
    samples = disagreement_records(records, limit)
    if len(samples) < 3:
        return {"summary": f"分歧样本只有 {len(samples)} 条，看不出稳定的规律。多跑一些数据，或换一批更容易分歧的样本再分析。",
                "patterns": [], "rule_edits": [], "n": len(samples)}
    obj = ask_json(config, model, prompt_suggest(task, samples), max_tokens=4000)
    patterns = [p for p in obj.get("patterns") or [] if isinstance(p, dict) and p.get("problem")]
    edits = [str(x).strip() for x in (obj.get("rule_edits") or []) if str(x).strip()]
    return {"summary": str(obj.get("summary", "")).strip(), "patterns": patterns, "rule_edits": edits, "n": len(samples)}


def label_share(records: list[dict]) -> dict[str, float]:
    labels = [r.get("label") or "（无标签）" for r in records]
    n = len(labels)
    if not n:
        return {}
    keys = list(dict.fromkeys(labels))
    return {k: labels.count(k) / n for k in keys}


def tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def drift_alerts(rows: list[dict]) -> list[str]:
    """rows 按时间从早到晚。每行含 name、human_rate、auto_rate、acc（可无）、labels（占比）、criteria_hash。

    只比较最近一次和它的前一次。
    """
    if len(rows) < 2:
        return []
    prev, last = rows[-2], rows[-1]
    alerts = []
    same_rule = prev.get("criteria_hash") and prev.get("criteria_hash") == last.get("criteria_hash")
    prefix = "" if same_rule else "这两次运行的分类标准不同，变化可能来自标准而不是数据。"
    dh = (last.get("human_rate") or 0) - (prev.get("human_rate") or 0)
    if dh >= HUMAN_UP:
        alerts.append(f"{prefix}需人工审核的比例从 {prev['human_rate'] * 100:.0f}% 升到 {last['human_rate'] * 100:.0f}%"
                      f"（{last.get('name')} 对比 {prev.get('name')}）。数据变难了，或模型变得更不一致。")
    da = (prev.get("auto_rate") or 0) - (last.get("auto_rate") or 0)
    if da >= AUTO_DOWN:
        alerts.append(f"{prefix}自动采纳比例下降了 {da * 100:.0f} 个百分点。")
    if prev.get("acc") is not None and last.get("acc") is not None and prev["acc"] - last["acc"] >= ACC_DOWN:
        alerts.append(f"{prefix}准确率从 {prev['acc'] * 100:.1f}% 降到 {last['acc'] * 100:.1f}%。")
    dist = tv_distance(prev.get("labels") or {}, last.get("labels") or {})
    if dist >= TV_DISTANCE:
        alerts.append(f"{prefix}最终标签的分布变了（总变差 {dist:.2f}）。如果数据来源没变，可能是口径或模型变了。")
    return [a.strip() for a in alerts]
