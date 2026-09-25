from __future__ import annotations

import hashlib
import random
import re
from typing import TYPE_CHECKING

from .config import LABEL_SEP, LEVEL_SEP, LabelDef, TaskConfig, parent_of

if TYPE_CHECKING:
    from .classifier import Prediction

SYSTEM_PROMPT = "你是一名严谨、客观的文本分类标注员。你必须严格依据给定的分类标准做判断，并且只输出 JSON。"


def prompt_seed(task: TaskConfig, model_name: str, text: str) -> str | None:
    """选项顺序随机化的种子：同一模型、同一文本的顺序固定（缓存可复用），不同模型之间不同。"""
    return f"{model_name}\n{text}" if task.shuffle_labels else None


def label_order(task: TaskConfig, seed: str | None = None) -> list[LabelDef]:
    """seed 为 None 时按配置顺序；否则按 seed 的哈希打乱（同一 seed 顺序固定）。层级分类时同一一级类的类别排在一起。"""
    labels = list(task.labels)
    if seed is not None:
        random.Random(int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16)).shuffle(labels)
    if task.hierarchical:
        first = {}
        for i, lab in enumerate(labels):
            first.setdefault(parent_of(lab.name), i)
        labels.sort(key=lambda lab: first[parent_of(lab.name)])
    return labels


def _label_block(lab: LabelDef, level: str) -> str:
    block = [f"{level} {lab.name}"]
    if lab.definition:
        block.append(f"定义：{lab.definition}")
    if lab.examples:
        block.append("正例：\n" + "\n".join(f"- {e}" for e in lab.examples))
    if lab.counter_examples:
        block.append("反例（不属于该类）：\n" + "\n".join(f"- {e}" for e in lab.counter_examples))
    return "\n".join(block)


def render_guide(task: TaskConfig, seed: str | None = None) -> str:
    parts = []
    if task.description:
        parts.append(f"# 任务\n{task.description}")
    parts.append("# 分类标准")
    if task.hierarchical:
        parts.append(f"类别分为两级，名称写作“一级类{LEVEL_SEP}二级类”。请先判断一级类，再在该一级类下选择最合适的二级类。")
        current = None
        for lab in label_order(task, seed):
            parent = parent_of(lab.name)
            if parent != current:
                parts.append(f"## 一级类：{parent}")
                current = parent
            parts.append(_label_block(lab, "###"))
    else:
        parts += [_label_block(lab, "##") for lab in label_order(task, seed)]
    if task.multi_label:
        parts.append(f"# 多标签\n一条文本可以同时属于多个类别（最多 {task.max_labels} 个）。只选确实符合定义的类别，不要为了凑数多选。")
    if task.rules:
        parts.append("# 边界规则\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(task.rules, 1)))
    return "\n\n".join(parts)


Examples = list[tuple[str, str]]


def _examples_block(examples: Examples | None, max_chars: int = 300) -> str | None:
    if not examples:
        return None

    def clip(t: str) -> str:
        t = " ".join(t.split())
        return t if len(t) <= max_chars else t[:max_chars] + "…"

    lines = [f"{i}. 【{show_label(label)}】{clip(t)}" for i, (t, label) in enumerate(examples, 1)]
    return (
        "# 相似的已标注样本（来自同一数据集的训练集，反映该数据集的标注习惯）\n"
        "这些样本与待分类文本最相似，其标签由人工标注。请参考它们理解标注尺度，"
        "但仍需根据待分类文本本身的内容判断，不要机械照搬。\n" + "\n".join(lines)
    )


def _text_block(text: str) -> str:
    return f"# 待分类文本\n<text>\n{text}\n</text>"


def show_label(label: str | None) -> str:
    return "、".join(label.split(LABEL_SEP)) if label else "无"


def _output_spec(task: TaskConfig, seed: str | None = None, evidence: bool = False) -> str:
    names = " / ".join(lab.name for lab in label_order(task, seed))
    ev = ', "evidence": "<从待分类文本中逐字摘抄的最关键片段，不超过40字>"' if evidence else ""
    if task.multi_label:
        spec = ('{"labels": ["<标签>", ...], "confidence": <0到1之间的小数>, '
                f'"reason": "<不超过60字的判断理由>"{ev}}}\n'
                f"labels 中每一项必须是以下之一，选 1 到 {task.max_labels} 个：{names}")
    else:
        spec = ('{"label": "<标签>", "confidence": <0到1之间的小数>, '
                f'"reason": "<不超过60字的判断理由>"{ev}}}\n'
                f"label 必须是以下之一：{names}")
    if evidence:
        spec += "\nevidence 必须是原文中连续出现的原话，不要改写或概括；原文中找不到能支撑判断的片段时填空字符串。"
    return "# 输出格式\n只输出一个 JSON 对象，不要输出任何其他内容：\n" + spec


def mask_labels(text: str, task: TaskConfig) -> str:
    """只看理由的复核：把理由里出现的类别名替换掉，避免复核者从理由中直接读到结论。"""
    names = set(task.label_names) | {parent_of(n) for n in task.label_names} | {n.split(LEVEL_SEP)[-1] for n in task.label_names}
    names = sorted((n for n in names if n), key=len, reverse=True)
    return re.sub("|".join(map(re.escape, names)), "［某类］", text) if names else text


def _opinions_block(title: str, opinions: list[Prediction], view: str = "full", task: TaskConfig | None = None) -> str:
    lines = []
    for i, p in enumerate(opinions):
        who = f"- 评审员{chr(ord('A') + i)}："
        if view == "labels":
            lines.append(f"{who}标签={show_label(p.label)}，置信度={p.confidence:.2f}")
            continue
        ev = f"，引用原文：「{p.evidence}」" if p.evidence else ""
        if view == "reasons":
            reason = mask_labels(p.reason, task) if task is not None else p.reason
            lines.append(f"{who}理由：{reason}{mask_labels(ev, task) if task is not None else ev}")
        else:
            lines.append(f"{who}标签={show_label(p.label)}，置信度={p.confidence:.2f}，理由：{p.reason}{ev}")
    return f"# {title}\n<peer_opinions>\n" + "\n".join(lines) + "\n</peer_opinions>"


_VIEW_NOTE = {
    "reasons": "（为避免被他人结论带偏，这里只给出理由，理由中的类别名已隐去）",
    "labels": "（这里只给出他人的结论，不给理由，请独立思考判断依据）",
}


def _head(task: TaskConfig, text: str, examples: Examples | None, max_chars: int, seed: str | None) -> list[str]:
    parts = [render_guide(task, seed)]
    block = _examples_block(examples, max_chars)
    if block:
        parts.append(block)
    parts.append(_text_block(text))
    return parts


def build_classify_prompt(task: TaskConfig, text: str, examples: Examples | None = None, max_chars: int = 300,
                          seed: str | None = None, evidence: bool = False) -> str:
    return "\n\n".join([*_head(task, text, examples, max_chars, seed), _output_spec(task, seed, evidence)])


def build_review_prompt(
    task: TaskConfig, text: str, own: Prediction | None, peers: list[Prediction],
    examples: Examples | None = None, max_chars: int = 300, seed: str | None = None,
    view: str = "full", evidence: bool = False, round_no: int = 1, devil: Prediction | None = None,
) -> str:
    parts = _head(task, text, examples, max_chars, seed)
    if own is not None and own.ok:
        mine = "你上一轮的判断" if round_no > 1 else "你之前的判断"
        parts.append(f"# {mine}\n标签={show_label(own.label)}，置信度={own.confidence:.2f}，理由：{own.reason}")
    title = f"其他评审员第 {round_no - 1} 轮复核后的意见（匿名）" if round_no > 1 else "其他评审员的意见（匿名）"
    parts.append(_opinions_block(title + _VIEW_NOTE.get(view, ""), peers, view, task))
    if devil is not None and devil.ok:
        parts.append(
            "# 反方意见（魔鬼代言人）\n下面的意见来自专门负责挑刺的评审员：他被要求反对当前的多数意见、寻找其漏洞。"
            "请逐条检验这些质疑是否真的成立；成立就修正，不成立就坚持，不要因为有人反对就动摇。\n"
            f"<devil>\n主张：{show_label(devil.label)}；理由：{devil.reason}\n</devil>"
        )
    parts.append(
        "# 要求\n请结合分类标准重新审视这条文本：\n"
        "1. 逐条对照其他评审员的理由，判断其是否比你的判断更符合分类标准和边界规则；\n"
        "2. 只有当对方理由确实更符合分类标准时才修改答案，不要因为多数人的选择而盲从；\n"
        "3. 如果坚持原判断，请在理由中简要指出对方理由的问题。"
    )
    parts.append(_output_spec(task, seed, evidence))
    return "\n\n".join(parts)


def build_arbiter_prompt(
    task: TaskConfig, text: str, opinions: list[Prediction],
    examples: Examples | None = None, max_chars: int = 300, seed: str | None = None,
    view: str = "full", evidence: bool = False,
) -> str:
    return "\n\n".join([
        *_head(task, text, examples, max_chars, seed),
        _opinions_block("各评审员的意见（匿名，存在分歧）" + _VIEW_NOTE.get(view, ""), opinions, view, task),
        "# 要求\n你是最终仲裁者。请严格依据分类标准和边界规则给出最终判断。"
        "评审员的意见仅供参考，多数意见不一定正确。如果文本本身确实模糊、难以判断，请给出较低的置信度。",
        _output_spec(task, seed, evidence),
    ])


def build_devil_prompt(
    task: TaskConfig, text: str, majority: str, supporters: list[Prediction],
    examples: Examples | None = None, max_chars: int = 300, seed: str | None = None,
) -> str:
    reasons = "\n".join(f"- {p.reason}" for p in supporters if p.reason) or "- （未给出理由）"
    return "\n\n".join([
        *_head(task, text, examples, max_chars, seed),
        f"# 当前多数意见\n多数评审员认为标签是：{show_label(majority)}。他们的理由：\n{reasons}",
        "# 你的角色：魔鬼代言人\n你的任务不是附和，而是尽最大努力找出上述多数意见的漏洞："
        "对照分类标准和边界规则，指出它忽略了文本中的哪些信息、误读了哪条规则，"
        f"并给出你认为最有可能的其他标签（必须不同于“{show_label(majority)}”）。"
        "如果确实找不到有力的反驳，也请给出最接近的其他标签，并把置信度填得很低（不超过 0.3）。",
        _output_spec(task, seed),
    ])
