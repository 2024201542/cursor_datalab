from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

from .config import LabelDef, TaskConfig

if TYPE_CHECKING:
    from .classifier import Prediction

SYSTEM_PROMPT = "你是一名严谨、客观的文本分类标注员。你必须严格依据给定的分类标准做判断，并且只输出 JSON。"


def prompt_seed(task: TaskConfig, model_name: str, text: str) -> str | None:
    """选项顺序随机化的种子：同一模型、同一文本的顺序固定（缓存可复用），不同模型之间不同。"""
    return f"{model_name}\n{text}" if task.shuffle_labels else None


def label_order(task: TaskConfig, seed: str | None = None) -> list[LabelDef]:
    """seed 为 None 时按配置顺序；否则按 seed 的哈希打乱（同一 seed 顺序固定）。"""
    labels = list(task.labels)
    if seed is not None:
        random.Random(int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16)).shuffle(labels)
    return labels


def render_guide(task: TaskConfig, seed: str | None = None) -> str:
    parts = []
    if task.description:
        parts.append(f"# 任务\n{task.description}")
    parts.append("# 分类标准")
    for lab in label_order(task, seed):
        block = [f"## {lab.name}"]
        if lab.definition:
            block.append(f"定义：{lab.definition}")
        if lab.examples:
            block.append("正例：\n" + "\n".join(f"- {e}" for e in lab.examples))
        if lab.counter_examples:
            block.append("反例（不属于该类）：\n" + "\n".join(f"- {e}" for e in lab.counter_examples))
        parts.append("\n".join(block))
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

    lines = [f"{i}. 【{label}】{clip(t)}" for i, (t, label) in enumerate(examples, 1)]
    return (
        "# 相似的已标注样本（来自同一数据集的训练集，反映该数据集的标注习惯）\n"
        "这些样本与待分类文本最相似，其标签由人工标注。请参考它们理解标注尺度，"
        "但仍需根据待分类文本本身的内容判断，不要机械照搬。\n" + "\n".join(lines)
    )


def _text_block(text: str) -> str:
    return f"# 待分类文本\n<text>\n{text}\n</text>"


def _output_spec(task: TaskConfig, seed: str | None = None) -> str:
    return (
        "# 输出格式\n只输出一个 JSON 对象，不要输出任何其他内容：\n"
        '{"label": "<标签>", "confidence": <0到1之间的小数>, "reason": "<不超过60字的判断理由>"}\n'
        f"label 必须是以下之一：{' / '.join(lab.name for lab in label_order(task, seed))}"
    )


def _opinions_block(title: str, opinions: list[Prediction]) -> str:
    lines = [
        f"- 评审员{chr(ord('A') + i)}：标签={p.label}，置信度={p.confidence:.2f}，理由：{p.reason}"
        for i, p in enumerate(opinions)
    ]
    return f"# {title}\n<peer_opinions>\n" + "\n".join(lines) + "\n</peer_opinions>"


def _head(task: TaskConfig, text: str, examples: Examples | None, max_chars: int, seed: str | None) -> list[str]:
    parts = [render_guide(task, seed)]
    block = _examples_block(examples, max_chars)
    if block:
        parts.append(block)
    parts.append(_text_block(text))
    return parts


def build_classify_prompt(task: TaskConfig, text: str, examples: Examples | None = None, max_chars: int = 300,
                          seed: str | None = None) -> str:
    return "\n\n".join([*_head(task, text, examples, max_chars, seed), _output_spec(task, seed)])


def build_review_prompt(
    task: TaskConfig, text: str, own: Prediction | None, peers: list[Prediction],
    examples: Examples | None = None, max_chars: int = 300, seed: str | None = None,
) -> str:
    parts = _head(task, text, examples, max_chars, seed)
    if own is not None and own.ok:
        parts.append(f"# 你之前的判断\n标签={own.label}，置信度={own.confidence:.2f}，理由：{own.reason}")
    parts.append(_opinions_block("其他评审员的意见（匿名）", peers))
    parts.append(
        "# 要求\n请结合分类标准重新审视这条文本：\n"
        "1. 逐条对照其他评审员的理由，判断其是否比你的判断更符合分类标准和边界规则；\n"
        "2. 只有当对方理由确实更符合分类标准时才修改答案，不要因为多数人的选择而盲从；\n"
        "3. 如果坚持原判断，请在理由中简要指出对方理由的问题。"
    )
    parts.append(_output_spec(task, seed))
    return "\n\n".join(parts)


def build_arbiter_prompt(
    task: TaskConfig, text: str, opinions: list[Prediction],
    examples: Examples | None = None, max_chars: int = 300, seed: str | None = None,
) -> str:
    return "\n\n".join([
        *_head(task, text, examples, max_chars, seed),
        _opinions_block("各评审员的意见（匿名，存在分歧）", opinions),
        "# 要求\n你是最终仲裁者。请严格依据分类标准和边界规则给出最终判断。"
        "评审员的意见仅供参考，多数意见不一定正确。如果文本本身确实模糊、难以判断，请给出较低的置信度。",
        _output_spec(task, seed),
    ])
