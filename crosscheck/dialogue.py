"""把多轮对话整理成可送进现有分类流水线的文本。

支持两种原始格式：一行一句（会话编号 + 角色 + 内容），或一行一段已写好角色的对话。
分类粒度可以是整段对话一个类别，或只判断其中某一句（带上前文）。
"""
from __future__ import annotations

import re

DEFAULT_MAX_CHARS = 4000
DEFAULT_CONTEXT_TURNS = 6

_ROLE_MAP = {
    "user": "用户", "customer": "用户", "client": "用户", "用户": "用户", "客户": "用户", "顾客": "用户",
    "agent": "客服", "assistant": "客服", "service": "客服", "客服": "客服", "坐席": "客服", "商家": "客服",
    "system": "系统", "系统": "系统",
}
_SPEAKER = re.compile(r"^\s*([^：:\n]{1,16})[：:]\s*(.*)$")


def normalize_role(role: str) -> str:
    text = str(role or "").strip()
    return _ROLE_MAP.get(text.lower(), text or "未知")


def parse_transcript(text: str) -> list[dict]:
    """把“角色：内容”的多行文本拆成一轮轮发言。没有角色标记的行并入上一句，或单独作为“文本”。"""
    turns: list[dict] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _SPEAKER.match(stripped)
        if m and m.group(2).strip():
            turns.append({"role": normalize_role(m.group(1)), "text": m.group(2).strip()})
        elif turns:
            turns[-1]["text"] += "\n" + stripped
        else:
            turns.append({"role": "文本", "text": stripped})
    return turns


def group_turns(rows: list[dict], session_col: str, role_col: str, text_col: str,
                time_col: str | None = None, label_col: str | None = None) -> list[dict]:
    """一行一句 → 按会话合并。同一会话都有时间列时按时间排序。"""
    order: list[str] = []
    bucket: dict[str, list[dict]] = {}
    for i, row in enumerate(rows):
        sid = str(row.get(session_col) or "").strip() or f"row{i + 1}"
        text = str(row.get(text_col) or "").strip()
        if not text:
            continue
        if sid not in bucket:
            bucket[sid] = []
            order.append(sid)
        turn = {"role": normalize_role(str(row.get(role_col) or "")), "text": text}
        if time_col and str(row.get(time_col) or "").strip():
            turn["time"] = str(row.get(time_col)).strip()
        if label_col and str(row.get(label_col) or "").strip():
            turn["label"] = str(row.get(label_col)).strip()
        bucket[sid].append(turn)
    sessions = []
    for sid in order:
        turns = bucket[sid]
        if turns and all("time" in t for t in turns):
            turns = sorted(turns, key=lambda t: t["time"])
        sessions.append({"id": sid, "turns": turns})
    return sessions


def sessions_from_transcripts(rows: list[dict], id_col: str | None, text_col: str,
                              label_col: str | None = None) -> list[dict]:
    """一行一段对话。能识别“角色：内容”就拆开，否则整段当作一句。"""
    sessions = []
    for i, row in enumerate(rows):
        text = str(row.get(text_col) or "").strip()
        if not text:
            continue
        turns = parse_transcript(text) or [{"role": "文本", "text": text}]
        if label_col and str(row.get(label_col) or "").strip():
            label = str(row.get(label_col)).strip()
            for t in turns:
                t["label"] = label
        sid = str(row.get(id_col) or "").strip() if id_col else ""
        sessions.append({"id": sid or str(i + 1), "turns": turns})
    return sessions


def _tail(turns: list[dict], max_chars: int) -> tuple[list[dict], int]:
    """保留最近的若干句，使总字数不超过 max_chars。至少保留最后一句。"""
    kept: list[dict] = []
    total = 0
    for t in reversed(turns):
        n = len(t["text"]) + len(t["role"]) + 2
        if kept and total + n > max_chars:
            break
        kept.append(t)
        total += n
    kept.reverse()
    return kept, len(turns) - len(kept)


def _render(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}：{t['text']}" for t in turns)


def _unique_label(turns: list[dict]) -> tuple[str | None, str | None]:
    vals = list(dict.fromkeys(t["label"] for t in turns if t.get("label")))
    if not vals:
        return None, None
    if len(vals) > 1:
        return None, f"同一段对话里的标签不一致（{' / '.join(vals)}），本次不计算准确率"
    return vals[0], None


def build_items(sessions: list[dict], *, granularity: str = "session", target_roles: list[str] | None = None,
                max_chars: int = DEFAULT_MAX_CHARS, context_turns: int = DEFAULT_CONTEXT_TURNS
                ) -> tuple[list[dict], dict, list[str]]:
    """返回 (待分类条目, 审核用的原文索引, 警告)。

    granularity: session 整段一个类别；turn 只判断某一句，前文作为语境。
    target_roles: 逐句模式下只分类这些角色；空表示每一句都分类。
    """
    items: list[dict] = []
    index: dict[str, dict] = {}
    warnings: list[str] = []
    roles = set(target_roles or [])
    for s in sessions:
        turns = s["turns"]
        if granularity == "turn":
            for i, t in enumerate(turns):
                if roles and t["role"] not in roles:
                    continue
                ctx, omitted = _tail(turns[max(0, i - context_turns):i], max_chars)
                parts = ["下面是一段对话。请只判断【待分类】的那一句，前文只用来理解语境，不要给前文分类。"]
                if omitted:
                    parts.append(f"（前文较长，已省略更早的 {omitted} 句。）")
                if ctx:
                    parts.append("前文：\n" + _render(ctx))
                parts.append(f"【待分类】{t['role']}：{t['text']}")
                item_id = f"{s['id']}#{i + 1}"
                item = {"id": item_id, "text": "\n".join(parts)}
                if t.get("label"):
                    item["label"] = t["label"]
                items.append(item)
                index[item_id] = {"session_id": s["id"], "turns": turns, "focus": i}
            continue
        shown, omitted = _tail(turns, max_chars)
        parts = ["这是一段多轮对话，每行格式为“角色：内容”。请给整段对话一个类别。"]
        if omitted:
            parts.append(f"（对话较长，已省略开头 {omitted} 句，下面是最近的内容。）")
        parts.append(_render(shown))
        item = {"id": s["id"], "text": "\n".join(parts)}
        label, warn = _unique_label(turns)
        if warn:
            warnings.append(f"{s['id']}：{warn}")
        elif label:
            item["label"] = label
        items.append(item)
        index[s["id"]] = {"session_id": s["id"], "turns": turns, "focus": None, "omitted": omitted}
    extra = {"kind": "dialogue", "granularity": granularity, "items": index}
    return items, extra, warnings


def cited_indexes(turns: list[dict], reasons: list[str]) -> set[int]:
    """模型理由里逐字引用了哪几句（连续 8 个字重合即算引用）。"""
    blob = "\n".join(r for r in reasons if r)
    cited = set()
    if not blob:
        return cited
    for i, t in enumerate(turns):
        text = t["text"]
        if len(text) < 8:
            if text and text in blob:
                cited.add(i)
            continue
        step = 8
        for n in range(0, len(text) - 7, step):
            if text[n:n + 8] in blob:
                cited.add(i)
                break
    return cited
