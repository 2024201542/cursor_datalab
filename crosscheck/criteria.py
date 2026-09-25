"""分类标准（task 部分）的版本管理：哈希、保存版本、逐条对比差异。"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

VERSIONS_ROOT = Path("configs/versions")
_FIELDS = ("name", "description", "labels", "rules")


def criteria_of(task: dict | None) -> dict:
    """只保留影响分类标准含义的字段（不含 shuffle_labels 这类运行策略）。"""
    task = task or {}
    labels = [{"name": lab.get("name", ""), "definition": lab.get("definition", ""),
               "examples": list(lab.get("examples") or []), "counter_examples": list(lab.get("counter_examples") or [])}
              for lab in task.get("labels") or []]
    return {"name": task.get("name", ""), "description": task.get("description", ""),
            "labels": labels, "rules": list(task.get("rules") or [])}


def criteria_hash(task: dict | None) -> str:
    payload = json.dumps(criteria_of(task), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def versions_dir(config_path: str | Path, root: str | Path = VERSIONS_ROOT) -> Path:
    return Path(root) / Path(config_path).stem


def list_versions(config_path: str | Path, root: str | Path = VERSIONS_ROOT) -> list[dict]:
    d = versions_dir(config_path, root)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("v*.json"), key=lambda p: int(p.stem.split("_")[0][1:])):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return out


def save_version(config_path: str | Path, task: dict, note: str = "", root: str | Path = VERSIONS_ROOT) -> dict | None:
    """标准与最新版本不同时保存为新版本并返回；没有变化返回 None。"""
    versions = list_versions(config_path, root)
    h = criteria_hash(task)
    if versions and versions[-1]["hash"] == h:
        return None
    ver = {"version": (versions[-1]["version"] + 1) if versions else 1, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "hash": h, "note": note, "task": criteria_of(task)}
    d = versions_dir(config_path, root)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"v{ver['version']}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(ver, ensure_ascii=False, indent=1), encoding="utf-8")
    return ver


def version_label(config_path: str | Path, h: str, root: str | Path = VERSIONS_ROOT) -> str:
    """把哈希翻译成版本号，例如 v3；不在版本记录中时返回哈希本身。"""
    for v in list_versions(config_path, root):
        if v["hash"] == h:
            return f"v{v['version']}"
    return h


def _clip(s: str, n: int = 60) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def diff_criteria(old: dict | None, new: dict | None) -> list[str]:
    """两版分类标准的差异，每条一句话。"""
    a, b = criteria_of(old), criteria_of(new)
    out = []
    for key, title in (("name", "任务名称"), ("description", "任务说明")):
        if a[key] != b[key]:
            out.append(f"{title}：{_clip(a[key]) or '（空）'} → {_clip(b[key]) or '（空）'}")
    la = {lab["name"]: lab for lab in a["labels"]}
    lb = {lab["name"]: lab for lab in b["labels"]}
    for name in lb.keys() - la.keys():
        out.append(f"新增类别「{name}」：{_clip(lb[name]['definition'])}")
    for name in la.keys() - lb.keys():
        out.append(f"删除类别「{name}」")
    for name in [n for n in lb if n in la]:
        x, y = la[name], lb[name]
        if x["definition"] != y["definition"]:
            out.append(f"「{name}」定义：{_clip(x['definition']) or '（空）'} → {_clip(y['definition']) or '（空）'}")
        for key, title in (("examples", "正例"), ("counter_examples", "反例")):
            added = [e for e in y[key] if e not in x[key]]
            removed = [e for e in x[key] if e not in y[key]]
            if added:
                out.append(f"「{name}」新增{title} {len(added)} 条：{_clip('；'.join(added))}")
            if removed:
                out.append(f"「{name}」删除{title} {len(removed)} 条：{_clip('；'.join(removed))}")
    if [lab["name"] for lab in a["labels"] if lab["name"] in lb] != [lab["name"] for lab in b["labels"] if lab["name"] in la]:
        out.append("类别顺序调整")
    for r in b["rules"]:
        if r not in a["rules"]:
            out.append(f"新增规则：{_clip(r, 80)}")
    for r in a["rules"]:
        if r not in b["rules"]:
            out.append(f"删除规则：{_clip(r, 80)}")
    return out


def changed_labels(old: dict | None, new: dict | None) -> list[str]:
    """定义、正反例有变化或新增 / 删除的类别；这些类别的旧金标准可能需要复核。"""
    a = {lab["name"]: lab for lab in criteria_of(old)["labels"]}
    b = {lab["name"]: lab for lab in criteria_of(new)["labels"]}
    return sorted(n for n in a.keys() | b.keys() if a.get(n) != b.get(n))
