"""运行记录：每次运行写一份元信息（时间、数据、配置快照、费用），用于历史查看和多次运行对比。"""
from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path

from .criteria import criteria_hash, version_label
from .pipeline import STATUS_HUMAN

LOCAL_PROVIDERS = ("local", "mock")


def meta_path(run_path: str | Path) -> Path:
    p = Path(run_path)
    return p.with_name(f"{p.stem}.meta.json")


def _snapshot(raw: dict | None) -> dict | None:
    """配置快照，去掉直接写在配置里的 api_key。"""
    if not raw:
        return None
    raw = copy.deepcopy(raw)
    for m in (raw.get("models") or []) + ([raw["arbiter"]] if raw.get("arbiter") else []):
        m.pop("api_key", None)
    return raw


def write_meta(run_path: str | Path, *, source: str, input_name: str, config, raw: dict | None,
               stats: dict, elapsed: float, n: int, has_gold: bool, config_path: str = "", note: str = "") -> dict:
    pc, fs = config.pipeline, config.fewshot
    h = criteria_hash((raw or {}).get("task")) if raw else ""
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "input": input_name,
        "config_path": config_path,
        "task_name": config.task.name,
        "criteria_hash": h,
        "criteria_version": version_label(config_path, h) if h and config_path else h,
        "n": n,
        "has_gold": has_gold,
        "elapsed": round(elapsed, 1),
        "note": note,
        "models": [{"name": m.name, "provider": m.provider, "model": m.model, "logprobs": m.logprobs}
                   for m in config.models],
        "arbiter": config.arbiter.name if config.arbiter and pc.use_arbiter else None,
        "settings": {
            "disagreement_action": pc.disagreement_action,
            "cascade": list(pc.cascade),
            "cascade_min_confidence": pc.cascade_min_confidence,
            "shuffle_labels": config.task.shuffle_labels,
            "fewshot": fs.path if fs.enabled else "",
            "retriever": fs.retriever if fs.enabled and fs.retriever != "tfidf" else "",
            "accept_threshold": pc.accept_threshold,
            "arbiter_threshold": pc.arbiter_threshold,
            "calibration": pc.calibration,
            "min_posterior": pc.min_posterior,
            "samples": pc.samples,
            "debate_rounds": pc.debate_rounds,
            "devil_advocate": pc.devil_advocate,
            "review_view": pc.review_view,
            "require_evidence": pc.require_evidence,
            "jury": [j.name for j in config.jury] if pc.use_arbiter else [],
            "multi_label": config.task.multi_label,
            "hierarchical": config.task.hierarchical,
        },
        "cost": round(sum(s.get("cost", 0.0) for s in stats.values()), 6),
        "saved": round(sum(s.get("saved", 0.0) for s in stats.values()), 6),
        "stats": stats,
        "config": _snapshot(raw),
    }
    meta_path(run_path).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def load_meta(run_path: str | Path) -> dict:
    p = meta_path(run_path)
    if p.exists():
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not meta.get("criteria_hash") and (meta.get("config") or {}).get("task"):
            meta["criteria_hash"] = criteria_hash(meta["config"]["task"])
        h, cp = meta.get("criteria_hash"), meta.get("config_path")
        if h and cp:
            meta["criteria_version"] = version_label(cp, h)  # 运行后才保存的版本也能对上号
        return meta
    return {}


def run_criteria(meta: dict) -> dict | None:
    return (meta.get("config") or {}).get("task")


def set_note(run_path: str | Path, note: str) -> None:
    meta = load_meta(run_path)
    meta["note"] = note
    meta_path(run_path).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def extra_path(run_path: str | Path) -> Path:
    p = Path(run_path)
    return p.with_name(f"{p.stem}_extra.json")


def save_extra(run_path: str | Path, extra: dict | None) -> None:
    """对话原文或文档段落位置，供审核页按气泡 / 上下文显示。没有则不写。"""
    if not extra:
        return
    extra_path(run_path).write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")


def load_extra(run_path: str | Path) -> dict | None:
    p = extra_path(run_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def delete_run(run_path: str | Path, web_root: str | Path = "output/web") -> None:
    """删除一次运行：网页运行删除整个时间戳目录；命令行运行只删除同名的结果文件，保留目录里的其他内容。"""
    p = Path(run_path)
    if p.parent.parent.resolve() == Path(web_root).resolve():
        shutil.rmtree(p.parent, ignore_errors=True)
        return
    for f in (p, p.with_suffix(".csv"), p.with_name(f"{p.stem}_need_human.csv"),
              meta_path(p), p.with_name(f"{p.stem}_reviews.json"), extra_path(p)):
        f.unlink(missing_ok=True)


def settings_text(meta: dict, records: list[dict]) -> str:
    """一句话描述运行策略，没有元信息时从结果推断。"""
    s = meta.get("settings") or {}
    models = [m["name"] for m in meta.get("models") or []] or ([p["model"] for p in records[0]["round1"]] if records else [])
    parts = [" / ".join(models)]
    if s.get("fewshot"):
        parts.append("动态示例" + ("·向量" if s.get("retriever") == "embedding" else ""))
    if s.get("shuffle_labels"):
        parts.append("选项随机")
    if s.get("cascade"):
        th = s.get("cascade_min_confidence") or 0
        parts.append(f"级联[{'+'.join(s['cascade'])}{f' ≥{th:g}' if th else ''}]")
    if (s.get("samples") or 1) > 1:
        parts.append(f"采样×{s['samples']}")
    if s.get("require_evidence"):
        parts.append("证据引用")
    if s.get("calibration"):
        parts.append("校准" + (f"[后验≥{s['min_posterior']:g}]" if s.get("min_posterior") else ""))
    action = s.get("disagreement_action")
    if action:
        parts.append({"human": "分歧转人工", "arbiter": "分歧仲裁", "review": "复核投票"}.get(action, action))
    if action == "review":
        if (s.get("debate_rounds") or 1) > 1:
            parts.append(f"辩论≤{s['debate_rounds']}轮")
        if s.get("devil_advocate"):
            parts.append("魔鬼代言人")
        if s.get("review_view") in ("reasons", "labels"):
            parts.append({"reasons": "只看理由", "labels": "只看标签"}[s["review_view"]])
    if s.get("jury") and action != "human":
        parts.append(f"评审团[{'+'.join(s['jury'])}]")
    if s.get("multi_label"):
        parts.append("多标签")
    if s.get("hierarchical"):
        parts.append("层级")
    return "，".join(parts)


def _gold_of(records: list[dict], gold: dict[str, str] | None) -> dict[str, str]:
    if gold:
        return {r["id"]: gold[r["id"]] for r in records if r["id"] in gold}
    return {r["id"]: r["gold"] for r in records if r.get("gold")}


def summarize(records: list[dict], meta: dict | None = None, gold: dict[str, str] | None = None) -> dict:
    """一次运行的汇总指标。gold 不传时使用结果文件中保存的真实标签（评估运行才有）。"""
    meta = meta or {}
    n = len(records)
    auto = [r for r in records if r["status"] != STATUS_HUMAN]
    local = {m["name"] for m in meta.get("models") or [] if m.get("provider") in LOCAL_PROVIDERS} or {"local"}
    def calls_of(r: dict) -> int:
        rounds = r.get("debate") or [r.get("round2") or []]
        n = sum(len(p.get("samples") or [0]) for p in r["round1"] if p["model"] not in local)
        n += sum(1 for rnd in rounds for p in rnd if p["model"] not in local)
        n += len(r.get("jury") or []) or (1 if r.get("arbiter") else 0)
        return n + (1 if r.get("devil") else 0)

    llm_calls = sum(calls_of(r) for r in records)
    out = {
        "n": n,
        "auto_rate": len(auto) / n if n else 0.0,
        "human": n - len(auto),
        "llm_calls": llm_calls,
        "calls_per_item": llm_calls / n if n else 0.0,
        "cost": meta.get("cost"),
        "full_cost": (meta["cost"] + meta.get("saved", 0.0)) if meta.get("cost") is not None else None,
    }
    g = _gold_of(records, gold)
    out["gold_n"] = len(g)
    if g:
        rs = [r for r in records if r["id"] in g]
        auto_g = [r for r in rs if r["status"] != STATUS_HUMAN]
        ok_auto = sum(r["label"] == g[r["id"]] for r in auto_g)
        out.update({
            "acc": sum(r["label"] == g[r["id"]] for r in rs) / len(rs),
            "auto_acc": ok_auto / len(auto_g) if auto_g else None,
            "with_human": (ok_auto + len(rs) - len(auto_g)) / len(rs),
        })
        per_model = {}
        for name in ([p["model"] for p in rs[0]["round1"]] if rs else []):
            called = [(r, p) for r in rs for p in r["round1"] if p["model"] == name]
            ok = [(r, p) for r, p in called if p.get("label") and not p.get("error")]
            per_model[name] = sum(p["label"] == g[r["id"]] for r, p in ok) / len(called) if called else None
        out["per_model"] = per_model
    return out


def item_correct(rec: dict, gold: str, metric: str) -> bool:
    """metric：acc 全自动（分歧样本也取模型建议）；with_human 人工兜底（需人工的样本视为判对）。"""
    if metric == "with_human" and rec["status"] == STATUS_HUMAN:
        return True
    return rec["label"] == gold


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """配对样本的 McNemar 精确检验：只看两次运行结论不同的样本。"""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    m = only_a + only_b
    if m == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m)
    return {"only_a": only_a, "only_b": only_b, "p": p}


def compare_items(runs: dict[str, list[dict]], gold: dict[str, str] | None = None) -> tuple[list[str], list[dict]]:
    """多次运行在共同样本上的逐条对照。返回 (共同 id 列表, 每条一行的字典)。"""
    maps = {name: {r["id"]: r for r in recs} for name, recs in runs.items()}
    names = list(maps)
    common = [i for i in maps[names[0]] if all(i in m for m in maps.values())]
    if gold is None:
        gold = {}
        for m in maps.values():
            gold.update({i: r["gold"] for i, r in m.items() if r.get("gold")})
    rows = []
    for i in common:
        first = maps[names[0]][i]
        row = {"id": i, "text": first["text"], "gold": gold.get(i)}
        for name in names:
            r = maps[name][i]
            row[name] = r["label"]
            row[f"{name}|status"] = r["status"]
        rows.append(row)
    return common, rows
