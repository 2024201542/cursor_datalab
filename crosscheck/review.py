"""人工审核：读取运行结果、筛选审核范围、保存审核记录、统计、回流金标准。"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

SCOPES = {
    "need_human": "需人工审核的样本",
    "disagree": "首轮有分歧的全部样本",
    "spot": "抽检：首轮一致通过的样本",
    "all": "全部样本",
}


def list_runs(root: str | Path = "output") -> list[Path]:
    files = [p for p in Path(root).rglob("*.jsonl") if p.name in ("results.jsonl", "eval.jsonl")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_run(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_model_names(records: list[dict]) -> list[str]:
    return [p["model"] for p in records[0]["round1"]] if records else []


def reviews_path(run_path: str | Path) -> Path:
    p = Path(run_path)
    return p.with_name(f"{p.stem}_reviews.json")


def load_reviews(run_path: str | Path) -> dict[str, dict]:
    p = reviews_path(run_path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_reviews(run_path: str | Path, reviews: dict[str, dict]) -> None:
    p = reviews_path(run_path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(reviews, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def set_review(reviews: dict[str, dict], rec: dict, label: str, note: str = "") -> None:
    reviews[rec["id"]] = {
        "label": label,
        "note": note,
        "machine_label": rec["label"],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def round1_votes(rec: dict) -> dict[str, int]:
    votes: dict[str, int] = {}
    for p in rec["round1"]:
        if p.get("label") and not p.get("error"):
            votes[p["label"]] = votes.get(p["label"], 0) + 1
    return votes


def is_disagreement(rec: dict) -> bool:
    votes = round1_votes(rec)
    return len(votes) > 1 or sum(votes.values()) < len(rec["round1"])


def select_queue(records: list[dict], scope: str, spot_n: int = 20, seed: int = 42) -> list[dict]:
    if scope == "need_human":
        return [r for r in records if r["status"] == "need_human"]
    if scope == "disagree":
        return [r for r in records if is_disagreement(r)]
    if scope == "spot":
        pool = [i for i, r in enumerate(records) if r["status"] == "consensus"]
        picked = sorted(random.Random(seed).sample(pool, min(spot_n, len(pool))))
        return [records[i] for i in picked]
    return list(records)


def review_stats(records: list[dict], reviews: dict[str, dict], model_names: list[str]) -> dict:
    reviewed = [r for r in records if r["id"] in reviews]
    n = len(reviewed)
    if not n:
        return {"n": 0}
    human = {r["id"]: reviews[r["id"]]["label"] for r in reviewed}
    agree = {"互检系统": sum(r["label"] == human[r["id"]] for r in reviewed) / n}
    for m in model_names:
        agree[m] = sum(
            any(p["model"] == m and p.get("label") == human[r["id"]] for p in r["round1"]) for r in reviewed
        ) / n
    by_status: dict[str, dict] = {}
    for r in reviewed:
        s = by_status.setdefault(r["status"], {"n": 0, "agree": 0})
        s["n"] += 1
        s["agree"] += r["label"] == human[r["id"]]
    return {
        "n": n,
        "changed": sum(r["label"] != human[r["id"]] for r in reviewed),
        "agree": agree,
        "by_status": by_status,
    }


def merged_rows(records: list[dict], reviews: dict[str, dict]) -> list[dict]:
    rows = []
    for r in records:
        rv = reviews.get(r["id"])
        rows.append({
            "id": r["id"],
            "text": r["text"],
            "machine_label": r["label"],
            "status": r["status"],
            "human_label": rv["label"] if rv else "",
            "final_label": rv["label"] if rv else r["label"],
            "source": "人工" if rv else "模型",
            "note": rv.get("note", "") if rv else "",
        })
    return rows


def _read_csv_any(path: Path) -> tuple[list[str], list[dict]]:
    for enc in ("utf-8-sig", "gbk"):
        try:
            with path.open(encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader.fieldnames or []), list(reader)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别 {path} 的编码")


def append_to_gold(records: list[dict], reviews: dict[str, dict], gold_path: str | Path) -> tuple[int, int]:
    """把已审核样本写入金标准 csv（按文本去重），返回 (新增条数, 因重复跳过条数)。"""
    path = Path(gold_path)
    fieldnames, rows = (["id", "text", "label"], [])
    if path.exists():
        fieldnames, rows = _read_csv_any(path)
        missing = {"text", "label"} - set(fieldnames)
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    existing = {row["text"].strip() for row in rows}
    added = skipped = 0
    for r in records:
        rv = reviews.get(r["id"])
        if not rv:
            continue
        if r["text"].strip() in existing:
            skipped += 1
            continue
        rows.append({"id": r["id"], "text": r["text"], "label": rv["label"]})
        existing.add(r["text"].strip())
        added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return added, skipped
