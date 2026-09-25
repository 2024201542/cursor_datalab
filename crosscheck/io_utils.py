from __future__ import annotations

import csv
import json
from pathlib import Path

from .pipeline import STATUS_HUMAN, STATUS_TEXT, ItemResult


def _read_csv(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "gbk"):
        try:
            with path.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别 {path} 的编码，请另存为 UTF-8")


def read_items(path: str | Path, text_col: str = "text", id_col: str = "id", label_col: str | None = None) -> list[dict]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(p)
    elif suffix == ".jsonl":
        with p.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError("输入文件仅支持 .csv 或 .jsonl")

    items = []
    for n, row in enumerate(rows, 1):
        if text_col not in row:
            raise ValueError(f"第 {n} 行缺少文本列 '{text_col}'，现有列: {list(row)}")
        text = str(row[text_col] or "").strip()
        if not text:
            continue
        item = {"id": str(row.get(id_col) or n), "text": text}
        if label_col:
            if label_col not in row:
                raise ValueError(f"第 {n} 行缺少标签列 '{label_col}'")
            item["label"] = str(row[label_col] or "").strip()
        items.append(item)
    return items


def _pred_cells(preds, name):
    for p in preds:
        if p.model == name:
            return (p.label or f"错误:{(p.error or '')[:40]}"), (round(p.confidence, 2) if p.ok else "")
    return "", ""


def write_results(results: list[ItemResult], out_dir: str | Path, model_names: list[str], prefix: str = "results") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "jsonl": out / f"{prefix}.jsonl",
        "csv": out / f"{prefix}.csv",
        "human": out / f"{prefix}_need_human.csv",
    }

    with paths["jsonl"].open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    header = ["id", "text", "label", "status", "status_text", "confidence"]
    for m in model_names:
        header += [f"{m}_r1", f"{m}_r1_conf", f"{m}_r2"]
    header += ["arbiter", "arbiter_conf", "note"]

    def row_of(r: ItemResult) -> list:
        row = [r.id, r.text, r.label or "", r.status, STATUS_TEXT[r.status], r.confidence]
        for m in model_names:
            l1, c1 = _pred_cells(r.round1, m)
            l2, _ = _pred_cells(r.round2, m)
            row += [l1, c1, l2]
        a = r.arbiter
        row += [(a.label if a and a.ok else ""), (round(a.confidence, 2) if a and a.ok else ""), r.note]
        return row

    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(row_of(r) for r in results)

    with paths["human"].open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "suggested_label", "human_label", "opinions"])
        for r in results:
            if r.status != STATUS_HUMAN:
                continue
            final = r.round2 or r.round1
            ops = [f"{p.model}: {p.label}({p.confidence:.2f}) {p.reason}" for p in final if p.ok]
            if r.arbiter and r.arbiter.ok:
                ops.append(f"仲裁: {r.arbiter.label}({r.arbiter.confidence:.2f}) {r.arbiter.reason}")
            w.writerow([r.id, r.text, r.label or "", "", " | ".join(ops)])
    return paths
