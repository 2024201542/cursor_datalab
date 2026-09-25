from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from .aggregate import dawid_skene
from .config import Config, load_config, load_dotenv
from .evaluate import evaluate, format_report
from .io_utils import read_items, write_results
from .llm import LLMError, create_llm
from .pipeline import STATUS_TEXT, CrossCheckPipeline, ItemResult
from .report import build_report


def _load_weights(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    return {k: float(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def _print_summary(results: list[ItemResult], stats: dict, elapsed: float) -> None:
    counts = Counter(r.status for r in results)
    print(f"\n共处理 {len(results)} 条，用时 {elapsed:.1f}s")
    for status, text in STATUS_TEXT.items():
        n = counts.get(status, 0)
        print(f"  {text:<10} {n:>5} 条 ({n / max(len(results), 1) * 100:.1f}%)")
    print("模型调用统计（实际请求 / 缓存命中 / 失败）:")
    for name, s in stats.items():
        print(f"  {name:<16} {s['calls']:>5} / {s['cache_hits']:>5} / {s['errors']:>5}")


async def _run_pipeline(config: Config, items: list[dict], weights) -> tuple[list[ItemResult], dict, float]:
    start = time.monotonic()
    async with CrossCheckPipeline(config, weights) as pipe:
        results = await pipe.run(items)
        return results, pipe.stats(), time.monotonic() - start


def cmd_classify(args) -> None:
    config = load_config(args.config, mock=args.mock)
    items = read_items(args.input, args.text_col, args.id_col)
    if args.limit:
        items = items[: args.limit]
    results, stats, elapsed = asyncio.run(_run_pipeline(config, items, _load_weights(args.weights)))
    paths = write_results(results, args.output, [m.name for m in config.models])
    _print_summary(results, stats, elapsed)
    print(f"\n结果: {paths['csv']}\n明细: {paths['jsonl']}\n人工审核: {paths['human']}")


def cmd_evaluate(args) -> None:
    config = load_config(args.config, mock=args.mock)
    items = read_items(args.input, args.text_col, args.id_col, label_col=args.label_col)
    labels = config.task.label_names
    bad = sorted({it["label"] for it in items} - set(labels))
    if bad:
        raise ValueError(f"金标准中存在配置里没有的标签: {bad}")

    loaded = _load_weights(args.weights)
    results, stats, elapsed = asyncio.run(_run_pipeline(config, items, loaded))
    model_names = [m.name for m in config.models]
    write_results(results, args.output, model_names, prefix="eval")
    _print_summary(results, stats, elapsed)

    weights = {m.name: m.weight for m in config.models} | (loaded or {})
    rep = evaluate(results, {it["id"]: it["label"] for it in items}, model_names, labels, weights)
    text = format_report(rep, labels)
    out = Path(args.output)
    (out / "weights.json").write_text(json.dumps(rep["suggested_weights"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "eval_report.txt").write_text(text, encoding="utf-8")
    html_path = build_report(rep, out)
    print("\n" + text)
    print(f"\n文字报告: {out / 'eval_report.txt'}\n图表报告: {html_path.resolve()}")


def cmd_aggregate(args) -> None:
    config = load_config(args.config)
    labels = config.task.label_names
    records = []
    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    annotations = [
        (rec["id"], p["model"], p["label"])
        for rec in records
        for p in rec["round1"]
        if p.get("label") and not p.get("error")
    ]
    ds = dawid_skene(annotations, labels)

    print("[Dawid-Skene 估计的模型准确率]（无需标准答案，基于首轮结果）")
    for w, acc in sorted(ds.worker_accuracy.items(), key=lambda x: -x[1]):
        print(f"  {w:<12} {acc * 100:.1f}%")
    print("[估计的类别分布]")
    for lab, p in zip(labels, ds.class_prior):
        print(f"  {lab:<8} {p * 100:.1f}%")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ds_results.csv"
    diff = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "pipeline_label", "pipeline_status", "ds_label", "ds_prob", "same"])
        for rec in records:
            ds_label = ds.labels.get(rec["id"], "")
            same = ds_label == rec["label"]
            diff += not same
            w.writerow([rec["id"], rec["text"], rec["label"], rec["status"], ds_label,
                        round(ds.probs.get(rec["id"], 0.0), 3), "是" if same else "否"])
    print(f"\n与互检流水线结果不一致: {diff}/{len(records)} 条\n已保存: {path}")


async def _ping(config: Config) -> None:
    targets = config.models + ([config.arbiter] if config.arbiter else [])
    async with httpx.AsyncClient(timeout=config.pipeline.timeout) as http:
        for m in targets:
            start = time.monotonic()
            try:
                llm = create_llm(m, http, config)
                reply = await llm.chat("You are a helpful assistant.", "请只回复两个字母：OK")
                print(f"[成功] {m.name:<10} {m.provider}/{m.model}  {time.monotonic() - start:.1f}s  回复: {reply.strip()[:60]}")
            except LLMError as e:
                print(f"[失败] {m.name:<10} {m.provider}/{m.model}  {e}")


def cmd_ping(args) -> None:
    asyncio.run(_ping(load_config(args.config, mock=args.mock)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m crosscheck", description="多模型互检分类")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, with_io=True):
        p.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
        p.add_argument("--mock", action="store_true", help="所有模型切换为 mock，不调用真实 API")
        if with_io:
            p.add_argument("-o", "--output", default="output", help="输出目录")
            p.add_argument("--text-col", default="text", help="文本列名")
            p.add_argument("--id-col", default="id", help="ID 列名")
            p.add_argument("--weights", help="模型权重 json（evaluate 生成的 weights.json）")

    p = sub.add_parser("classify", help="对数据进行互检分类")
    p.add_argument("input", help="输入 .csv / .jsonl")
    p.add_argument("--limit", type=int, help="只处理前 N 条（试跑用）")
    common(p)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("evaluate", help="在人工标注的金标准数据上评估，并生成模型权重")
    p.add_argument("input", help="带标签列的 .csv / .jsonl")
    p.add_argument("--label-col", default="label", help="标签列名")
    common(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("aggregate", help="对 classify 的结果做 Dawid-Skene 无监督聚合")
    p.add_argument("input", help="classify 输出的 results.jsonl")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-o", "--output", default="output")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("ping", help="测试各模型 API 是否可用")
    common(p, with_io=False)
    p.set_defaults(func=cmd_ping)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (LLMError, ValueError, FileNotFoundError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
