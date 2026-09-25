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
from .config import Config, load_config, load_dotenv, read_raw_config
from .cost import estimate_run, format_estimate
from .evaluate import evaluate, format_report
from .history import write_meta
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
    print("模型调用统计（实际请求 / 缓存命中 / 失败 | 输入 token / 输出 token | 费用 / 缓存节省）:")
    for name, s in stats.items():
        est = f"（{s['estimated_calls']} 次接口未返回用量，按字数估算）" if s.get("estimated_calls") else ""
        print(f"  {name:<16} {s['calls']:>5} / {s['cache_hits']:>5} / {s['errors']:>5} | "
              f"{s['in_tokens']:>9} / {s['out_tokens']:>8} | {s['cost']:.4f} 元 / {s['saved']:.4f} 元{est}")
    total, saved = sum(s["cost"] for s in stats.values()), sum(s["saved"] for s in stats.values())
    print(f"  合计费用 {total:.4f} 元，缓存节省 {saved:.4f} 元（未设置单价的模型按 0 计算）")


def _checkpoint(out: str | Path, prefix: str) -> Path:
    return Path(out) / f"{prefix}.checkpoint.jsonl"


def _load_checkpoint(path: Path, items: list[dict], resume: bool) -> list[ItemResult]:
    if not path.exists():
        return []
    if not resume:
        print(f"提示：发现上次未完成的运行 {path}，本次从头开始（加 --resume 可接着跑）。", file=sys.stderr)
        path.unlink()
        return []
    ids = {it["id"] for it in items}
    done = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = ItemResult.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # 中断时最后一行可能没写完整
            if r.id in ids:
                done.append(r)
    done = list({r.id: r for r in done}.values())
    print(f"断点续跑：已完成 {len(done)} 条，剩余 {len(items) - len(done)} 条。", file=sys.stderr)
    return done


async def _run_pipeline(config: Config, items: list[dict], weights, checkpoint: Path | None = None,
                        resume: bool = False) -> tuple[list[ItemResult], dict, float]:
    start = time.monotonic()
    done = _load_checkpoint(checkpoint, items, resume) if checkpoint else []
    finished = {r.id for r in done}
    todo = [it for it in items if it["id"] not in finished]
    fh = None
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        fh = checkpoint.open("a", encoding="utf-8")

    def save(res: ItemResult) -> None:
        fh.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
        fh.flush()

    try:
        async with CrossCheckPipeline(config, weights) as pipe:
            new = await pipe.run(todo, on_result=save if fh else None)
            stats = pipe.stats()
    finally:
        if fh:
            fh.close()
    by_id = {r.id: r for r in done + new}
    return [by_id[it["id"]] for it in items], stats, time.monotonic() - start


def _check_budget(config: Config, items: list[dict], args) -> None:
    """--budget：预计费用超出预算时不开始运行；--estimate-only：只打印预估。"""
    if not (args.budget or args.estimate_only):
        return
    est = estimate_run(config, items, args.disagree_rate)
    print(format_estimate(est), file=sys.stderr)
    if args.estimate_only:
        sys.exit(0)
    if est["expected"]["cost"] > args.budget:
        raise ValueError(f"预计费用 {est['expected']['cost']:.4f} 元超过预算 {args.budget} 元，未开始运行。"
                         "可以先用 --limit 试跑，或开启级联 / 调整模型。")


def _save_meta(args, config: Config, path: Path, source: str, stats: dict, elapsed: float, n: int, has_gold: bool) -> None:
    write_meta(path, source=source, input_name=Path(args.input).as_posix(), config=config,
               raw=read_raw_config(args.config), stats=stats, elapsed=elapsed, n=n, has_gold=has_gold,
               config_path=Path(args.config).as_posix(), note=args.note)


def cmd_classify(args) -> None:
    config = load_config(args.config, mock=args.mock)
    items = read_items(args.input, args.text_col, args.id_col)
    if args.limit:
        items = items[: args.limit]
    _check_budget(config, items, args)
    ckpt = _checkpoint(args.output, "results")
    results, stats, elapsed = asyncio.run(
        _run_pipeline(config, items, _load_weights(args.weights), ckpt, args.resume))
    paths = write_results(results, args.output, [m.name for m in config.models])
    _save_meta(args, config, paths["jsonl"], "classify", stats, elapsed, len(items), False)
    ckpt.unlink(missing_ok=True)
    _print_summary(results, stats, elapsed)
    print(f"\n结果: {paths['csv']}\n明细: {paths['jsonl']}\n人工审核: {paths['human']}")


def cmd_evaluate(args) -> None:
    config = load_config(args.config, mock=args.mock)
    items = read_items(args.input, args.text_col, args.id_col, label_col=args.label_col)
    labels = config.task.label_names
    bad = sorted({it["label"] for it in items if config.task.normalize_gold(it["label"]) is None})
    if bad:
        raise ValueError(f"金标准中存在配置里没有的标签: {bad}")
    for it in items:
        it["label"] = config.task.normalize_gold(it["label"])

    loaded = _load_weights(args.weights)
    _check_budget(config, items, args)
    ckpt = _checkpoint(args.output, "eval")
    results, stats, elapsed = asyncio.run(_run_pipeline(config, items, loaded, ckpt, args.resume))
    model_names = [m.name for m in config.models]
    gold = {it["id"]: it["label"] for it in items}
    paths = write_results(results, args.output, model_names, prefix="eval", gold=gold)
    _save_meta(args, config, paths["jsonl"], "evaluate", stats, elapsed, len(items), True)
    ckpt.unlink(missing_ok=True)
    _print_summary(results, stats, elapsed)

    weights = {m.name: m.weight for m in config.models} | (loaded or {})
    rep = evaluate(results, gold, model_names, labels, weights, config.task.multi_label, config.task.hierarchical)
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


def cmd_calibrate(args) -> None:
    from .calibrate import (calibration_report, compare_aggregators, crossfit_posteriors, fit_calibration,
                            pick_threshold, posterior_curve, simulate_thresholds)

    config = load_config(args.config)
    labels = config.task.label_names
    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not any(r.get("gold") in labels for r in records):
        raise ValueError("结果文件中没有可用的标准答案（需要 evaluate 生成的 eval.jsonl）")
    pct = lambda x: "—" if x is None else f"{x * 100:.1f}%"

    cmp = compare_aggregators(records, labels)
    print(f"[聚合方式对比] {cmp['n']} 条金标准；需要标准答案的方法用 {cmp['folds']} 折交叉验证")
    for r in cmp["rows"]:
        print(f"  {r['方法']:<18} {pct(r['准确率'])}{'  (交叉验证)' if r['需要标准答案'] else ''}")
    print(f"  推荐：{cmp['best']}")

    print("\n[置信度校准] ECE 越小越好（校准后的数字为交叉拟合）")
    for r in calibration_report(records, labels):
        after = "—" if r["校准后 ECE"] is None else f"{r['校准后 ECE']:.3f}"
        print(f"  {r['模型']:<12} 准确率 {pct(r['准确率'])}  平均置信度 {pct(r['平均置信度'])}  "
              f"ECE {r['校准前 ECE']:.3f} → {after}（{r['方法']}）")

    target = args.target
    curve = posterior_curve(crossfit_posteriors(records, labels))
    best = pick_threshold(curve, target)
    print(f"\n[阈值搜索] 目标：自动采纳部分的准确率 ≥ {pct(target)}")
    if best:
        print(f"  min_posterior = {best['门槛']:g}：自动采纳 {pct(best['自动采纳比例'])}，准确率 {pct(best['自动采纳准确率'])}")
    else:
        top = max((r for r in curve if r["自动采纳准确率"] is not None), key=lambda r: r["自动采纳准确率"], default=None)
        print("  按后验概率无法达到目标" + (f"（最高 {pct(top['自动采纳准确率'])}，自动采纳 {pct(top['自动采纳比例'])}）" if top else ""))
    sim = pick_threshold(simulate_thresholds(records, labels), target)
    if sim and (sim["accept_threshold"] is not None or sim["arbiter_threshold"] is not None):
        print(f"  accept_threshold = {sim['accept_threshold']}，arbiter_threshold = {sim['arbiter_threshold']}："
              f"自动采纳 {pct(sim['自动采纳比例'])}，准确率 {pct(sim['自动采纳准确率'])}（按已有复核 / 仲裁记录模拟）")

    cal = fit_calibration(records, labels, source=Path(args.input).as_posix())
    path = cal.save(args.output)
    print(f"\n校准文件：{path}\n使用方法：在配置的 pipeline 中加入 calibration: {path.as_posix()}"
          + (f" 和 min_posterior: {best['门槛']:g}" if best else ""))


async def _ping(config: Config) -> None:
    targets = config.models + ([config.arbiter] if config.arbiter else []) + config.jury
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


def _read_eval(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_evolve(args) -> None:
    """不调用模型：优先审核能抓住多少错，以及哪些标准答案和模型的一致意见相反。"""
    from .evolve import audit, local_gain, noise_candidates

    records = _read_eval(args.input)
    pct = lambda x: f"{x * 100:.1f}%"
    rows = audit(records)
    print(f"[主动学习] {sum(1 for r in records if r.get('gold'))} 条有标准答案")
    if not rows:
        print("  没有可以比较的判错（缺少 gold，或系统和标准答案完全一致）。")
    else:
        print(f"  {'排序':<24} {'审核':>4}  抓住  召回    随机期望")
        for r in rows:
            print(f"  {r['排序']:<24} {r['审核条数']:>4}  {r['抓住的错']:>3}/{r['错误总数']:<3}  {pct(r['召回']):>6}  {r['随机期望']:>6}")
    noise = noise_candidates(records)
    print(f"\n[可能标错的标签] {len(noise)} 条：至少 2 个模型以 ≥ 0.75 的平均置信度同意另一个标签")
    for r in noise[:20]:
        text = (r["text"] or "").replace("\n", " ")
        if len(text) > 60:
            text = text[:60] + "…"
        print(f"  {r['gold']} → {r['suspect']}  {r['同意该标签的模型数']}/{r['模型数']} 个模型  置信度 {r['平均置信度']:.2f}  {text}")
    if not noise:
        print("  没有。")
    if args.train:
        from .local_model import load_examples

        labels = sorted({r.get("gold") for r in records if r.get("gold")})
        texts, ys = load_examples(args.train, labels)
        pool = [r for r in records if r.get("gold") and r.get("text")]
        print(f"\n[加进本地小模型] 训练集 {len(texts)} 条；把金标准里的 k 条加进去后，在其余金标准上的准确率")
        for k in (20, 50):
            if len(pool) <= k + 1:
                continue
            active = local_gain(texts, ys, pool, k, "active")
            rand = [local_gain(texts, ys, pool, k, "random", seed=s) for s in range(5)]
            rand_s = sum(x or 0 for x in rand) / len(rand)
            print(f"  加 {k} 条：按不确定度 {pct(active or 0)}，随机 5 次平均 {pct(rand_s)}")


def _optimize_items(path: str, text_col: str, label_col: str) -> list[dict]:
    if str(path).endswith(".jsonl"):
        items = []
        for r in _read_eval(path):
            gold = r.get("gold") or r.get(label_col)
            if gold and r.get(text_col):
                items.append({"id": str(r.get("id")), "text": r[text_col], "label": gold})
        return items
    return read_items(path, text_col=text_col, label_col=label_col)


def cmd_optimize(args) -> None:
    from .optimize import format_optimize, optimize_rules

    raw = read_raw_config(args.config)
    if args.mock:
        for key in ("models", "jury"):
            for m in raw.get(key) or []:
                if m.get("provider") != "local":
                    m["provider"] = "mock"
        if isinstance(raw.get("arbiter"), dict) and raw["arbiter"].get("provider") != "local":
            raw["arbiter"]["provider"] = "mock"
    items = _optimize_items(args.input, args.text_col, args.label_col)
    if not items:
        raise ValueError("没有带标准答案的样本")
    rep = optimize_rules(raw, items, limit=args.limit, holdout_ratio=args.holdout, log=print)
    text = format_optimize(rep)
    print("\n" + text)
    out = Path(args.output)
    if args.output == "output" or out.is_dir():
        out = out / "optimize_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n\n胜出的规则：\n" + "\n".join(f"- {r}" for r in rep["winner_rules"]), encoding="utf-8")
    print(f"\n已写入 {out}")
    if rep["winner"] != "当前规则":
        print("规则没有自动写进配置。确认留出集上确实更好之后，再复制到分类任务里。")


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
            p.add_argument("--resume", action="store_true", help="从上次中断的位置继续（读取输出目录中的断点文件）")
            p.add_argument("--budget", type=float, default=0, help="费用预算（元），预计费用超出时不运行")
            p.add_argument("--estimate-only", action="store_true", help="只预估调用次数和费用，不运行")
            p.add_argument("--disagree-rate", type=float, default=0.25, help="预估时假设的首轮分歧比例（默认 0.25）")
            p.add_argument("--note", default="", help="本次运行的备注，显示在网页的历史记录中")

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

    p = sub.add_parser("calibrate", help="用 evaluate 的结果拟合校准文件、比较聚合方式、搜索阈值（不调用模型）")
    p.add_argument("input", help="evaluate 输出的 eval.jsonl")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-o", "--output", default="output/calibration.json", help="校准文件保存路径")
    p.add_argument("--target", type=float, default=0.95, help="自动采纳部分的目标准确率（默认 0.95）")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("ping", help="测试各模型 API 是否可用")
    common(p, with_io=False)
    p.set_defaults(func=cmd_ping)

    p = sub.add_parser("evolve", help="不调用模型：主动学习能抓住多少错、哪些标签可能标错")
    p.add_argument("input", help="evaluate 输出的 eval.jsonl")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("--train", help="训练集 csv。给出后，比较把优先样本加进本地小模型的效果")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("optimize", help="在金标准上搜索边界规则（会调用模型，留出集只评一次）")
    p.add_argument("input", help="金标准 .csv / .jsonl，或 evaluate 的 eval.jsonl")
    p.add_argument("--label-col", default="label")
    p.add_argument("--limit", type=int, default=36, help="最多用多少条（默认 36），按类别比例抽取")
    p.add_argument("--holdout", type=float, default=0.34, help="留出集比例，只用于报告、不参与挑选")
    common(p)
    p.set_defaults(func=cmd_optimize)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (LLMError, ValueError, FileNotFoundError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
