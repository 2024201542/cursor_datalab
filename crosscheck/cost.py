"""token 估算、费用计算，以及运行前的调用次数与费用预估。"""
from __future__ import annotations

import re

from .config import Config, ModelConfig

_CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]")
DEFAULT_OUT_TOKENS = 50  # 投票模型一次回复（JSON + 短理由）的典型输出长度，实测 40~52


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 0.7 token / 字，其他字符约 0.35 token / 字（按 DeepSeek / Kimi / Qwen 实测用量校准，误差约 ±10%）。"""
    cjk = len(_CJK_RE.findall(text))
    return int(cjk * 0.7 + (len(text) - cjk) * 0.35) + 1


def cost_of(cfg: ModelConfig, in_tokens: float, out_tokens: float) -> float:
    return (in_tokens * cfg.price_in + out_tokens * cfg.price_out) / 1_000_000


def priced(cfg: ModelConfig) -> bool:
    return cfg.price_in > 0 or cfg.price_out > 0


def arbiter_out_tokens(cfg: ModelConfig) -> int:
    """仲裁模型常开思考，输出长度按 max_tokens 的一半估算。"""
    return max(DEFAULT_OUT_TOKENS, cfg.max_tokens // 2)


def estimate_run(config: Config, items: list[dict], disagree_rate: float = 0.25) -> dict:
    """按当前配置估算处理 items 需要的调用次数、token 和费用。

    首轮提示词逐条构造并查缓存，结果是准确的；后续环节（级联的其余模型、复核、仲裁）取决于模型之间
    是否一致，事先无法知道，所以给出三档：最少（全部一致）、预计（按 disagree_rate 的比例出现分歧）、
    最多（全部分歧），这些环节一律按未命中缓存计算。
    """
    from .classifier import LLMCache
    from .local_model import get_bank
    from .prompts import SYSTEM_PROMPT, build_classify_prompt, prompt_seed

    pc, fs = config.pipeline, config.fewshot
    cache = LLMCache(config.cache.path, config.cache.enabled)
    bank = get_bank(fs.path, tuple(config.task.label_names), fs.text_col, fs.label_col) if fs.enabled else None
    first = set(pc.cascade) if pc.cascade else {m.name for m in config.models}
    uses_arbiter = config.arbiter is not None and pc.use_arbiter and pc.disagreement_action != "human"
    n = len(items)

    rows = {}
    for m in config.models:
        rows[m.name] = {"billable": m.provider not in ("local", "mock"), "r1_calls": 0, "r1_cached": 0, "r1_in": 0}
    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    review_extra = 0
    for it in items:
        ex = bank.nearest(it["text"], fs.k) if bank is not None else None
        user = build_classify_prompt(config.task, it["text"], ex, fs.max_chars)
        tokens = sys_tokens + estimate_tokens(user)
        for m in config.models:
            r = rows[m.name]
            if not r["billable"]:
                continue
            if config.task.shuffle_labels:
                user = build_classify_prompt(config.task, it["text"], ex, fs.max_chars,
                                             prompt_seed(config.task, m.name, it["text"]))
            if cache.get(LLMCache.make_key(m, SYSTEM_PROMPT, user)) is not None:
                r["r1_cached"] += 1
            else:
                r["r1_calls"] += 1
                r["r1_in"] += tokens
        review_extra += tokens
    cache.close()
    avg_in = review_extra / n if n else 0
    # 复核 / 仲裁提示词在首轮提示词基础上附带其他模型的意见，约多 60 token / 条意见
    opinions = len(config.models) - 1
    review_in = avg_in + 60 * opinions
    arb_in = avg_in + 60 * len(config.models)

    def scenario(rate: float) -> dict:
        out, total, total_calls = [], 0.0, 0
        for m in config.models:
            r = rows[m.name]
            if not r["billable"]:
                out.append({"模型": m.name, "调用次数": 0, "缓存命中": 0, "输入 token": 0, "输出 token": 0, "费用": 0.0})
                continue
            share = 1.0 if m.name in first else rate
            calls = r["r1_calls"] * share
            tin = r["r1_in"] * share
            if pc.disagreement_action == "review" and pc.cross_review:
                calls += n * rate
                tin += n * rate * review_in
            tout = calls * DEFAULT_OUT_TOKENS
            c = cost_of(m, tin, tout)
            out.append({"模型": m.name, "调用次数": round(calls), "缓存命中": round(r["r1_cached"] * share),
                        "输入 token": round(tin), "输出 token": round(tout), "费用": c})
            total += c
            total_calls += calls
        a = config.arbiter
        if uses_arbiter:
            calls = n * rate
            tin, tout = calls * arb_in, calls * arbiter_out_tokens(a)
            c = cost_of(a, tin, tout)
            out.append({"模型": f"{a.name}（仲裁）", "调用次数": round(calls), "缓存命中": 0,
                        "输入 token": round(tin), "输出 token": round(tout), "费用": c})
            total += c
            total_calls += calls
        return {"rows": out, "cost": total, "calls": round(total_calls)}

    targets = config.models + ([config.arbiter] if uses_arbiter else [])
    unpriced = [m.name for m in targets if m.provider not in ("local", "mock") and not priced(m)]
    return {
        "n": n,
        "disagree_rate": disagree_rate,
        "min": scenario(0.0),
        "expected": scenario(disagree_rate),
        "max": scenario(1.0),
        "unpriced": unpriced,
    }


def format_estimate(est: dict) -> str:
    lines = [f"待处理 {est['n']} 条。首轮提示词已逐条构造并查询缓存；后续环节按“分歧比例”估算。"]
    for key, title in (("min", "最少（全部一致）"), ("expected", f"预计（{est['disagree_rate']:.0%} 分歧）"), ("max", "最多（全部分歧）")):
        s = est[key]
        lines.append(f"  {title:<14} 实际请求 {s['calls']:>7} 次   费用 {s['cost']:.4f} 元")
    lines.append("  按模型（预计）：")
    for r in est["expected"]["rows"]:
        lines.append(f"    {r['模型']:<16} 请求 {r['调用次数']:>6}  缓存命中 {r['缓存命中']:>6}  "
                     f"输入 {r['输入 token']:>9}  输出 {r['输出 token']:>8}  {r['费用']:.4f} 元")
    if est["unpriced"]:
        lines.append(f"  注意：{est['unpriced']} 没有设置单价（price_in / price_out），费用按 0 计算，只统计 token。")
    lines.append("  token 为按字数的粗略估算，误差约 ±10~20%；运行结束后会显示接口返回的实际用量。")
    return "\n".join(lines)
