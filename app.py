"""多模型互检分类平台（本地使用）。启动：streamlit run app.py"""
from __future__ import annotations

import asyncio
import copy
import io
import json
import math
import os
import time
from dataclasses import MISSING, fields
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from crosscheck.config import (
    DISAGREEMENT_ACTIONS,
    PROVIDERS,
    ModelConfig,
    config_from_dict,
    load_dotenv,
    read_raw_config,
    save_config,
    save_dotenv,
)
from crosscheck.evaluate import evaluate
from crosscheck.io_utils import write_results
from crosscheck.llm import LLMError, create_llm
from crosscheck.pipeline import STATUS_HUMAN, STATUS_TEXT, CrossCheckPipeline
from crosscheck.prompts import build_classify_prompt

st.set_page_config(page_title="多模型互检分类平台", page_icon="🔍", layout="wide")
load_dotenv()

ROLE_VOTER, ROLE_ARBITER = "投票", "仲裁"
MODEL_COLS = {
    "enabled": "启用", "role": "角色", "name": "名称", "provider": "接口类型", "base_url": "接口地址",
    "model": "模型", "api_key_env": "Key 环境变量", "weight": "权重", "temperature": "温度",
    "max_tokens": "最大输出", "max_concurrency": "并发上限", "rpm": "每分钟请求", "extra_body": "额外参数(JSON)",
}
MODEL_DEFAULTS = {
    f.name: (f.default if f.default is not MISSING else f.default_factory())
    for f in fields(ModelConfig) if f.name != "name"
}
ESSENTIAL = {"name", "provider", "base_url", "model", "api_key_env"}


# ---------------------------------------------------------------- 配置 <-> 表格
def models_to_df(raw: dict) -> pd.DataFrame:
    entries = [(m, ROLE_VOTER) for m in raw.get("models") or []]
    if raw.get("arbiter"):
        entries.append((raw["arbiter"], ROLE_ARBITER))
    rows = []
    for m, role in entries:
        d = {**MODEL_DEFAULTS, **m, "role": role}
        d["extra_body"] = json.dumps(d["extra_body"], ensure_ascii=False) if d["extra_body"] else ""
        rows.append({col: d.get(key) for key, col in MODEL_COLS.items()})
    return pd.DataFrame(rows, columns=list(MODEL_COLS.values()))


def _clean(v, default):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return default
    return v


def df_to_models(df: pd.DataFrame, raw: dict) -> tuple[list[dict], dict | None]:
    originals = {m["name"]: m for m in (raw.get("models") or []) + ([raw["arbiter"]] if raw.get("arbiter") else [])}
    models, arbiters = [], []
    for _, row in df.iterrows():
        name = str(_clean(row[MODEL_COLS["name"]], "")).strip()
        if not name:
            continue
        m = dict(originals.get(name, {}))
        m["name"] = name
        for key, col in MODEL_COLS.items():
            if key in ("name", "role", "extra_body"):
                continue
            default = MODEL_DEFAULTS[key]
            v = _clean(row[col], default)
            m[key] = type(default)(v) if isinstance(default, (int, float, bool)) and not isinstance(v, bool) else v
        text = str(_clean(row[MODEL_COLS["extra_body"]], "")).strip()
        try:
            m["extra_body"] = json.loads(text) if text else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"模型 {name} 的额外参数不是合法 JSON：{e}") from e
        m = {k: v for k, v in m.items() if k in ESSENTIAL or v != MODEL_DEFAULTS.get(k)}
        (arbiters if row[MODEL_COLS["role"]] == ROLE_ARBITER else models).append(m)
    if len(arbiters) > 1:
        raise ValueError("最多只能设置一个仲裁模型")
    return models, (arbiters[0] if arbiters else None)


def labels_to_df(raw: dict) -> pd.DataFrame:
    labels = (raw.get("task") or {}).get("labels") or []
    return pd.DataFrame(
        [{
            "类别": lab.get("name", ""),
            "定义": lab.get("definition", ""),
            "正例（用 | 分隔）": " | ".join(lab.get("examples") or []),
            "反例（用 | 分隔）": " | ".join(lab.get("counter_examples") or []),
        } for lab in labels],
        columns=["类别", "定义", "正例（用 | 分隔）", "反例（用 | 分隔）"],
    )


def df_to_labels(df: pd.DataFrame) -> list[dict]:
    split = lambda s: [x.strip() for x in str(_clean(s, "")).split("|") if x.strip()]
    out = []
    for _, row in df.iterrows():
        name = str(_clean(row["类别"], "")).strip()
        if not name:
            continue
        lab = {"name": name, "definition": str(_clean(row["定义"], "")).strip()}
        if split(row["正例（用 | 分隔）"]):
            lab["examples"] = split(row["正例（用 | 分隔）"])
        if split(row["反例（用 | 分隔）"]):
            lab["counter_examples"] = split(row["反例（用 | 分隔）"])
        out.append(lab)
    return out


def load_into_state(path: str) -> None:
    raw = read_raw_config(path)
    st.session_state.raw = raw
    st.session_state.models_df = models_to_df(raw)
    st.session_state.labels_df = labels_to_df(raw)
    st.session_state.ver = st.session_state.get("ver", 0) + 1


def config_files() -> list[str]:
    return ["config.yaml"] + sorted(p.as_posix() for p in Path("configs").glob("*.yaml"))


# ---------------------------------------------------------------- 运行
async def ping_all(config) -> list[dict]:
    targets = [(m, ROLE_VOTER) for m in config.models] + ([(config.arbiter, ROLE_ARBITER)] if config.arbiter else [])
    async with httpx.AsyncClient(timeout=config.pipeline.timeout) as http:
        async def one(m, role):
            start = time.monotonic()
            try:
                reply = await create_llm(m, http, config).chat("You are a helpful assistant.", "请只回复两个字母：OK")
                return {"名称": m.name, "角色": role, "模型": m.model, "结果": "✅ 成功",
                        "耗时(秒)": round(time.monotonic() - start, 1), "回复 / 错误": reply.strip()[:80]}
            except LLMError as e:
                return {"名称": m.name, "角色": role, "模型": m.model, "结果": "❌ 失败",
                        "耗时(秒)": round(time.monotonic() - start, 1), "回复 / 错误": str(e)[:200]}
        return list(await asyncio.gather(*(one(m, r) for m, r in targets)))


async def run_pipeline(config, items, on_progress):
    start = time.monotonic()
    async with CrossCheckPipeline(config) as pipe:
        results = await pipe.run(items, progress=False, on_progress=on_progress)
        return results, pipe.stats(), time.monotonic() - start


def read_upload(file) -> pd.DataFrame:
    name = file.name.lower()
    data = file.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    if name.endswith(".jsonl"):
        return pd.read_json(io.BytesIO(data), lines=True)
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码，请另存为 UTF-8")


# ---------------------------------------------------------------- 页面
if "raw" not in st.session_state:
    load_into_state("config.yaml")

with st.sidebar:
    st.title("🔍 多模型互检分类")
    files = config_files()
    current = st.session_state.get("cfg_path", "config.yaml")
    picked = st.selectbox("配置文件", files, index=files.index(current) if current in files else 0)
    if picked != current:
        st.session_state.cfg_path = picked
        load_into_state(picked)
        st.rerun()
    st.caption("在页面上修改后立即生效；需要长期保留时点击保存。")
    default_save = "configs/web.yaml" if picked == "config.yaml" else picked
    save_path = st.text_input("保存为", value=default_save)
    save_clicked = st.button("💾 保存配置", width="stretch")
    st.caption("保存会按页面内容重写该文件，原文件中的注释不会保留。")

raw = st.session_state.raw
ver = st.session_state.ver
config_error = None
tab_models, tab_task, tab_run = st.tabs(["① 模型配置", "② 分类任务", "③ 运行与结果"])

# ---------------- ① 模型配置
with tab_models:
    st.subheader("参与互检的模型")
    st.caption("建议 3 个来自不同厂商的“投票”模型 + 1 个“仲裁”模型。接口类型：openai = 任何 OpenAI 兼容接口（DeepSeek / 百炼 / Kimi / 大部分中转站）。")
    edited_models = st.data_editor(
        st.session_state.models_df, key=f"models_editor_{ver}", num_rows="dynamic", width="stretch",
        column_config={
            MODEL_COLS["enabled"]: st.column_config.CheckboxColumn(default=True),
            MODEL_COLS["role"]: st.column_config.SelectboxColumn(options=[ROLE_VOTER, ROLE_ARBITER], default=ROLE_VOTER, required=True),
            MODEL_COLS["provider"]: st.column_config.SelectboxColumn(options=list(PROVIDERS), default="openai", required=True),
            MODEL_COLS["weight"]: st.column_config.NumberColumn(min_value=0.01, step=0.1, default=1.0),
            MODEL_COLS["temperature"]: st.column_config.NumberColumn(min_value=0.0, max_value=2.0, step=0.1, default=0.0),
            MODEL_COLS["max_tokens"]: st.column_config.NumberColumn(min_value=16, step=128, default=512),
            MODEL_COLS["max_concurrency"]: st.column_config.NumberColumn(min_value=0, step=1, default=0, help="0 表示不限制"),
            MODEL_COLS["rpm"]: st.column_config.NumberColumn(min_value=0, step=1, default=0, help="0 表示不限制"),
            MODEL_COLS["extra_body"]: st.column_config.TextColumn(help='例如关闭思考：{"enable_thinking": false}'),
        },
    )
    try:
        raw["models"], raw["arbiter"] = df_to_models(edited_models, raw)
    except ValueError as e:
        config_error = str(e)
        st.error(config_error)

    st.subheader("API Key")
    st.caption("Key 只保存在本机项目目录的 .env 文件中（已被 git 忽略）。")
    envs = sorted({str(_clean(v, "")) for v in edited_models[MODEL_COLS["api_key_env"]]} - {""})
    new_keys = {}
    for env in envs:
        cur = os.environ.get(env, "")
        status = f"已配置（…{cur[-4:]}）" if cur else "未配置"
        new_keys[env] = st.text_input(f"{env}　·　{status}", type="password", key=f"key_{env}", placeholder="留空表示不修改")
    c1, c2 = st.columns(2)
    if c1.button("保存 Key", width="stretch"):
        updates = {k: v.strip() for k, v in new_keys.items() if v.strip()}
        if updates:
            save_dotenv(updates)
            st.success(f"已保存：{', '.join(updates)}")
        else:
            st.info("没有填写新的 Key")
    if c2.button("🔌 测试所有模型连通性", width="stretch", type="primary"):
        try:
            with st.spinner("正在逐个调用模型…"):
                st.session_state.ping = asyncio.run(ping_all(config_from_dict(raw)))
        except ValueError as e:
            st.error(f"配置有误：{e}")
    if st.session_state.get("ping"):
        st.dataframe(pd.DataFrame(st.session_state.ping), width="stretch", hide_index=True)

# ---------------- ② 分类任务
with tab_task:
    task = raw.setdefault("task", {})
    c1, c2 = st.columns([1, 2])
    task["name"] = c1.text_input("任务名称", value=task.get("name", ""), key=f"task_name_{ver}")
    task["description"] = c2.text_input("任务说明", value=task.get("description", ""), key=f"task_desc_{ver}")
    st.subheader("类别定义")
    st.caption("定义越清晰、正反例越贴近真实数据，模型之间的分歧越少。")
    edited_labels = st.data_editor(st.session_state.labels_df, key=f"labels_editor_{ver}", num_rows="dynamic", width="stretch")
    task["labels"] = df_to_labels(edited_labels)
    st.subheader("边界规则")
    rules_text = st.text_area("每行一条：写清楚容易混淆的情况该怎么判", value="\n".join(task.get("rules") or []), height=140, key=f"rules_{ver}")
    task["rules"] = [r.strip() for r in rules_text.splitlines() if r.strip()]
    with st.expander("预览发给模型的提示词"):
        try:
            demo = config_from_dict(raw, mock=True).task
            st.code(build_classify_prompt(demo, "（这里是待分类文本）"), language="markdown")
        except ValueError as e:
            st.warning(str(e))

if save_clicked:
    if config_error:
        st.sidebar.error("配置有误，未保存")
    else:
        save_config(raw, save_path)
        st.sidebar.success(f"已保存到 {save_path}")

# ---------------- ③ 运行与结果
with tab_run:
    source = st.radio("数据来源", ["上传文件", "项目 data 目录中的文件"], horizontal=True)
    up = None
    if source == "上传文件":
        up = st.file_uploader("上传数据文件（CSV / Excel / JSONL）", type=["csv", "xlsx", "xls", "jsonl"])
    else:
        local = sorted(p.as_posix() for p in Path("data").glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".jsonl"))
        picked_file = st.selectbox("选择文件", local) if local else None
        if picked_file:
            up = io.BytesIO(Path(picked_file).read_bytes())
            up.name = picked_file
    df = None
    if up is not None:
        try:
            df = read_upload(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"读取失败：{e}")
    if df is not None:
        st.caption(f"共 {len(df)} 行，预览前 5 行：")
        st.dataframe(df.head(), width="stretch", hide_index=True)
        cols = [str(c) for c in df.columns]
        df.columns = cols
        guess = lambda names, fallback: next((c for c in cols if c.lower() in names), fallback)
        c1, c2, c3 = st.columns(3)
        text_col = c1.selectbox("文本列", cols, index=cols.index(guess({"text", "文本", "内容", "sentence"}, cols[0])))
        id_opts = ["（自动编号）"] + cols
        id_col = c2.selectbox("ID 列", id_opts, index=id_opts.index(guess({"id", "编号"}, "（自动编号）")))
        label_opts = ["（无，只做分类）"] + cols
        label_col = c3.selectbox("标签列（有则计算准确率）", label_opts, index=label_opts.index(guess({"label", "标签", "类别"}, "（无，只做分类）")))

        st.markdown("**运行设置**")
        p = raw.get("pipeline") or {}
        c1, c2, c3, c4 = st.columns(4)
        actions = list(DISAGREEMENT_ACTIONS)
        action = c1.selectbox("首轮出现分歧时", actions, index=actions.index(p.get("disagreement_action", "review")),
                              format_func=DISAGREEMENT_ACTIONS.get)
        accept = c2.slider("复核后自动采纳阈值", 0.5, 1.0, float(p.get("accept_threshold", 0.6)), 0.05,
                           help="过半模型同意且加权得票占比 ≥ 该值时自动采纳", disabled=action != "review")
        arb_th = c3.slider("仲裁采纳阈值", 0.0, 1.0, float(p.get("arbiter_threshold", 0.7)), 0.05,
                           help="仲裁模型置信度 ≥ 该值才采纳，否则转人工", disabled=action == "human")
        limit = c4.number_input("只处理前 N 条（0 = 全部）", min_value=0, value=0, step=10)
        c1, c2 = st.columns(2)
        concurrency = c1.number_input("总并发请求数", min_value=1, max_value=64, value=int(p.get("concurrency", 8)))
        mock = c2.toggle("mock 模式（不调用真实 API，只测流程）", value=False)

        sub = df if not limit else df.head(int(limit))
        items, bad = [], set()
        allowed = {lab["name"] for lab in raw["task"]["labels"]}
        for n, row in enumerate(sub.itertuples(index=False), 1):
            r = dict(zip(cols, row))
            text = str(_clean(r[text_col], "")).strip()
            if not text:
                continue
            item = {"id": str(r[id_col]) if id_col != "（自动编号）" else str(n), "text": text}
            if label_col != "（无，只做分类）":
                item["label"] = str(_clean(r[label_col], "")).strip()
                if item["label"] not in allowed:
                    bad.add(item["label"])
            items.append(item)
        if bad:
            st.error(f"标签列中有配置里不存在的类别：{sorted(bad)}。请在“分类任务”页添加这些类别，或检查标签列是否选对。")

        if st.button(f"🚀 开始运行（{len(items)} 条）", type="primary", disabled=bool(bad or config_error) or not items):
            run_raw = copy.deepcopy(raw)
            run_raw["pipeline"] = {**p, "disagreement_action": action, "accept_threshold": accept,
                                   "arbiter_threshold": arb_th, "concurrency": int(concurrency)}
            try:
                config = config_from_dict(run_raw, mock=mock)
            except ValueError as e:
                st.error(f"配置有误：{e}")
                st.stop()
            bar = st.progress(0.0, text="准备中…")
            t0 = time.monotonic()
            cb = lambda done, total: bar.progress(done / total, text=f"{done}/{total}　已用 {time.monotonic() - t0:.0f} 秒")
            try:
                results, stats, elapsed = asyncio.run(run_pipeline(config, items, cb))
            except LLMError as e:
                st.error(str(e))
                st.stop()
            out = Path("output/web") / datetime.now().strftime("%Y%m%d_%H%M%S")
            model_names = [m.name for m in config.models]
            paths = write_results(results, out, model_names)
            rep = None
            if label_col != "（无，只做分类）":
                from crosscheck.report import build_report
                weights = {m.name: m.weight for m in config.models}
                rep = evaluate(results, {it["id"]: it["label"] for it in items}, model_names, config.task.label_names, weights)
                paths["html"] = build_report(rep, out)
            st.session_state.last = {"results": results, "stats": stats, "elapsed": elapsed, "rep": rep,
                                     "paths": paths, "out": out, "model_names": model_names}
            bar.empty()

    last = st.session_state.get("last")
    if last:
        results, rep, model_names = last["results"], last["rep"], last["model_names"]
        st.divider()
        st.subheader("结果")
        n = len(results)
        counts = pd.Series([r.status for r in results]).value_counts()
        cols_m = st.columns(len(STATUS_TEXT) + 1)
        cols_m[0].metric("总条数", n, f"用时 {last['elapsed']:.0f} 秒", delta_color="off")
        for c, (s, t) in zip(cols_m[1:], STATUS_TEXT.items()):
            c.metric(t, int(counts.get(s, 0)), f"{counts.get(s, 0) / n * 100:.0f}%", delta_color="off")

        if rep:
            strat = rep["strategies"]
            best = max(model_names, key=lambda m: strat[m]["acc"])
            c = st.columns(5)
            c[0].metric(f"最佳单模型（{best}）", f"{strat[best]['acc'] * 100:.1f}%")
            c[1].metric("多数投票", f"{strat['多数投票']['acc'] * 100:.1f}%", f"{(strat['多数投票']['acc'] - strat[best]['acc']) * 100:+.1f}")
            c[2].metric("加权投票", f"{strat['加权投票']['acc'] * 100:.1f}%", f"{(strat['加权投票']['acc'] - strat[best]['acc']) * 100:+.1f}")
            c[3].metric("互检系统", f"{strat['互检系统']['acc'] * 100:.1f}%", f"{(strat['互检系统']['acc'] - strat[best]['acc']) * 100:+.1f}")
            c[4].metric("理论上限（任一模型答对）", f"{rep['any_correct'] * 100:.1f}%")

            table = pd.DataFrame(
                [{"方案": name, "类型": "单模型" if v["kind"] == "model" else "组合", "整体": v["acc"] * 100,
                  **{lab: (v["per_class"][lab] * 100 if v["per_class"][lab] is not None else None) for lab in rep["labels"]}}
                 for name, v in strat.items()]
            )
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in ["整体"] + rep["labels"]})

            charts = Path(last["out"]) / "charts"
            c1, c2 = st.columns(2)
            c1.image(str(charts / "accuracy.png"), width="stretch")
            c2.image(str(charts / "status.png"), width="stretch")
            st.image(str(charts / "per_class.png"), width="stretch")
            c1, c2 = st.columns(2)
            c1.image(str(charts / "review.png"), width="stretch")
            c2.image(str(charts / "confusion.png"), width="stretch")
        else:
            st.markdown("**模型两两一致率**（首轮）")
            agree = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
            for a in model_names:
                for b in model_names:
                    pairs = [(pa, pb) for r in results
                             for pa in r.round1 if pa.model == a and pa.ok
                             for pb in r.round1 if pb.model == b and pb.ok]
                    agree.loc[a, b] = sum(pa.label == pb.label for pa, pb in pairs) / len(pairs) * 100 if pairs else None
            st.dataframe(agree.style.format("{:.1f}%").background_gradient(cmap="Greens", vmin=50, vmax=100), width="stretch")
            st.markdown("**各模型给出的类别分布**（首轮）")
            dist = pd.DataFrame({m: pd.Series([p.label for r in results for p in r.round1 if p.model == m and p.ok]).value_counts()
                                 for m in model_names}).fillna(0)
            st.bar_chart(dist, stack=False)

        st.markdown("**逐条结果**")
        status_pick = st.multiselect("按处理环节筛选", list(STATUS_TEXT.values()), default=list(STATUS_TEXT.values()))
        gold = {it["id"]: it for it in rep["items"]} if rep else {}
        rows = []
        for r in results:
            if STATUS_TEXT[r.status] not in status_pick:
                continue
            row = {"ID": r.id, "文本": r.text}
            if rep:
                row["真实标签"] = gold[r.id]["gold"]
            for m in model_names:
                p1 = next((p for p in r.round1 if p.model == m), None)
                row[m] = p1.label if p1 and p1.ok else "失败"
            row.update({"最终标签": r.label, "处理环节": STATUS_TEXT[r.status], "置信度": r.confidence})
            rows.append(row)
        detail = pd.DataFrame(rows)
        if rep and not detail.empty:
            wrong = lambda col: [("background-color:#fdecea;color:#c0392b" if v != g else "") for v, g in zip(col, detail["真实标签"])]
            st.dataframe(detail.style.apply(wrong, subset=model_names + ["最终标签"]), hide_index=True, width="stretch")
        else:
            st.dataframe(detail, hide_index=True, width="stretch")

        paths = last["paths"]
        c = st.columns(4)
        c[0].download_button("⬇ 结果 CSV", paths["csv"].read_bytes(), paths["csv"].name, width="stretch")
        c[1].download_button("⬇ 待人工审核 CSV", paths["human"].read_bytes(), paths["human"].name, width="stretch",
                             disabled=not any(r.status == STATUS_HUMAN for r in results))
        c[2].download_button("⬇ 完整明细 JSONL", paths["jsonl"].read_bytes(), paths["jsonl"].name, width="stretch")
        if "html" in paths:
            c[3].download_button("⬇ 图表报告 HTML", paths["html"].read_bytes(), "report.html", width="stretch")
        with st.expander("模型调用统计"):
            st.dataframe(pd.DataFrame([{"模型": k, "实际请求": v["calls"], "缓存命中": v["cache_hits"], "失败": v["errors"]}
                                       for k, v in last["stats"].items()]), hide_index=True)
        st.caption(f"文件已保存在：{Path(last['out']).resolve()}")
