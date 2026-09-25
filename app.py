"""多模型互检分类平台（本地使用）。启动：streamlit run app.py"""
from __future__ import annotations

import asyncio
import copy
import html
import io
import json
import math
import os
import re
import time
from dataclasses import MISSING, fields
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
import yaml

from crosscheck.config import (
    DISAGREEMENT_ACTIONS,
    PROVIDERS,
    ModelConfig,
    base_of,
    config_from_dict,
    load_dotenv,
    read_raw_config,
    save_config_diff,
    save_dotenv,
)
from crosscheck.criteria import changed_labels, criteria_hash, diff_criteria, list_versions, save_version, version_label
from crosscheck.dialogue import build_items as build_dialogue_items, cited_indexes, group_turns, sessions_from_transcripts
from crosscheck.documents import aggregate_documents, filter_paragraphs, split_document
from crosscheck.drafting import draft_from_document, draft_from_examples, drafting_models, read_document, review_criteria
from crosscheck.monitor import drift_alerts, label_share, suggest_rules
from crosscheck.evaluate import evaluate
from crosscheck.validation import ci_halfwidth, sample_rows, split_dev_test
from crosscheck.history import (
    compare_items,
    delete_run,
    item_correct,
    load_extra,
    load_meta,
    mcnemar,
    run_criteria,
    save_extra,
    set_note,
    settings_text,
    summarize,
    write_meta,
)
from crosscheck.io_utils import write_results
from crosscheck.llm import LLMError, create_llm
from crosscheck.cost import estimate_run
from crosscheck.local_model import save_examples
from crosscheck.pipeline import STATUS_HUMAN, STATUS_TEXT, CrossCheckPipeline
from crosscheck.prompts import build_classify_prompt
from crosscheck.review import (
    SCOPES,
    append_to_gold,
    list_runs,
    load_reviews,
    load_run,
    merged_rows,
    review_stats,
    round1_votes,
    run_model_names,
    save_reviews,
    select_queue,
    set_review,
)

st.set_page_config(page_title="多模型互检分类平台", page_icon="🔍", layout="wide")
load_dotenv()

ROLE_VOTER, ROLE_ARBITER = "投票", "仲裁"
MODEL_COLS = {
    "enabled": "启用", "role": "角色", "name": "名称", "provider": "接口类型", "base_url": "接口地址",
    "model": "模型", "api_key_env": "Key 环境变量", "weight": "权重", "temperature": "温度",
    "max_tokens": "最大输出", "max_concurrency": "并发上限", "rpm": "每分钟请求", "logprobs": "读取概率",
    "price_in": "输入单价(元/百万token)", "price_out": "输出单价(元/百万token)", "extra_body": "额外参数(JSON)",
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


def apply_task(raw: dict, fields_: dict) -> None:
    """用新的分类标准（草稿或历史版本）替换页面上的标准，并刷新相关表格。"""
    raw["task"] = {**raw.get("task", {}), **{k: copy.deepcopy(v) for k, v in fields_.items()
                                             if k in ("name", "description", "labels", "rules")}}
    names = {lab["name"] for lab in raw["task"]["labels"]}
    kw = (raw.get("mock") or {}).get("keywords") or {}
    raw["mock"] = {**(raw.get("mock") or {}), "keywords": {k: v for k, v in kw.items() if k in names}}
    st.session_state.labels_df = labels_to_df(raw)
    st.session_state.models_df = models_to_df(raw)
    st.session_state.ver = st.session_state.get("ver", 0) + 1


def load_into_state(path: str) -> None:
    raw = read_raw_config(path)
    st.session_state.raw = raw
    st.session_state.models_df = models_to_df(raw)
    st.session_state.labels_df = labels_to_df(raw)
    st.session_state.ver = st.session_state.get("ver", 0) + 1


def config_files() -> list[str]:
    return ["config.yaml"] + sorted(p.as_posix() for p in Path("configs").glob("*.yaml"))


@st.cache_data(show_spinner=False)
def _task_title(path: str, stamp: int) -> str:
    try:
        return (read_raw_config(path).get("task") or {}).get("name") or Path(path).stem
    except (OSError, ValueError, yaml.YAMLError):
        return Path(path).stem


def task_title(path: str) -> str:
    return _task_title(path, Path(path).stat().st_mtime_ns if Path(path).exists() else 0)


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


@st.cache_data(show_spinner=False)
def read_training(path: str, stamp: int) -> pd.DataFrame:
    """stamp 是文件修改时间，只用于让缓存在文件变化后失效。"""
    buf = io.BytesIO(Path(path).read_bytes())
    buf.name = path
    df = read_upload(buf)
    df.columns = [str(c) for c in df.columns]
    return df


def training_df(fs: dict) -> pd.DataFrame | None:
    path = str(fs.get("path") or "")
    if not path or not Path(path).exists():
        return None
    return read_training(path, Path(path).stat().st_mtime_ns)


def guess_col(cols: list[str], names: set[str], fallback: str) -> str:
    return next((c for c in cols if c.lower() in names), fallback)


@st.cache_data(show_spinner=False)
def cached_run(path: str, stamp: int) -> list[dict]:
    """stamp 是文件修改时间，只用于让缓存在文件变化后失效。"""
    return load_run(path)


def run_records(path: str) -> list[dict]:
    return cached_run(path, Path(path).stat().st_mtime_ns)


def read_gold_file(path: str) -> dict[str, str]:
    buf = io.BytesIO(Path(path).read_bytes())
    buf.name = path
    df = read_upload(buf)
    df.columns = [str(c).lower() for c in df.columns]
    if not {"id", "label"} <= set(df.columns):
        raise ValueError("标签文件需要 id 和 label 两列")
    return dict(zip(df["id"].astype(str), df["label"].astype(str)))


def _bubble_html(turns: list[dict], focus, reasons: list[str]) -> str:
    cited = cited_indexes(turns, reasons)
    lo, hi = 0, len(turns)
    if focus is not None and len(turns) > 24:
        lo, hi = max(0, focus - 8), min(len(turns), focus + 9)
    parts = []
    if lo:
        parts.append(f"<div style='color:#888;font-size:12px'>已省略前面 {lo} 句</div>")
    for i in range(lo, hi):
        t = turns[i]
        right = t["role"] in ("客服", "商家", "坐席")
        bg = "#fef3c7" if i in cited else ("#e9f8ef" if right else "#e8f1ff")
        border = "2px solid #2563eb" if i == focus else "1px solid transparent"
        mark = " · 模型引用" if i in cited else ""
        focus_mark = " · 待分类" if i == focus else ""
        parts.append(
            f"<div style='display:flex;justify-content:{'flex-end' if right else 'flex-start'};margin:4px 0'>"
            f"<div style='max-width:78%;background:{bg};border:{border};border-radius:10px;padding:8px 12px'>"
            f"<div style='font-size:12px;color:#64748b'>{html.escape(t['role'])}{focus_mark}{mark}</div>"
            f"<div>{html.escape(t['text'])}</div></div></div>"
        )
    if hi < len(turns):
        parts.append(f"<div style='color:#888;font-size:12px'>已省略后面 {len(turns) - hi} 句</div>")
    return "\n".join(parts)


def _reasons_of(rec: dict) -> list[str]:
    out = [p.get("reason") or "" for p in rec.get("round1") or []]
    out += [p.get("reason") or "" for p in rec.get("round2") or []]
    if rec.get("arbiter"):
        out.append(rec["arbiter"].get("reason") or "")
    return out


def show_item_body(rec: dict, extra: dict | None) -> None:
    """审核时：对话显示气泡，长文档显示章节和相邻段落，其余显示原文。"""
    info = ((extra or {}).get("items") or {}).get(rec["id"]) if extra else None
    if extra and extra.get("kind") == "dialogue" and info:
        st.markdown(_bubble_html(info["turns"], info.get("focus"), _reasons_of(rec)), unsafe_allow_html=True)
        return
    if extra and extra.get("kind") == "document" and info:
        page = f"第 {info['page']} 页" if info.get("page") else "无分页"
        st.caption(f"{info.get('doc', '')}　·　{info.get('chapter') or '（未识别章节）'}　·　{page}")
        if info.get("prev"):
            st.caption("上文：" + info["prev"])
        st.markdown(f"<div style='font-size:1.15rem;padding:14px 18px;background:#fff7ed;border-left:4px solid #ea580c;"
                    f"border-radius:8px;margin:6px 0'>{html.escape(info.get('text') or rec['text'])}</div>",
                    unsafe_allow_html=True)
        if info.get("next"):
            st.caption("下文：" + info["next"])
        return
    st.markdown(f"<div style='font-size:1.25rem;padding:14px 18px;background:#f4f7fb;border-radius:8px;"
                f"margin:6px 0 12px'>{html.escape(rec['text'])}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _split_doc_cached(name: str, data: bytes, min_chars: int):
    return split_document(name, data, min_chars)


def pct(v) -> float | None:
    return None if v is None else v * 100


# ---------------------------------------------------------------- 页面
if "raw" not in st.session_state:
    load_into_state("config.yaml")

with st.sidebar:
    st.title("🔍 多模型互检分类")
    files = config_files()
    current = st.session_state.get("cfg_path", "config.yaml")
    picked = st.selectbox("分类任务（配置文件）", files, index=files.index(current) if current in files else 0,
                          format_func=lambda f: f"{task_title(f)}　·　{f}")
    if picked != current:
        st.session_state.cfg_path = picked
        load_into_state(picked)
        st.rerun()
    version_slot = st.empty()
    default_save = "configs/web.yaml" if picked == "config.yaml" else picked
    save_path = st.text_input("保存为", value=default_save)
    save_note = st.text_input("本次修改说明（可选）", placeholder="例如：拆分“投诉”为两类 / 补充反讽规则",
                              help="分类标准（类别、定义、正反例、规则）有变化时会自动保存为新版本，运行记录里会标明用的是哪个版本")
    save_clicked = st.button("💾 保存", width="stretch")
    st.caption("有 base 的任务文件只保存与 base 不同的部分（模型配置随 config.yaml 更新）；原文件中的注释不会保留。")

    with st.expander("➕ 新建分类任务"):
        new_title = st.text_input("任务名称", placeholder="例如：客服对话质检")
        new_file = st.text_input("文件名（字母、数字、下划线）", placeholder="例如：dialog_qc")
        new_from = st.radio("从哪里开始", ["空白任务（只沿用模型配置）", "复制当前任务的标准和设置"])
        if st.button("创建", width="stretch", disabled=not (new_title.strip() and new_file.strip())):
            slug = re.sub(r"[^0-9A-Za-z_\-]", "_", new_file.strip())
            target = Path("configs") / f"{slug}.yaml"
            if target.exists():
                st.error(f"{target.as_posix()} 已存在，换一个文件名")
            else:
                if new_from.startswith("空白"):
                    new_raw = {**read_raw_config("config.yaml"), "mock": {"keywords": {}}, "fewshot": {},
                               "task": {"name": new_title.strip(), "description": "", "rules": [],
                                        "labels": [{"name": "类别A", "definition": ""}, {"name": "类别B", "definition": ""}]}}
                else:
                    new_raw = copy.deepcopy(st.session_state.raw)
                    new_raw["task"] = {**new_raw.get("task", {}), "name": new_title.strip()}
                save_config_diff(new_raw, target, Path("config.yaml"))
                save_version(target, new_raw["task"], "创建任务")
                st.session_state.cfg_path = target.as_posix()
                load_into_state(target.as_posix())
                st.rerun()
        st.caption("新任务保存在 configs/ 下，模型配置继承 config.yaml。接着到“② 分类任务”页写标准，或用文档自动生成草稿。")

raw = st.session_state.raw
ver = st.session_state.ver
config_error = None
tab_models, tab_task, tab_train, tab_run, tab_review, tab_history = st.tabs(
    ["① 模型配置", "② 分类任务", "③ 训练数据（可选）", "④ 运行与结果", "⑤ 人工审核", "⑥ 历史与对比"])

# ---------------- ① 模型配置
with tab_models:
    st.subheader("参与互检的模型")
    st.caption("建议 3 个来自不同厂商的“投票”模型 + 1 个“仲裁”模型。接口类型：openai = 任何 OpenAI 兼容接口（DeepSeek / 百炼 / Kimi / 大部分中转站）；"
               "local = 用训练数据训练的本地小模型，在“③ 训练数据”页一键添加。")
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
            MODEL_COLS["logprobs"]: st.column_config.CheckboxColumn(
                default=False, help="读取标签的输出概率（logprobs）作为级联的置信度，仅 OpenAI 兼容接口。"
                                    "千问、Kimi 可用；DeepSeek 在温度 0 时只返回 0 / 1，没有区分度"),
            MODEL_COLS["price_in"]: st.column_config.NumberColumn(min_value=0.0, step=0.1, default=0.0, format="%.2f",
                                                                  help="用于费用预估和统计，按官网价格填写；0 表示不计费用、只统计 token"),
            MODEL_COLS["price_out"]: st.column_config.NumberColumn(min_value=0.0, step=0.1, default=0.0, format="%.2f"),
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
    try:
        draft_cfg = config_from_dict(raw)
        draft_models = drafting_models(draft_cfg)
    except ValueError:
        draft_cfg, draft_models = None, []

    with st.expander("🪄 用标准文档或已标注数据生成标准草稿", expanded=not (task.get("labels") and any(
            lab.get("definition") for lab in task.get("labels") or []))):
        st.caption("由大模型把你的分类标准整理成下面的“类别定义 + 正反例 + 边界规则”，生成后先预览，确认后再替换，之后仍可手动修改。")
        if not draft_models:
            st.warning("需要至少一个可用的大模型（在“① 模型配置”页配置并填好 Key）。")
        else:
            c1, c2 = st.columns([2, 1])
            src = c1.radio("草稿来源", ["上传标准文档", "粘贴标准文字", "从已标注数据归纳"], horizontal=True,
                           help="文档 / 文字：适合已有书面分类口径；已标注数据：适合只有标好的样本、没有成文标准的情况")
            names = [m.name for m in draft_models]
            draft_model = c2.selectbox("用哪个模型起草", names,
                                       index=next((i for i, n in enumerate(names) if "qwen" in n.lower()), 0))
            model_cfg = next(m for m in draft_models if m.name == draft_model)
            doc_text, pairs, truncated = "", [], False
            if src == "上传标准文档":
                f = st.file_uploader("标准文档（Word / PDF / TXT / Markdown）", type=["docx", "pdf", "txt", "md"], key="std_doc")
                if f is not None:
                    try:
                        doc_text, truncated = read_document(f.name, f.getvalue())
                        st.caption(f"读取到 {len(doc_text)} 字" + ("（超过 4 万字，只使用前 4 万字）" if truncated else ""))
                    except (ValueError, ImportError) as err:
                        st.error(str(err))
            elif src == "粘贴标准文字":
                doc_text = st.text_area("把分类标准粘贴到这里", height=200, key="std_paste",
                                        placeholder="例如：一、投诉：用户对商品或服务表达不满……\n二、咨询：……")
            else:
                files = sorted(p.as_posix() for p in Path("data").glob("**/*") if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".jsonl"))
                lf = st.selectbox("已标注的数据文件（data 目录）", files, key="std_labeled") if files else None
                if lf:
                    ldf = read_training(lf, Path(lf).stat().st_mtime_ns)
                    lcols = list(ldf.columns)
                    c1, c2, c3 = st.columns(3)
                    tcol = c1.selectbox("文本列", lcols, index=lcols.index(guess_col(lcols, {"text", "文本", "内容"}, lcols[0])), key="std_tcol")
                    lcol = c2.selectbox("标签列", lcols, index=lcols.index(guess_col(lcols, {"label", "标签", "类别"}, lcols[-1])), key="std_lcol")
                    per_label = c3.slider("每类抽多少条给模型看", 10, 60, 25, 5, key="std_per")
                    sub = ldf[[tcol, lcol]].dropna()
                    pairs = list(zip(sub[tcol].astype(str), sub[lcol].astype(str)))
                    counts = sub[lcol].astype(str).value_counts()
                    st.caption(f"{len(pairs)} 条，{len(counts)} 个类别：" + "，".join(f"{k} {v}" for k, v in counts.items()))
                    if len(counts) > 30:
                        st.warning("类别超过 30 个，请确认选对了标签列。")
            revise = st.checkbox("在当前标准基础上修订（保留现有类别名称）", value=False, disabled=src == "从已标注数据归纳",
                                 help="不勾选：按文档重新整理一份完整标准")
            ready = bool(doc_text.strip()) if src != "从已标注数据归纳" else len({lab for _, lab in pairs}) >= 2
            if st.button("🪄 生成草稿", disabled=not ready, type="primary"):
                try:
                    with st.spinner(f"{draft_model} 正在整理标准，通常需要 20~90 秒…"):
                        if src == "从已标注数据归纳":
                            st.session_state.draft = draft_from_examples(draft_cfg, model_cfg, pairs, task.get("description", ""), per_label)
                        else:
                            st.session_state.draft = draft_from_document(draft_cfg, model_cfg, doc_text, task if revise else None)
                except LLMError as err:
                    st.error(str(err))
        draft = st.session_state.get("draft")
        if draft:
            st.markdown(f"**草稿：{draft['name'] or '（未命名）'}**　{draft['description']}")
            st.dataframe(labels_to_df({"task": draft}), hide_index=True, width="stretch")
            if draft["rules"]:
                st.markdown("**边界规则**\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(draft["rules"], 1)))
            for note in draft["notes"]:
                st.warning(f"需要确认：{note}")
            if task.get("labels"):
                changes = diff_criteria(task, draft)
                with st.expander(f"与当前标准相比的变化（{len(changes)} 处）"):
                    st.markdown("\n".join(f"- {c}" for c in changes) or "没有变化")
            c1, c2 = st.columns(2)
            if c1.button("✅ 用草稿替换当前标准", width="stretch"):
                apply_task(raw, {k: draft[k] for k in ("name", "description", "labels", "rules") if draft[k] or k != "name"})
                st.session_state.draft = None
                st.rerun()
            if c2.button("丢弃草稿", width="stretch"):
                st.session_state.draft = None
                st.rerun()

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
    names_now = {lab["name"] for lab in task["labels"]}
    kw = (raw.get("mock") or {}).get("keywords") or {}
    if set(kw) - names_now:  # 改名或删类别后，去掉 mock 模式里已经不存在的类别，避免配置报错
        raw["mock"] = {**(raw.get("mock") or {}), "keywords": {k: v for k, v in kw.items() if k in names_now}}
    with st.expander("预览发给模型的提示词"):
        try:
            demo = config_from_dict(raw, mock=True).task
            st.code(build_classify_prompt(demo, "（这里是待分类文本）"), language="markdown")
        except ValueError as e:
            st.warning(str(e))

    # -------- 标准体检
    st.subheader("🩺 标准体检")
    st.caption("让大模型从标注员的角度检查：类别是否重叠、定义是否含糊、常见情况是否没被覆盖、规则是否冲突。建议写完标准、正式运行前做一次。")
    c1, c2 = st.columns([1, 3])
    if c1.button("开始体检", disabled=not draft_models or len(task["labels"]) < 2, width="stretch"):
        names = [m.name for m in draft_models]
        m = draft_models[next((i for i, n in enumerate(names) if "qwen" in n.lower()), 0)]
        try:
            with st.spinner(f"{m.name} 正在审查标准…"):
                st.session_state.checkup = {"hash": criteria_hash(task), **review_criteria(draft_cfg, m, task)}
        except LLMError as err:
            st.error(str(err))
    chk = st.session_state.get("checkup")
    if chk:
        if chk["hash"] != criteria_hash(task):
            c2.caption("标准在体检后又修改过，结果可能已过时。")
        if chk["summary"]:
            st.info(chk["summary"])
        if chk["issues"]:
            st.dataframe(pd.DataFrame([{"类型": i.get("type", ""), "涉及类别": "、".join(i.get("labels") or []),
                                        "问题": i.get("problem", ""), "建议": i.get("suggestion", "")} for i in chk["issues"]]),
                         hide_index=True, width="stretch",
                         column_config={"问题": st.column_config.TextColumn(width="large"),
                                        "建议": st.column_config.TextColumn(width="large")})
        else:
            st.success("没有发现明显问题。")

    # -------- 标准版本
    cfg_now = st.session_state.get("cfg_path", "config.yaml")
    vers = list_versions(cfg_now)
    with st.expander(f"🕘 分类标准版本（{len(vers)} 个）"):
        if not vers:
            st.caption("还没有保存过版本。点击侧边栏“保存”时，标准有变化就会自动存为新版本。")
        else:
            st.dataframe(pd.DataFrame([{"版本": f"v{v['version']}", "时间": v["time"], "说明": v["note"],
                                        "类别数": len(v["task"]["labels"]), "规则数": len(v["task"]["rules"])}
                                       for v in reversed(vers)]), hide_index=True, width="stretch")
            pick_v = st.selectbox("查看某个版本", [v["version"] for v in reversed(vers)], format_func=lambda x: f"v{x}")
            vsel = next(v for v in vers if v["version"] == pick_v)
            prev = next((v for v in vers if v["version"] == pick_v - 1), None)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**v{pick_v} 相比上一版**")
                st.markdown("\n".join(f"- {x}" for x in diff_criteria(prev["task"], vsel["task"])) if prev else "（第一个版本）")
            with c2:
                st.markdown("**当前页面上的标准相比 v%d**" % pick_v)
                st.markdown("\n".join(f"- {x}" for x in diff_criteria(vsel["task"], task)) or "完全相同")
            if st.button(f"↩ 恢复为 v{pick_v}", help="用这个版本替换页面上的标准；需要长期保留时再点侧边栏保存"):
                apply_task(raw, vsel["task"])
                st.rerun()

    # -------- 新任务验证
    with st.expander("✅ 新任务 / 标准修改后怎么验证"):
        st.markdown(
            "1. **抽样**：从要分类的数据里随机抽 100~200 条（下面的工具）。\n"
            "2. **先让模型分**：在“④ 运行与结果”页运行抽样文件。\n"
            "3. **人工标注**：在“⑤ 人工审核”页选“全部样本”逐条确认，再“把已审核样本加入金标准”。模型建议只是参考，拿不准的样本正好暴露标准的漏洞。\n"
            "4. **划分调参集 / 验证集**（下面的工具）：只用调参集反复修改标准、看判错样本；验证集只在最后跑一次，"
            "这样得到的准确率才能代表新数据上的表现，而不是“把标准调得只适合这批样本”。\n"
            "5. **改标准 → 重跑 → 对比**：每次修改后在侧边栏保存（自动存版本），重跑调参集，在“⑥ 历史与对比”页勾选前后两次，"
            "看准确率变化、显著性和标准差异。已调用过的提示词命中缓存，只有改动影响到的部分才重新付费。\n"
            "6. **上线后抽检**：定期在“⑤ 人工审核”页抽检“首轮一致通过”的样本；人工改动率明显上升，说明数据变了或标准需要更新。"
        )
        n_tbl = pd.DataFrame([{"样本量": n, **{f"准确率 {p:.0%}": f"±{ci_halfwidth(p, n) * 100:.1f} 个百分点" for p in (0.7, 0.8, 0.9)}}
                              for n in (50, 100, 200, 400, 1000)])
        st.markdown("**样本量与误差**（95% 置信区间半宽）：比较两个版本时，差异小于误差范围就可能只是随机波动，以“⑥ 历史与对比”页的显著性检验为准。")
        st.dataframe(n_tbl, hide_index=True, width="stretch")

        st.markdown("**抽样工具**")
        files = sorted(p.as_posix() for p in Path("data").glob("**/*") if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".jsonl"))
        c1, c2, c3 = st.columns([2, 1, 1])
        sf = c1.selectbox("从哪个文件抽样（data 目录）", files, key="pilot_src") if files else None
        sn = c2.number_input("抽多少条", 10, 5000, 150, 10, key="pilot_n")
        seed = c3.number_input("随机种子", 0, 9999, 42, key="pilot_seed")
        slug = re.sub(r"[^0-9A-Za-z_]", "_", Path(cfg_now).stem)
        if sf and st.button("抽样并保存"):
            rows = read_training(sf, Path(sf).stat().st_mtime_ns).to_dict("records")
            picked_rows = sample_rows(rows, int(sn), int(seed))
            outp = Path("data") / f"{slug}_pilot_{len(picked_rows)}.csv"
            pd.DataFrame(picked_rows).to_csv(outp, index=False, encoding="utf-8-sig")
            st.success(f"已保存 {outp.as_posix()}（{len(picked_rows)} 条），到“④ 运行与结果”页选“项目 data 目录中的文件”即可运行。")

        st.markdown("**划分调参集 / 验证集**（需要有标签列）")
        c1, c2, c3 = st.columns([2, 1, 1])
        gf = c1.selectbox("已标注文件", files, key="split_src") if files else None
        ratio = c2.slider("验证集比例", 0.2, 0.7, 0.5, 0.05, key="split_ratio")
        gcols = list(read_training(gf, Path(gf).stat().st_mtime_ns).columns) if gf else []
        glab = c3.selectbox("标签列", gcols, index=gcols.index(guess_col(gcols, {"label", "标签", "类别"}, gcols[-1])) if gcols else 0,
                            key="split_lab") if gcols else None
        if gf and glab and st.button("分层划分并保存"):
            rows = read_training(gf, Path(gf).stat().st_mtime_ns).to_dict("records")
            dev, test = split_dev_test(rows, glab, ratio)
            stem = Path(gf).with_suffix("")
            pd.DataFrame(dev).to_csv(f"{stem}_dev.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(test).to_csv(f"{stem}_test.csv", index=False, encoding="utf-8-sig")
            st.success(f"调参集 {stem.as_posix()}_dev.csv（{len(dev)} 条），验证集 {stem.as_posix()}_test.csv（{len(test)} 条）。按类别分层，各类比例与原文件一致。")

# ---------------- ③ 训练数据（可选）
with tab_train:
    label_names = [lab["name"] for lab in raw["task"].get("labels") or []]
    fs = raw.get("fewshot") or {}
    tpath = str(fs.get("path") or "")
    tcol, lcol = fs.get("text_col", "text"), fs.get("label_col", "label")
    st.info(
        "**没有训练数据也可以直接使用**：模型只按“② 分类任务”页的类别定义和边界规则判断，本页可以跳过。\n\n"
        "如果手上有同类数据的已标注样本（历史人工标注、公开数据集的训练集、人工审核过的结果），加入后可以开启两项增强：\n"
        "- **动态示例**：每条待分类文本自动附上训练数据中最相似的几条已标注样本，让模型学到这批数据的标注尺度；\n"
        "- **本地小模型**：用训练数据训练一个分类器作为额外一票，它和大模型的出错方式不同，“全票一致”更可信。\n\n"
        "两项都在本机运行，不额外调用 API。"
    )

    st.subheader("当前训练数据")
    tdf = training_df(fs)
    if tdf is not None and {tcol, lcol} - set(tdf.columns):
        st.error(f"{tpath} 缺少 {tcol} / {lcol} 列，无法使用")
        tdf = None
    if not tpath:
        st.caption("未使用训练数据。")
    elif tdf is None and not Path(tpath).exists():
        st.warning(f"配置中的训练数据文件 {tpath} 不存在，可以在下方重新加入。")
    if tdf is not None:
        counts = tdf[lcol].astype(str).str.strip().value_counts()
        known = counts[counts.index.isin(label_names)]
        c = st.columns(3)
        c[0].metric("样本数", len(tdf))
        c[1].metric("可用样本（标签属于当前类别）", int(known.sum()))
        c[2].metric("覆盖类别", f"{len(known)} / {len(label_names)}")
        st.caption(f"文件：{tpath}")
        unknown = counts[~counts.index.isin(label_names)]
        if len(unknown):
            st.warning(f"{int(unknown.sum())} 条样本的标签不在当前类别中（如 {list(unknown.index[:6])}），使用时会被忽略。")
        missing = [lab for lab in label_names if lab not in counts.index]
        if missing:
            st.warning(f"这些类别没有训练样本：{missing}")
        few = [lab for lab in label_names if 0 < counts.get(lab, 0) < 30]
        if few:
            st.caption(f"样本较少（< 30 条）的类别：{few}。动态示例仍然有用，本地小模型在这些类别上会不太准。")
        st.bar_chart(known.reindex(label_names).fillna(0), horizontal=True)
        with st.expander("预览训练数据"):
            st.dataframe(tdf[[tcol, lcol]].head(200), hide_index=True, width="stretch")

    st.subheader("加入训练数据")
    tv = st.session_state.setdefault("train_ver", 0)
    how = st.radio("方式", ["上传文件", "手动录入"], horizontal=True, key="train_how")
    new_texts, new_labels = [], []
    if how == "上传文件":
        tf = st.file_uploader("上传已标注数据（CSV / Excel / JSONL，需要有文本列和标签列）",
                              type=["csv", "xlsx", "xls", "jsonl"], key=f"train_up_{tv}")
        if tf is not None:
            try:
                udf = read_upload(tf)
            except Exception as e:  # noqa: BLE001
                st.error(f"读取失败：{e}")
                udf = None
            if udf is not None:
                ucols = [str(c) for c in udf.columns]
                udf.columns = ucols
                c1, c2 = st.columns(2)
                utc = c1.selectbox("文本列", ucols, key="train_tc",
                                   index=ucols.index(guess_col(ucols, {"text", "文本", "内容", "sentence"}, ucols[0])))
                ulc = c2.selectbox("标签列", ucols, key="train_lc",
                                   index=ucols.index(guess_col(ucols, {"label", "标签", "类别"}, ucols[-1])))
                new_texts, new_labels = list(udf[utc]), list(udf[ulc])
    else:
        st.caption("在表格中逐行填写，点表格下方的“+”增加一行；也可以从 Excel 复制两列直接粘贴进来。")
        manual = st.data_editor(
            pd.DataFrame({"文本": pd.Series(dtype=str), "标签": pd.Series(dtype=str)}),
            key=f"train_manual_{tv}", num_rows="dynamic", width="stretch",
            column_config={"文本": st.column_config.TextColumn(width="large"),
                           "标签": st.column_config.SelectboxColumn(options=label_names)},
        )
        new_texts, new_labels = list(manual["文本"]), list(manual["标签"])

    pairs = [(str(_clean(t, "")).strip(), str(_clean(y, "")).strip()) for t, y in zip(new_texts, new_labels)]
    pairs = [(t, y) for t, y in pairs if t and y]
    good = [(t, y) for t, y in pairs if y in label_names]
    bad_labels = sorted({y for _, y in pairs if y not in label_names})
    if pairs:
        msg = f"可加入 {len(good)} 条"
        if bad_labels:
            msg += f"；{len(pairs) - len(good)} 条的标签不在当前类别中，将被跳过：{bad_labels[:8]}"
        (st.warning if bad_labels else st.caption)(msg)

    slug = re.sub(r'[\\/:*?"<>|\s]+', "_", str(raw["task"].get("name") or "")).strip("_") or "task"
    c1, c2 = st.columns([2, 1])
    target = c1.text_input("保存到", value=tpath if tpath.lower().endswith(".csv") else f"data/train/{slug}.csv",
                           key=f"train_target_{ver}_{tv}")
    exists = Path(target).exists()
    replace = c2.radio("写入方式", ["追加（按文本去重）", "替换原有数据"], key="train_mode",
                       disabled=not exists) == "替换原有数据" and exists
    if st.button("💾 保存训练数据", type="primary", disabled=not good):
        try:
            added, skipped = save_examples(target, [t for t, _ in good], [y for _, y in good], append=not replace)
        except (OSError, ValueError) as e:
            st.error(str(e))
        else:
            fs = {k: v for k, v in fs.items() if k not in ("text_col", "label_col")}
            raw["fewshot"] = {**fs, "path": target, "k": int(fs.get("k") or 6)}
            st.session_state.train_ver = tv + 1
            st.session_state.train_msg = (f"已{'替换' if replace else '写入'} {target}：新增 {added} 条"
                                          + (f"，重复跳过 {skipped} 条" if skipped else "")
                                          + "。已自动开启动态示例；需要长期保留时请在左侧保存配置。")
            st.rerun()
    if st.session_state.get("train_msg"):
        st.success(st.session_state.pop("train_msg"))

    st.subheader("使用方式")
    has_data = tdf is not None
    fs = raw.get("fewshot") or {}
    if not has_data:
        st.caption("加入训练数据后才能开启下面两项。")
    c1, c2, c3 = st.columns([2, 1, 1])
    use_fs = c1.toggle("动态示例：提示词里附上最相似的已标注样本", value=has_data and int(fs.get("k", 6)) > 0,
                       disabled=not has_data, key=f"use_fs_{tv}_{has_data}")
    k = c2.number_input("每条附几个示例", min_value=1, max_value=20, value=int(fs.get("k") or 6), disabled=not use_fs,
                        key=f"fs_k_{tv}")
    max_chars = c3.number_input("每个示例最多字符", min_value=50, max_value=2000, step=50,
                                value=int(fs.get("max_chars", 300)), disabled=not use_fs, key=f"fs_mc_{tv}")
    if tpath:
        raw["fewshot"] = {**fs, "k": int(k) if use_fs else 0, "max_chars": int(max_chars)}

    has_local = any(m.get("provider") == "local" for m in raw.get("models") or [])
    use_local = st.toggle(
        "本地小模型：用训练数据训练一个分类器（TF-IDF + 逻辑回归）作为额外一票", value=has_local,
        disabled=not (has_data or has_local), key=f"use_local_{ver}_{has_local}",
        help="建议每个类别至少 30 条样本。开启后首轮需要全部投票者一致才自动通过，自动通过的样本更准，但转人工的会变多。",
    )
    if use_local != has_local:
        if use_local:
            names = {m["name"] for m in raw["models"]}
            raw["models"].append({"name": next(n for n in ("local", "local2", "local3") if n not in names), "provider": "local"})
        else:
            raw["models"] = [m for m in raw["models"] if m.get("provider") != "local"]
        st.session_state.models_df = models_to_df(raw)
        st.session_state.labels_df = labels_to_df(raw)
        st.session_state.ver = ver + 1
        st.rerun()
    if has_local and not has_data and not all(m.get("train_path") for m in raw["models"] if m.get("provider") == "local"):
        st.error("模型列表中有本地小模型，但没有可用的训练数据，运行会失败。请加入训练数据，或关闭上面的开关。")

    if has_data:
        with st.expander("🔎 试一试：输入一段文本，看看会检索到哪些示例、本地小模型怎么判"):
            q = st.text_area("文本", key="train_try", height=80).strip()
            if q:
                from crosscheck.local_model import get_bank, get_classifier
                try:
                    with st.spinner("正在加载训练数据…"):
                        ex = get_bank(tpath, tuple(label_names), tcol, lcol).nearest(q, int(k))
                        pred, prob = get_classifier(tpath, tuple(label_names), tcol, lcol).predict(q)
                    st.markdown(f"本地小模型判断：**{pred}**（概率 {prob:.2f}）")
                    st.dataframe(pd.DataFrame(ex, columns=["最相似的已标注样本", "标签"]), hide_index=True, width="stretch")
                except (OSError, ValueError, ImportError) as e:
                    st.error(str(e))

if save_clicked:
    if config_error:
        st.sidebar.error("配置有误，未保存")
    else:
        target = Path(save_path)
        if target.exists():
            base = base_of(target)
        else:
            cur = st.session_state.get("cfg_path", "config.yaml")
            base = base_of(cur) or (None if target.resolve() == Path("config.yaml").resolve() else Path("config.yaml"))
        save_config_diff(raw, target, base)
        v = save_version(save_path, raw.get("task"), save_note.strip())
        st.sidebar.success(f"已保存到 {save_path}" + (f"，分类标准保存为 v{v['version']}" if v else "，分类标准没有变化"))
        if v and v["version"] > 1:
            prev = list_versions(save_path)[-2]
            changed = changed_labels(prev["task"], v["task"])
            if changed:
                st.sidebar.warning(f"类别 {changed} 的定义或正反例有变化：按旧标准标注的金标准 / 训练数据中这些类别的样本可能需要复核。")

versions = list_versions(picked)
saved_as = next((v for v in versions if v["hash"] == criteria_hash(raw.get("task"))), None)
state = f"v{saved_as['version']}" if saved_as else ("有未保存的修改" if versions else "尚未保存版本")
version_slot.caption(f"分类标准：{state}（共 {len(versions)} 个版本）。在页面上修改后立即生效；需要长期保留时点击保存。")

# ---------------- ③ 运行与结果
with tab_run:
    kind = st.radio("要分类的是", ["普通文本", "多轮对话", "长文档"], horizontal=True, key="run_kind")
    extra_payload = None
    input_name = "上传文件"
    df = None
    if kind == "长文档":
        st.caption("上传 PDF / Word / 网页 / TXT。先按章节和关键词缩小范围，一份年报往往有几百段，直接全部分类花费会比较高。扫描版 PDF 读不出文字。")
        doc_files = st.file_uploader("上传文档（可多份）", type=["pdf", "docx", "html", "htm", "txt", "md"],
                                     accept_multiple_files=True, key="doc_files")
        local_docs = sorted(p.as_posix() for p in Path("data").glob("*")
                            if p.suffix.lower() in (".pdf", ".docx", ".html", ".htm", ".txt", ".md"))
        picked_docs = st.multiselect("或选择 data 目录中的文档", local_docs, key="doc_local") if local_docs else []
        c1, c2 = st.columns(2)
        min_chars = c1.number_input("短于多少字的段落丢掉", 10, 500, 40, 10, key="doc_min")
        kw_text = c2.text_input("只保留包含这些词的段落（可选，逗号分隔）", key="doc_kw", placeholder="例如：气候, 排放, 风险")
        blobs = [(f.name, f.getvalue()) for f in (doc_files or [])]
        blobs += [(Path(p).name, Path(p).read_bytes()) for p in picked_docs]
        paras, stats = [], []
        for name, data in blobs:
            try:
                got, stat = _split_doc_cached(name, data, int(min_chars))
                paras.extend(got)
                stats.append(f"{name}：读到 {stat['blocks']} 块，留下 {stat['kept']} 段")
            except (ValueError, OSError) as e:
                st.error(f"{name}：{e}")
        if paras:
            chapters = list(dict.fromkeys(p["chapter"] for p in paras))
            keep_ch = st.multiselect("只分类这些章节（不选 = 全部）", chapters, key="doc_chapters")
            keywords = re.split(r"[,，、\s]+", kw_text)
            paras = filter_paragraphs(paras, keywords, keep_ch or None)
            df = pd.DataFrame([{"id": p["id"], "text": p["text"], "文档": p["doc"], "章节": p["chapter"], "页码": p["page"]}
                               for p in paras])
            extra_payload = {"kind": "document", "items": {p["id"]: p for p in paras}}
            input_name = "、".join(n for n, _ in blobs)[:80]
            st.caption("；".join(stats) + f"。筛选后 {len(paras)} 段。")
            st.dataframe(df.head(8), width="stretch", hide_index=True)
    else:
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
        if up is not None:
            input_name = getattr(up, "name", input_name)
            try:
                df = read_upload(up)
            except Exception as e:  # noqa: BLE001
                st.error(f"读取失败：{e}")
        if df is not None and kind == "多轮对话":
            st.caption(f"共 {len(df)} 行。先选格式，再整理成待分类文本。")
            st.dataframe(df.head(), width="stretch", hide_index=True)
            layout = st.radio("对话格式", ["一行一句（有会话编号和角色）", "一行一段完整对话"], horizontal=True, key="dlg_layout")
            cols0 = [str(c) for c in df.columns]
            df.columns = cols0
            if layout.startswith("一行一句"):
                c1, c2, c3, c4 = st.columns(4)
                sess_c = c1.selectbox("会话编号列", cols0, index=cols0.index(guess_col(cols0, {"session_id", "会话", "conversation_id", "dialogue_id"}, cols0[0])), key="dlg_sess")
                role_c = c2.selectbox("角色列", cols0, index=cols0.index(guess_col(cols0, {"role", "角色", "speaker"}, cols0[min(1, len(cols0) - 1)])), key="dlg_role")
                text_c = c3.selectbox("内容列", cols0, index=cols0.index(guess_col(cols0, {"content", "text", "内容", "文本"}, cols0[0])), key="dlg_text")
                time_opts = ["（无）"] + cols0
                time_c = c4.selectbox("时间列", time_opts, index=time_opts.index(guess_col(time_opts, {"time", "时间"}, "（无）")), key="dlg_time")
                label_opts0 = ["（无）"] + cols0
                lab_c = st.selectbox("标签列（可选，整段一个标签，或每句一个标签）", label_opts0,
                                     index=label_opts0.index(guess_col(label_opts0, {"label", "标签", "类别"}, "（无）")), key="dlg_lab")
                sessions = group_turns(df.to_dict("records"), sess_c, role_c, text_c,
                                       None if time_c == "（无）" else time_c, None if lab_c == "（无）" else lab_c)
            else:
                c1, c2, c3 = st.columns(3)
                text_c = c1.selectbox("对话文本列", cols0, index=cols0.index(guess_col(cols0, {"text", "内容", "对话", "content"}, cols0[0])), key="dlg_text2")
                id_opts0 = ["（自动编号）"] + cols0
                id_c = c2.selectbox("会话编号列", id_opts0, index=id_opts0.index(guess_col(id_opts0, {"id", "编号", "session_id"}, "（自动编号）")), key="dlg_id2")
                label_opts0 = ["（无）"] + cols0
                lab_c = c3.selectbox("标签列", label_opts0, index=label_opts0.index(guess_col(label_opts0, {"label", "标签", "类别"}, "（无）")), key="dlg_lab2")
                sessions = sessions_from_transcripts(df.to_dict("records"), None if id_c == "（自动编号）" else id_c, text_c,
                                                     None if lab_c == "（无）" else lab_c)
            roles = sorted({t["role"] for s in sessions for t in s["turns"]})
            c1, c2, c3 = st.columns(3)
            granularity = c1.radio("分类粒度", ["整段对话一个类别", "逐句分类"], horizontal=True, key="dlg_grain")
            target = c2.multiselect("逐句时只分类这些角色（不选 = 每一句）", roles, key="dlg_roles",
                                    disabled=not granularity.startswith("逐句"))
            max_chars = c3.number_input("单条最多字数（超出则保留最近的内容）", 500, 20000, 4000, 500, key="dlg_chars")
            built, extra_payload, warns = build_dialogue_items(
                sessions, granularity="turn" if granularity.startswith("逐句") else "session",
                target_roles=target, max_chars=int(max_chars))
            for w in warns:
                st.warning(w)
            if not built:
                st.error("没有整理出可分类的对话，请检查列是否选对。")
                df = None
            else:
                if not all(it.get("label") for it in built):
                    if any(it.get("label") for it in built):
                        st.caption("只有部分对话有标签，本次不计算准确率。")
                    built = [{k: v for k, v in it.items() if k != "label"} for it in built]
                df = pd.DataFrame(built)
                st.caption(f"整理成 {len(sessions)} 段对话、{len(built)} 条待分类文本。角色：{'、'.join(roles) or '无'}。"
                           "审核页会按聊天气泡显示，并标出模型引用到的句子。")
        elif df is not None:
            st.caption(f"共 {len(df)} 行，预览前 5 行：")
            st.dataframe(df.head(), width="stretch", hide_index=True)
    if df is not None:
        cols = [str(c) for c in df.columns]
        df.columns = cols
        guess = lambda names, fallback: next((c for c in cols if c.lower() in names), fallback)
        if extra_payload:
            text_col, id_col = "text", "id"
            label_col = "label" if "label" in cols else "（无，只做分类）"
            st.caption(f"待分类 {len(df)} 条。对话和长文档使用整理后的文本，列已自动对应。")
        else:
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
        action = c1.selectbox("首轮出现分歧时", actions, index=actions.index(p.get("disagreement_action", "human")),
                              format_func=DISAGREEMENT_ACTIONS.get)
        accept = c2.slider("复核后自动采纳阈值", 0.5, 1.0, float(p.get("accept_threshold", 0.6)), 0.05,
                           help="过半模型同意且加权得票占比 ≥ 该值时自动采纳", disabled=action != "review")
        arb_th = c3.slider("仲裁采纳阈值", 0.0, 1.0, float(p.get("arbiter_threshold", 0.7)), 0.05,
                           help="仲裁模型置信度 ≥ 该值才采纳，否则转人工", disabled=action == "human")
        limit = c4.number_input("只处理前 N 条（0 = 全部）", min_value=0, value=0, step=10)
        c1, c2 = st.columns(2)
        concurrency = c1.number_input("总并发请求数", min_value=1, max_value=64, value=int(p.get("concurrency", 8)))
        mock = c2.toggle("mock 模式（不调用真实 API，只测流程）", value=False)
        voters = [m["name"] for m in raw.get("models") or [] if m.get("enabled", True)]
        c1, c2 = st.columns([3, 1])
        cascade = c1.multiselect(
            "级联调用（可选）：首轮先只问这几个模型，满足条件就直接采纳，不再调用其余模型",
            voters, default=[n for n in p.get("cascade") or [] if n in voters], key=f"cascade_{ver}",
            help="有本地小模型时推荐“本地小模型 + 一个大模型”：实测费用降到约 40%，人工兜底后的准确率只比四票全一致低约 2 个百分点。"
                 "没有训练数据时可选两个大模型，约省 30%，但准确率会下降几个百分点。",
        )
        casc_th = c2.slider("级联置信度门槛", 0.0, 1.0, float(p.get("cascade_min_confidence", 0.0)), 0.05,
                            key=f"casc_th_{ver}", disabled=not cascade,
                            help="0 = 首批模型一致就采纳。> 0 时还要求每个模型的置信度都达到门槛，此时首批可以只选 1 个模型。"
                                 "置信度优先用 logprobs 概率（模型配置里勾选“读取概率”），其次是模型自报的置信度；本地小模型为预测概率")
        need = 1 if casc_th > 0 else 2
        if cascade and not need <= len(cascade) < len(voters):
            st.warning(f"级联至少选 {need} 个模型，且要少于全部投票模型，否则不生效。")
            cascade = []
        shuffle = st.toggle("选项顺序随机化", value=bool((raw.get("task") or {}).get("shuffle_labels", False)),
                            key=f"shuffle_{ver}",
                            help="每个模型、每条文本看到的类别顺序不同（按哈希固定，重跑仍命中缓存），消除模型对靠前选项的偏好。"
                                 "开启后提示词变化，已有缓存不再命中")
        run_note = st.text_input("本次运行备注（可选，显示在“⑥ 历史与对比”页）", key="run_note",
                                 placeholder="例如：换了千问新版本 / 加了 3 条边界规则")

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

        fs_now = raw.get("fewshot") or {}
        tdf_now = training_df(fs_now)
        fs_on = tdf_now is not None and int(fs_now.get("k", 6)) > 0
        local_on = any(m.get("provider") == "local" and m.get("enabled", True) for m in raw.get("models") or [])
        if fs_on or local_on:
            st.caption(f"训练数据：{fs_now.get('path', '（本地模型自带）')}　·　动态示例 {'开' if fs_on else '关'}"
                       f"　·　本地小模型 {'开' if local_on else '关'}（在“③ 训练数据”页修改）")
        else:
            st.caption("未使用训练数据：模型只按分类任务中的定义和规则判断。")
        if tdf_now is not None and label_col != "（无，只做分类）" and fs_now.get("text_col", "text") in tdf_now.columns:
            seen_texts = set(tdf_now[fs_now.get("text_col", "text")].astype(str).str.strip())
            overlap = sum(it["text"] in seen_texts for it in items)
            if overlap:
                st.warning(f"有 {overlap} 条待评估样本也在训练数据中。动态示例会自动跳过完全相同的文本，"
                           "但本地小模型已经见过这些答案，评估出的准确率会偏高。评估时建议用不在训练数据里的样本。")

        run_raw = copy.deepcopy(raw)
        run_raw["pipeline"] = {**p, "disagreement_action": action, "accept_threshold": accept,
                               "arbiter_threshold": arb_th, "concurrency": int(concurrency), "cascade": cascade,
                               "cascade_min_confidence": casc_th if cascade else 0.0}
        run_raw["task"] = {**run_raw.get("task", {}), "shuffle_labels": shuffle}

        c1, c2, c3 = st.columns([1, 1, 2])
        rate = c2.slider("预估时假设的分歧比例", 0.0, 1.0, 0.25, 0.05,
                         help="首轮提示词会逐条查缓存、准确计算；后续环节（级联的其余模型、复核、仲裁）取决于模型是否一致，按这个比例估算")
        if c1.button("💰 预估调用次数和费用", disabled=bool(bad or config_error) or not items, width="stretch"):
            try:
                with st.spinner("正在逐条构造提示词并查询缓存…"):
                    st.session_state.estimate = estimate_run(config_from_dict(run_raw, mock=mock), items, rate)
            except (ValueError, OSError, ImportError) as e:
                st.error(f"无法预估：{e}")
        est = st.session_state.get("estimate")
        if est and est["n"] == len(items):
            m = st.columns(3)
            for col, key, title in zip(m, ("min", "expected", "max"),
                                       ("最少（全部一致）", f"预计（{est['disagree_rate']:.0%} 分歧）", "最多（全部分歧）")):
                col.metric(title, f"{est[key]['cost']:.3f} 元", f"实际请求 {est[key]['calls']} 次", delta_color="off")
            st.dataframe(pd.DataFrame(est["expected"]["rows"]), hide_index=True, width="stretch",
                         column_config={"费用": st.column_config.NumberColumn(format="%.4f 元")})
            if est["unpriced"]:
                st.caption(f"{est['unpriced']} 没有设置单价，费用按 0 计算（可在“① 模型配置”页填写）。")
            st.caption("token 按字数粗略估算，误差约 ±20%；运行结束后显示接口返回的实际用量。")

        st.caption("中途关闭页面或出错后重新运行同一份数据时，已完成的模型调用会命中缓存，不会重复付费。")
        if st.button(f"🚀 开始运行（{len(items)} 条）", type="primary", disabled=bool(bad or config_error) or not items):
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
            gold = {it["id"]: it["label"] for it in items} if label_col != "（无，只做分类）" else None
            paths = write_results(results, out, model_names, gold=gold)
            save_extra(paths["jsonl"], extra_payload)
            write_meta(paths["jsonl"], source="web", input_name=input_name, config=config,
                       raw=run_raw, stats=stats, elapsed=elapsed, n=len(items), has_gold=bool(gold),
                       config_path=st.session_state.get("cfg_path", "config.yaml"), note=run_note.strip())
            rep = None
            if gold:
                from crosscheck.report import build_report
                weights = {m.name: m.weight for m in config.models}
                rep = evaluate(results, gold, model_names, config.task.label_names, weights)
                paths["html"] = build_report(rep, out)
            st.session_state.last = {"results": results, "stats": stats, "elapsed": elapsed, "rep": rep,
                                     "paths": paths, "out": out, "model_names": model_names, "extra": extra_payload}
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

        extra_now = last.get("extra") or load_extra(last["paths"]["jsonl"])
        if extra_now and extra_now.get("kind") == "document":
            agg_rows, tone = aggregate_documents(results, extra_now)
            st.markdown("**按文档汇总**")
            if tone:
                st.caption(tone)
            agg_df = pd.DataFrame(agg_rows)
            share_cols = [c for c in agg_df.columns if c.endswith("占比") or c == "净语调"]
            st.dataframe(agg_df, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="percent") if c.endswith("占比")
                                        else st.column_config.NumberColumn(format="%+.2f") for c in share_cols})
            st.download_button("⬇ 文档汇总 CSV", agg_df.to_csv(index=False).encode("utf-8-sig"), "document_summary.csv")

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
        n_human = sum(r.status == STATUS_HUMAN for r in results)
        if n_human:
            st.info(f"有 {n_human} 条需要人工审核，可到“⑤ 人工审核”页逐条处理。")
        st.markdown("**调用次数与费用**")
        stats = last["stats"]
        c = st.columns(4)
        c[0].metric("实际请求", sum(v["calls"] for v in stats.values()))
        c[1].metric("缓存命中", sum(v["cache_hits"] for v in stats.values()))
        c[2].metric("本次费用", f"{sum(v.get('cost', 0) for v in stats.values()):.4f} 元")
        c[3].metric("缓存节省", f"{sum(v.get('saved', 0) for v in stats.values()):.4f} 元")
        st.dataframe(pd.DataFrame([{"模型": k, "实际请求": v["calls"], "缓存命中": v["cache_hits"], "失败": v["errors"],
                                    "输入 token": v.get("in_tokens", 0), "输出 token": v.get("out_tokens", 0),
                                    "费用": v.get("cost", 0.0), "缓存节省": v.get("saved", 0.0)}
                                   for k, v in stats.items()]), hide_index=True, width="stretch",
                     column_config={"费用": st.column_config.NumberColumn(format="%.4f 元"),
                                    "缓存节省": st.column_config.NumberColumn(format="%.4f 元")})
        st.caption("token 为接口返回的实际用量（接口未返回时按字数估算）；未设置单价的模型费用按 0 计算。")
        st.caption(f"文件已保存在：{Path(last['out']).resolve()}")

# ---------------- ⑥ 历史与对比（写在审核页之前：审核页没有结果时会 st.stop）
with tab_history:
    entries = []
    for rp in (p.as_posix() for p in list_runs()):
        try:
            recs = run_records(rp)
        except (OSError, ValueError):
            continue
        if recs:
            meta = load_meta(rp)
            entries.append({"path": rp, "meta": meta, "records": recs, "summary": summarize(recs, meta)})
    if not entries:
        st.info("还没有运行记录。在“④ 运行与结果”页运行一次，或用命令行 classify / evaluate。")
    else:
        by_path = {e["path"]: e for e in entries}

        def run_time(e: dict) -> str:
            return e["meta"].get("time") or f"{datetime.fromtimestamp(Path(e['path']).stat().st_mtime):%Y-%m-%d %H:%M:%S}"

        run_name, used = {}, set()
        for e in entries:
            name = f"{run_time(e)[5:16]} {e['meta'].get('note') or Path(e['path']).parent.name}"
            while name in used:
                name += "*"
            used.add(name)
            run_name[e["path"]] = name

        st.caption("每次运行（网页或命令行）都会记录时间、数据、配置快照和费用。勾选“对比”列选择要对比的运行，备注可直接编辑。")
        inputs = sorted({e["meta"].get("input") or "（未记录）" for e in entries})
        c1, c2 = st.columns([3, 1])
        pick_inputs = c1.multiselect("按数据筛选", inputs, placeholder="全部数据")
        only_gold = c2.toggle("只看有真实标签的运行", value=False)
        shown = [e for e in entries
                 if (not pick_inputs or (e["meta"].get("input") or "（未记录）") in pick_inputs)
                 and (not only_gold or e["summary"]["gold_n"])]
        hist = pd.DataFrame([{
            "对比": False,
            "时间": run_time(e)[:16],
            "备注": e["meta"].get("note", ""),
            "数据": e["meta"].get("input") or "（未记录）",
            "条数": e["summary"]["n"],
            "方案": settings_text(e["meta"], e["records"]),
            "标准版本": e["meta"].get("criteria_version") or "",
            "自动采纳": pct(e["summary"]["auto_rate"]),
            "全自动准确率": pct(e["summary"].get("acc")),
            "自动采纳准确率": pct(e["summary"].get("auto_acc")),
            "人工兜底后": pct(e["summary"].get("with_human")),
            "大模型调用/条": e["summary"]["calls_per_item"],
            "本次费用": e["summary"]["cost"],
            "不计缓存费用": e["summary"]["full_cost"],
            "路径": e["path"],
        } for e in shown], columns=["对比", "时间", "备注", "数据", "条数", "方案", "标准版本", "自动采纳", "全自动准确率",
                                     "自动采纳准确率", "人工兜底后", "大模型调用/条", "本次费用", "不计缓存费用", "路径"])
        pct_cfg = st.column_config.NumberColumn(format="%.1f%%")
        yuan_cfg = st.column_config.NumberColumn(format="%.4f 元")
        edited_hist = st.data_editor(
            hist, hide_index=True, width="stretch", key=f"hist_{len(entries)}_{'|'.join(pick_inputs)}_{only_gold}",
            disabled=[c for c in hist.columns if c not in ("对比", "备注")],
            column_config={
                "对比": st.column_config.CheckboxColumn(width="small"),
                "方案": st.column_config.TextColumn(width="large"),
                "自动采纳": pct_cfg, "全自动准确率": pct_cfg, "自动采纳准确率": pct_cfg, "人工兜底后": pct_cfg,
                "大模型调用/条": st.column_config.NumberColumn(format="%.2f"),
                "本次费用": yuan_cfg,
                "不计缓存费用": st.column_config.NumberColumn(
                    format="%.4f 元", help="本次费用 + 缓存节省：假如所有调用都是新请求需要花的钱，用来公平比较不同方案的成本"),
            },
        )
        st.caption("全自动准确率：分歧样本也取模型建议；人工兜底后：需人工的样本假设人工判对。早期运行的费用没有记录。")
        note_changes = [(r["路径"], str(_clean(r["备注"], "")).strip()) for _, r in edited_hist.iterrows()
                        if str(_clean(r["备注"], "")).strip() != (by_path[r["路径"]]["meta"].get("note") or "")]
        if note_changes and st.button(f"💾 保存备注（{len(note_changes)} 条）"):
            for rp, note in note_changes:
                set_note(rp, note)
            st.rerun()
        selected = [r["路径"] for _, r in edited_hist.iterrows() if r["对比"]]

        # -------- 运行详情
        st.divider()
        st.subheader("运行详情")
        detail_path = st.selectbox("选择运行", [e["path"] for e in shown] or [entries[0]["path"]], format_func=run_name.get)
        e = by_path[detail_path]
        meta, s = e["meta"], e["summary"]
        c = st.columns(5)
        c[0].metric("条数", s["n"], f"需人工 {s['human']} 条", delta_color="off")
        c[1].metric("自动采纳", f"{s['auto_rate'] * 100:.1f}%")
        if s["gold_n"]:
            c[2].metric("全自动准确率", f"{s['acc'] * 100:.1f}%")
            c[3].metric("人工兜底后", f"{s['with_human'] * 100:.1f}%")
        c[4].metric("本次费用", "未记录" if s["cost"] is None else f"{s['cost']:.4f} 元",
                    None if s["full_cost"] is None else f"不计缓存 {s['full_cost']:.4f} 元", delta_color="off")
        st.caption(f"方案：{settings_text(meta, e['records'])}　·　数据：{meta.get('input') or '未记录'}　·　"
                   f"配置文件：{meta.get('config_path') or '未记录'}　·　来源：{meta.get('source') or '未记录'}　·　"
                   f"文件：{detail_path}")
        if meta.get("backfilled"):
            st.caption("这次运行早于运行记录功能，元信息为事后补录：策略按结果推断，配置快照是补录时的配置文件内容。")
        c1, c2 = st.columns(2)
        if s.get("per_model"):
            c1.markdown("**各模型首轮准确率**")
            c1.dataframe(pd.DataFrame([{"模型": k, "准确率": pct(v)} for k, v in s["per_model"].items()]),
                         hide_index=True, width="stretch", column_config={"准确率": pct_cfg})
        if meta.get("stats"):
            c2.markdown("**调用与费用**")
            c2.dataframe(pd.DataFrame([{"模型": k, "请求": v["calls"], "缓存命中": v["cache_hits"],
                                        "输入 token": v.get("in_tokens", 0), "输出 token": v.get("out_tokens", 0),
                                        "费用": v.get("cost", 0.0)} for k, v in meta["stats"].items()]),
                         hide_index=True, width="stretch", column_config={"费用": yuan_cfg})
        if meta.get("config"):
            with st.expander("配置快照（这次运行实际使用的模型、任务和策略）"):
                st.json(meta["config"], expanded=False)
        c1, c2, c3 = st.columns([2, 1, 1])
        if c1.button("📥 载入这次运行的配置", disabled=not meta.get("config"), width="stretch",
                     help="把模型、类别定义、规则、训练数据和运行策略恢复为这次运行时的设置，用于复现或在此基础上修改"):
            snap = copy.deepcopy(meta["config"])
            st.session_state.raw = snap
            st.session_state.models_df = models_to_df(snap)
            st.session_state.labels_df = labels_to_df(snap)
            st.session_state.ver = st.session_state.get("ver", 0) + 1
            st.toast("已载入，可在前四页查看；需要长期保留时在侧边栏保存")
            st.rerun()
        confirm = c2.checkbox("确认删除", key=f"del_ok|{detail_path}", help="删除这次运行的结果文件和审核记录，不可恢复")
        if c3.button("🗑 删除这次运行", disabled=not confirm, width="stretch"):
            delete_run(detail_path)
            st.cache_data.clear()
            st.rerun()

        # -------- 多次运行对比
        st.divider()
        st.subheader("多次运行对比")
        if len(selected) < 2:
            st.caption("在上方表格的“对比”列勾选 2 个或更多运行。最适合同一份数据换模型、换策略、改规则前后的对比。")
        else:
            runs = {run_name[rp]: by_path[rp]["records"] for rp in selected}
            gold_ext = None
            if not all(by_path[rp]["summary"]["gold_n"] for rp in selected):
                files = sorted(p.as_posix() for p in Path("data").glob("**/*") if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".jsonl"))
                gf = st.selectbox("部分运行没有真实标签。可选一个含 id、label 列的标签文件作为对照（按 id 匹配）", ["（不使用）"] + files)
                if gf != "（不使用）":
                    try:
                        gold_ext = read_gold_file(gf)
                    except (ValueError, OSError) as err:
                        st.error(str(err))
            common, rows = compare_items(runs, gold_ext)
            gold_map = {r["id"]: r["gold"] for r in rows if r["gold"]}
            if not common:
                st.warning("这些运行没有共同样本（id 不一致），无法对比。")
            else:
                st.caption(f"共同样本 {len(common)} 条（按 id 匹配），其中 {len(gold_map)} 条有真实标签；以下指标只在共同样本上计算，费用为整次运行。")
                cid = set(common)
                comp = []
                for rp in selected:
                    name = run_name[rp]
                    sub = summarize([r for r in by_path[rp]["records"] if r["id"] in cid], by_path[rp]["meta"], gold_map or None)
                    comp.append({"运行": name, "方案": settings_text(by_path[rp]["meta"], by_path[rp]["records"]),
                                 "自动采纳": pct(sub["auto_rate"]), "全自动准确率": pct(sub.get("acc")),
                                 "自动采纳准确率": pct(sub.get("auto_acc")), "人工兜底后": pct(sub.get("with_human")),
                                 "大模型调用/条": sub["calls_per_item"], "不计缓存费用": by_path[rp]["summary"]["full_cost"],
                                 **{f"{m}": pct(v) for m, v in (sub.get("per_model") or {}).items()}})
                comp_df = pd.DataFrame(comp)
                model_cols = [c for c in comp_df.columns if c not in ("运行", "方案", "自动采纳", "全自动准确率", "自动采纳准确率",
                                                                     "人工兜底后", "大模型调用/条", "不计缓存费用")]
                st.dataframe(comp_df, hide_index=True, width="stretch",
                             column_config={**{c: pct_cfg for c in ["自动采纳", "全自动准确率", "自动采纳准确率", "人工兜底后"] + model_cols},
                                            "大模型调用/条": st.column_config.NumberColumn(format="%.2f"),
                                            "不计缓存费用": yuan_cfg, "方案": st.column_config.TextColumn(width="large")})
                if model_cols:
                    st.caption(f"{' / '.join(model_cols)} 列为各模型首轮单独的准确率（级联下只统计被调用的样本）。")
                ref = selected[0]
                ref_task = run_criteria(by_path[ref]["meta"])
                crit_diffs = {run_name[rp]: diff_criteria(ref_task, run_criteria(by_path[rp]["meta"]))
                              for rp in selected[1:] if ref_task and run_criteria(by_path[rp]["meta"])}
                if any(crit_diffs.values()):
                    with st.expander(f"📝 分类标准差异（相对 {run_name[ref]}）", expanded=True):
                        for n, lines in crit_diffs.items():
                            st.markdown(f"**{n}**：" + ("与参照相同" if not lines else ""))
                            for line in lines:
                                st.markdown(f"- {line}")
                if gold_map:
                    st.bar_chart(comp_df.set_index("运行")[["全自动准确率", "自动采纳准确率", "人工兜底后"]], stack=False, horizontal=True)

                    st.markdown("**差异是否显著**")
                    names = list(runs)
                    c1, c2 = st.columns([1, 2])
                    base_name = c1.selectbox("基准运行", names)
                    metric = c2.radio("比较指标", ["acc", "with_human"], horizontal=True,
                                      format_func={"acc": "全自动准确率", "with_human": "人工兜底后准确率"}.get)
                    maps = {n: {r["id"]: r for r in recs} for n, recs in runs.items()}
                    gid = [i for i in common if i in gold_map]
                    base_ok = [item_correct(maps[base_name][i], gold_map[i], metric) for i in gid]
                    tests = []
                    for n in names:
                        if n == base_name:
                            continue
                        ok = [item_correct(maps[n][i], gold_map[i], metric) for i in gid]
                        t = mcnemar(ok, base_ok)
                        tests.append({"对比运行": n, "准确率差（百分点）": (sum(ok) - sum(base_ok)) / len(gid) * 100,
                                      "仅它判对": t["only_a"], "仅基准判对": t["only_b"], "p 值": t["p"],
                                      "结论": "显著" if t["p"] < 0.05 else "不显著，可能是随机波动"})
                    st.dataframe(pd.DataFrame(tests), hide_index=True, width="stretch",
                                 column_config={"准确率差（百分点）": st.column_config.NumberColumn(format="%+.1f"),
                                                "p 值": st.column_config.NumberColumn(format="%.3f")})
                    st.caption("McNemar 精确检验：在同一批样本上，只看两次运行一个判对、一个判错的样本。p < 0.05 表示差异不太可能是随机波动；"
                               "150 条样本上相差 3~5 个百分点通常还不显著。")

                st.markdown("**逐条对照**")
                view = st.radio("显示", ["结论不同的样本", "有运行判错的样本", "全部"], horizontal=True, key="cmp_view")
                names = list(runs)
                disp, mask = [], []
                for r in rows:
                    labels_now = [r[n] for n in names]
                    wrong = [bool(r["gold"]) and r[n] != r["gold"] for n in names]
                    if view == "结论不同的样本" and len(set(labels_now)) == 1:
                        continue
                    if view == "有运行判错的样本" and not any(wrong):
                        continue
                    disp.append({"ID": r["id"], "文本": r["text"], "真实标签": r["gold"] or "",
                                 **{n: f"{r[n]}{'（人工）' if r[f'{n}|status'] == STATUS_HUMAN else ''}" for n in names}})
                    mask.append(wrong)
                st.caption(f"{len(disp)} 条。标红为判错；“（人工）”表示该运行把这条交给人工审核。")
                if disp:
                    ddf = pd.DataFrame(disp)
                    mdf = pd.DataFrame(mask, columns=names, index=ddf.index)
                    css = "background-color:#fdecea;color:#c0392b"
                    st.dataframe(ddf.style.apply(lambda _: mdf.map(lambda w: css if w else ""), axis=None, subset=names),
                                 hide_index=True, width="stretch", column_config={"文本": st.column_config.TextColumn(width="large")})
                    st.download_button("⬇ 下载对照表 CSV", ddf.to_csv(index=False).encode("utf-8-sig"), "compare.csv")

        st.divider()
        st.subheader("漂移监控")
        st.caption("同一分类任务按时间排列。最近一次和前一次比：人工比例升高、准确率下降或标签分布明显变化时会提示。"
                   "数据来源不同（比如一次是新闻、一次是股吧）也会触发，先看清“数据”列再下结论。")
        task_names = sorted({e["meta"].get("task_name") or "（未记录任务）" for e in entries})
        drift_task = st.selectbox("任务", task_names, key="drift_task")
        drift_entries = [e for e in entries if (e["meta"].get("task_name") or "（未记录任务）") == drift_task]
        drift_entries.sort(key=run_time)
        drift_rows = []
        for e in drift_entries:
            s = e["summary"]
            drift_rows.append({
                "name": run_name[e["path"]],
                "time": run_time(e),
                "input": e["meta"].get("input") or "",
                "human_rate": 1 - s["auto_rate"],
                "auto_rate": s["auto_rate"],
                "acc": s.get("acc"),
                "labels": label_share(e["records"]),
                "criteria_hash": e["meta"].get("criteria_hash") or "",
            })
        drift_df = pd.DataFrame([{
            "时间": r["time"], "运行": r["name"], "数据": r["input"],
            "需人工": r["human_rate"], "自动采纳": r["auto_rate"], "准确率": r["acc"], "标准": (r["criteria_hash"] or "")[:8],
        } for r in drift_rows])
        st.dataframe(drift_df, hide_index=True, width="stretch",
                     column_config={"需人工": st.column_config.NumberColumn(format="percent"),
                                    "自动采纳": st.column_config.NumberColumn(format="percent"),
                                    "准确率": st.column_config.NumberColumn(format="percent")})
        if len(drift_rows) >= 2:
            chart = pd.DataFrame({"需人工 %": [r["human_rate"] * 100 for r in drift_rows],
                                  "准确率 %": [(r["acc"] * 100 if r["acc"] is not None else None) for r in drift_rows]},
                                 index=[r["time"][5:16] for r in drift_rows])
            st.line_chart(chart)
            for alert in drift_alerts(drift_rows):
                st.warning(alert)
        else:
            st.caption("这个任务至少要有两次运行才能比较。")

        st.subheader("规则建议")
        st.caption("挑一次有分歧的运行，让模型归纳分歧集中在哪几条边界，并给出可以直接追加的规则。只分析，不会自动改标准。")
        suggest_path = st.selectbox("分析哪次运行", [e["path"] for e in entries], format_func=lambda p: run_name[p], key="suggest_run")
        suggest_models = []
        try:
            suggest_models = drafting_models(config_from_dict(raw))
        except ValueError:
            suggest_models = []
        if not suggest_models:
            st.caption("当前没有可用于分析的大模型（本地小模型和 mock 不行）。")
        elif st.button("🪄 根据分歧建议规则", key="suggest_btn"):
            task_now = run_criteria(by_path[suggest_path]["meta"]) or raw.get("task") or {}
            model = next((m for m in suggest_models if m.name == "qwen"), suggest_models[0])
            try:
                with st.spinner("正在阅读分歧样本…"):
                    advice = suggest_rules(config_from_dict(raw), model, task_now, by_path[suggest_path]["records"])
                st.session_state.rule_advice = advice
            except LLMError as e:
                st.error(str(e))
        advice = st.session_state.get("rule_advice")
        if advice:
            st.write(advice.get("summary") or "")
            st.caption(f"分析了 {advice.get('n', 0)} 条分歧样本。")
            if advice.get("patterns"):
                st.dataframe(pd.DataFrame(advice["patterns"]), hide_index=True, width="stretch")
            edits = [r for r in advice.get("rule_edits") or [] if r not in (raw.get("task") or {}).get("rules", [])]
            if edits:
                st.markdown("**建议追加的规则**\n" + "\n".join(f"- {r}" for r in edits))
                if st.button("追加到当前分类标准（还要在侧边栏保存才会写入文件）", key="apply_rules"):
                    raw["task"]["rules"] = list((raw.get("task") or {}).get("rules") or []) + edits
                    st.session_state.ver = st.session_state.get("ver", 0) + 1
                    st.toast("已追加到“② 分类任务”页，确认后在侧边栏保存")
                    st.rerun()
            elif advice.get("n", 0) >= 3:
                st.caption("模型没有给出可直接追加的规则。")

# ---------------- ⑤ 人工审核
with tab_review:
    runs = [p.as_posix() for p in list_runs()]
    if not runs:
        st.info("还没有运行结果。先在“④ 运行与结果”页运行一次。")
        st.stop()

    last = st.session_state.get("last")
    last_path = Path(last["paths"]["jsonl"]).as_posix() if last else None
    run_path = st.selectbox(
        "选择要审核的运行结果", runs, index=runs.index(last_path) if last_path in runs else 0,
        format_func=lambda s: f"{s}　（{datetime.fromtimestamp(Path(s).stat().st_mtime):%m-%d %H:%M}）",
    )
    records = load_run(run_path)
    reviews = load_reviews(run_path)
    model_names = run_model_names(records)
    seen = [lab for r in records for lab in [r["label"]] + [p.get("label") for p in r["round1"]] if lab]
    labels = list(dict.fromkeys([lab["name"] for lab in raw["task"]["labels"]] + seen))

    c1, c2, c3 = st.columns([2, 1, 1])
    scope = c1.selectbox("审核范围", list(SCOPES), format_func=SCOPES.get,
                         help="建议除了“需人工审核”外，也定期抽检“首轮一致通过”的样本：模型可能一致地答错")
    spot_n = c2.number_input("抽检条数", min_value=1, max_value=1000, value=20, disabled=scope != "spot")
    only_pending = c3.toggle("只显示未审核", value=True)
    queue = select_queue(records, scope, int(spot_n))
    todo = [r for r in queue if r["id"] not in reviews] if only_pending else queue
    done_n = sum(r["id"] in reviews for r in queue)
    st.progress(done_n / len(queue) if queue else 1.0, text=f"本范围共 {len(queue)} 条，已审核 {done_n} 条")

    mode = st.radio("审核方式", ["逐条审核", "表格批量审核"], horizontal=True)
    rv_ver = st.session_state.setdefault("rv_ver", 0)
    idx_key = f"rv_idx|{run_path}|{scope}|{only_pending}"

    if not todo:
        st.success("本范围内的样本都已审核完 🎉" if queue else "本范围内没有样本")
    elif mode == "逐条审核":
        idx = min(st.session_state.get(idx_key, 0), len(todo) - 1)
        rec = todo[idx]
        prev = reviews.get(rec["id"], {})
        head = f"**第 {idx + 1} / {len(todo)} 条**　·　ID `{rec['id']}`　·　{STATUS_TEXT[rec['status']]}"
        if rec.get("note"):
            head += f"　·　{rec['note']}"
        if prev:
            head += f"　·　✅ 已审核为「{prev['label']}」"
        st.markdown(head)
        show_item_body(rec, load_extra(run_path))

        r2 = {p["model"]: p for p in rec.get("round2") or []}
        ops = []
        for p in rec["round1"]:
            q = r2.get(p["model"], {})
            ops.append({
                "模型": p["model"],
                "首轮": p.get("label") or "失败",
                "置信度": p.get("confidence") if p.get("label") else None,
                "首轮理由": p.get("reason") or p.get("error") or "",
                "复核后": q.get("label") or ("—" if not q else "失败"),
                "复核理由": q.get("reason", ""),
            })
        arb = rec.get("arbiter")
        if arb:
            ops.append({"模型": f"{arb['model']}（仲裁）", "首轮": arb.get("label") or "失败",
                        "置信度": arb.get("confidence") if arb.get("label") else None,
                        "首轮理由": arb.get("reason") or arb.get("error") or "", "复核后": "", "复核理由": ""})
        st.dataframe(pd.DataFrame(ops), hide_index=True, width="stretch",
                     column_config={"置信度": st.column_config.NumberColumn(format="%.2f"),
                                    "首轮理由": st.column_config.TextColumn(width="large"),
                                    "复核理由": st.column_config.TextColumn(width="large")})

        votes = round1_votes(rec)
        current = prev.get("label") or rec["label"]

        def fmt(lab: str) -> str:
            tags = [f"{votes[lab]} 票"] if lab in votes else []
            if lab == rec["label"]:
                tags.append("模型建议")
            return f"{lab}（{'，'.join(tags)}）" if tags else lab

        choice = st.radio("人工判定", labels, index=labels.index(current) if current in labels else 0,
                          horizontal=True, format_func=fmt, key=f"rv_choice|{run_path}|{rec['id']}")
        note = st.text_input("备注（可选）", value=prev.get("note", ""), key=f"rv_note|{run_path}|{rec['id']}")
        b1, b2, b3 = st.columns([1, 2, 1])
        if b1.button("⬅ 上一条", disabled=idx == 0, width="stretch"):
            st.session_state[idx_key] = idx - 1
            st.rerun()
        if b2.button("✅ 确认并下一条", type="primary", width="stretch"):
            set_review(reviews, rec, choice, note.strip())
            save_reviews(run_path, reviews)
            if not only_pending:
                st.session_state[idx_key] = idx + 1
            st.rerun()
        if b3.button("跳过 ➡", disabled=idx >= len(todo) - 1, width="stretch"):
            st.session_state[idx_key] = idx + 1
            st.rerun()
    else:
        table = pd.DataFrame([{
            "ID": r["id"],
            "文本": r["text"],
            "模型建议": r["label"],
            "各模型首轮": " / ".join(f"{p['model']}:{p.get('label') or '失败'}" for p in r["round1"]),
            "人工标签": reviews.get(r["id"], {}).get("label"),
            "备注": reviews.get(r["id"], {}).get("note", ""),
        } for r in todo])
        edited = st.data_editor(
            table, key=f"rv_table|{run_path}|{scope}|{only_pending}|{rv_ver}", hide_index=True, width="stretch",
            disabled=["ID", "文本", "模型建议", "各模型首轮"],
            column_config={"人工标签": st.column_config.SelectboxColumn(options=labels),
                           "文本": st.column_config.TextColumn(width="large")},
        )
        by_id = {r["id"]: r for r in todo}
        c1, c2 = st.columns(2)
        if c1.button("💾 保存表格中已填写的人工标签", type="primary", width="stretch"):
            n = 0
            for _, row in edited.iterrows():
                lab = _clean(row["人工标签"], "")
                if lab:
                    set_review(reviews, by_id[row["ID"]], lab, str(_clean(row["备注"], "")).strip())
                    n += 1
            save_reviews(run_path, reviews)
            st.session_state.rv_ver = rv_ver + 1
            st.toast(f"已保存 {n} 条")
            st.rerun()
        if c2.button("✔ 未填写的全部采用模型建议", width="stretch",
                     help="适合抽检：只改有问题的几条，其余一键确认"):
            n = 0
            for _, row in edited.iterrows():
                lab = _clean(row["人工标签"], "") or row["模型建议"]
                set_review(reviews, by_id[row["ID"]], lab, str(_clean(row["备注"], "")).strip())
                n += 1
            save_reviews(run_path, reviews)
            st.session_state.rv_ver = rv_ver + 1
            st.toast(f"已确认 {n} 条")
            st.rerun()

    # -------- 统计
    st.divider()
    st.subheader("审核统计")
    stats = review_stats(records, reviews, model_names)
    if not stats["n"]:
        st.caption("还没有审核记录。")
    else:
        c = st.columns(3)
        c[0].metric("已审核", f"{stats['n']} / {len(records)}")
        c[1].metric("互检系统与人工一致", f"{stats['agree']['互检系统'] * 100:.1f}%")
        c[2].metric("人工改动", f"{stats['changed']} 条")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**与人工判定的一致率**（仅统计已审核样本）")
            st.bar_chart(pd.DataFrame({"一致率 %": {k: v * 100 for k, v in stats["agree"].items()}}), horizontal=True)
        with c2:
            st.markdown("**按处理环节**")
            st.dataframe(pd.DataFrame([{"处理环节": STATUS_TEXT[s], "已审核": v["n"],
                                        "模型结果被确认": v["agree"], "确认率": v["agree"] / v["n"] * 100}
                                       for s, v in stats["by_status"].items()]),
                         hide_index=True, width="stretch",
                         column_config={"确认率": st.column_config.NumberColumn(format="%.1f%%")})
        st.caption("如果审核范围只包含难例（需人工审核、首轮有分歧），这里的一致率会明显低于整体准确率，属正常现象。")

    # -------- 导出与回流
    st.divider()
    st.subheader("导出与回流")
    merged = pd.DataFrame(merged_rows(records, reviews))
    c1, c2 = st.columns([1, 2])
    c1.download_button("⬇ 下载最终结果（合并人工审核）", merged.to_csv(index=False).encode("utf-8-sig"),
                       f"{Path(run_path).stem}_final.csv", width="stretch")
    with c2:
        golds = sorted(p.as_posix() for p in Path("data").glob("**/*.csv"))
        train_path = (raw.get("fewshot") or {}).get("path", "")
        default = golds.index(train_path) if train_path in golds else len(golds)
        target = st.selectbox("加入到金标准 / 训练数据文件", golds + ["（新建文件）"], index=default,
                              help="选当前的训练数据文件，之后分类时就会参考这些人工确认过的样本")
        if target == "（新建文件）":
            target = st.text_input("新文件路径", value="data/gold_reviewed.csv")
        if st.button("📥 把已审核样本加入金标准", disabled=not stats["n"], width="stretch"):
            try:
                added, skipped = append_to_gold(records, reviews, target)
                st.success(f"已写入 {target}：新增 {added} 条，重复跳过 {skipped} 条。"
                           "金标准可在“④ 运行与结果”页用来评估；训练数据文件会在下次分类时自动生效。")
            except ValueError as e:
                st.error(str(e))
