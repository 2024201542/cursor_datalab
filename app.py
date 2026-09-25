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
    "max_tokens": "最大输出", "max_concurrency": "并发上限", "rpm": "每分钟请求",
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
tab_models, tab_task, tab_train, tab_run, tab_review = st.tabs(
    ["① 模型配置", "② 分类任务", "③ 训练数据（可选）", "④ 运行与结果", "⑤ 人工审核"])

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
        cascade = st.multiselect(
            "级联调用（可选）：首轮先只问这几个模型，它们全部一致就直接采纳，不再调用其余模型",
            voters, default=[n for n in p.get("cascade") or [] if n in voters], key=f"cascade_{ver}",
            help="有本地小模型时推荐“本地小模型 + 一个大模型”：实测大模型调用量降到约 57%，人工兜底后的准确率与四票全一致接近。"
                 "没有训练数据时可选两个大模型，约省 30%，但准确率会下降几个百分点。",
        )
        if cascade and not 2 <= len(cascade) < len(voters):
            st.warning("级联至少选 2 个模型，且要少于全部投票模型，否则不生效。")
            cascade = []

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
                               "arbiter_threshold": arb_th, "concurrency": int(concurrency), "cascade": cascade}

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

# ---------------- ④ 人工审核
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
        st.markdown(f"<div style='font-size:1.25rem;padding:14px 18px;background:#f4f7fb;border-radius:8px;"
                    f"margin:6px 0 12px'>{html.escape(rec['text'])}</div>", unsafe_allow_html=True)

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
