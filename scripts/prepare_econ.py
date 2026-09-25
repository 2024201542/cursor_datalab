"""下载并整理三个经济/金融文本分类公开数据集，生成金标准测试集和提示词示例。

    python scripts/prepare_econ.py            # 每类抽 50 条测试样本
    python scripts/prepare_econ.py --per-class 100

数据来源（均为公开数据集，已划分训练集/测试集）：
  climate  ClimateBERT climate_sentiment：上市公司年报/可持续发展报告中的气候相关段落
           标签 风险/中性/机遇（train 1000 / test 320，CC BY-NC-SA 4.0）
  fomc     Trillion Dollar Words（ACL 2023）：美联储 FOMC 会议纪要/讲话/发布会句子
           标签 鸽派/鹰派/中性（train 1984 / test 496，CC BY-NC 4.0）
  finfe    BBT-FinCUGE FinFE：中文股吧/财经社交媒体帖子
           标签 消极/中性/积极（train 16157 / eval 2020，测试集未公开，用 eval 代替）
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / ".cache" / "econ"
OUT = ROOT / "data"

HF = "https://hf-mirror.com/datasets"
GH = "https://raw.githubusercontent.com/supersymmetry-technologies/BBT-FinCUGE-Applications/main/FinCUGE_Publish/finfe"

SOURCES = {
    "climate_train.parquet": f"{HF}/climatebert/climate_sentiment/resolve/main/data/train-00000-of-00001-04b49ae22f595095.parquet",
    "climate_test.parquet": f"{HF}/climatebert/climate_sentiment/resolve/main/data/test-00000-of-00001-3f9f7af4f5914b8e.parquet",
    "fomc_train.csv": f"{HF}/gtfintechlab/fomc_communication/resolve/main/train.csv",
    "fomc_test.csv": f"{HF}/gtfintechlab/fomc_communication/resolve/main/test.csv",
    "finfe_train_list.json": f"{GH}/train_list.json",
    "finfe_eval_list.json": f"{GH}/eval_list.json",
}

LABELS = {
    "climate": {0: "风险", 1: "中性", 2: "机遇"},
    "fomc": {0: "鸽派", 1: "鹰派", 2: "中性"},
    "finfe": {0: "消极", 1: "中性", 2: "积极"},
}


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = RAW / name
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"下载 {name} ...")
        urllib.request.urlretrieve(url, path)


def load(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (train, test)，统一为 text/label 两列，label 为中文。"""
    if name == "climate":
        train = pd.read_parquet(RAW / "climate_train.parquet")
        test = pd.read_parquet(RAW / "climate_test.parquet")
    elif name == "fomc":
        train = pd.read_csv(RAW / "fomc_train.csv").rename(columns={"sentence": "text"})
        test = pd.read_csv(RAW / "fomc_test.csv").rename(columns={"sentence": "text"})
    else:
        def read(f: str) -> pd.DataFrame:
            rows = json.loads((RAW / f).read_text(encoding="utf-8"))
            return pd.DataFrame(rows, columns=["text", "label"])
        train, test = read("finfe_train_list.json"), read("finfe_eval_list.json")
    out = []
    for df in (train, test):
        df = df[["text", "label"]].copy()
        df["text"] = df["text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        df = df[df["text"].str.len() >= 4].drop_duplicates("text")
        df["label"] = df["label"].map(LABELS[name])
        out.append(df.reset_index(drop=True))
    return out[0], out[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    download()
    examples: dict[str, dict[str, list[str]]] = {}
    for name in LABELS:
        train, test = load(name)
        test_texts = set(test["text"])
        n = min(args.per_class, test["label"].value_counts().min())
        gold = test.groupby("label").sample(n=n, random_state=args.seed)
        gold = gold.sample(frac=1, random_state=args.seed).reset_index()
        gold["id"] = [f"{name}{i}" for i in gold["index"]]
        path = OUT / f"econ_{name}_gold.csv"
        gold[["id", "text", "label"]].to_csv(path, index=False, encoding="utf-8-sig")
        # 训练集去掉与测试集重复的文本，供动态示例检索和本地模型训练使用
        pool = train[~train["text"].isin(test_texts)]
        pool[["text", "label"]].to_csv(OUT / f"econ_{name}_train.csv", index=False, encoding="utf-8-sig")
        print(f"{path.name}: {len(gold)} 条（每类 {n}），测试集共 {len(test)} 条；训练集 {len(pool)} 条 → econ_{name}_train.csv")

        # 静态提示词示例只取训练集中的短样本
        short = pool[pool["text"].str.len().between(15, 220)]
        examples[name] = {
            lab: g.sample(n=min(2, len(g)), random_state=args.seed)["text"].tolist()
            for lab, g in short.groupby("label")
        }

    path = RAW / "examples.json"
    path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练集示例已写入 {path}")


if __name__ == "__main__":
    main()
