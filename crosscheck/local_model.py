"""基于训练集的本地文本能力：相似样本检索（动态 few-shot）与 TF-IDF + 逻辑回归分类器。

两者都在本机运行，不调用任何 API；依赖 scikit-learn。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError("动态示例和本地模型需要 scikit-learn：pip install scikit-learn") from e


def load_examples(path: str | Path, labels: list[str], text_col: str = "text", label_col: str = "label") -> tuple[list[str], list[str]]:
    """读取训练集，只保留标签在候选集合内的样本，并按文本去重。"""
    from .io_utils import read_items  # io_utils 依赖 pipeline，延迟导入避免循环

    items = read_items(path, text_col=text_col, label_col=label_col)
    allowed, seen = set(labels), set()
    texts, ys = [], []
    for it in items:
        if it["label"] in allowed and it["text"] not in seen:
            seen.add(it["text"])
            texts.append(it["text"])
            ys.append(it["label"])
    if not texts:
        raise ValueError(f"{path} 中没有标签属于 {labels} 的样本")
    return texts, ys


def _char_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1, max_features=300_000)


class ExampleBank:
    """按字符 n-gram TF-IDF 余弦相似度，从训练集中找出与待分类文本最相似的已标注样本。"""

    def __init__(self, texts: list[str], labels: list[str]):
        _require_sklearn()
        self.texts, self.labels = texts, labels
        self.vec = _char_vectorizer()
        self.matrix = self.vec.fit_transform(texts)
        self.index = {t: i for i, t in enumerate(texts)}

    def nearest(self, text: str, k: int) -> list[tuple[str, str]]:
        if k <= 0:
            return []
        sims = (self.matrix @ self.vec.transform([text]).T).toarray().ravel()
        same = self.index.get(text)
        if same is not None:
            sims[same] = -1.0
        top = sims.argsort()[::-1][:k]
        return [(self.texts[i], self.labels[i]) for i in top if sims[i] > 0]


class TextClassifier:
    """词 + 字符 n-gram TF-IDF 特征上的逻辑回归，类别按样本量加权。"""

    def __init__(self, texts: list[str], labels: list[str]):
        _require_sklearn()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union

        if len(set(labels)) < 2:
            raise ValueError("本地小模型至少需要 2 个类别的训练样本")

        features = make_union(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            _char_vectorizer(),
        )
        self.model = make_pipeline(features, LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))
        self.model.fit(texts, labels)

    def predict(self, text: str) -> tuple[str, float]:
        probs = self.model.predict_proba([text])[0]
        i = int(probs.argmax())
        return str(self.model.classes_[i]), float(probs[i])


def save_examples(path: str | Path, texts: list[str], labels: list[str], append: bool = True) -> tuple[int, int]:
    """写入 text,label 两列的训练集 csv，按文本去重。append=False 时覆盖原文件。返回 (新增条数, 重复跳过条数)。"""
    import csv

    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("训练数据文件必须是 .csv")
    rows: list[dict] = []
    if append and path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = [{"text": r.get("text", ""), "label": r.get("label", "")} for r in csv.DictReader(f)]
    seen = {r["text"].strip() for r in rows}
    added = skipped = 0
    for t, y in zip(texts, labels):
        t, y = str(t).strip(), str(y).strip()
        if not t or not y:
            continue
        if t in seen:
            skipped += 1
            continue
        rows.append({"text": t, "label": y})
        seen.add(t)
        added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)
    return added, skipped


def _stamp(path: str) -> int:
    p = Path(path)
    return p.stat().st_mtime_ns if p.exists() else 0


# 缓存键包含文件修改时间：训练数据被追加或替换后自动重新加载
def get_bank(path: str, labels: tuple[str, ...], text_col: str = "text", label_col: str = "label") -> ExampleBank:
    return _bank(path, _stamp(path), labels, text_col, label_col)


def get_classifier(path: str, labels: tuple[str, ...], text_col: str = "text", label_col: str = "label") -> TextClassifier:
    return _classifier(path, _stamp(path), labels, text_col, label_col)


@lru_cache(maxsize=8)
def _bank(path, stamp, labels, text_col, label_col) -> ExampleBank:
    return ExampleBank(*load_examples(path, list(labels), text_col, label_col))


@lru_cache(maxsize=8)
def _classifier(path, stamp, labels, text_col, label_col) -> TextClassifier:
    return TextClassifier(*load_examples(path, list(labels), text_col, label_col))
