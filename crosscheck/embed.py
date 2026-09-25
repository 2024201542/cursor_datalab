"""向量检索：调用 OpenAI 兼容的 embeddings 接口，把训练集和待分类文本变成向量再比相似度。

默认用阿里云百炼的 text-embedding-v4（北京地域）。向量按文本哈希缓存在 .cache/embeddings，
相同文本不重复计费。不缓存原文。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

from .config import FewShotConfig
from .llm import LLMError
from .local_model import _stamp, load_examples


class EmbedError(LLMError):
    pass


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(vectors) -> np.ndarray:
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class EmbeddingCache:
    """哈希 → 向量。文件只追加，已经有的哈希不再写入。"""

    def __init__(self, model: str, dim: int, root: str | Path = ".cache/embeddings"):
        safe = re.sub(r"[^\w.\-]+", "_", model)
        self.path = Path(root) / f"{safe}_{dim}.jsonl"
        self.dim = dim
        self.vecs: dict[str, list[float]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vec = obj.get("v")
                if isinstance(vec, list) and len(vec) == dim and obj.get("h"):
                    self.vecs[obj["h"]] = vec

    def get(self, text: str) -> list[float] | None:
        return self.vecs.get(_hash(text))

    def add(self, pairs: list[tuple[str, list[float]]]) -> None:
        if not pairs:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for text, vec in pairs:
                h = _hash(text)
                if h in self.vecs:
                    continue
                stored = [round(float(x), 6) for x in vec]
                self.vecs[h] = stored
                f.write(json.dumps({"h": h, "v": stored}, ensure_ascii=False) + "\n")


def embed_texts(texts: list[str], fs: FewShotConfig, timeout: float = 60) -> tuple[list[list[float]], int]:
    """按配置调用向量接口。返回与 texts 等长的向量，以及本次消耗的 token 数。"""
    import os

    key = os.environ.get(fs.embed_api_key_env, "")
    if not key:
        raise EmbedError(f"向量检索需要环境变量 {fs.embed_api_key_env}（与千问同一把百炼密钥）")
    url = fs.embed_base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {key}"}
    out: list[list[float] | None] = [None] * len(texts)
    tokens = 0
    batch = 10  # 百炼 text-embedding 单次最多 10 条
    with httpx.Client(timeout=timeout) as http:
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            body = {"model": fs.embed_model, "input": chunk, "dimensions": fs.embed_dimensions,
                    "encoding_format": "float"}
            last = ""
            for attempt in range(4):
                try:
                    resp = http.post(url, headers=headers, json=body)
                except httpx.TransportError as e:
                    last = f"网络错误: {e!r}"
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    rows = data.get("data") or []
                    if rows and "index" in rows[0]:
                        rows = sorted(rows, key=lambda d: d["index"])
                    if len(rows) != len(chunk):
                        raise EmbedError(f"向量接口返回 {len(rows)} 条，请求的是 {len(chunk)} 条")
                    for i, row in enumerate(rows):
                        vec = row.get("embedding")
                        if not isinstance(vec, list) or len(vec) != fs.embed_dimensions:
                            raise EmbedError(f"向量维度不是 {fs.embed_dimensions}（实际 {len(vec) if isinstance(vec, list) else '无'}）")
                        out[start + i] = vec
                    tokens += int((data.get("usage") or {}).get("total_tokens") or 0)
                    break
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in (408, 429, 500, 502, 503):
                    raise EmbedError(f"向量接口失败（{fs.embed_model}）：{last}")
                time.sleep(2 ** attempt)
            else:
                raise EmbedError(f"向量接口重试后仍失败：{last}")
    if any(v is None for v in out):
        raise EmbedError("向量接口有文本没有返回结果")
    return [v for v in out if v is not None], tokens


class VectorBank:
    """用余弦相似度找最相近的已标注样本。接口与 ExampleBank.nearest 相同。"""

    def __init__(self, texts: list[str], labels: list[str], cache: EmbeddingCache, fs: FewShotConfig,
                 embedder=None):
        self.texts, self.labels = texts, labels
        self.cache, self.fs = cache, fs
        self.embedder = embedder  # 测试用：texts -> (vectors, tokens)。不传则调用向量接口
        self.index = {t: i for i, t in enumerate(texts)}
        self._query: dict[str, np.ndarray] = {}
        self.matrix: np.ndarray | None = None
        self.tokens = 0
        self.new_texts = 0
        self._fit()

    def _fit(self) -> None:
        vecs = [self.cache.get(t) for t in self.texts]
        if vecs and all(v is not None for v in vecs):
            self.matrix = _normalize(vecs)

    def prepare(self, texts: list[str]) -> tuple[int, int]:
        """补齐训练集和这批文本的向量。返回 (新计算的条数, token 数)。"""
        missing = [t for t in list(dict.fromkeys(list(self.texts) + list(texts))) if self.cache.get(t) is None]
        tokens = 0
        if missing:
            vecs, tokens = self.embedder(missing) if self.embedder else embed_texts(missing, self.fs)
            self.cache.add(list(zip(missing, vecs)))
            self.tokens += tokens
            self.new_texts += len(missing)
        self._fit()
        for t in texts:
            vec = self.cache.get(t)
            if vec is not None:
                self._query[t] = _normalize([vec])[0]
        return len(missing), tokens

    def _vector(self, text: str) -> np.ndarray:
        if text in self._query:
            return self._query[text]
        if self.matrix is not None and text in self.index:
            return self.matrix[self.index[text]]
        self.prepare([text])
        if text not in self._query:
            raise EmbedError("没有拿到这段文本的向量")
        return self._query[text]

    def nearest(self, text: str, k: int) -> list[tuple[str, str]]:
        if k <= 0 or not self.texts:
            return []
        if self.matrix is None:
            self.prepare([text])
        if self.matrix is None:
            return []
        q = self._vector(text)
        sims = self.matrix @ q
        same = self.index.get(text)
        if same is not None:
            sims = sims.copy()
            sims[same] = -1.0
        top = np.argsort(-sims)[:k]
        return [(self.texts[i], self.labels[i]) for i in top if sims[i] > 0]


def get_vector_bank(fs: FewShotConfig, labels: tuple[str, ...]) -> VectorBank:
    return _cached_bank(
        fs.path, _stamp(fs.path), tuple(labels), fs.text_col, fs.label_col,
        fs.embed_model, fs.embed_dimensions, fs.embed_base_url, fs.embed_api_key_env, fs.embed_price,
    )


@lru_cache(maxsize=4)
def _cached_bank(path, stamp, labels, text_col, label_col, model, dim, base, key_env, price) -> VectorBank:
    fs = FewShotConfig(
        path=path, text_col=text_col, label_col=label_col, retriever="embedding",
        embed_model=model, embed_dimensions=dim, embed_base_url=base,
        embed_api_key_env=key_env, embed_price=price,
    )
    texts, ys = load_examples(path, list(labels), text_col, label_col)
    return VectorBank(texts, ys, EmbeddingCache(model, dim), fs)
