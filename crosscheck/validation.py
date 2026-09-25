"""新任务验证用的小工具：抽样、分层划分调参集 / 验证集、准确率置信区间。"""
from __future__ import annotations

import math
import random


def sample_rows(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """随机抽 n 条（保持原顺序）。"""
    if n >= len(rows):
        return list(rows)
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def split_dev_test(rows: list[dict], label_key: str = "label", test_ratio: float = 0.5,
                   seed: int = 42) -> tuple[list[dict], list[dict]]:
    """按类别分层划分：调参集用来反复修改标准，验证集只在最后检查一次，避免把标准“调”得只适合这批样本。"""
    by: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(str(r.get(label_key, "")), []).append(i)
    rng = random.Random(seed)
    test = set()
    for idx in by.values():
        idx = idx[:]
        rng.shuffle(idx)
        k = round(len(idx) * test_ratio)
        if len(idx) >= 2:
            k = min(max(k, 1), len(idx) - 1)
        test.update(idx[:k])
    return [r for i, r in enumerate(rows) if i not in test], [r for i, r in enumerate(rows) if i in test]


def ci_halfwidth(p: float, n: int) -> float:
    """准确率 p、样本 n 时 95% 置信区间的半宽（正态近似）。"""
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n) if n else 1.0
