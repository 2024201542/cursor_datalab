"""用金标准做校准：按类别的混淆矩阵权重、置信度校准、聚合算法对比和阈值搜索。

输入都是结果文件（eval.jsonl）中的记录字典，只用已有的模型回答，不调用任何模型。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .aggregate import VoteResult, dawid_skene, glad, mace, wawa, weighted_vote
from .classifier import Prediction


def _certainty(p: dict | Prediction) -> float:
    d = p if isinstance(p, dict) else asdict(p)
    if d.get("samples"):
        return float(d.get("confidence") or 0.0)
    return float(d["prob"]) if d.get("prob") is not None else float(d.get("confidence") or 0.0)


def _ok(p: dict) -> bool:
    return bool(p.get("label")) and not p.get("error")


# ---------------------------------------------------------------- 置信度校准
def _pav(points: list[tuple[float, int]]) -> list[list[float]]:
    """保序回归（Pool Adjacent Violators）：返回 [[x_lo, x_hi, 正确数, 样本数], ...]，正确率随置信度单调不减。"""
    grouped: dict[float, list[float]] = {}
    for x, y in points:  # 相同置信度的样本先合并，否则会按到达顺序被错误地两两合并
        g = grouped.setdefault(x, [0.0, 0.0])
        g[0] += y
        g[1] += 1
    blocks: list[list[float]] = []
    for x in sorted(grouped):
        blocks.append([x, x, grouped[x][0], grouped[x][1]])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] >= blocks[-1][2] / blocks[-1][3]:
            last = blocks.pop()
            blocks[-1][1] = last[1]
            blocks[-1][2] += last[2]
            blocks[-1][3] += last[3]
    return blocks


def fit_isotonic(points: list[tuple[float, int]], prior: float = 2.0) -> list[list[float]]:
    """每段的正确率向整体正确率收缩（相当于加 prior 条伪样本），避免小样本时出现 0% / 100%。"""
    if not points:
        return []
    base = sum(y for _, y in points) / len(points)
    return [[lo, hi, (c + prior * base) / (n + prior), n] for lo, hi, c, n in _pav(points)]


def apply_isotonic(curve: list[list[float]], x: float) -> float | None:
    if not curve:
        return None
    for _, hi, y, _ in curve:
        if x <= hi:
            return y
    return curve[-1][2]


def fit_platt(points: list[tuple[float, int]], n_iter: int = 200) -> tuple[float, float]:
    """Platt / 温度缩放：p = sigmoid(a · logit(置信度) + b)。a < 1 相当于把过度自信的分数“降温”。"""
    a, b = 1.0, 0.0
    if not points:
        return a, b
    xs = [math.log(min(max(x, 1e-4), 1 - 1e-4) / (1 - min(max(x, 1e-4), 1 - 1e-4))) for x, _ in points]
    ys = [y for _, y in points]
    lam = 1.0  # 向 a=1、b=0（不校准）收缩的 L2 正则，防止样本可分时发散
    for _ in range(n_iter):  # 牛顿法
        ga, gb = lam * (a - 1), lam * b
        haa = hbb = lam
        hab = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-max(min(a * x + b, 30), -30)))
            ga += (p - y) * x
            gb += p - y
            w = p * (1 - p)
            haa += w * x * x
            hab += w * x
            hbb += w
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        step = max(abs(da), abs(db))
        if step > 1:
            da, db = da / step, db / step
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-8:
            break
    return a, b


def apply_platt(ab: tuple[float, float], x: float) -> float:
    x = min(max(x, 1e-4), 1 - 1e-4)
    z = ab[0] * math.log(x / (1 - x)) + ab[1]
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


def ece(points: list[tuple[float, int]], bins: int = 10) -> float | None:
    """期望校准误差：按置信度分 10 档，每档“平均置信度”与“实际正确率”之差的加权平均。"""
    if not points:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sub = [(x, y) for x, y in points if lo <= x < hi or (b == bins - 1 and x == 1.0)]
        if sub:
            total += len(sub) * abs(sum(x for x, _ in sub) / len(sub) - sum(y for _, y in sub) / len(sub))
    return total / len(points)


def reliability(points: list[tuple[float, int]], bins: int = 5) -> list[dict]:
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sub = [(x, y) for x, y in points if lo <= x < hi or (b == bins - 1 and x == 1.0)]
        if sub:
            rows.append({"区间": f"{lo:.1f}–{hi:.1f}", "样本数": len(sub),
                         "平均置信度": sum(x for x, _ in sub) / len(sub), "实际正确率": sum(y for _, y in sub) / len(sub)})
    return rows


# ---------------------------------------------------------------- 校准文件
@dataclass
class Calibration:
    labels: list[str]
    prior: list[float]
    confusion: dict[str, list[list[float]]]  # 模型 -> [真实类][预测类] 的条件概率
    curves: dict[str, dict] = field(default_factory=dict)  # 模型 -> {"method": "isotonic"|"platt", ...}
    acc: dict[str, float] = field(default_factory=dict)  # 模型 -> 金标准上的准确率
    n: int = 0
    source: str = ""
    use_confidence: bool = True

    def calibrate(self, model: str, certainty: float) -> float | None:
        c = self.curves.get(model)
        if not c:
            return None
        return round(_apply_curve(c, certainty), 4)

    def posterior(self, preds: list[Prediction], evidence_penalty: float = 1.0,
                  use_confidence: bool | None = None) -> dict[str, float]:
        """朴素贝叶斯：P(真实=k | 各模型的回答) ∝ 先验(k) × Π 模型 m 在真实为 k 时给出该回答的概率。

        模型 A 判“投诉”很准、判“建议”常错，则它说“投诉”时证据强、说“建议”时证据弱——这就是按类别的权重。
        use_confidence（默认跟随 self.use_confidence）：再按校准后的置信度调整——同一个模型置信度低的回答证据更弱。
        """
        idx = {lab: k for k, lab in enumerate(self.labels)}
        K = len(self.labels)
        use_confidence = self.use_confidence if use_confidence is None else use_confidence
        logp = [math.log(max(p, 1e-9)) for p in self.prior]
        for p in preds:
            conf = self.confusion.get(p.model)
            if not p.ok or conf is None or p.label not in idx:
                continue
            j = idx[p.label]
            power = evidence_penalty if p.evidence_ok is False else 1.0
            scale = None
            if use_confidence and self.acc.get(p.model):
                c = self.calibrate(p.model, _certainty(p))
                if c is not None:
                    scale = c / self.acc[p.model]
            for k in range(K):
                like = conf[k][j]
                if scale is not None:
                    diag = min(max(conf[k][k] * scale, 0.01), 0.99)
                    like = diag if k == j else (1 - diag) * conf[k][j] / max(1 - conf[k][k], 1e-9)
                logp[k] += power * math.log(max(like, 1e-9))
        m = max(logp)
        e = [math.exp(x - m) for x in logp]
        s = sum(e)
        return {lab: v / s for lab, v in zip(self.labels, e)}

    def vote(self, preds: list[Prediction], evidence_penalty: float = 1.0, use_confidence: bool | None = None) -> VoteResult:
        valid = [p for p in preds if p.ok]
        if not valid:
            return VoteResult(None, 0.0, 0, 0)
        post = self.posterior(valid, evidence_penalty, use_confidence)
        label = max(post, key=post.get)
        return VoteResult(label, post[label], sum(p.label == label for p in valid), len(valid), post)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _gold_records(records: list[dict], labels: list[str]) -> list[dict]:
    return [r for r in records if r.get("gold") in labels]


def _points(records: list[dict], model: str, use: str = "round1") -> list[tuple[float, int]]:
    out = []
    for r in records:
        for p in r.get(use) or []:
            if p["model"] == model and _ok(p):
                out.append((_certainty(p), int(p["label"] == r["gold"])))
    return out


def model_names_of(records: list[dict]) -> list[str]:
    names: dict[str, None] = {}
    for r in records:
        for p in r["round1"]:
            names.setdefault(p["model"])
    return list(names)


def fit_calibration(records: list[dict], labels: list[str], smoothing: float = 1.0, method: str = "auto",
                    source: str = "") -> Calibration:
    """records 为带 gold 字段的结果记录。smoothing 是混淆矩阵每格的伪计数（样本少时防止出现 0 概率）。"""
    recs = _gold_records(records, labels)
    K = len(labels)
    idx = {lab: k for k, lab in enumerate(labels)}
    counts = [0.0] * K
    for r in recs:
        counts[idx[r["gold"]]] += 1
    prior = [(c + smoothing) / (len(recs) + K * smoothing) for c in counts]
    confusion, curves, acc = {}, {}, {}
    for m in model_names_of(recs):
        mat = [[smoothing] * K for _ in range(K)]
        for r in recs:
            p = next((x for x in r["round1"] if x["model"] == m), None)
            if p is not None and _ok(p) and p["label"] in idx:
                mat[idx[r["gold"]]][idx[p["label"]]] += 1
        confusion[m] = [[v / sum(row) for v in row] for row in mat]
        pts = _points(recs, m)
        if pts:
            acc[m] = sum(y for _, y in pts) / len(pts)
        if len(pts) >= 10:
            curves[m] = _fit_curve(pts, method)
    return Calibration(labels, prior, confusion, curves, acc, len(recs), source)


def _fit_curve(points: list[tuple[float, int]], method: str) -> dict:
    if method == "auto":
        # 原本就校准得好的（ECE < 0.05，如本地小模型的预测概率）保持不变；否则两折交叉验证比较 Brier 分数选一种
        if (ece(points) or 0) < 0.05:
            return {"method": "none"}
        a, b = points[::2], points[1::2]
        brier = {}
        for m in ("isotonic", "platt"):
            err = 0.0
            for train, test in ((a, b), (b, a)):
                c = _fit_curve(train, m)
                err += sum((_apply_curve(c, x) - y) ** 2 for x, y in test)
            brier[m] = err
        method = min(brier, key=brier.get)
    if method == "none":
        return {"method": "none"}
    if method == "platt":
        return {"method": "platt", "ab": list(fit_platt(points))}
    return {"method": "isotonic", "curve": fit_isotonic(points)}


def _apply_curve(curve: dict, x: float) -> float:
    if curve["method"] == "none":
        return x
    if curve["method"] == "platt":
        return apply_platt(tuple(curve["ab"]), x)
    return apply_isotonic(curve["curve"], x)


def _fold_of(item_id: str, folds: int) -> int:
    return int(hashlib.md5(str(item_id).encode("utf-8")).hexdigest(), 16) % folds


def _pred(d: dict) -> Prediction:
    return Prediction(**{k: v for k, v in d.items() if k in Prediction.__dataclass_fields__})


def _preds(r: dict) -> list[Prediction]:
    return [_pred(p) for p in r["round1"]]


def calibration_report(records: list[dict], labels: list[str], folds: int = 5) -> list[dict]:
    """每个模型校准前后的 ECE（校准后的值用交叉拟合计算：每条样本的校准曲线都不是用它自己拟合的）。"""
    recs = _gold_records(records, labels)
    rows = []
    for m in model_names_of(recs):
        before = _points(recs, m)
        after = []
        for f in range(folds):
            train = [r for r in recs if _fold_of(r["id"], folds) != f]
            test = [r for r in recs if _fold_of(r["id"], folds) == f]
            pts = _points(train, m)
            if len(pts) < 10:
                continue
            curve = _fit_curve(pts, "auto")
            after += [(_apply_curve(curve, x), y) for x, y in _points(test, m)]
        if not before:
            continue
        rows.append({
            "模型": m, "样本数": len(before), "准确率": sum(y for _, y in before) / len(before),
            "平均置信度": sum(x for x, _ in before) / len(before),
            "校准前 ECE": ece(before), "校准后 ECE": ece(after) if after else None,
            "方法": _fit_curve(before, "auto")["method"] if len(before) >= 10 else "样本太少",
        })
    return rows


# ---------------------------------------------------------------- 聚合算法对比
def compare_aggregators(records: list[dict], labels: list[str], folds: int = 5) -> dict:
    """在金标准上比较各聚合方式的准确率。

    需要标准答案来拟合的方法（加权投票、按类别加权）用 K 折交叉验证：每一折用其余样本拟合、在本折上评估，
    所以数字不会因为“用答案拟合再用答案评估”而偏高。Dawid-Skene / Wawa / MACE / GLAD 不用标准答案，直接在全部样本上运行。
    """
    from .evaluate import log_odds_weight, majority_vote

    recs = _gold_records(records, labels)
    if not recs:
        return {"rows": [], "best": None, "n": 0}
    gold = {r["id"]: r["gold"] for r in recs}
    models = model_names_of(recs)
    preds = {r["id"]: _preds(r) for r in recs}
    out: dict[str, dict[str, str | None]] = {}

    for m in models:
        out[m] = {i: next((p.label for p in ps if p.model == m and p.ok), None) for i, ps in preds.items()}
    out["多数投票"] = {i: majority_vote(ps) for i, ps in preds.items()}
    out["加权投票（按模型）"], out["按类别加权（混淆矩阵）"], out["按类别加权 + 校准置信度"] = {}, {}, {}
    for f in range(folds):
        train = [r for r in recs if _fold_of(r["id"], folds) != f]
        test = [r for r in recs if _fold_of(r["id"], folds) == f]
        if not train or not test:
            continue
        acc = {m: sum(1 for r in train for p in r["round1"] if p["model"] == m and _ok(p) and p["label"] == r["gold"])
               / max(1, sum(1 for r in train if any(p["model"] == m for p in r["round1"]))) for m in models}
        w = {m: log_odds_weight(a, len(labels)) for m, a in acc.items()}
        cal = fit_calibration(train, labels)
        for r in test:
            out["加权投票（按模型）"][r["id"]] = weighted_vote(preds[r["id"]], w).label
            out["按类别加权（混淆矩阵）"][r["id"]] = cal.vote(preds[r["id"]], use_confidence=False).label
            out["按类别加权 + 校准置信度"][r["id"]] = cal.vote(preds[r["id"]], use_confidence=True).label

    ann = [(i, p.model, p.label) for i, ps in preds.items() for p in ps if p.ok]
    out["Dawid-Skene"] = dawid_skene(ann, labels).labels
    out["Wawa"] = wawa(ann, labels).labels
    out["MACE"] = mace(ann, labels).labels
    out["GLAD"] = glad(ann, labels).labels

    rows = []
    for name, lab in out.items():
        rows.append({"方法": name, "类型": "单模型" if name in models else "聚合",
                     "需要标准答案": name in ("加权投票（按模型）", "按类别加权（混淆矩阵）", "按类别加权 + 校准置信度"),
                     "准确率": sum(lab.get(i) == g for i, g in gold.items()) / len(gold)})
    rows.sort(key=lambda x: -x["准确率"])
    best = next((r["方法"] for r in rows if r["类型"] == "聚合"), None)
    return {"rows": rows, "best": best, "n": len(recs), "folds": folds}


# ---------------------------------------------------------------- 阈值搜索
def crossfit_posteriors(records: list[dict], labels: list[str], folds: int = 5,
                        use_confidence: bool = True) -> list[tuple[str, float, bool]]:
    """每条样本的 (id, 后验概率, 是否正确)，后验由不含该样本的其余折拟合。"""
    recs = _gold_records(records, labels)
    out = []
    for f in range(folds):
        train = [r for r in recs if _fold_of(r["id"], folds) != f]
        test = [r for r in recs if _fold_of(r["id"], folds) == f]
        if not train or not test:
            continue
        cal = fit_calibration(train, labels)
        for r in test:
            v = cal.vote(_preds(r), use_confidence=use_confidence)
            out.append((r["id"], v.share, v.label == r["gold"]))
    return out


def posterior_curve(points: list[tuple[str, float, bool]], grid: list[float] | None = None) -> list[dict]:
    grid = grid or [round(0.5 + 0.01 * i, 2) for i in range(50)]
    n = len(points)
    rows = []
    for t in grid:
        acc = [c for _, p, c in points if p >= t]
        rows.append({"门槛": t, "自动采纳比例": len(acc) / n if n else 0.0,
                     "自动采纳准确率": sum(acc) / len(acc) if acc else None, "自动采纳条数": len(acc)})
    return rows


def pick_threshold(rows: list[dict], target: float, cov_key: str = "自动采纳比例", acc_key: str = "自动采纳准确率") -> dict | None:
    """满足“自动采纳准确率 ≥ 目标”的设置中，自动采纳比例最高的一个（同比例取准确率更高的）。"""
    ok = [r for r in rows if r[acc_key] is not None and r[acc_key] >= target]
    return max(ok, key=lambda r: (r[cov_key], r[acc_key])) if ok else None


def _find(preds: list[dict], name: str) -> dict | None:
    return next((p for p in preds if p["model"] == name), None)


def simulate_thresholds(records: list[dict], labels: list[str]) -> list[dict]:
    """用已有记录模拟不同 accept_threshold / arbiter_threshold 下的结果（不重新调用模型）。

    首轮一致的样本保持自动采纳；有复核记录的按复核后加权投票判断；没有仲裁记录却需要仲裁的样本按转人工计算（偏保守）。
    模型权重按 1 计算。
    """
    recs = _gold_records(records, labels)
    n = len(recs)
    base = []
    for r in recs:
        r1 = _preds(r)
        valid = [p for p in r1 if p.ok]
        if r["status"] == "consensus":
            base.append(("auto", r["label"] == r["gold"], None, None))
            continue
        r2 = {p["model"]: p for p in r.get("round2") or []}
        final = [_pred(r2[p.model]) if p.model in r2 and _ok(r2[p.model]) else p for p in r1]
        vote = weighted_vote(final, {}) if r.get("round2") else None
        arb = r.get("arbiter")
        arb_t = (arb["label"] == r["gold"], float(arb.get("confidence") or 0)) if arb and _ok(arb) else None
        rev_t = ((vote.label == r["gold"]), vote.share, vote.agree * 2 > vote.total) if vote and valid else None
        base.append(("flow", None, rev_t, arb_t))
    rows = []
    has_review = any(b[2] for b in base)
    has_arb = any(b[3] for b in base)
    a_grid = [round(0.5 + 0.05 * i, 2) for i in range(11)] if has_review else [None]
    b_grid = [round(0.05 * i, 2) for i in range(21)] if has_arb else [None]
    for a in a_grid:
        for b in b_grid:
            auto, correct = 0, 0
            for kind, ok, rev, arb in base:
                if kind == "auto":
                    auto += 1
                    correct += ok
                    continue
                if rev and a is not None and rev[2] and rev[1] >= a:
                    auto += 1
                    correct += rev[0]
                elif arb and b is not None and arb[1] >= b:
                    auto += 1
                    correct += arb[0]
            rows.append({"accept_threshold": a, "arbiter_threshold": b, "自动采纳比例": auto / n if n else 0.0,
                         "自动采纳准确率": correct / auto if auto else None, "自动采纳条数": auto})
    return rows
