"""把 evaluate 的结果画成图，并生成一份可直接用浏览器打开的 HTML 报告。"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .pipeline import STATUS_TEXT  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_COLOR = "#8fb3d9"
ENSEMBLE_COLORS = {"多数投票": "#f2b36f", "加权投票": "#e8894a", "互检系统": "#2e9e6b"}
PALETTE = ["#5b8fd6", "#e07b54", "#8c6bb1", "#d4a72c", "#4aa3a2", "#c85d8a"]


def _bar_labels(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10)


def chart_accuracy(rep: dict, path: Path) -> None:
    names = list(rep["strategies"])
    accs = [rep["strategies"][n]["acc"] * 100 for n in names]
    colors = [MODEL_COLOR if rep["strategies"][n]["kind"] == "model" else ENSEMBLE_COLORS.get(n, "#e8894a") for n in names]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.3), 4.8))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", width=0.62)
    _bar_labels(ax, bars)
    best_single = max(rep["per_model"][m]["round1_acc"] for m in rep["model_names"]) * 100
    upper = rep["any_correct"] * 100
    ax.axhline(best_single, color="#5b7fa8", ls="--", lw=1, label=f"最佳单模型 {best_single:.1f}%")
    ax.axhline(upper, color="#999", ls=":", lw=1.2, label=f"理论上限（任一模型答对）{upper:.1f}%")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax.set_ylim(max(0, min(accs) - 20), 108)
    ax.set_ylabel("准确率 (%)")
    ax.set_title(f"单模型 vs 投票 vs 互检系统（金标准 {rep['n']} 条）", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_per_class(rep: dict, path: Path) -> None:
    labels = rep["labels"]
    series = rep["model_names"] + ["互检系统"]
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(min(15, max(8, len(labels) * 1.6)), 5))
    for i, name in enumerate(series):
        vals = [(rep["strategies"][name]["per_class"].get(lab) or 0) * 100 for lab in labels]
        xs = [j + (i - (len(series) - 1) / 2) * width for j in range(len(labels))]
        color = ENSEMBLE_COLORS["互检系统"] if name == "互检系统" else PALETTE[i % len(PALETTE)]
        ax.bar(xs, vals, width=width * 0.95, label=name, color=color)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("该类别的准确率（召回率 %）")
    ax.set_title("各类别准确率：哪个模型在哪类上更强", fontsize=13)
    ax.legend(ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_review(rep: dict, path: Path) -> None:
    models = rep["model_names"]
    r1 = [rep["per_model"][m]["round1_acc"] * 100 for m in models]
    r2 = [rep["per_model"][m]["round2_acc"] * 100 for m in models]
    xs = range(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.6), 4.5))
    b1 = ax.bar([x - 0.18 for x in xs], r1, width=0.36, label="独立判断（首轮）", color=MODEL_COLOR)
    b2 = ax.bar([x + 0.18 for x in xs], r2, width=0.36, label="交叉复核后", color="#2e9e6b")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.set_xticks(list(xs), models)
    ax.set_ylim(max(0, min(r1 + r2) - 20), 105)
    ax.set_ylabel("准确率 (%)")
    ax.set_title("交叉复核的效果：看到其他模型的理由后是否改对", fontsize=13)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_confusion(rep: dict, path: Path) -> None:
    labels = rep["labels"]
    matrix = [[rep["confusion"].get(g, {}).get(p, 0) for p in labels] for g in labels]
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    im = ax.imshow(matrix, cmap="Greens")
    vmax = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > vmax * 0.6 else "#333", fontsize=11)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("互检系统判断")
    ax.set_ylabel("真实标签")
    ax.set_title("互检系统混淆矩阵", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_status(rep: dict, path: Path) -> None:
    statuses = [s for s in STATUS_TEXT if s in rep["by_status"]]
    counts = [rep["by_status"][s]["count"] for s in statuses]
    accs = [rep["by_status"][s]["acc"] * 100 for s in statuses]
    names = [STATUS_TEXT[s] for s in statuses]
    colors = ["#2e9e6b", "#7cc49a", "#e8894a", "#d9534f"][: len(statuses)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.pie(counts, labels=[f"{n}\n{c} 条" for n, c in zip(names, counts)], colors=colors,
           autopct="%1.0f%%", startangle=90, wedgeprops={"edgecolor": "white"})
    a1.set_title("样本在各环节的去向", fontsize=13)
    bars = a2.bar(names, accs, color=colors)
    _bar_labels(a2, bars)
    a2.set_ylim(0, 110)
    a2.set_title("各环节的准确率", fontsize=13)
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


CHARTS = [
    ("accuracy.png", "准确率总览", chart_accuracy),
    ("per_class.png", "各类别准确率", chart_per_class),
    ("review.png", "交叉复核效果", chart_review),
    ("status.png", "处理环节分布", chart_status),
    ("confusion.png", "混淆矩阵", chart_confusion),
]


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_report(rep: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    chart_dir = out / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    for fname, _, fn in CHARTS:
        fn(rep, chart_dir / fname)

    esc = html.escape
    pct = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    names = list(rep["strategies"])

    acc_rows = "".join(
        f"<tr class='{v['kind']}'><td>{'单模型' if v['kind'] == 'model' else '组合'}</td><td>{esc(n)}</td><td>{pct(v['acc'])}</td>"
        + "".join(f"<td>{pct(v['per_class'].get(lab))}</td>" for lab in rep["labels"]) + "</tr>"
        for n, v in rep["strategies"].items()
    )
    item_rows = []
    for it in rep["items"]:
        cells = "".join(
            f"<td class='{'ok' if it[n] == it['gold'] else 'bad'}'>{esc(it[n] or '-')}</td>" for n in names
        )
        item_rows.append(
            f"<tr><td>{esc(it['id'])}</td><td class='text'>{esc(it['text'])}</td><td><b>{esc(it['gold'])}</b></td>"
            f"{cells}<td>{esc(STATUS_TEXT[it['status']])}</td></tr>"
        )
    k = rep["fleiss_kappa_round1"]
    best = max(rep["model_names"], key=lambda m: rep["strategies"][m]["acc"])
    sys_acc = rep["strategies"]["互检系统"]["acc"]
    best_acc = rep["strategies"][best]["acc"]

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>多模型互检评估报告</title>
<style>
body{{font-family:"Microsoft YaHei",sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#222}}
h1{{margin-bottom:4px}} .sub{{color:#777;margin-top:0}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
.card{{flex:1;min-width:170px;background:#f5f8f6;border-radius:10px;padding:14px 18px}}
.card .v{{font-size:28px;font-weight:bold;color:#2e9e6b}} .card .k{{color:#666;font-size:13px}}
table{{border-collapse:collapse;width:100%;margin:10px 0 24px;font-size:13px}}
th,td{{border:1px solid #e3e3e3;padding:6px 8px;text-align:center}} th{{background:#fafafa}}
tr.ensemble td{{background:#fff7ef}} td.text{{text-align:left;max-width:320px}}
td.ok{{color:#2e7d4f}} td.bad{{background:#fdecea;color:#c0392b;font-weight:bold}}
img{{max-width:100%;border:1px solid #eee;border-radius:8px;margin:8px 0 20px}}
</style></head><body>
<h1>多模型互检评估报告</h1>
<p class="sub">金标准 {rep['n']} 条 · 参与模型：{esc('、'.join(rep['model_names']))}</p>
<div class="cards">
<div class="card"><div class="k">最佳单模型（{esc(best)}）</div><div class="v">{pct(best_acc)}</div></div>
<div class="card"><div class="k">多数投票</div><div class="v">{pct(rep['strategies']['多数投票']['acc'])}</div></div>
<div class="card"><div class="k">互检系统（复核 + 仲裁）</div><div class="v">{pct(sys_acc)}</div></div>
<div class="card"><div class="k">相对最佳单模型</div><div class="v">{(sys_acc - best_acc) * 100:+.1f} 个百分点</div></div>
<div class="card"><div class="k">模型间一致性 Kappa</div><div class="v">{'-' if k is None else f'{k:.2f}'}</div></div>
</div>
<h2>准确率对比</h2>
<table><tr><th>类型</th><th>方案</th><th>整体准确率</th>{''.join(f'<th>{esc(l)}</th>' for l in rep['labels'])}</tr>{acc_rows}
<tr><td>参考</td><td>任一模型答对（理论上限）</td><td>{pct(rep['any_correct'])}</td>{'<td></td>' * len(rep['labels'])}</tr></table>
{''.join(f'<h2>{title}</h2><img src="data:image/png;base64,{_b64(chart_dir / fname)}">' for fname, title, _ in CHARTS)}
<h2>逐条明细（红色为判错）</h2>
<table><tr><th>ID</th><th>文本</th><th>真实</th>{''.join(f'<th>{esc(n)}</th>' for n in names)}<th>处理环节</th></tr>
{''.join(item_rows)}</table>
</body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
