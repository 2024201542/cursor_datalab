"""把年报等长文档切成带章节、页码的段落，供现有分类流水线逐段处理，并汇总成文档级指标。"""
from __future__ import annotations

import io
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

DEFAULT_MIN_CHARS = 40

_HEADING = re.compile(
    r"^(第[一二三四五六七八九十百零0-9]{1,4}[章节部分篇]|"
    r"[一二三四五六七八九十]{1,3}、.{1,24}|"
    r"\d{1,2}[\.、]\s*\S{1,24})$"
)
_HEADING_WORD = re.compile(r"节|章|风险|讨论与分析|社会责任|公司治理|重要事项|财务报告")
_NOISE = (
    re.compile(r"\.{5,}|…{2,}|⋯{2,}"),
    re.compile(r"本公司董事会"),
    re.compile(r"^第?\d{1,4}页$"),
    re.compile(r"^目录$"),
)
_POS = ("积极", "正面", "机遇", "看多")
_NEG = ("消极", "负面", "风险", "看空")


def _clean(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return ("".join(lines) if cjk * 2 >= max(len(text), 1) else " ".join(lines)).strip()


def is_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 40 or re.search(r"[。！？；]$", t):
        return False
    return bool(_HEADING.match(t) or (len(t) <= 24 and _HEADING_WORD.search(t)))


def _is_noise(text: str, repeated: set[str]) -> bool:
    t = re.sub(r"\s+", "", text)
    if not t or t in repeated:
        return True
    return any(p.search(t) for p in _NOISE)


def _repeated_headers(pages: list[list[str]]) -> set[str]:
    """在 3 页以上都出现的短句，多半是页眉页脚。"""
    counts: Counter[str] = Counter()
    for blocks in pages:
        for t in {re.sub(r"\s+", "", b) for b in blocks if len(b) <= 30}:
            counts[t] += 1
    return {t for t, n in counts.items() if n >= 3}


class _HTMLBlocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[tuple[str, str]] = []
        self._buf: list[str] = []
        self._skip = 0
        self._tag = ""

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        self._tag = tag

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "br", "tr") and self._buf:
            text = "".join(self._buf).strip()
            if text:
                self.parts.append(("heading" if tag.startswith("h") else "text", text))
            self._buf = []

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def close(self):
        super().close()
        if self._buf and "".join(self._buf).strip():
            self.parts.append(("text", "".join(self._buf).strip()))


def _text_poor(text: str) -> bool:
    n = sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return n < 20


def _page_is_scan(page, text: str) -> bool:
    """几乎没有文字层，且有一张铺满版面的图，才当作扫描页。带小图标的正文页不识别。"""
    if not _text_poor(text):
        return False
    area = float(page.rect.width * page.rect.height) or 1.0
    try:
        infos = page.get_image_info()
    except Exception:
        return False
    for info in infos:
        box = info.get("bbox") or (0, 0, 0, 0)
        if (box[2] - box[0]) * (box[3] - box[1]) >= 0.4 * area:
            return True
    return False


_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ValueError("这份 PDF 是扫描件，需要先安装文字识别：pip install rapidocr-onnxruntime") from e
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_page(page) -> list[str]:
    """把扫描页渲染成图片再识别。同一段里竖向挨得很近的行拼在一起。"""
    import fitz

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    result, _elapsed = _get_ocr()(pix.tobytes("png"))
    rows = []
    for item in result or []:
        if len(item) < 2:
            continue
        box, text = item[0], str(item[1]).strip()
        score = float(item[2]) if len(item) > 2 else 1.0
        if not text or score < 0.5:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        rows.append((min(ys), max(ys), min(xs), text))
    rows.sort()
    paras: list[str] = []
    buf: list[str] = []
    prev_bottom = None
    prev_h = 16.0
    for top, bottom, _x, text in rows:
        gap = 0.0 if prev_bottom is None else top - prev_bottom
        if buf and gap > max(prev_h * 0.8, 10):
            paras.append(_clean("".join(buf)))
            buf = []
        buf.append(text)
        prev_bottom = bottom
        prev_h = max(bottom - top, 8)
    if buf:
        paras.append(_clean("".join(buf)))
    return [p for p in paras if p]


def _pdf_blocks(data: bytes) -> list[dict]:
    import fitz

    blocks: list[dict] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc, 1):
            page_blocks = []
            for b in page.get_text("blocks"):
                if len(b) >= 7 and b[6] != 0:
                    continue
                text = str(b[4]).strip()
                if text:
                    page_blocks.append({"page": i, "text": text, "heading": False})
            joined = "\n".join(b["text"] for b in page_blocks)
            if _page_is_scan(page, joined):
                lines = _ocr_page(page)
                if lines:
                    page_blocks = [{"page": i, "text": ln, "heading": False, "ocr": True} for ln in lines]
            blocks.extend(page_blocks)
    return blocks


def _blocks_of(name: str, data: bytes) -> list[dict]:
    """统一成 {page, text, heading}。page 从 1 计，没有分页的格式为 None。"""
    suffix = Path(name).suffix.lower()
    blocks: list[dict] = []
    if suffix == ".pdf":
        blocks.extend(_pdf_blocks(data))
    elif suffix == ".docx":
        import docx

        d = docx.Document(io.BytesIO(data))
        for p in d.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name or "") if p.style else ""
            heading = style.lower().startswith("heading") or style.startswith("标题")
            blocks.append({"page": None, "text": text, "heading": heading})
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append({"page": None, "text": " | ".join(cells), "heading": False})
    elif suffix in (".html", ".htm"):
        parser = _HTMLBlocks()
        for enc in ("utf-8-sig", "gbk"):
            try:
                html = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法识别网页编码，请另存为 UTF-8")
        parser.feed(html)
        parser.close()
        for kind, text in parser.parts:
            blocks.append({"page": None, "text": _clean(text), "heading": kind == "heading"})
    elif suffix in (".txt", ".md", ".markdown"):
        for enc in ("utf-8-sig", "gbk"):
            try:
                raw = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法识别文本编码，请另存为 UTF-8")
        for chunk in re.split(r"\n\s*\n", raw):
            text = _clean(chunk)
            if text:
                blocks.append({"page": None, "text": text, "heading": False})
    else:
        raise ValueError("长文档支持 .pdf / .docx / .html / .txt / .md")
    return blocks


def split_document(name: str, data: bytes, min_chars: int = DEFAULT_MIN_CHARS) -> tuple[list[dict], dict]:
    """切成段落。返回 (段落列表, 统计)。段落还没有按关键词或章节筛选。"""
    blocks = _blocks_of(name, data)
    if not blocks:
        raise ValueError(f"没有从 {name} 中读到文字。扫描件需要安装 rapidocr-onnxruntime 后再上传。")
    by_page: dict[int, list[str]] = {}
    for b in blocks:
        if b["page"]:
            by_page.setdefault(b["page"], []).append(b["text"])
    repeated = _repeated_headers(list(by_page.values())) if len(by_page) >= 3 else set()

    chapter = "（开篇）"
    paras: list[dict] = []
    dropped = 0
    doc = Path(name).name
    for b in blocks:
        text = b["text"]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()] or [text]
        first, rest = lines[0], lines[1:]
        if b["heading"] or is_heading(first):
            chapter = first[:40]
            body = _clean("\n".join(rest))
            if not body:
                continue
            text = body
        else:
            text = _clean(text)
        if _is_noise(text, repeated) or len(text) < min_chars:
            dropped += 1
            continue
        paras.append({
            "id": f"{doc}#{len(paras) + 1}",
            "doc": doc,
            "chapter": chapter,
            "page": b["page"],
            "text": text,
        })
    stat = {"blocks": len(blocks), "kept": len(paras), "dropped": dropped,
            "ocr_pages": len({b["page"] for b in blocks if b.get("ocr")}),
            "chapters": list(dict.fromkeys(p["chapter"] for p in paras))}
    return paras, stat


def filter_paragraphs(paras: list[dict], keywords: list[str] | None = None,
                      chapters: list[str] | None = None) -> list[dict]:
    """按关键词（任一命中）和章节筛选，并补上同一文档中相邻段落，供审核时对照。"""
    keys = [k.strip() for k in (keywords or []) if k.strip()]
    chosen = []
    for p in paras:
        if chapters and p["chapter"] not in chapters:
            continue
        if keys and not any(k.lower() in p["text"].lower() for k in keys):
            continue
        chosen.append(dict(p))
    by_doc: dict[str, list[dict]] = {}
    for p in chosen:
        by_doc.setdefault(p["doc"], []).append(p)
    for group in by_doc.values():
        for i, p in enumerate(group):
            p["prev"] = group[i - 1]["text"][:160] if i else ""
            p["next"] = group[i + 1]["text"][:160] if i + 1 < len(group) else ""
    return chosen


def _tone_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    pos = [l for l in labels if any(w in l for w in _POS)]
    neg = [l for l in labels if any(w in l for w in _NEG)]
    return pos, neg


def aggregate_documents(records: list, extra: dict) -> tuple[list[dict], str]:
    """按文档汇总各类段落数、占比。类别里同时有积极/机遇和消极/风险时给出净语调。"""
    items = (extra or {}).get("items") or {}

    def get(r, key):
        return r[key] if isinstance(r, dict) else getattr(r, key)

    def labs(r):
        # 多标签结果写成 "A|B"，每个标签分别计数
        return [x for x in (get(r, "label") or "").split("|") if x]

    labels = list(dict.fromkeys(x for r in records for x in labs(r)))
    pos, neg = _tone_labels(labels)
    by_doc: dict[str, list] = {}
    for r in records:
        info = items.get(get(r, "id")) or {}
        by_doc.setdefault(info.get("doc") or "（未分组）", []).append(r)
    rows = []
    for doc, recs in by_doc.items():
        counts = Counter(x for r in recs for x in (labs(r) or ["（无标签）"]))
        n = len(recs)
        row = {"文档": doc, "段落数": n}
        for lab in labels:
            row[lab] = counts.get(lab, 0)
            row[f"{lab}占比"] = counts.get(lab, 0) / n if n else 0
        if pos and neg and n:
            row["净语调"] = (sum(counts.get(l, 0) for l in pos) - sum(counts.get(l, 0) for l in neg)) / n
        rows.append(row)
    note = ""
    if pos and neg:
        note = f"净语调 =（{'+'.join(pos)} − {'+'.join(neg)}）/ 段落数，范围 −1 到 1。"
    elif rows:
        note = "当前类别名称里没有成对的“积极/机遇”和“消极/风险”，所以不算净语调，只给各类数量和占比。"
    return rows, note
