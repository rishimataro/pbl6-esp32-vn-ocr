"""Nhận diện bố cục trang từ các dòng chữ đã đọc, rồi dựng tài liệu Word (.docx) có định dạng gần đúng bản gốc.

Không dùng model bố cục tải thêm; mọi thứ suy ra từ hình học của các khung dòng và từ ảnh trang:
  - bảng: tìm các đường kẻ ngang và dọc bằng phép mở hình thái, dựng lưới ô, gán dòng chữ vào ô;
  - hình/dấu triện: mảng đỏ lớn (dấu triện) và thành phần liên thông lớn được cắt thành ảnh chèn vào tài liệu;
  - cột: khe trống dọc chạy suốt phần thân trang;
  - đoạn, tiêu đề, danh sách: gộp dòng theo khoảng cách dòng, lề trái, độ dài dòng, cỡ chữ;
  - căn lề (trái, giữa, phải, đều hai bên), thụt đầu dòng, thụt treo của mục đánh số, tab cho dòng "nhãn : giá trị";
  - cỡ chữ (từ chu kỳ dòng và tỉ lệ chiều cao), chữ đậm (độ dày nét), giãn dòng, khoảng cách giữa các đoạn, lề trang.
Hạn chế: không nhận ra chữ nghiêng/gạch chân, chữ đậm chỉ xét theo cả dòng, bảng không có đường kẻ không được nhận ra
(dòng "nhãn : giá trị" được giữ bằng dấu tab), phông chữ do người dùng chọn.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

A4 = (595.3, 841.9)
LIST_RE = re.compile(r"^\s*((\d+(\.\d+)*[.)]?)|([a-zA-Zđ][.)])|([-–•*·+]))\s*$")        # đoạn chỉ gồm số mục / gạch đầu dòng
LIST_START = re.compile(r"^\s*((\d+(\.\d+)*[.)]?)|([a-zA-Zđ][.)])|([-–•*·+]))\s+\S")     # dòng bắt đầu bằng số mục
HEAD_WORD = re.compile(r"^\s*(điều|chương|mục|phần|phụ lục|article|chapter|section)\b", re.I)
SIZES = [7, 8, 9, 10, 10.5, 11, 12, 13, 14, 16, 18, 20, 22, 24, 28, 32, 36]


@dataclass
class Row:
    segs: list                 # [(x0, x1, text)] các đoạn trên cùng một hàng, theo x
    y0: int
    y1: int
    h: float = 0.0             # chiều cao mực ước lượng
    bold: float = 0.0          # độ dày nét tương đối
    is_bold: bool = False
    size: float = 12.0
    align: str = "left"
    kind: str = "text"         # text | heading
    level: int = 0
    full: bool = False         # chạy hết bề rộng thân trang
    region: int | None = None
    rlabel: str | None = None

    @property
    def x0(self): return self.segs[0][0]
    @property
    def x1(self): return self.segs[-1][1]
    @property
    def text(self): return " ".join(s[2] for s in self.segs).strip()
    @property
    def cy(self): return (self.y0 + self.y1) / 2


@dataclass
class Block:
    type: str                  # paragraph | heading | list | table | figure
    rows: list = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    table: list | None = None  # [[text ô]] với type == table
    cells: list | None = None  # [(x0,y0,x1,y1)] theo hàng
    image: np.ndarray | None = None
    column: int = 0


def _snap(pt: float) -> float:
    return float(min(SIZES, key=lambda s: abs(s - pt)))


def _ink(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    k = max(15, (min(h, w) // 40) | 1)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, k, 12)


def _stroke(ink: np.ndarray, box) -> float:
    x0, y0, x1, y1 = box
    sub = ink[y0:y1, x0:x1]
    area = int(sub.sum())
    if area < 20:
        return 0.0
    er = cv2.erode(sub, np.ones((3, 3), np.uint8))
    boundary = int(sub.sum() - er.sum())
    return 2.0 * area / max(boundary, 1)


# ------------------------------------------------------------------ bảng và hình
def _cluster(vals, tol):
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(c)) for c in out]


def find_tables(ink: np.ndarray, pitch: float):
    """Trả về danh sách bảng: (bbox, xs, ys) với xs, ys là toạ độ các đường kẻ dọc / ngang đã gộp."""
    H, W = ink.shape
    lh = max(30, int(max(3 * pitch, 0.06 * W)))
    lv = max(30, int(max(3 * pitch, 0.04 * H)))
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (lh, 1)))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, lv)))
    grid = cv2.dilate(cv2.bitwise_or(hor, ver), np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    tables = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if w < 4 * pitch or h < 2 * pitch:
            continue
        comp_h = hor[y:y + h, x:x + w]
        comp_v = ver[y:y + h, x:x + w]
        ys = _cluster(np.where(comp_h.sum(1) > 0.5 * w * 0.2)[0] + y, max(4, pitch * 0.25))
        xs = _cluster(np.where(comp_v.sum(0) > 0.5 * h * 0.2)[0] + x, max(4, pitch * 0.25))
        if len(ys) >= 2 and len(xs) >= 2 and (len(ys) - 1) * (len(xs) - 1) >= 2:
            tables.append(((int(x), int(y), int(x + w), int(y + h)), xs, ys))
    return tables


def find_figures(page_bgr: np.ndarray, ink: np.ndarray, pitch: float, exclude: list):
    """Hình cần chèn nguyên: mảng đỏ lớn (dấu triện) và thành phần liên thông rất lớn không phải bảng."""
    H, W = ink.shape
    figs = []
    b, g, r = [page_bgr[..., i].astype(np.int16) for i in range(3)]
    red = ((r - np.maximum(g, b)) > 14).astype(np.uint8)
    if red.mean() < 0.03:
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((int(pitch), int(pitch)), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if w > 2.5 * pitch and h > 2.5 * pitch and a > 0.08 * w * h:
                figs.append((int(x), int(y), int(x + w), int(y + h)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w > 4 * pitch and h > 4 * pitch and a > 0.25 * w * h:
            figs.append((int(x), int(y), int(x + w), int(y + h)))
    def inter(a, b_):
        ix = min(a[2], b_[2]) - max(a[0], b_[0]); iy = min(a[3], b_[3]) - max(a[1], b_[1])
        return max(ix, 0) * max(iy, 0)
    keep = []
    for f in figs:
        if any(inter(f, e) > 0.5 * (f[2] - f[0]) * (f[3] - f[1]) for e in exclude):
            continue
        if not any(inter(f, k) > 0.5 * (f[2] - f[0]) * (f[3] - f[1]) for k in keep):
            keep.append(f)
    return keep


# ------------------------------------------------------------------ dựng hàng và khối
def build_rows(boxes_text, ink):
    items = [dict(b=b, t=t.strip()) for b, t in boxes_text if t.strip()]
    for it in items:
        x0, y0, x1, y1 = it["b"]
        it["h"] = max(4.0, (y1 - y0) / 1.24)
        it["rel"] = _stroke(ink, it["b"]) / it["h"]
    items.sort(key=lambda it: (it["b"][1] + it["b"][3]))
    rows = []
    for it in items:
        x0, y0, x1, y1 = it["b"]
        placed = False
        for r in reversed(rows[-4:]):
            ov = min(y1, r.y1) - max(y0, r.y0)
            if ov > 0.55 * min(y1 - y0, r.y1 - r.y0) and all(x0 >= s[1] - 2 or x1 <= s[0] + 2 for s in r.segs):
                r.segs.append((x0, x1, it["t"])); r.y0 = min(r.y0, y0); r.y1 = max(r.y1, y1)
                r._items.append(it); placed = True
                break
        if not placed:
            r = Row(segs=[(x0, x1, it["t"])], y0=y0, y1=y1)
            r._items = [it]
            rows.append(r)
    for r in rows:
        r.segs.sort(key=lambda s: s[0])
        main = max(r._items, key=lambda it: (it["b"][2] - it["b"][0]))
        r.h = float(main["h"]); r.bold = float(main["rel"])
    rows.sort(key=lambda r: r.cy)
    return rows


def _mode_x(xs, W, tol=0.012):
    if not xs:
        return 0.0
    xs = np.array(xs, float)
    best = max(xs, key=lambda v: np.sum(np.abs(xs - v) < tol * W))
    return float(np.median(xs[np.abs(xs - best) < tol * W]))


def analyse(rows, W, H, ptpx):
    """Điền cỡ chữ, đậm, căn lề, loại (thân/tiêu đề) cho từng hàng. Trả về số đo toàn trang."""
    longrows = [r for r in rows if (r.x1 - r.x0) > 0.35 * W and len(r.text) > 25] or rows
    body_h = float(np.median([r.h for r in longrows]))
    body_rel = float(np.median([r.bold for r in longrows if r.bold > 0] or [1.0]))
    cys = [r.cy for r in rows]
    diffs = [b - a for a, b in zip(cys[:-1], cys[1:]) if 0.8 * body_h * 1.24 < b - a < 2.2 * body_h * 1.24]
    pitch = float(np.median(diffs)) if diffs else body_h * 1.3
    left = _mode_x([r.x0 for r in longrows], W)
    right_vals = sorted(r.x1 for r in longrows)
    right = float(np.percentile(right_vals, 80)) if right_vals else W
    body_pt = _snap(pitch * ptpx / 1.15)
    mid = (left + right) / 2
    for r in rows:
        ratio = r.h / body_h
        r.size = _snap(body_pt * ratio) if (ratio >= 1.25 or ratio <= 0.72) else _snap(body_pt)
        r.is_bold = len(r.text) >= 8 and r.bold > 1.22 * body_rel
        width = r.x1 - r.x0
        r.full = r.x1 >= right - 0.035 * W and r.x0 <= left + 0.06 * W
        centered = abs((r.x0 + r.x1) / 2 - mid) < 0.035 * W and r.x0 - left > 0.06 * W and right - r.x1 > 0.06 * W
        if centered and len(r.segs) == 1:
            r.align = "center"
        elif len(r.segs) == 1 and r.x1 >= right - 0.02 * W and r.x0 - left > 0.3 * W:
            r.align = "right"
        t = r.text
        short = width < 0.8 * (right - left)
        up = sum(c.isupper() for c in t) > 0.6 * max(1, sum(c.isalpha() for c in t))
        if len(t) < 5 or sum(c.isalpha() for c in t) < 3:
            pass
        elif ratio >= 1.45:
            r.kind, r.level = "heading", 0
        elif ratio >= 1.25 and short:
            r.kind, r.level = "heading", 1
        elif r.align == "center" and up and len(t) < 90:
            r.kind, r.level = "heading", 1
        elif short and len(r.segs) == 1 and len(t) < 90 and ((r.is_bold and (up or HEAD_WORD.match(t))) or (up and HEAD_WORD.match(t))):
            r.kind, r.level = "heading", 2
        elif r.align == "center" and r.is_bold and short:
            r.kind, r.level = "heading", 1
    return dict(body_h=body_h, pitch=pitch, left=left, right=right, body_pt=body_pt, body_rel=body_rel)


def to_blocks(rows, m, W, ptpx):
    """Gộp các hàng thành khối: tiêu đề, đoạn, mục danh sách."""
    blocks = []
    cur = None
    L, R, pitch = m["left"], m["right"], m["pitch"]
    for r in rows:
        first_seg_is_num = len(r.segs) >= 2 and LIST_RE.match(r.segs[0][2])
        starts_list = first_seg_is_num or (len(r.segs) == 1 and LIST_START.match(r.text))
        new = False
        if cur is None or r.kind == "heading" or cur.type == "heading":
            new = True
        elif starts_list:
            new = True
        else:
            prev = cur.rows[-1]
            gap = r.cy - prev.cy
            if gap > 1.45 * pitch:
                new = True
            elif len(r.segs) > 1 or len(prev.segs) > 1 and cur.type != "list":
                new = True
            elif prev.x1 < R - 0.10 * W and cur.type != "list":
                new = True                              # dòng trước ngắn: đoạn đã kết thúc
            elif prev.x1 < R - 0.10 * W and cur.type == "list":
                new = True
            elif r.align != prev.align or abs(r.size - prev.size) > 0.9:
                new = True
            elif cur.type == "paragraph":
                first = cur.rows[0]
                # dòng đầu thụt vào so với các dòng sau thì dòng sau không phải đầu đoạn mới;
                # dòng thụt vào so với dòng trước là đầu đoạn mới
                if len(cur.rows) >= 2 and r.x0 > cur.rows[1].x0 + 0.02 * W:
                    new = True
                elif len(cur.rows) == 1 and r.x0 > first.x0 + 0.02 * W and first.x0 <= L + 0.02 * W:
                    new = True
            elif cur.type == "list":
                tx = cur.rows[0].segs[1][0] if len(cur.rows[0].segs) >= 2 else cur.rows[0].x0
                if abs(r.x0 - tx) > 0.03 * W and r.x0 < tx - 0.03 * W:
                    new = True
        if new:
            t = "heading" if r.kind == "heading" else "list" if starts_list else "paragraph"
            cur = Block(type=t, rows=[r])
            blocks.append(cur)
        else:
            cur.rows.append(r)
    for b in blocks:
        b.bbox = (min(r.x0 for r in b.rows), min(r.y0 for r in b.rows), max(r.x1 for r in b.rows), max(r.y1 for r in b.rows))
        if b.type == "paragraph" and len(b.rows) > 1:
            for r in b.rows[:-1]:
                if r.full:
                    pass
    return blocks


# ------------------------------------------------------------------ cột
def find_columns(boxes, W, H):
    """Tìm khe trống dọc phân cột. Trả về list vị trí x giữa các khe (rỗng nếu trang một cột)."""
    if len(boxes) < 12:
        return []
    cover = np.zeros(W + 1, np.int32)
    for x0, y0, x1, y1 in boxes:
        cover[max(0, x0):min(W, x1) + 1] += 1
    xs = np.where(cover > 0)[0]
    if not len(xs):
        return []
    a, b = xs[0], xs[-1]
    gutters = []
    d = np.diff(np.concatenate([[0], (cover[a:b + 1] <= max(1, int(0.03 * len(boxes)))).astype(np.int8), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s >= 0.025 * W:
            xc = a + (s + e) // 2
            left = sum(1 for bb in boxes if bb[2] <= xc); right = sum(1 for bb in boxes if bb[0] >= xc)
            if left >= 6 and right >= 6:
                gutters.append(int(xc))
    return gutters


# ------------------------------------------------------------------ tổng hợp
def model_ready() -> bool:
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    py = root / ".venv-paddle" / "Scripts" / "python.exe"
    return py.exists() and all((root / "models" / "layout_pp" / m / "inference.pdiparams").exists() for m in ("PP-DocLayout-M", "SLANet_plus"))


def run_model(page_bgr: np.ndarray, timeout: int = 300):
    """Chạy PP-DocLayout-M và SLANet_plus trong .venv-paddle. Trả về None nếu chưa cài hoặc lỗi."""
    import json, os, subprocess, sys, tempfile
    from pathlib import Path
    if not model_ready():
        return None
    root = Path(__file__).resolve().parents[1]
    py = root / ".venv-paddle" / "Scripts" / "python.exe"
    fd, tmp = tempfile.mkstemp(suffix=".png"); os.close(fd)
    try:
        cv2.imwrite(tmp, page_bgr)
        r = subprocess.run([str(py), "-X", "utf8", str(root / "src" / "layout_worker.py"), tmp],
                           capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        for ln in reversed(r.stdout.splitlines()):
            if ln.startswith("{"):
                return json.loads(ln)
    except Exception:  # noqa: BLE001
        return None
    finally:
        try: os.remove(tmp)
        except OSError: pass
    return None


def _tables_from_model(model_tables, pitch):
    out = []
    for t in model_tables:
        cells = t["cells"]
        if len(cells) < 2:
            continue
        tol = max(6.0, 0.6 * pitch)
        xs = _cluster([c[0] for c in cells] + [c[2] for c in cells], tol)
        ys = _cluster([c[1] for c in cells] + [c[3] for c in cells], tol)
        if len(xs) >= 2 and len(ys) >= 2:
            out.append((tuple(int(v) for v in t["box"]), xs, ys))
    return out


FIG_LABELS = {"image", "figure", "chart", "seal", "formula"}


def analyse_page(page_bgr: np.ndarray, lines: list, page_size=A4, model=None):
    """model: kết quả của layout_worker ({'regions','tables'}) hoặc None (dùng hình học thuần)."""
    """lines: [{'box': [x0,y0,x1,y1], 'text': str}]. Trả về dict gồm khối, số đo trang."""
    H, W = page_bgr.shape[:2]
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    ink = _ink(gray)
    portrait = H >= W
    pw, ph = (page_size if portrait else page_size[::-1])
    ptpx = pw / W
    items = [(tuple(l["box"]), l["text"]) for l in lines if l["text"].strip()]
    # đo sơ bộ chu kỳ dòng để dò bảng và hình
    rows0 = build_rows(items, ink) if items else []
    if not rows0:
        return dict(blocks=[], W=W, H=H, ptpx=ptpx, page=(pw, ph), measure=None)
    m0 = analyse(rows0, W, H, ptpx)
    pitch = m0["pitch"]
    regions = (model or {}).get("regions", [])
    if model and model.get("tables"):
        tables = _tables_from_model(model["tables"], pitch)
    else:
        tables = find_tables(ink, pitch)
    if model:
        figs = [tuple(r["box"]) for r in regions if r["label"] in FIG_LABELS and r["score"] >= 0.4]
    else:
        figs = find_figures(page_bgr, ink, pitch, [t[0] for t in tables])
    blocks = []
    used = set()
    inside = lambda c, bb: bb[0] <= c[0] <= bb[2] and bb[1] <= c[1] <= bb[3]
    for bb, xs, ys in tables:
        ncol, nrow = len(xs) - 1, len(ys) - 1
        grid = [["" for _ in range(ncol)] for _ in range(nrow)]
        cells_items = [[[] for _ in range(ncol)] for _ in range(nrow)]
        for i, (box, t) in enumerate(items):
            c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if not inside(c, bb):
                continue
            ci = next((k for k in range(ncol) if xs[k] <= c[0] <= xs[k + 1]), None)
            ri = next((k for k in range(nrow) if ys[k] <= c[1] <= ys[k + 1]), None)
            if ci is None or ri is None:
                continue
            cells_items[ri][ci].append((box, t)); used.add(i)
        for ri in range(nrow):
            for ci in range(ncol):
                cells_items[ri][ci].sort(key=lambda bt: (bt[0][1], bt[0][0]))
                grid[ri][ci] = "\n".join(t for _, t in cells_items[ri][ci])
        sizes = [(b_[3] - b_[1]) / 1.24 for r_ in cells_items for c_ in r_ for b_, _ in c_]
        blocks.append(Block(type="table", bbox=bb, table=grid,
                            cells=[[(xs[k], ys[r_], xs[k + 1], ys[r_ + 1]) for k in range(ncol)] for r_ in range(nrow)],
                            rows=[]))
        blocks[-1].size_px = float(np.median(sizes)) if sizes else pitch / 1.3
    for fb in figs:
        c0 = page_bgr[fb[1]:fb[3], fb[0]:fb[2]].copy()
        blocks.append(Block(type="figure", bbox=fb, image=c0, rows=[]))
        for i, (box, t) in enumerate(items):
            c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if inside(c, fb):
                used.add(i)
    rest = [it for i, it in enumerate(items) if i not in used]
    rows = build_rows(rest, ink) if rest else []
    text_regs = [(i, r) for i, r in enumerate(regions) if r["label"] not in FIG_LABELS and r["label"] != "table"]
    for r_ in rows:
        cx, cy = (r_.x0 + r_.x1) / 2, r_.cy
        best = None
        for i, rg in text_regs:
            x0, y0, x1, y1 = rg["box"]
            if x0 - 4 <= cx <= x1 + 4 and y0 - 4 <= cy <= y1 + 4:
                if best is None or (x1 - x0) * (y1 - y0) < best[1]:
                    best = (i, (x1 - x0) * (y1 - y0), rg["label"])
        r_.region, r_.rlabel = (best[0], best[2]) if best else (None, None)
    measure = analyse(rows, W, H, ptpx) if rows else m0
    if model and rows:
        for r_ in rows:
            if r_.rlabel == "doc_title" and len(r_.text) >= 5:
                r_.kind, r_.level = "heading", 0
            elif r_.rlabel in ("paragraph_title", "figure_title") and len(r_.text) >= 5 and len(r_.text) < 120:
                r_.kind, r_.level = "heading", max(r_.level, 2) if r_.kind == "heading" else 2
            elif r_.rlabel in ("footnote", "footer", "header", "number") and r_.size > measure["body_pt"]:
                r_.size = _snap(measure["body_pt"] * 0.85)
    if rows:
        gutters = find_columns([(sg[0], 0, sg[1], 0) for r_ in rows for sg in r_.segs if (sg[1] - sg[0]) >= 0.15 * W and len(sg[2]) >= 12], W, H)
    else:
        gutters = []
    text_blocks = []
    cols = [(0, W)]
    if gutters:
        edges = [0] + gutters + [W]
        cols = list(zip(edges[:-1], edges[1:]))
    for ci, (cx0, cx1) in enumerate(cols):
        sub = [r_ for r_ in rows if (len(cols) == 1) or (cx0 <= (r_.x0 + r_.x1) / 2 < cx1)]
        if len(cols) > 1 and sub:
            # đo lại lề trong cột để căn lề và thụt dòng tính theo cột
            sm = analyse(sub, W, H, ptpx) | {"left": min(r_.x0 for r_ in sub), "right": max(r_.x1 for r_ in sub)}
            sm["pitch"] = measure["pitch"]
        else:
            sm = measure
        for b in to_blocks(sub, sm, W, ptpx):
            b.column = ci; b.measure = sm
            text_blocks.append(b)
    for b in text_blocks:
        if not hasattr(b, "measure"):
            b.measure = measure
    allb = blocks + text_blocks
    allb.sort(key=lambda b: (b.column if b.type not in ("table", "figure") and len(cols) > 1 else 0, b.bbox[1]) if len(cols) == 1 else (b.bbox[1] if b.type in ("table", "figure") else b.column * 10 ** 6 + b.bbox[1]))
    return dict(blocks=allb, W=W, H=H, ptpx=ptpx, page=(pw, ph), measure=measure, columns=len(cols), gutters=gutters, rows=rows)


def describe(res) -> list:
    """Danh sách khối ở dạng JSON để giao diện vẽ chồng lên ảnh."""
    out = []
    for b in res["blocks"]:
        txt = b.table and " | ".join(" / ".join(r_) for r_ in b.table)[:80] or " ".join(r_.text for r_ in b.rows)[:80]
        out.append({"type": b.type, "box": list(map(int, b.bbox)), "text": txt, "column": b.column})
    return out


# ------------------------------------------------------------------ .docx
def build_docx(res, font: str = "Times New Roman", page_size=A4) -> bytes:
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Emu, Pt

    ptpx = res["ptpx"]; W = res["W"]; pw, ph = res["page"]
    m = res["measure"]
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Pt(pw), Pt(ph)
    blocks = res["blocks"]
    if m:
        xs0 = [b.bbox[0] for b in blocks]; xs1 = [b.bbox[2] for b in blocks]; ys0 = [b.bbox[1] for b in blocks]; ys1 = [b.bbox[3] for b in blocks]
        lm = float(np.clip(min(xs0) * ptpx, 28, 120)); rm = float(np.clip((W - max(xs1)) * ptpx, 28, 120))
        tm = float(np.clip(min(ys0) * ptpx, 28, 110)); bm = float(np.clip((res["H"] - max(ys1)) * ptpx, 28, 110))
    else:
        lm = rm = tm = bm = 72.0
    sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Pt(lm), Pt(rm), Pt(tm), Pt(bm)
    origin = lm / ptpx                                                  # x (px) ứng với lề trái trang Word
    st = doc.styles["Normal"]
    st.font.name = font; st.font.size = Pt(m["body_pt"] if m else 12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), font); st.element.rPr.rFonts.set(qn("w:cs"), font)
    st.paragraph_format.space_after = Pt(0); st.paragraph_format.space_before = Pt(0)

    def set_cols(section, n, space_pt=18):
        sp = section._sectPr
        for c in sp.findall(qn("w:cols")):
            sp.remove(c)
        c = OxmlElement("w:cols"); c.set(qn("w:num"), str(n)); c.set(qn("w:space"), str(int(space_pt * 20)))
        sp.append(c)

    def add_run(p, text, size, bold):
        run = p.add_run(text)
        run.font.size = Pt(size); run.font.bold = bool(bold); run.font.name = font
        run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
        return run

    def line_spacing(p, pitch_px, size):
        pt = pitch_px * ptpx
        if size * 1.0 <= pt <= size * 2.2:
            p.paragraph_format.line_spacing = Pt(pt)
            p.paragraph_format.line_spacing_rule = 4          # WD_LINE_SPACING.EXACTLY

    prev_bottom = None
    prev_col = None
    cur_cols = 1
    first = True
    ncols = res.get("columns", 1)
    if ncols > 1:
        doc.add_section(WD_SECTION.CONTINUOUS)
        set_cols(doc.sections[-1], ncols)
        cur_cols = ncols
    pitch = m["pitch"] if m else 20
    for bi, b in enumerate(blocks):
        if ncols > 1 and b.type not in ("table", "figure"):
            if prev_col is not None and b.column != prev_col:
                doc.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)
            prev_col = b.column
        if b.type == "table":
            rows_n, cols_n = len(b.table), len(b.table[0])
            tbl = doc.add_table(rows=rows_n, cols=cols_n)
            tbl.style = "Table Grid"
            size = _snap(max(8, (b.size_px * 1.24 / 1.24) * ptpx / 0.8)) if getattr(b, "size_px", 0) else 11
            size = min(size, (m["body_pt"] if m else 12))
            for ri in range(rows_n):
                for ci in range(cols_n):
                    x0, y0, x1, y1 = b.cells[ri][ci]
                    cell = tbl.cell(ri, ci)
                    cell.width = Pt(max(10, (x1 - x0) * ptpx))
                    cell.text = ""
                    parts = b.table[ri][ci].split("\n") if b.table[ri][ci] else [""]
                    for pi, part in enumerate(parts):
                        p = cell.paragraphs[0] if pi == 0 else cell.add_paragraph()
                        add_run(p, part, size, False)
            doc.add_paragraph().paragraph_format.space_after = Pt(0)
            prev_bottom = b.bbox[3]
            continue
        if b.type == "figure":
            p = doc.add_paragraph()
            wpx = b.bbox[2] - b.bbox[0]
            ok, buf = cv2.imencode(".png", b.image)
            p.add_run().add_picture(io.BytesIO(buf.tobytes()), width=Pt(max(20, wpx * ptpx)))
            p.paragraph_format.left_indent = Pt(max(0, (b.bbox[0] - origin) * ptpx))
            prev_bottom = b.bbox[3]
            continue
        rows = b.rows
        r0 = rows[0]
        bm_ = b.measure
        L = bm_["left"]; R = bm_["right"]
        size = r0.size
        p = doc.add_paragraph()
        pf = p.paragraph_format
        # khoảng cách trước đoạn
        if prev_bottom is not None and prev_col == b.column or (prev_bottom is not None and ncols == 1):
            gap = r0.y0 - prev_bottom - (pitch - m["body_h"] * 1.24)
            if gap > 0.25 * pitch:
                pf.space_before = Pt(min(36, gap * ptpx))
        # căn lề
        if b.type == "heading" or r0.align != "left":
            al = r0.align
            pf.alignment = {"center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}.get(al, WD_ALIGN_PARAGRAPH.LEFT)
        elif len(rows) >= 2 and sum(1 for r in rows[:-1] if r.full or r.x1 >= R - 0.035 * W) >= max(1, len(rows[:-1]) - 1):
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        # thụt lề
        if b.type == "list" and len(r0.segs) >= 2:
            tx = r0.segs[1][0]
            pf.left_indent = Pt(max(0, (tx - origin) * ptpx))
            pf.first_line_indent = Pt(-(tx - r0.segs[0][0]) * ptpx)
            pf.tab_stops.add_tab_stop(Pt(max(0, (tx - origin) * ptpx)), WD_TAB_ALIGNMENT.LEFT)
            first_text = r0.segs[0][2] + "\t" + " ".join(s[2] for s in r0.segs[1:])
        else:
            base_x = rows[1].x0 if len(rows) >= 2 else r0.x0
            li = (base_x - origin) * ptpx
            pf.left_indent = Pt(max(0, li))
            fi = (r0.x0 - base_x) * ptpx
            if abs(fi) > 3:
                pf.first_line_indent = Pt(fi)
            if len(r0.segs) >= 2:
                for s in r0.segs[1:]:
                    pf.tab_stops.add_tab_stop(Pt(max(0, (s[0] - origin) * ptpx)), WD_TAB_ALIGNMENT.LEFT)
                first_text = "\t".join(s[2] for s in r0.segs)
            else:
                first_text = r0.text
        texts = [first_text] + [r.text for r in rows[1:]]
        joined = ""
        for t in texts:
            if joined.endswith("-") and not joined.endswith(" -"):
                joined = joined[:-1] + t
            else:
                joined = (joined + " " + t) if joined and not joined.endswith("\t") else joined + t
        bold = sum(1 for r in rows if r.is_bold) > len(rows) / 2
        add_run(p, joined.strip(), size, bold or b.type == "heading")
        line_spacing(p, pitch, size)
        pf.keep_together = False
        prev_bottom = max(r.y1 for r in rows)
        first = False
    if ncols > 1:
        doc.add_section(WD_SECTION.CONTINUOUS); set_cols(doc.sections[-1], 1)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
