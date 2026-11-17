"""N2: sinh dữ liệu từ kết quả căn chỉnh N1.

Đầu ra (mặc định data/processed/esp32_pairs):
  pages/{esp_raw,esp_1a,scan}/0000NN.png   trang đã nắn (khung render 144 DPI)
  crops/{esp_raw,esp_1a,scan}/0000NN_LLL.png  crop dòng
  labels/{train,val,test}_{variant}.txt    định dạng VietOCR: path<TAB>text
  lines.jsonl                               metadata từng dòng (bbox, score, cờ)
  patches/{split}/{esp,scan}/*.png          cặp patch 256x256 đã căn cho khâu 1b
  summary.json

Chạy:
  .venv-ocr/Scripts/python.exe -X utf8 src/n2_build_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhance import enhance_1a  # noqa: E402
from n1_align import flat_gray, imread, imwrite  # noqa: E402

import paths  # noqa: E402

ROOT = paths.ROOT
SPLITS = {
    "test": [3, 4, 7, 9, 11, 14, 25, 32, 38, 44],
    "val": [6, 17, 21, 29, 45],
}
VARIANTS = ("esp_raw", "esp_1a", "scan")
CHAR_MAP = {"‘": "'", "’": "'", "‚": ",", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-", "−": "-",
            "…": "...", " ": " ", "•": "-", "·": ".", "×": "x", "Ð": "Đ", "ð": "đ"}


def split_of(idx: int) -> str:
    for s, ids in SPLITS.items():
        if idx in ids:
            return s
    return "train"


def vocab() -> set[str]:
    from vietocr.tool.config import Cfg
    return set(Cfg.load_config_from_name("vgg_transformer")["vocab"])


def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = "".join(CHAR_MAP.get(c, c) for c in t)
    return " ".join(t.split())


# ----------------------------------------------------------------------------- gộp đoạn thành dòng

def merge_segments(segs: list[dict]) -> list[dict]:
    """Gộp các đoạn PDF cùng đường chân chữ, liền kề theo chiều ngang thành một dòng."""
    segs = sorted(segs, key=lambda s: (s["bbox"][1] + s["bbox"][3]) / 2)
    lines: list[dict] = []
    for s in segs:
        x0, y0, x1, y1 = s["bbox"]
        h = y1 - y0
        best = None
        for ln in lines:
            lx0, ly0, lx1, ly1 = ln["bbox"]
            lh = ly1 - ly0
            ov = min(y1, ly1) - max(y0, ly0)
            if ov < 0.6 * min(h, lh) or abs(h - lh) > 0.6 * max(h, lh):
                continue
            gap = max(x0 - lx1, lx0 - x1)  # khoảng cách ngang (âm = chồng lấn)
            if -0.3 * h <= gap <= 1.2 * max(h, lh):
                best = ln
                break
        if best is None:
            lines.append({"bbox": [x0, y0, x1, y1], "segs": [s]})
        else:
            best["segs"].append(s)
            b = best["bbox"]
            best["bbox"] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
    out = []
    for ln in lines:
        segs_ = sorted(ln["segs"], key=lambda s: s["bbox"][0])
        text = segs_[0]["text"]
        for prev, cur in zip(segs_, segs_[1:]):
            gap = cur["bbox"][0] - prev["bbox"][2]
            h = cur["bbox"][3] - cur["bbox"][1]
            # chỉ là phương án dự phòng: text chính lấy lại từ PDF theo khung cả dòng (pdf_line_text)
            text += (" " if gap > 0.12 * h else "") + cur["text"]
        out.append({"bbox": ln["bbox"], "text_pdf": text, "n_segments": len(segs_)})
    out.sort(key=lambda l: (round(l["bbox"][1] / 10), l["bbox"][0]))
    return out


def pdf_line_text(textpage, page_h_pt: float, bbox_render, scale=2.0) -> str:
    """Lấy text của cả dòng trực tiếp từ PDF theo khung (giữ đúng dấu cách, dấu câu theo thứ tự nội dung).
    Bỏ 35% phía trên khung để không lấy dấu/đuôi chữ của dòng kề trên."""
    x0, y0, x1, y1 = [v / scale for v in bbox_render]
    h = y1 - y0
    return textpage.get_text_bounded(x0 - 0.5, page_h_pt - (y1 - 0.05 * h), x1 + 0.5, page_h_pt - (y0 + 0.35 * h))


def match_line(esp_flat: np.ndarray, ref_flat: np.ndarray, bbox, search=8):
    h, w = ref_flat.shape
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0, x1, y1 = max(x0 - 2, 0), max(y0 - 2, 0), min(x1 + 2, w), min(y1 + 2, h)
    tpl = ref_flat[y0:y1, x0:x1]
    sx0, sy0, sx1, sy1 = max(x0 - search, 0), max(y0 - search, 0), min(x1 + search, w), min(y1 + search, h)
    win = esp_flat[sy0:sy1, sx0:sx1]
    if tpl.size == 0 or win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1] or tpl.std() < 5 or win.std() < 2:
        return 0, 0, 0.0
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return loc[0] + sx0 - x0, loc[1] + sy0 - y0, float(mx)


def crop(img: np.ndarray, bbox, dx=0, dy=0) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    py, px = 0.2 * h, 0.25 * h
    H, W = img.shape[:2]
    a, b = int(max(x0 + dx - px, 0)), int(min(x1 + dx + px, W))
    c, d = int(max(y0 + dy - py, 0)), int(min(y1 + dy + py, H))
    return img[c:d, a:b]


# ----------------------------------------------------------------------------- patch cho 1b

def make_patches(esp: np.ndarray, scan: np.ndarray, valid: np.ndarray, out: Path, idx: int,
                 size=256, stride=128, search=6):
    esp_f, scan_f = flat_gray(esp), flat_gray(scan)
    H, W = esp.shape[:2]
    n = 0
    for y in range(0, H - size - 2 * search, stride):
        for x in range(0, W - size - 2 * search, stride):
            ys, xs = y + search, x + search
            if valid[y:y + size + 2 * search, x:x + size + 2 * search].min() == 0:
                continue
            tpl = scan_f[ys:ys + size, xs:xs + size]
            if tpl.std() < 12:  # vùng gần như trắng
                continue
            win = esp_f[y:y + size + 2 * search, x:x + size + 2 * search]
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, (lx, ly) = cv2.minMaxLoc(res)
            if mx < 0.35:
                continue
            ep = esp[y + ly:y + ly + size, x + lx:x + lx + size]
            sp = scan[ys:ys + size, xs:xs + size]
            name = f"{idx:06d}_{y:04d}_{x:04d}.png"
            imwrite(out / "esp" / name, ep)
            imwrite(out / "scan" / name, sp)
            n += 1
    return n


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=Path, default=paths.ESP32_ALIGN)
    ap.add_argument("--out", type=Path, default=paths.ESP32_PAIRS)
    ap.add_argument("--pdf", type=Path, default=paths.SOURCE_PDF)
    ap.add_argument("--layout", type=Path, default=paths.ESP32_PAIRS / "layout" / "layout_draft.json",
                    help="nhãn layout nháp; dòng nằm trong vùng figure bị loại")
    ap.add_argument("--no-patches", action="store_true")
    args = ap.parse_args()
    figures: dict[int, list] = {}
    if args.layout and args.layout.exists():
        for k, regs in json.loads(args.layout.read_text("utf-8"))["pages"].items():
            figures[int(k)] = [r["bbox_render"] for r in regs if r["class"] == "figure"]
    out = args.out
    voc = vocab()
    align = {int(Path(r["image"]).stem): r for r in json.loads((args.n1 / "alignment.json").read_text("utf-8"))}
    doc = pdfium.PdfDocument(str(args.pdf))
    meta, labels = [], {(s, v): [] for s in ("train", "val", "test") for v in VARIANTS}
    stats = {"pages": 0, "lines_total": 0, "lines_kept": 0, "dropped": {}, "oov_chars": {}, "patches": {}}

    def drop(reason):
        stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1

    for idx in sorted(align):
        rec = align[idx]
        if rec.get("status") != "OK":
            continue
        split = split_of(idx)
        esp = imread(args.n1 / "rectified_esp" / f"{idx:06d}.png")
        scan = imread(args.n1 / "rectified_scan" / f"{idx:06d}.png")
        render = imread(args.n1 / "render" / f"{idx:06d}.png")
        # vùng có dữ liệu ESP32 = khung ảnh gốc chiếu sang khung render (không suy từ độ sáng:
        # chữ đen trên ảnh tối có giá trị 0 nhưng vẫn hợp lệ)
        ew, eh = rec["esp_size"]
        rw, rh = rec["render_size"]
        valid = cv2.warpPerspective(np.ones((eh, ew), np.uint8), np.array(rec["H_esp_to_render"]), (rw, rh),
                                    flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
        e1a = enhance_1a(esp)
        e1a[valid == 0] = 0
        pages = {"esp_raw": esp, "esp_1a": e1a, "scan": scan}
        for v, img in pages.items():
            imwrite(out / "pages" / v / f"{idx:06d}.png", img)
        esp_f, ref_f = flat_gray(esp), flat_gray(render)
        segs = json.loads((args.n1 / "lines" / f"{idx:06d}.json").read_text("utf-8"))["lines"]
        pdf_page = doc[rec["pdf_page"] - 1]
        page_h_pt = pdf_page.get_size()[1]
        textpage = pdf_page.get_textpage()
        lines = merge_segments(segs)
        stats["pages"] += 1
        for k, ln in enumerate(lines):
            stats["lines_total"] += 1
            x0, y0, x1, y1 = ln["bbox"]
            h, w = y1 - y0, x1 - x0
            text = clean_text(pdf_line_text(textpage, page_h_pt, ln["bbox"])) or clean_text(ln["text_pdf"])
            oov = sorted(set(text) - voc)
            if h < 8 or h > 70:
                drop("height"); continue
            if w < 2 * h or len(text) < 2:
                drop("short_or_narrow"); continue  # nhãn trục biểu đồ, số thứ tự…
            cy, cx = int((y0 + y1) / 2), int((x0 + x1) / 2)
            if not (valid[int(y0), int(x0)] and valid[int(y1) - 1, int(x1) - 1] and valid[cy, cx]):
                drop("outside_esp_frame"); continue
            if any(fx0 <= cx <= fx1 and fy0 <= cy <= fy1 for fx0, fy0, fx1, fy1 in figures.get(idx, [])):
                drop("in_figure"); continue  # chữ trong biểu đồ: giữ nguyên làm hình, PDF hay lặp số
            if oov:
                for c in oov:
                    stats["oov_chars"][c] = stats["oov_chars"].get(c, 0) + 1
                drop("oov_char"); continue
            dx, dy, score = match_line(esp_f, ref_f, ln["bbox"])
            if score < 0.3:
                dx = dy = 0  # không tin độ dịch khi khớp kém
            lid = f"{idx:06d}_{k:03d}"
            row = {"id": lid, "page": idx, "split": split, "text": text, "bbox_render": [round(v, 1) for v in ln["bbox"]],
                   "dx": dx, "dy": dy, "match_score": round(score, 3), "n_segments": ln["n_segments"],
                   "legibility": "good" if score >= 0.5 else "medium" if score >= 0.3 else "poor"}
            for v, img in pages.items():
                sdx, sdy = (dx, dy) if v.startswith("esp") else (0, 0)
                c = crop(img, ln["bbox"], sdx, sdy)
                rel = f"crops/{v}/{lid}.png"
                imwrite(out / rel, c)
                labels[(split, v)].append(f"{rel}\t{text}")
            meta.append(row)
            stats["lines_kept"] += 1
        if not args.no_patches:
            n = make_patches(esp, scan, valid, out / "patches" / split, idx)
            stats["patches"][split] = stats["patches"].get(split, 0) + n
        print(idx, split, len(lines), "lines", flush=True)

    (out / "labels").mkdir(parents=True, exist_ok=True)
    for (s, v), rows in labels.items():
        (out / "labels" / f"{s}_{v}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with open(out / "lines.jsonl", "w", encoding="utf-8") as f:
        for r in meta:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("train", "val", "test"):
        rows = [r for r in meta if r["split"] == s]
        stats[f"lines_{s}"] = len(rows)
        stats[f"legibility_{s}"] = {g: sum(r["legibility"] == g for r in rows) for g in ("good", "medium", "poor")}
    (out / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
