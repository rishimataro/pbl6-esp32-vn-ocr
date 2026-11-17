"""N1: căn chỉnh ESP32 ↔ scan ↔ PDF và đo khâu tìm trang.

Với mỗi ảnh i (1..45):
  - render trang PDF gốc tương ứng (khung tọa độ chuẩn "render"),
  - homography scan → render (SIFT + RANSAC, tinh chỉnh ECC),
  - homography ESP32 → scan (SIFT + RANSAC, tinh chỉnh ECC),
  - ghép: ESP32 → render; nắn ảnh ESP32 về khung render,
  - lấy khung dòng chữ từ lớp text PDF, chiếu sang ảnh ESP32,
  - đo độ khớp từng dòng bằng template matching (độ lệch dư, điểm tương quan),
  - tìm trang tự động trên ảnh ESP32 (khâu 0) và so với góc trang suy ra từ căn chỉnh.

Chạy:
  .venv-ocr/Scripts/python.exe -X utf8 src/n1_align.py            # ghi vào data/interim/esp32_alignment
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

ROOT = paths.ROOT
DEFAULT_DATA = paths.ESP32_RAW
DEFAULT_PDF = paths.SOURCE_PDF
DEFAULT_MAP = paths.PAGE_ORDER_CSV
RENDER_SCALE = 2.0  # 72 pt * 2 = 144 DPI, trang ~1179 x 1616 px


# ----------------------------------------------------------------------------- io

def imread(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)  # hỗ trợ đường dẫn Unicode trên Windows
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite(path: Path, img: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))


def load_mapping(path: Path) -> dict[int, int]:
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            idx = int(Path(row["image_file"]).stem)
            out[idx] = int(row["selected_original_pdf_page_1based"])
    return out


# ----------------------------------------------------------------------------- pdf

def render_page(doc: pdfium.PdfDocument, page_no: int):
    """Trả về ảnh render BGR và danh sách đoạn chữ (text, bbox render px)."""
    page = doc[page_no - 1]
    w_pt, h_pt = page.get_size()
    bmp = page.render(scale=RENDER_SCALE)
    img = cv2.cvtColor(np.asarray(bmp.to_pil().convert("RGB")), cv2.COLOR_RGB2BGR)
    tp = page.get_textpage()
    lines = []
    for i in range(tp.count_rects()):
        l, b, r, t = tp.get_rect(i)
        if r <= 0 or t <= 0 or l >= w_pt or b >= h_pt:  # ngoài trang (text ẩn)
            continue
        text = tp.get_text_bounded(l, b, r, t)
        text = unicodedata.normalize("NFC", " ".join(text.split()))
        if not text:
            continue
        x0, x1 = l * RENDER_SCALE, r * RENDER_SCALE
        y0, y1 = (h_pt - t) * RENDER_SCALE, (h_pt - b) * RENDER_SCALE
        lines.append({"text": text, "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]})
    return img, lines


# ----------------------------------------------------------------------------- alignment

def flat_gray(img: np.ndarray) -> np.ndarray:
    """Gray, chia cho nền ước lượng (bỏ vignetting/bóng), giãn tương phản."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = g.astype(np.float32) + 1.0
    k = max(31, (min(g.shape) // 20) | 1)
    bg = cv2.medianBlur(np.clip(g, 0, 255).astype(np.uint8), k).astype(np.float32) + 1.0
    n = g / bg
    lo, hi = np.percentile(n, [1, 99])
    n = np.clip((n - lo) / max(hi - lo, 1e-6), 0, 1)
    return (n * 255).astype(np.uint8)


_SIFT = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.01)
_MATCHER = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})


def sift_homography(src: np.ndarray, dst: np.ndarray, src_mask=None, ratio=0.8):
    """Homography src → dst. src/dst là ảnh gray uint8."""
    k1, d1 = _SIFT.detectAndCompute(src, src_mask)
    k2, d2 = _SIFT.detectAndCompute(dst, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, {"matches": 0, "inliers": 0}
    knn = _MATCHER.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, {"matches": len(good), "inliers": 0}
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 4.0, maxIters=20000, confidence=0.9995)
    inl = int(mask.sum()) if mask is not None else 0
    return H, {"matches": len(good), "inliers": inl}


def ecc_refine(src: np.ndarray, dst: np.ndarray, H: np.ndarray, scale=0.5, iters=200):
    """Tinh chỉnh homography src→dst bằng ECC trên ảnh đã thu nhỏ. Trả về (H, cc) hoặc (H gốc, None)."""
    S = np.diag([scale, scale, 1.0])
    Si = np.diag([1 / scale, 1 / scale, 1.0])
    s_small = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    d_small = cv2.resize(dst, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    s_small = cv2.GaussianBlur(s_small, (0, 0), 1.2).astype(np.float32)
    d_small = cv2.GaussianBlur(d_small, (0, 0), 1.2).astype(np.float32)
    # ECC tìm warp W sao cho src(W·x) ≈ dst(x) → W là ánh xạ dst→src (nghịch đảo của H)
    W0 = (S @ np.linalg.inv(H) @ Si).astype(np.float32)
    W0 /= W0[2, 2]
    try:
        cc, W = cv2.findTransformECC(d_small, s_small, W0, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6), None, 5)
    except cv2.error:
        return H, None
    Hn = np.linalg.inv(Si @ W.astype(np.float64) @ S)
    return Hn / Hn[2, 2], float(cc)


def warp_ncc(src_gray, dst_gray, H) -> float:
    """Tương quan chuẩn hóa giữa src đã warp sang dst và dst, chỉ trong vùng hợp lệ."""
    h, w = dst_gray.shape
    wa = cv2.warpPerspective(src_gray, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpPerspective(np.full(src_gray.shape, 255, np.uint8), H, (w, h)) > 0
    a = cv2.GaussianBlur(wa, (0, 0), 1.5).astype(np.float32)[valid]
    b = cv2.GaussianBlur(dst_gray, (0, 0), 1.5).astype(np.float32)[valid]
    if a.size < 1000:
        return 0.0
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def homography_sane(H, src_shape, dst_shape) -> bool:
    if H is None or not np.all(np.isfinite(H)):
        return False
    h, w = src_shape[:2]
    c = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H)[0]
    if not cv2.isContourConvex(c.astype(np.float32)):
        return False
    area = cv2.contourArea(c)
    return 0.05 * dst_shape[0] * dst_shape[1] < area < 20 * dst_shape[0] * dst_shape[1]


# ----------------------------------------------------------------------------- khâu 0: tìm trang

def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])  # tl, tr, br, bl


def detect_page(esp: np.ndarray):
    """Tìm tứ giác trang giấy sáng trên nền tối. Trả về 4 góc (tl,tr,br,bl) hoặc None."""
    hsv = cv2.cvtColor(esp, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[..., 2], (0, 0), 3)
    _, th = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * esp.shape[0] * esp.shape[1]:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def quad_iou(a: np.ndarray, b: np.ndarray, shape) -> float:
    h, w = shape[:2]
    pad = 400  # cho phép góc nằm ngoài ảnh
    ma = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    mb = ma.copy()
    cv2.fillPoly(ma, [np.int32(a + pad)], 1)
    cv2.fillPoly(mb, [np.int32(b + pad)], 1)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union else 0.0


# ----------------------------------------------------------------------------- line check

def line_residuals(esp_rect_flat: np.ndarray, render_flat: np.ndarray, lines, search=10):
    """Template matching từng dòng: dòng render trong ảnh ESP32 đã nắn, ±search px."""
    h, w = render_flat.shape
    out = []
    for ln in lines:
        x0, y0, x1, y1 = [int(round(v)) for v in ln["bbox"]]
        x0, y0 = max(x0 - 2, 0), max(y0 - 2, 0)
        x1, y1 = min(x1 + 2, w), min(y1 + 2, h)
        if x1 - x0 < 40 or y1 - y0 < 8:
            out.append(None)
            continue
        tpl = render_flat[y0:y1, x0:x1]
        sx0, sy0 = max(x0 - search, 0), max(y0 - search, 0)
        sx1, sy1 = min(x1 + search, w), min(y1 + search, h)
        win = esp_rect_flat[sy0:sy1, sx0:sx1]
        if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1] or tpl.std() < 5 or win.std() < 3:
            out.append(None)
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        dx, dy = loc[0] + sx0 - x0, loc[1] + sy0 - y0
        # điểm tại vị trí chiếu (không dịch) để biết căn chỉnh thô tốt đến đâu
        zx, zy = x0 - sx0, y0 - sy0
        z = float(res[zy, zx]) if 0 <= zy < res.shape[0] and 0 <= zx < res.shape[1] else float("nan")
        out.append({"dx": int(dx), "dy": int(dy), "score": round(float(mx), 3), "score_at_zero": round(z, 3)})
    return out


# ----------------------------------------------------------------------------- main per page

def process(idx: int, pdf_page: int, doc, data_dir: Path, out: Path):
    t0 = time.time()
    esp = imread(data_dir / "ảnh esp" / f"{idx:06d}.jpg")
    scan = imread(data_dir / "ảnh scan" / f"Adobe Scan 9 Nov 2026 (1)_Page{idx}.png")
    render, lines = render_page(doc, pdf_page)
    rh, rw = render.shape[:2]
    imwrite(out / "render" / f"{idx:06d}.png", render)

    esp_f, scan_f, render_f = flat_gray(esp), flat_gray(scan), flat_gray(render)
    rec = {"image": f"{idx:06d}.jpg", "pdf_page": pdf_page, "esp_size": list(esp.shape[1::-1]),
           "scan_size": list(scan.shape[1::-1]), "render_size": [rw, rh], "n_pdf_lines": len(lines)}

    # scan → render
    H_sr, st = sift_homography(scan_f, render_f)
    rec["scan_to_render_sift"] = st
    if not homography_sane(H_sr, scan.shape, render.shape):
        rec["status"] = "FAIL_SCAN_RENDER"
        return rec, None
    ncc0 = warp_ncc(scan_f, render_f, H_sr)
    H2, cc = ecc_refine(scan_f, render_f, H_sr)
    ncc1 = warp_ncc(scan_f, render_f, H2) if cc is not None else -1
    if ncc1 > ncc0:
        H_sr = H2
    rec["scan_to_render_ncc"] = round(max(ncc0, ncc1), 4)

    # ESP32 → scan: chỉ lấy đặc trưng trong vùng trang (bỏ nền tối nhiễu)
    quad = detect_page(esp)
    mask = None
    if quad is not None:
        mask = np.zeros(esp.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.int32(quad)], 255)
        mask = cv2.dilate(mask, np.ones((41, 41), np.uint8))
    H_es, st = sift_homography(esp_f, scan_f, mask)
    rec["esp_to_scan_sift"] = st
    method = "sift"
    if not homography_sane(H_es, esp.shape, scan.shape) and quad is not None:
        # fallback: góc trang → góc scan
        sh, sw = scan.shape[:2]
        H_es = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]))
        method = "quad"
    if not homography_sane(H_es, esp.shape, scan.shape):
        rec["status"] = "FAIL_ESP_SCAN"
        return rec, None
    ncc0 = warp_ncc(esp_f, scan_f, H_es)
    H2, cc = ecc_refine(esp_f, scan_f, H_es)
    ncc1 = warp_ncc(esp_f, scan_f, H2) if cc is not None else -1
    if ncc1 > ncc0:
        H_es, method = H2, method + "+ecc"
    rec["esp_to_scan_method"] = method
    rec["esp_to_scan_ncc"] = round(max(ncc0, ncc1), 4)

    # ESP32 → render
    H_er = H_sr @ H_es
    H_er /= H_er[2, 2]
    # tinh chỉnh trực tiếp ESP32 → render (bỏ sai số tích lũy qua scan)
    ncc0 = warp_ncc(esp_f, render_f, H_er)
    H2, cc = ecc_refine(esp_f, render_f, H_er)
    ncc1 = warp_ncc(esp_f, render_f, H2) if cc is not None else -1
    if ncc1 > ncc0:
        H_er = H2
    rec["esp_to_render_ncc"] = round(max(ncc0, ncc1), 4)
    rec["H_scan_to_render"] = np.round(H_sr, 8).tolist()
    rec["H_esp_to_scan"] = np.round(H_es, 8).tolist()
    rec["H_esp_to_render"] = np.round(H_er, 8).tolist()

    esp_rect = cv2.warpPerspective(esp, H_er, (rw, rh), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full(esp.shape[:2], 255, np.uint8), H_er, (rw, rh), flags=cv2.INTER_NEAREST)
    imwrite(out / "rectified_esp" / f"{idx:06d}.png", esp_rect)
    scan_rect = cv2.warpPerspective(scan, H_sr, (rw, rh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    imwrite(out / "rectified_scan" / f"{idx:06d}.png", scan_rect)

    # dòng chữ: kiểm tra độ khớp, chiếu sang ảnh ESP32 gốc
    esp_rect_f = flat_gray(esp_rect)
    resid = line_residuals(esp_rect_f, render_f, lines)
    H_re = np.linalg.inv(H_er)
    for ln, r in zip(lines, resid):
        x0, y0, x1, y1 = ln["bbox"]
        q = cv2.perspectiveTransform(np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]), H_re)[0]
        ln["esp_quad"] = np.round(q, 1).tolist()
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        ln["in_esp_frame"] = bool(0 <= cy < rh and 0 <= cx < rw and valid[cy, cx] > 0
                                  and valid[int(min(y1, rh - 1)), int(min(x1, rw - 1))] > 0
                                  and valid[int(max(y0, 0)), int(max(x0, 0))] > 0)
        ln["match"] = r
    (out / "lines").mkdir(parents=True, exist_ok=True)
    (out / "lines" / f"{idx:06d}.json").write_text(json.dumps(
        {"image": rec["image"], "pdf_page": pdf_page, "frame": "render", "render_scale": RENDER_SCALE,
         "lines": lines}, ensure_ascii=False, indent=1), encoding="utf-8")

    m = [ln["match"] for ln in lines if ln["match"] and ln["in_esp_frame"]]
    if m:
        shift = np.hypot([x["dx"] for x in m], [x["dy"] for x in m])
        rec["lines_checked"] = len(m)
        rec["line_shift_px_median"] = round(float(np.median(shift)), 2)
        rec["line_shift_px_p90"] = round(float(np.percentile(shift, 90)), 2)
        rec["line_score_median"] = round(float(np.median([x["score"] for x in m])), 3)
        rec["lines_score_ge_0_5"] = int(sum(x["score"] >= 0.5 for x in m))
    rec["lines_in_esp_frame"] = int(sum(ln["in_esp_frame"] for ln in lines))

    # khâu 0: góc trang thật = góc ảnh scan chiếu ngược về ESP32
    sh, sw = scan.shape[:2]
    gt = cv2.perspectiveTransform(np.float32([[[0, 0], [sw, 0], [sw, sh], [0, sh]]]), np.linalg.inv(H_es))[0]
    eh, ew = esp.shape[:2]
    rec["page_gt_corners_esp"] = np.round(gt, 1).tolist()
    rec["page_fully_in_frame"] = bool(np.all((gt[:, 0] >= -2) & (gt[:, 0] <= ew + 2) & (gt[:, 1] >= -2) & (gt[:, 1] <= eh + 2)))
    if quad is not None:
        rec["page_detect_corners"] = np.round(quad, 1).tolist()
        rec["page_detect_iou"] = round(quad_iou(quad, gt, esp.shape), 4)
        rec["page_detect_corner_err_px"] = round(float(np.linalg.norm(quad - gt, axis=1).mean()), 1)
    else:
        rec["page_detect_iou"] = 0.0

    # overlay kiểm tra
    ov = esp_rect.copy()
    for ln in lines:
        if not ln["in_esp_frame"]:
            continue
        x0, y0, x1, y1 = [int(v) for v in ln["bbox"]]
        sc = ln["match"]["score"] if ln["match"] else 0
        color = (0, 200, 0) if sc >= 0.5 else (0, 165, 255) if sc >= 0.3 else (0, 0, 255)
        cv2.rectangle(ov, (x0, y0), (x1, y1), color, 1)
    imwrite(out / "overlays" / f"{idx:06d}.jpg", ov, 85)
    eo = esp.copy()
    cv2.polylines(eo, [np.int32(gt)], True, (0, 255, 0), 3)
    if quad is not None:
        cv2.polylines(eo, [np.int32(quad)], True, (0, 0, 255), 2)
    imwrite(out / "page_detect" / f"{idx:06d}.jpg", cv2.resize(eo, None, fx=0.5, fy=0.5), 80)

    rec["status"] = "OK"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec, (esp_rect, scan_rect, ov)


def contact_sheet(out: Path, items: list[tuple[int, np.ndarray]], name: str, cols=9, tw=200):
    if not items:
        return
    th = int(tw * 1616 / 1179)
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * (th + 22), cols * tw, 3), 255, np.uint8)
    for k, (idx, img) in enumerate(items):
        r, c = divmod(k, cols)
        t = cv2.resize(img, (tw - 4, th), interpolation=cv2.INTER_AREA)
        y, x = r * (th + 22) + 20, c * tw + 2
        sheet[y:y + th, x:x + tw - 4] = t
        cv2.putText(sheet, str(idx), (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    imwrite(out / name, sheet, 85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    ap.add_argument("--out", type=Path, default=paths.ESP32_ALIGN)
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(args.mapping)
    doc = pdfium.PdfDocument(str(args.pdf))
    ids = args.only or sorted(mapping)
    records, ovs, rects = [], [], []
    for idx in ids:
        try:
            rec, imgs = process(idx, mapping[idx], doc, args.data, args.out)
        except Exception as e:  # ghi lỗi, chạy tiếp các trang khác
            rec, imgs = {"image": f"{idx:06d}.jpg", "status": f"ERROR {type(e).__name__}: {e}"}, None
        records.append(rec)
        if imgs:
            rects.append((idx, imgs[0])); ovs.append((idx, imgs[2]))
        print(idx, rec.get("status"), rec.get("esp_to_scan_method"), rec.get("esp_to_render_ncc"),
              rec.get("line_shift_px_median"), rec.get("line_score_median"), rec.get("page_detect_iou"), flush=True)
    (args.out / "alignment.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.only:
        contact_sheet(args.out, rects, "contact_rectified_esp.jpg")
        contact_sheet(args.out, ovs, "contact_overlays.jpg")


if __name__ == "__main__":
    main()
