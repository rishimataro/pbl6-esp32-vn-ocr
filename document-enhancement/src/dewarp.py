"""Làm phẳng ảnh tài liệu (bước hình học, chạy TRƯỚC model nâng chất lượng).

Hai cách:
  - "corners": tìm 4 góc trang rồi biến đổi phối cảnh (OpenCV). Chỉ sửa chụp nghiêng, không sửa giấy cong.
  - "uvdoc":   UVDoc (Verhoeven et al., SIGGRAPH Asia 2023), model đã train sẵn, dự đoán lưới 2D 45×31
               rồi kéo ảnh về phẳng. Sửa được cả phối cảnh lẫn giấy cong/nhăn. Không train lại.

Chạy:
  .venv-ocr/Scripts/python.exe -X utf8 src/dewarp.py --inp <ảnh hoặc thư mục> --out <thư mục> \
      --method uvdoc [--enhance models/retrain/m4_nafnet_all/best.pt] [--long-side 1024]
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

UVDOC_CKPT = paths.MODELS / "dewarp/uvdoc_best_model.pkl"
UVDOC_IN = (488, 712)  # (rộng, cao) ảnh vào của UVDoc


@lru_cache(maxsize=2)
def load_uvdoc(ckpt=str(UVDOC_CKPT), device="cpu"):
    from uvdoc_model import UVDocnet
    model = UVDocnet(num_filter=32, kernel_size=5)
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def dewarp_uvdoc(img_bgr: np.ndarray, device="cpu", model=None) -> np.ndarray:
    """Ảnh ra cùng kích thước ảnh vào (như demo gốc của UVDoc)."""
    model = model or load_uvdoc(device=device)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(rgb, UVDOC_IN, interpolation=cv2.INTER_AREA).transpose(2, 0, 1))[None]
    grid2d, _ = model(inp.to(device))                                     # 1×2×45×31, toạ độ [-1, 1]
    h, w = rgb.shape[:2]
    grid = F.interpolate(grid2d, size=(h, w), mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
    full = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(device)
    out = F.grid_sample(full, grid, align_corners=True)[0].permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((out * 255 + 0.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1)[:, 0]
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])  # tl, tr, br, bl


def find_page_quad(img_bgr: np.ndarray, min_area=0.2):
    """Tìm tứ giác trang lớn nhất. Trả về 4 điểm (tl, tr, br, bl) theo toạ độ ảnh gốc, hoặc None."""
    H, W = img_bgr.shape[:2]
    s = 800 / max(H, W)
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr.copy()
    s = min(s, 1.0)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    area = gray.shape[0] * gray.shape[1]
    masks = [cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((5, 5), np.uint8)),
             cv2.morphologyEx(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                              cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))]
    best = None
    for m in masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.02, 0.03, 0.05):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    a = cv2.contourArea(ap)
                    if a > min_area * area and a < 0.98 * area and (best is None or a > best[0]):
                        best = (a, ap.reshape(4, 2).astype(np.float32) / s)
                    break
    return None if best is None else _order(best[1])


def dewarp_corners(img_bgr: np.ndarray):
    """Trả về (ảnh, đã_tìm_thấy_trang). Không tìm thấy thì giữ nguyên ảnh."""
    q = find_page_quad(img_bgr)
    if q is None:
        return img_bgr, False
    tl, tr, br, bl = q
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    return cv2.warpPerspective(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), True


def dewarp_auto(img_bgr: np.ndarray, device="cpu", max_page_frac=0.6):
    """Ảnh còn nhiều nền (trang < 60 % khung) thì cắt 4 góc trước, rồi luôn chạy UVDoc.
    Ảnh đã cắt sát trang thì đưa thẳng vào UVDoc (cắt 4 góc trên trang cong dễ bắt nhầm mép).
    Trả về (ảnh, đã_cắt_4_góc)."""
    q = find_page_quad(img_bgr)
    cropped = False
    if q is not None and cv2.contourArea(q) < max_page_frac * img_bgr.shape[0] * img_bgr.shape[1]:
        img_bgr, cropped = dewarp_corners(img_bgr)
    return dewarp_uvdoc(img_bgr, device), cropped


def dewarp(img_bgr: np.ndarray, method="uvdoc", device="cpu") -> np.ndarray:
    if method in ("none", None):
        return img_bgr
    if method == "corners":
        return dewarp_corners(img_bgr)[0]
    if method == "uvdoc":
        return dewarp_uvdoc(img_bgr, device)
    if method == "auto":
        return dewarp_auto(img_bgr, device)[0]
    raise ValueError(f"cách làm phẳng không hỗ trợ: {method}")


def resize_long(img, long_side):
    h, w = img.shape[:2]
    if not long_side or max(h, w) <= long_side:
        return img
    s = long_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True, help="một ảnh hoặc thư mục ảnh")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", default="uvdoc", choices=["auto", "uvdoc", "corners", "none"])
    ap.add_argument("--enhance", type=Path, help="checkpoint nâng chất lượng chạy sau bước làm phẳng (vd. M4)")
    ap.add_argument("--long-side", type=int, default=0, help="thu nhỏ ảnh vào trước khi xử lý (0 = giữ nguyên)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = [args.inp] if args.inp.is_file() else sorted(
        p for p in args.inp.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    args.out.mkdir(parents=True, exist_ok=True)
    model = None
    if args.enhance:
        from restore_model import load_model, restore_image
        model = load_model(args.enhance, device)
    for f in files:
        img = resize_long(cv2.imread(str(f)), args.long_side)
        t = time.perf_counter()
        y = dewarp(img, args.method, device)
        t1 = time.perf_counter()
        if model is not None:
            y = restore_image(model, y, device)
        cv2.imwrite(str(args.out / (f.stem + ".png")), y)
        print(f"{f.name}: làm phẳng {t1 - t:.2f}s, nâng {time.perf_counter() - t1:.2f}s", flush=True)


if __name__ == "__main__":
    main()
