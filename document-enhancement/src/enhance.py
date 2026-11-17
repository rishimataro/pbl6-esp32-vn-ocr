"""Khâu 1a: nâng chất lượng ảnh trang theo cách cổ điển (không học, không nhị phân hóa).

Các bước (mỗi bước bật/tắt được để làm ablation):
  1. white balance theo màu giấy (bỏ ám xanh của OV2640),
  2. làm phẳng ánh sáng: chia cho nền ước lượng (bỏ tối mép/bóng),
  3. khử nhiễu nhẹ (Non-local Means),
  4. CLAHE nhẹ trên kênh sáng,
  5. unsharp mask nhẹ.
Đầu vào/ra: ảnh BGR uint8 (trang đã nắn).
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT = dict(white_balance=True, flatten=True, denoise=True, clahe=True, sharpen=True)


def _paper_mask(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(g[g > 20], 60) if (g > 20).any() else 128
    return g >= thr


def white_balance(img: np.ndarray, target=235.0) -> np.ndarray:
    m = _paper_mask(img)
    if m.sum() < 1000:
        return img
    ref = np.array([np.percentile(img[..., c][m], 75) for c in range(3)], np.float32)
    gain = target / np.maximum(ref, 1.0)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def flatten_illumination(img: np.ndarray, target=240.0) -> np.ndarray:
    """Ước lượng nền giấy bằng closing (xóa chữ tối) + blur lớn, rồi chia."""
    h, w = img.shape[:2]
    s = 0.25
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    k = max(7, int(min(small.shape[:2]) * 0.04) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, k if k <= 255 else 255)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    out = img.astype(np.float32) / np.maximum(bg, 20.0) * target
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise(img: np.ndarray, h=4) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 5, 15)


def clahe(img: np.ndarray, clip=1.5) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen(img: np.ndarray, amount=0.6, sigma=1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def enhance_1a(img: np.ndarray, **opts) -> np.ndarray:
    o = {**DEFAULT, **opts}
    out = img
    if o["white_balance"]:
        out = white_balance(out)
    if o["flatten"]:
        out = flatten_illumination(out)
    if o["denoise"]:
        out = denoise(out)
    if o["clahe"]:
        out = clahe(out)
    if o["sharpen"]:
        out = sharpen(out)
    return out
