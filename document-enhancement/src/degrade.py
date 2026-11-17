"""Sinh ảnh tài liệu bị suy biến có điều khiển từ ảnh sạch.

Sáu loại suy biến, mỗi loại có cường độ s trong [0, 1]:
  shading   sáng không đều / bóng mềm (nhân với bản đồ ánh sáng mượt)
  color     ám màu (hệ số riêng cho từng kênh, thường thiên xanh như camera rẻ)
  lowlight  thiếu sáng (giảm độ sáng + gamma)
  blur      nhòe Gauss hoặc nhòe chuyển động ngắn
  noise     nhiễu cảm biến Poisson-Gauss (vùng tối nhiễu tương đối mạnh hơn)
  jpeg      nén JPEG
Thứ tự áp dụng mô phỏng chuỗi chụp ảnh: ánh sáng → màu → phơi sáng → quang học → cảm biến → nén.

Dùng trong train (sinh ngay trên patch):
  from degrade import degrade, random_config
  inp = degrade(gt_rgb_uint8, random_config(rng), rng)

Tạo tập test cố định (mỗi loại × mức, cùng seed cho mọi model):
  python degrade.py --clean /content/realdae/test/gt --out /content/synth_test \
      --types shading color lowlight blur noise jpeg mix --levels 1 2 3 --seed 123
  Kết quả: <out>/<loại>/L<mức>/{inp,gt}/<tên>.png
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

TYPES = ("shading", "color", "lowlight", "blur", "noise", "jpeg")
ORDER = TYPES  # thứ tự áp dụng
LEVEL_S = {1: 0.33, 2: 0.66, 3: 1.0}  # mức test → cường độ
# xác suất mỗi loại xuất hiện khi train (ít nhất một loại luôn được chọn)
TRAIN_P = {"shading": 0.5, "color": 0.5, "lowlight": 0.6, "blur": 0.4, "noise": 0.6, "jpeg": 0.5}


def _lerp(a, b, s):
    return a + (b - a) * s


def _shading(x, s, rng):
    h, w = x.shape[:2]
    amp = _lerp(0.15, 0.65, s)
    g = rng.random((rng.integers(2, 5), rng.integers(2, 5))).astype(np.float32)
    m = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    th = rng.uniform(0, 2 * np.pi)  # thêm gradient tuyến tính theo hướng ngẫu nhiên
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = m + rng.uniform(0.5, 1.5) * ((xx / w) * np.cos(th) + (yy / h) * np.sin(th))
    if rng.random() < 0.4:  # vệt bóng mềm: nửa mặt phẳng với mép mờ
        a = rng.uniform(0, 2 * np.pi)
        d = (xx - rng.uniform(0, w)) * np.cos(a) + (yy - rng.uniform(0, h)) * np.sin(a)
        m = m + 1.5 * (1 / (1 + np.exp(-d / (0.05 * max(h, w)))))
    m = (m - m.min()) / max(float(m.max() - m.min()), 1e-6)
    return x * (1 - amp * m)[..., None]


def _color(x, s, rng):
    d = _lerp(0.08, 0.35, s)
    if rng.random() < 0.5:  # thiên xanh: giảm đỏ, tăng lam
        gain = np.array([1 - d, 1 - d / 3, 1 + d / 2], np.float32) * rng.uniform(0.95, 1.05, 3)
    else:
        gain = 1 + rng.uniform(-d, d, 3).astype(np.float32)
    return x * gain


def _lowlight(x, s, rng):
    k = _lerp(0.6, 0.12, s) * rng.uniform(0.9, 1.1)
    gamma = _lerp(1.2, 2.2, s) * rng.uniform(0.95, 1.05)
    return np.clip(x, 0, 1) ** gamma * k


def _blur(x, s, rng):
    if rng.random() < 0.5:
        return cv2.GaussianBlur(x, (0, 0), _lerp(0.5, 2.0, s))
    n = int(round(_lerp(3, 9, s))) | 1
    k = np.zeros((n, n), np.float32)
    k[n // 2] = 1
    rot = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1)
    k = cv2.warpAffine(k, rot, (n, n))
    return cv2.filter2D(x, -1, k / max(float(k.sum()), 1e-6))


def _noise(x, s, rng):
    x = np.clip(x, 0, 1)
    a = _lerp(0.0005, 0.008, s)  # nhiễu hạt (tỉ lệ với cường độ)
    b = _lerp(3, 20, s) / 255     # nhiễu đọc (cố định)
    std = np.sqrt(a * x + b * b)
    return x + rng.standard_normal(x.shape).astype(np.float32) * std


def _jpeg(x, s, rng):
    q = int(round(_lerp(80, 25, s) + rng.uniform(-3, 3)))
    u8 = (np.clip(x, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, max(5, q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255


FNS = {"shading": _shading, "color": _color, "lowlight": _lowlight, "blur": _blur, "noise": _noise, "jpeg": _jpeg}


def random_config(rng, p=None, s_range=(0.0, 1.0)) -> dict:
    """Cấu hình ngẫu nhiên cho train: mỗi loại bật với xác suất p[loại], cường độ đều trong s_range."""
    p = p or TRAIN_P
    cfg = {t: float(rng.uniform(*s_range)) for t in TYPES if rng.random() < p[t]}
    if not cfg:
        cfg[TYPES[rng.integers(len(TYPES))]] = float(rng.uniform(*s_range))
    return cfg


def level_config(kind: str, level: int, rng) -> dict:
    """Cấu hình cho tập test: một loại ở mức cố định, hoặc 'mix' = 3 loại ngẫu nhiên cùng mức."""
    s = LEVEL_S[level]
    if kind == "mix":
        picked = rng.choice(list(TYPES), size=3, replace=False)
        return {str(t): s for t in picked}
    return {kind: s}


def degrade(img: np.ndarray, cfg: dict, rng) -> np.ndarray:
    """img: uint8 RGB HxWx3 (ảnh sạch). cfg: {loại: cường độ}. Trả về uint8 RGB."""
    x = img.astype(np.float32) / 255
    for t in ORDER:
        if t in cfg:
            x = FNS[t](x, float(cfg[t]), rng)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, nargs="+", required=True, help="thư mục ảnh sạch")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--types", nargs="+", default=list(TYPES) + ["mix"])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--long-side", type=int, default=0, help="thu nhỏ để cạnh dài bằng giá trị này (0 = giữ)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    files = []
    for d in args.clean:
        files += sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.max_images:
        files = files[: args.max_images]
    manifest = []
    for f in files:
        img = cv2.imread(str(f))[..., ::-1]
        if args.long_side and max(img.shape[:2]) > args.long_side:
            r = args.long_side / max(img.shape[:2])
            img = cv2.resize(img, (round(img.shape[1] * r), round(img.shape[0] * r)), interpolation=cv2.INTER_AREA)
        name = f"{f.parent.name}_{f.stem}.png" if len(args.clean) > 1 else f"{f.stem}.png"
        for kind in args.types:
            for lv in args.levels:
                rng = np.random.default_rng(_seed(args.seed, name, kind, lv))
                cfg = level_config(kind, lv, rng)
                d = args.out / kind / f"L{lv}"
                (d / "inp").mkdir(parents=True, exist_ok=True)
                (d / "gt").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / "gt" / name), np.ascontiguousarray(img[..., ::-1]))
                cv2.imwrite(str(d / "inp" / name), np.ascontiguousarray(degrade(img, cfg, rng)[..., ::-1]))
                manifest.append({"name": name, "type": kind, "level": lv, "cfg": cfg})
        print(f.name, flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(files)} ảnh × {len(args.types)} loại × {len(args.levels)} mức → {args.out}")


if __name__ == "__main__":
    main()
