"""Đánh giá làm phẳng trên DocUNet benchmark (130 ảnh giấy cong/gập/nhăn).

Ảnh vào: data/external/docunet_raw/crop/<id> copy.png (crop.zip của DocUNet)
Ảnh chuẩn (scan phẳng): data/external/docaligner_raw/<id>_target.png

Chỉ số (theo tinh thần giao thức DocUNet, cài đặt lại bằng Python nên KHÔNG so trực tiếp với số công bố):
  - MS-SSIM: ảnh xám, ảnh ra và scan cùng đổi về diện tích 598 400 px theo tỉ lệ của scan,
    5 mức, trọng số (0.0448, 0.2856, 0.3001, 0.2363, 0.1333). Cao hơn = tốt hơn.
  - LD (local distortion): độ dời trung bình (pixel) của luồng quang học dày đặc từ scan sang ảnh ra.
    Bản gốc dùng SIFT Flow (MATLAB); ở đây dùng DIS optical flow của OpenCV. Thấp hơn = tốt hơn.

  .venv-ocr/Scripts/python.exe -X utf8 src/eval_docunet.py --methods none corners uvdoc corners+uvdoc
  # thêm nâng chất lượng sau làm phẳng: phương pháp "uvdoc+m4"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from dewarp import dewarp_auto, dewarp_corners, dewarp_uvdoc  # noqa: E402

AREA = 598400
MS_W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
M4_CKPT = paths.RETRAIN_CKPT / "m4_nafnet_all/best.pt"


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def _gauss(size=11, sigma=1.5):
    g = torch.exp(-((torch.arange(size) - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: ảnh xám uint8 cùng kích thước."""
    x = torch.from_numpy(a).float()[None, None] / 255
    y = torch.from_numpy(b).float()[None, None] / 255
    w = _gauss()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for i in range(len(MS_W)):
        mx, my = F.conv2d(x, w), F.conv2d(y, w)
        sxx = F.conv2d(x * x, w) - mx ** 2
        syy = F.conv2d(y * y, w) - my ** 2
        sxy = F.conv2d(x * y, w) - mx * my
        cs = ((2 * sxy + C2) / (sxx + syy + C2)).mean()
        if i == len(MS_W) - 1:
            l = ((2 * mx * my + C1) / (mx ** 2 + my ** 2 + C1)).mean()
            vals.append((l * cs).clamp(min=1e-6))
        else:
            vals.append(cs.clamp(min=1e-6))
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    return float(torch.prod(torch.stack([v ** wt for v, wt in zip(vals, MS_W)])))


def local_distortion(ref: np.ndarray, out: np.ndarray) -> float:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(ref, out, None)
    return float(np.linalg.norm(flow, axis=2).mean())


def score(out_bgr, ref_bgr):
    h, w = ref_bgr.shape[:2]
    s = (AREA / (h * w)) ** 0.5
    size = (round(w * s), round(h * s))
    r = cv2.resize(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    o = cv2.resize(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    return ms_ssim(o, r), local_distortion(r, o)


_m4 = None


def apply(img, method):
    global _m4
    out = img
    for step in method.split("+"):
        if step == "corners":
            out = dewarp_corners(out)[0]
        elif step == "uvdoc":
            out = dewarp_uvdoc(out)
        elif step == "auto":
            out = dewarp_auto(out)[0]
        elif step == "m4":
            from restore_model import load_model, restore_image
            _m4 = _m4 or load_model(M4_CKPT)
            h, w = out.shape[:2]
            s = min(1.0, 1024 / max(h, w))      # ENH-M4 được chấm ở cạnh dài 1024
            small = cv2.resize(out, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            out = restore_image(_m4, small)
        elif step != "none":
            raise ValueError(step)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, default=paths.EXTERNAL / "docunet_raw/crop")
    ap.add_argument("--gt", type=Path, default=paths.EXTERNAL / "docaligner_raw")
    ap.add_argument("--methods", nargs="*", default=["none", "corners", "uvdoc", "corners+uvdoc"])
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "dewarp/docunet.json")
    ap.add_argument("--save", type=Path, help="lưu ảnh ra (cạnh dài 1024) vào <save>/<method>/")
    ap.add_argument("--max-images", type=int)
    args = ap.parse_args()
    files = sorted(args.inp.glob("*.png"))[: args.max_images]
    prev = json.loads(args.out.read_text("utf-8")) if args.out.exists() else {"summary": {}, "images": {}}
    res = prev["images"]
    for m in args.methods:
        res[m] = {}
        t0 = time.perf_counter()
        for f in files:
            iid = f.stem.replace(" copy", "")
            img, ref = imread(f), imread(args.gt / f"{iid}_target.png")
            t = time.perf_counter()
            out = apply(img, m)
            dt = time.perf_counter() - t
            ms, ld = score(out, ref)
            res[m][iid] = {"ms_ssim": round(ms, 4), "ld": round(ld, 3), "seconds": round(dt, 2)}
            if args.save:
                d = args.save / m.replace("+", "_")
                d.mkdir(parents=True, exist_ok=True)
                h, w = out.shape[:2]
                s = min(1.0, 1024 / max(h, w))
                cv2.imencode(".jpg", cv2.resize(out, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA),
                             [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(d / f"{iid}.jpg"))
        v = list(res[m].values())
        prev["summary"][m] = {"images": len(v),
                              "ms_ssim": round(float(np.mean([x["ms_ssim"] for x in v])), 4),
                              "ld": round(float(np.mean([x["ld"] for x in v])), 3),
                              "seconds": round(float(np.mean([x["seconds"] for x in v])), 2)}
        print(m, prev["summary"][m], f"({time.perf_counter() - t0:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prev, indent=1, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
