"""Chạy GCDRNet (trọng số tác giả, suy luận zero-shot) trên một thư mục ảnh, cùng cấu trúc <set>/inp/*.

  .venv-ocr/Scripts/python.exe -X utf8 src/run_gcdrnet.py --set data/external/docaligner \
      --out outputs/gcdrnet/docaligner --weights <thư mục có gcnet/ và drnet/>
Rồi chấm bằng:  eval_sets.py --method pred --pred-root outputs/gcdrnet/docaligner
Mã GCDRNet (MIT) để ở third_party/GCDRNet-main; trọng số tải từ OneDrive của tác giả.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "GCDRNet-main"


def load(weights: Path, device):
    cwd = os.getcwd()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(1, str(REPO / "models" / "UNeXt"))
    from utils import convert_state_dict
    from models.UNeXt.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L
    os.chdir(cwd)
    m1 = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=512).to(device)
    m1.load_state_dict(convert_state_dict(torch.load(weights / "gcnet/checkpoint.pkl", map_location=device)["model_state"]))
    m2 = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512).to(device)
    m2.load_state_dict(convert_state_dict(torch.load(weights / "drnet/checkpoint.pkl", map_location=device)["model_state"]))
    return m1.eval(), m2.eval()


def stride_integral(img, stride=32):
    h, w = img.shape[:2]
    ph = (stride - h % stride) % stride
    pw = (stride - w % stride) % stride
    out = cv2.copyMakeBorder(img, ph, 0, pw, 0, cv2.BORDER_REFLECT) if (ph or pw) else img
    return out, ph, pw


@torch.no_grad()
def enhance(m1, m2, bgr, device):
    im, ph, pw = stride_integral(bgr)
    h, w = im.shape[:2]
    x = torch.from_numpy(im.transpose(2, 0, 1) / 255).unsqueeze(0).float().to(device)  # BGR như infer.py gốc
    shadow = F.interpolate(m1(x), (h, w))
    y = torch.clamp(x / shadow, 0, 1)
    pred = m2(torch.cat((x, y), 1))[0]
    out = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return out[ph:, pw:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m1, m2 = load(args.weights, device)
    files = sorted(p for g in [args.set] + [d for d in args.set.rglob("*") if d.is_dir()]
                   if g.name == "inp" or g == args.set for p in [])
    files = sorted(args.set.rglob("inp/*"))
    files = [p for p in files if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if args.limit:
        files = files[:args.limit]
    t0 = time.perf_counter()
    for i, p in enumerate(files):
        dst = args.out / p.relative_to(args.set).parent.parent / p.name  # bỏ cấp "inp" để eval_sets --pred-root đọc được
        if dst.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        out = enhance(m1, m2, img, device)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(dst.suffix or ".png", out)[1].tofile(str(dst))
        if i % 10 == 0:
            print(i, len(files), p.name, f"{(time.perf_counter() - t0) / (i + 1):.1f}s/ảnh", flush=True)
    print("xong", len(files), "ảnh,", round(time.perf_counter() - t0), "s")


if __name__ == "__main__":
    main()
