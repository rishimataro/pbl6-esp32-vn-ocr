"""N3: áp model 1b lên trang ESP32 đã nắn, đo PSNR/SSIM so với scan, cắt crop dòng để đo CER.

  python restore_eval.py --data /content/n2 --ckpt .../best.pt --variant esp_1b [--input esp_raw]

Tạo: <data>/pages/<variant>/*.png, <data>/crops/<variant>/*.png, <data>/labels/{split}_<variant>.txt,
     <data>/eval/restore_<variant>.json (PSNR/SSIM theo trang, so với ảnh vào).
Sau đó đo CER: python n2_eval.py --data <data> --skip-labels --variants <variant> --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restore_model import load_model, psnr, ssim  # noqa: E402


def crop(img, bbox, dx=0, dy=0):
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    py, px = 0.2 * h, 0.25 * h
    H, W = img.shape[:2]
    a, b = int(max(x0 + dx - px, 0)), int(min(x1 + dx + px, W))
    c, d = int(max(y0 + dy - py, 0)), int(min(y1 + dy + py, H))
    return img[c:d, a:b]


def to_t(img, device):
    return torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--variant", default="esp_1b")
    ap.add_argument("--input", default="esp_raw", help="biến thể trang làm đầu vào (esp_raw hoặc esp_1a)")
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    lines = [json.loads(l) for l in (args.data / "lines.jsonl").read_text("utf-8").splitlines()]
    pages = sorted({l["page"] for l in lines if l["split"] in args.splits})
    out_pages = args.data / "pages" / args.variant
    out_pages.mkdir(parents=True, exist_ok=True)
    per_page = {}
    for idx in pages:
        src = cv2.imread(str(args.data / "pages" / args.input / f"{idx:06d}.png"))
        scan = cv2.imread(str(args.data / "pages" / "scan" / f"{idx:06d}.png"))
        valid = (src.max(2) > 0).astype(np.uint8)
        # lấp lỗ do chữ đen (giá trị 0) rồi bỏ mép khung (vùng ngoài ảnh ESP32 = 0)
        valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        valid = cv2.erode(valid, np.ones((31, 31), np.uint8))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            y = model(to_t(src, device))
        y = (y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255 + 0.5).astype(np.uint8)
        y = np.ascontiguousarray(y)
        cv2.imwrite(str(out_pages / f"{idx:06d}.png"), y)
        ys, xs = np.where(valid > 0)
        r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1  # khung chữ nhật trong vùng hợp lệ
        t_s = to_t(scan[r0:r1, c0:c1], device)
        per_page[idx] = {
            "psnr": round(psnr(to_t(y[r0:r1, c0:c1], device), t_s).item(), 3),
            "ssim": round(ssim(to_t(y[r0:r1, c0:c1], device), t_s).item(), 4),
            "psnr_input": round(psnr(to_t(src[r0:r1, c0:c1], device), t_s).item(), 3),
            "ssim_input": round(ssim(to_t(src[r0:r1, c0:c1], device), t_s).item(), 4)}
        print(idx, per_page[idx], flush=True)
    rows = {s: [] for s in args.splits}
    (args.data / "crops" / args.variant).mkdir(parents=True, exist_ok=True)
    cache = {}
    for l in lines:
        if l["split"] not in args.splits:
            continue
        if l["page"] not in cache:
            cache[l["page"]] = cv2.imread(str(out_pages / f"{l['page']:06d}.png"))
        rel = f"crops/{args.variant}/{l['id']}.png"
        cv2.imwrite(str(args.data / rel), crop(cache[l["page"]], l["bbox_render"], l["dx"], l["dy"]))
        rows[l["split"]].append(f"{rel}\t{l['text']}")
    for s, r in rows.items():
        (args.data / "labels" / f"{s}_{args.variant}.txt").write_text("\n".join(r) + "\n", encoding="utf-8")
    agg = {k: round(float(np.mean([v[k] for v in per_page.values()])), 4) for k in next(iter(per_page.values()))}
    (args.data / "eval").mkdir(exist_ok=True)
    (args.data / "eval" / f"restore_{args.variant}.json").write_text(
        json.dumps({"ckpt": str(args.ckpt), "mean": agg, "pages": per_page}, indent=1), encoding="utf-8")
    print("mean", agg)


if __name__ == "__main__":
    main()
