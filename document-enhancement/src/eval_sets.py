"""Chấm một phương pháp nâng ảnh trên các tập {inp,gt}: PSNR, SSIM, LPIPS, thời gian/ảnh.

  # model đã train
  python eval_sets.py --set /content/realdae/test --method model --ckpt .../best.pt --out results/m4_realdae.json
  # cả cây tập tổng hợp (mỗi thư mục con có inp/ và gt/ là một nhóm, vd shading/L2)
  python eval_sets.py --set /content/synth_test --method model --ckpt .../best.pt --out results/m4_synth.json
  # đối chứng
  python eval_sets.py --set ... --method identity  --out results/m0_....json   # ảnh vào, không xử lý
  python eval_sets.py --set ... --method classic   --out results/m1_....json   # enhance.py (1a)
  python eval_sets.py --set ... --method pred --pred-root /content/gcdrnet_out --out results/m5_....json
       # ảnh đã xử lý sẵn bằng công cụ khác (vd infer.py của GCDRNet), cùng đường dẫn tương đối và tên tệp

Thêm --lpips để tính LPIPS (cần `pip install lpips`), --save-root để lưu ảnh kết quả.
PSNR tính trên RGB, SSIM trên kênh xám (cùng hàm với lúc train). Ảnh dự đoán khác kích thước ảnh chuẩn
sẽ được resize về kích thước ảnh chuẩn và đếm vào 'resized'.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restore_model import load_model, psnr, restore_image, ssim  # noqa: E402

EXT = (".png", ".jpg", ".jpeg")


def find_groups(root: Path):
    """Mọi thư mục (kể cả root) có cả inp/ và gt/ → {tên nhóm: thư mục}."""
    groups = {}
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if (d / "inp").is_dir() and (d / "gt").is_dir():
            groups[str(d.relative_to(root)).replace("\\", "/") or "."] = d
    return groups


def to_t(img_bgr, device):
    return torch.from_numpy(img_bgr[..., ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--method", choices=["model", "identity", "classic", "pred"], required=True)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--pred-root", type=Path)
    ap.add_argument("--tile", type=int, default=0,
                    help="chia ô khi suy luận (0 = cả ảnh). Chỉ dùng khi thiếu bộ nhớ, và >= 512: NAFNet có pooling "
                         "toàn cục nên ô nhỏ làm lệch kết quả (ô 512 lệch ~42 dB, ô 256 lệch ~29 dB so với cả ảnh)")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="none")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--save-root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0, help="giới hạn số ảnh mỗi nhóm (để chạy thử)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp] if device == "cuda" else None

    model, nparam = None, None
    if args.method == "model":
        assert args.ckpt, "--method model cần --ckpt"
        model = load_model(args.ckpt, device)
        nparam = sum(p.numel() for p in model.parameters())
    if args.method == "classic":
        from enhance import enhance_1a
    if args.method == "pred":
        assert args.pred_root, "--method pred cần --pred-root"
    lp = None
    if args.lpips:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    groups = find_groups(args.set)
    if not groups:
        raise SystemExit(f"Không có thư mục nào chứa inp/ và gt/ trong {args.set}")
    result = {"set": str(args.set), "method": args.method, "ckpt": str(args.ckpt or ""), "params": nparam,
              "tile": args.tile, "device": device, "groups": {}}
    for gname, gdir in groups.items():
        names = sorted(n.name for n in (gdir / "gt").iterdir() if n.suffix.lower() in EXT)
        if args.max_images:
            names = names[: args.max_images]
        rows, resized, secs = [], 0, []
        for n in names:
            gt = cv2.imread(str(gdir / "gt" / n))
            inp_path = gdir / "inp" / n
            if not inp_path.exists():  # cho phép ảnh vào khác đuôi tệp
                cand = [p for p in (gdir / "inp").glob(Path(n).stem + ".*")]
                inp_path = cand[0] if cand else inp_path
            inp = cv2.imread(str(inp_path))
            t0 = time.time()
            if args.method == "model":
                pred = restore_image(model, inp, device, tile=args.tile or None, amp=amp)
                if device == "cuda":
                    torch.cuda.synchronize()
            elif args.method == "identity":
                pred = inp
            elif args.method == "classic":
                pred = enhance_1a(inp)
            else:
                rel = Path(gname) / n if gname != "." else Path(n)
                cand = list((args.pred_root / rel.parent).glob(rel.stem + ".*"))
                if not cand:
                    raise FileNotFoundError(args.pred_root / rel)
                pred = cv2.imread(str(cand[0]))
            secs.append(time.time() - t0)
            if pred.shape[:2] != gt.shape[:2]:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
                resized += 1
            if args.save_root:
                d = args.save_root / gname
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / (Path(n).stem + ".png")), pred)
            with torch.no_grad():
                tp, tg = to_t(pred, device), to_t(gt, device)
                row = {"name": n, "psnr": round(psnr(tp, tg).item(), 4), "ssim": round(ssim(tp, tg).item(), 5)}
                if lp is not None:
                    row["lpips"] = round(lp(tp * 2 - 1, tg * 2 - 1).item(), 5)
            rows.append(row)
        keys = [k for k in rows[0] if k != "name"] if rows else []
        mean = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
        result["groups"][gname] = {"n": len(rows), "mean": mean, "sec_per_image": round(float(np.mean(secs)), 4),
                                   "resized": resized, "images": rows}
        print(f"{gname:20s} n={len(rows):4d} {mean} {np.mean(secs):.3f}s/ảnh" + (f" resized={resized}" if resized else ""),
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("→", args.out)


if __name__ == "__main__":
    main()
