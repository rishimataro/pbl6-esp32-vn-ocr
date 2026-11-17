"""Train model nâng chất lượng ảnh tài liệu (NAFNet hoặc U-Net) từ trọng số ngẫu nhiên. Chạy trên Colab GPU.

  python train_restore.py --data /content/realdae /content/esp32 --synth 0.5 \
      --out /content/drive/MyDrive/PBL6/checkpoints/m4_nafnet_all --iters 8000 --eval-every 500

Mỗi thư mục trong --data là một nguồn, theo một trong ba dạng:
  <root>/patches/{train,val}/{esp,scan}/*.png   cặp patch ESP32 → scan (data/processed/esp32_pairs)
  <root>/{train,val}/{inp,gt}/*.png             cặp ảnh vào → ảnh chuẩn, kích thước tùy ý (prep_realdae.py)
  <root>/{train,val}/gt/*.png                   chỉ ảnh sạch: ảnh vào luôn sinh bằng degrade.py
Khi train, mỗi mẫu chọn nguồn theo --data-weights (mặc định đều nhau), cắt ngẫu nhiên --crop px.
Với xác suất --synth, ảnh vào được thay bằng degrade(ảnh chuẩn) với cấu hình suy biến ngẫu nhiên.

Tùy chọn thêm từ M5 (sửa các lỗi thấy khi đánh giá M4):
  --edge-w      loss L1 trên gradient Sobel của ảnh xám: giữ nét mảnh, dấu tiếng Việt
  --paper-color xác suất nhuộm ảnh chuẩn (luồng tổng hợp) bằng màu giấy nhạt ngẫu nhiên: học giữ nền màu thật
  --identity-p  xác suất (trong luồng tổng hợp) đưa ảnh sạch vào và đòi ảnh ra y hệt: không làm hỏng ảnh đã tốt

Đầu ra: last.pt, best.pt (theo PSNR trung bình trên val thật của các nguồn), log.jsonl, config.json.
Colab ngắt giữa chừng: chạy lại đúng lệnh, thêm --resume (nạp last.pt của chính run này).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade, level_config, random_config  # noqa: E402
from restore_model import build_model, psnr, ssim  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg")


def _read_rgb(p: Path) -> np.ndarray:
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(p)
    return np.ascontiguousarray(img[..., ::-1])


def _list(d: Path):
    return sorted(n for n in os.listdir(d) if n.lower().endswith(IMG_EXT))


class Pairs:
    """Danh sách cặp (inp hoặc None, gt). Nếu mọi ảnh cùng kích thước thì gộp thành một mảng liền
    (gửi sang worker DataLoader nhanh và nhẹ hơn nhiều so với hàng nghìn mảng nhỏ)."""

    def __init__(self, items):
        same = items and all(g.shape == items[0][1].shape for _, g in items)
        has_inp = items and all(i is not None for i, _ in items)
        if same and has_inp:
            self.inp, self.gt = np.stack([i for i, _ in items]), np.stack([g for _, g in items])
            self.items = None
        elif same:
            self.inp, self.gt, self.items = None, np.stack([g for _, g in items]), None
        else:
            self.inp = self.gt = None
            self.items = items

    def __len__(self):
        return len(self.items) if self.items is not None else len(self.gt)

    def __getitem__(self, k):
        if self.items is not None:
            return self.items[k]
        return (None if self.inp is None else self.inp[k]), self.gt[k]

    def __iter__(self):
        return (self[k] for k in range(len(self)))


def load_pairs(root: Path, split: str):
    """Trả về Pairs (inp hoặc None, gt), ảnh uint8 RGB. None nếu nguồn không có split này."""
    if (root / "patches" / split / "esp").is_dir():
        a, b = root / "patches" / split / "esp", root / "patches" / split / "scan"
    elif (root / split / "gt").is_dir():
        a, b = root / split / "inp", root / split / "gt"
    else:
        return None
    names = _list(b)
    has_inp = a.is_dir()
    pairs = []
    for n in names:
        gt = _read_rgb(b / n)
        inp = _read_rgb(a / n) if has_inp else None
        if inp is not None and inp.shape != gt.shape:
            raise ValueError(f"{a / n} và {b / n} khác kích thước: {inp.shape} vs {gt.shape}")
        pairs.append((inp, gt))
    return Pairs(pairs)


class PairStream(IterableDataset):
    """Luồng mẫu vô hạn. Mỗi worker có rng riêng, tách theo seed, worker id và vòng bắt đầu (để --resume không lặp)."""

    def __init__(self, sources, weights, crop, synth, seed, start_iter=0, paper_color=0.0, identity_p=0.0):
        self.sources, self.crop, self.synth, self.seed, self.start = sources, crop, synth, seed, start_iter
        self.paper_color, self.identity_p = paper_color, identity_p
        w = np.asarray(weights, np.float64)
        self.w = w / w.sum()

    def _crop(self, inp, gt, rng):
        c = self.crop
        h, w = gt.shape[:2]
        if h < c or w < c:  # ảnh nhỏ hơn crop: pad phản chiếu
            ph, pw = max(0, c - h), max(0, c - w)
            gt = np.pad(gt, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            inp = None if inp is None else np.pad(inp, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            h, w = gt.shape[:2]
        y, x = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
        gt = gt[y:y + c, x:x + c]
        inp = None if inp is None else inp[y:y + c, x:x + c]
        if rng.random() < 0.5:
            gt = gt[:, ::-1]
            inp = None if inp is None else inp[:, ::-1]
        if rng.random() < 0.5:
            gt = gt[::-1]
            inp = None if inp is None else inp[::-1]
        return inp, np.ascontiguousarray(gt)

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0, self.start])
        while True:
            pairs = self.sources[rng.choice(len(self.sources), p=self.w)]
            inp, gt = pairs[rng.integers(len(pairs))]
            inp, gt = self._crop(inp, gt, rng)
            if inp is None or rng.random() < self.synth:
                if self.paper_color and rng.random() < self.paper_color:
                    gt = tint_paper(gt, rng)
                if self.identity_p and rng.random() < self.identity_p:
                    a = gt.astype(np.float32) / 255
                else:
                    a = degrade(gt, random_config(rng), rng).astype(np.float32) / 255
            else:  # ảnh thật: jitter nhẹ độ sáng/gamma như bản v1
                a = np.ascontiguousarray(inp).astype(np.float32) / 255
                a = np.clip(np.clip(a, 1e-4, 1) ** math.exp(rng.uniform(-0.15, 0.15)) * rng.uniform(0.85, 1.15), 0, 1)
            yield (torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1),
                   torch.from_numpy(gt).permute(2, 0, 1).float().div(255))


def tint_paper(gt: np.ndarray, rng) -> np.ndarray:
    """Nhân ảnh chuẩn với một màu giấy nhạt (hue ngẫu nhiên, độ bão hòa 0,05–0,30, độ sáng 0,85–1):
    nền trắng thành nền màu, chữ đen vẫn đen. Ảnh chuẩn giữ màu này nên model học không tẩy nền màu."""
    hsv = np.uint8([[[rng.integers(0, 180), int(255 * rng.uniform(0.05, 0.30)), int(255 * rng.uniform(0.85, 1.0))]]])
    c = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32) / 255
    return np.clip(gt.astype(np.float32) * c + 0.5, 0, 255).astype(np.uint8)


_SOBEL = None


def edge_loss(y, b):
    """L1 giữa gradient Sobel (x, y) của ảnh xám ra và ảnh chuẩn."""
    global _SOBEL
    if _SOBEL is None or _SOBEL.device != y.device:
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4
        _SOBEL = torch.stack([kx, kx.t()])[:, None].to(y.device)
    w = torch.tensor([0.299, 0.587, 0.114], device=y.device)[None, :, None, None]
    gy, gb = (y * w).sum(1, keepdim=True), (b * w).sum(1, keepdim=True)
    return F.l1_loss(F.conv2d(gy, _SOBEL, padding=1), F.conv2d(gb, _SOBEL, padding=1))


def make_synth_val(val_sets, n, size, seed):
    """Tập val tổng hợp cố định: n patch từ ảnh chuẩn của các nguồn, mỗi patch một cấu hình 'mix' mức ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    gts = [gt for pairs in val_sets.values() for _, gt in pairs if min(gt.shape[:2]) >= size]
    if not gts:
        return []
    out = []
    for _ in range(n):
        gt = gts[rng.integers(len(gts))]
        y, x = rng.integers(0, gt.shape[0] - size + 1), rng.integers(0, gt.shape[1] - size + 1)
        g = np.ascontiguousarray(gt[y:y + size, x:x + size])
        out.append((degrade(g, level_config("mix", int(rng.integers(1, 4)), rng), rng), g))
    return out


@torch.no_grad()
def evaluate(model, pairs, device, amp_dtype, bs=16):
    """PSNR/SSIM trung bình theo ảnh. Gom các ảnh cùng kích thước thành batch."""
    model.eval()
    groups = {}
    for inp, gt in pairs:
        groups.setdefault(inp.shape, []).append((inp, gt))
    ps, ss, ps_in, ss_in = [], [], [], []
    for items in groups.values():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            a = torch.from_numpy(np.stack([c[0] for c in chunk])).permute(0, 3, 1, 2).float().div(255).to(device)
            b = torch.from_numpy(np.stack([c[1] for c in chunk])).permute(0, 3, 1, 2).float().div(255).to(device)
            with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
                y = model(a)
            y = y.float().clamp(0, 1)
            for j in range(len(a)):
                ps.append(psnr(y[j:j + 1], b[j:j + 1]).item()); ss.append(ssim(y[j:j + 1], b[j:j + 1]).item())
                ps_in.append(psnr(a[j:j + 1], b[j:j + 1]).item()); ss_in.append(ssim(a[j:j + 1], b[j:j + 1]).item())
    model.train()
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
            "psnr_input": float(np.mean(ps_in)), "ssim_input": float(np.mean(ss_in)), "n": len(ps)}


class OCRFeatureLoss(nn.Module):
    """L1 trên feature 3 khối đầu của CNN VGG19-bn trong VietOCR (đóng băng): giữ nét chữ/dấu. Chỉ dùng cho ablation."""

    def __init__(self, weights_path: str):
        super().__init__()
        import torchvision
        feats = torchvision.models.vgg19_bn().features[:27]
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        sub = {k[len("cnn.model.features."):]: v for k, v in sd.items() if k.startswith("cnn.model.features.")}
        sub = {k: v for k, v in sub.items() if int(k.split(".")[0]) < 27}
        missing = feats.load_state_dict(sub, strict=False)
        assert not missing.missing_keys, missing.missing_keys
        self.f = feats.eval()
        for p in self.f.parameters():
            p.requires_grad_(False)

    def forward(self, x, y):
        loss, a, b = 0.0, x, y
        for i, layer in enumerate(self.f):
            a, b = layer(a), layer(b)
            if i in (5, 12, 25):  # relu1_2, relu2_2, relu3_4
                loss = loss + F.l1_loss(a, b)
        return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="một hoặc nhiều nguồn dữ liệu")
    ap.add_argument("--data-weights", type=float, nargs="*", default=None, help="tỉ lệ lấy mẫu mỗi nguồn")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", choices=["nafnet", "unet"], default="nafnet")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ssim-w", type=float, default=0.2)
    ap.add_argument("--synth", type=float, default=0.0, help="xác suất thay ảnh vào bằng ảnh suy biến tổng hợp")
    ap.add_argument("--synth-val", type=int, default=200, help="số patch val tổng hợp (0 = bỏ)")
    ap.add_argument("--edge-w", type=float, default=0.0, help="trọng số loss gradient Sobel (M5)")
    ap.add_argument("--paper-color", type=float, default=0.0, help="xác suất nhuộm màu giấy ảnh chuẩn (M5)")
    ap.add_argument("--identity-p", type=float, default=0.0, help="xác suất mẫu ảnh sạch → ảnh sạch (M5)")
    ap.add_argument("--ocr-loss", type=float, default=0.0)
    ap.add_argument("--vietocr-weights", default="")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    # Windows dùng spawn: worker phải nhận bản sao dữ liệu, chậm và từng lỗi → mặc định 0 trên Windows
    ap.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else min(8, os.cpu_count() or 1))
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp] if device == "cuda" else None
    if amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("GPU này không hỗ trợ bf16 bằng phần cứng → chuyển sang fp16", flush=True)
        amp_dtype = torch.float16

    # dữ liệu
    names = [p.name or str(p) for p in args.data]
    train_sets, val_sets = [], {}
    for name, root in zip(names, args.data):
        tr = load_pairs(root, "train")
        if not tr:
            raise SystemExit(f"Không tìm thấy dữ liệu train trong {root}")
        train_sets.append(tr)
        va = load_pairs(root, "val")
        real = [(i, g) for i, g in (va or []) if i is not None]
        if real:
            val_sets[name] = real
        n_inp = sum(i is not None for i, _ in tr)
        print(f"[{name}] train {len(tr)} ảnh/patch ({n_inp} có ảnh vào thật), val {len(real)}", flush=True)
    weights = args.data_weights or [1.0] * len(train_sets)
    assert len(weights) == len(train_sets), "--data-weights phải có đúng một số cho mỗi nguồn"
    synth_val = make_synth_val({n: load_pairs(r, "val") or [] for n, r in zip(names, args.data)},
                               args.synth_val, 256, seed=999) if args.synth > 0 and args.synth_val else []

    # model
    model = build_model(args.arch, args.width).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1, (it + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(it, args.iters) / args.iters)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    ocr = OCRFeatureLoss(args.vietocr_weights).to(device) if args.ocr_loss > 0 else None
    start, best = 0, -1.0
    if args.resume and (args.out / "last.pt").exists():
        ck = torch.load(args.out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start, best = ck["iter"], ck.get("best", -1.0)
        print(f"resume từ vòng {start}, best {best:.3f}", flush=True)
    cfg = {**{k: str(v) for k, v in vars(args).items()}, "params": nparam, "device": device,
           "amp_dtype": str(amp_dtype), "gpu": torch.cuda.get_device_name() if device == "cuda" else ""}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"{args.arch} width={args.width} params={nparam / 1e6:.2f}M device={device} amp={amp_dtype}", flush=True)

    stream = PairStream(train_sets, weights, args.crop, args.synth, args.seed, start,
                        paper_color=args.paper_color, identity_p=args.identity_p)
    loader = DataLoader(stream, batch_size=args.bs, num_workers=args.num_workers,
                        pin_memory=device == "cuda", persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    log = open(args.out / "log.jsonl", "a")
    t0, run = time.time(), {"l1": 0.0, "ssim": 0.0, "edge": 0.0, "ocr": 0.0, "n": 0, "data_wait": 0.0}
    model.train()
    it_loader = iter(loader)
    for it in range(start, args.iters):
        tw = time.time()
        a, b = next(it_loader)
        a, b = a.to(device, non_blocking=True), b.to(device, non_blocking=True)
        run["data_wait"] += time.time() - tw
        with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            y = model(a)
        y = y.float()
        l1 = F.l1_loss(y, b)
        ls = 1 - ssim(y.clamp(0, 1), b)
        loss = l1 + args.ssim_w * ls
        le = edge_loss(y, b) if args.edge_w > 0 else torch.zeros((), device=device)
        loss = loss + args.edge_w * le
        lo = torch.zeros((), device=device)
        if ocr is not None:
            lo = ocr(y.clamp(0, 1), b)
            loss = loss + args.ocr_loss * lo
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(opt); scaler.update(); sched.step()
        run["l1"] += l1.item(); run["ssim"] += ls.item(); run["edge"] += le.item(); run["ocr"] += lo.item(); run["n"] += 1
        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
            evs = {n: evaluate(model, p, device, amp_dtype) for n, p in val_sets.items()}
            if synth_val:
                evs["synth"] = evaluate(model, synth_val, device, amp_dtype)
            real = [evs[n]["psnr"] for n in val_sets] or [evs["synth"]["psnr"]]
            score = float(np.mean(real))
            dt = time.time() - t0  # thời gian train của đoạn này (gồm cả eval vừa chạy)
            rec = {"iter": it + 1, "lr": sched.get_last_lr()[0], "sec": round(dt, 1),
                   "it_per_s": round(run["n"] / max(dt, 1e-6), 2),
                   "data_wait_frac": round(run["data_wait"] / max(dt, 1e-6), 3),
                   **{f"train_{k}": round(v / run["n"], 5) for k, v in run.items() if k not in ("n", "data_wait")},
                   "val_score": round(score, 4),
                   **{f"val_{n}_{k}": round(v, 4) for n, ev in evs.items() for k, v in ev.items() if k != "n"}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            run = {"l1": 0.0, "ssim": 0.0, "edge": 0.0, "ocr": 0.0, "n": 0, "data_wait": 0.0}
            t0 = time.time()
            meta = {"arch": args.arch, "width": args.width}
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), **meta, "iter": it + 1, "val": evs, "val_score": score},
                           args.out / "best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "iter": it + 1, "best": best, **meta}, args.out / "last.pt")
    print("done best val score (PSNR)", round(best, 4), flush=True)


if __name__ == "__main__":
    main()
