"""Chuẩn bị RealDAE (hoặc bộ cặp ảnh tài liệu tương tự) về cấu trúc chung cho train_restore.py / eval_sets.py.

  python prep_realdae.py --src /content/realdae_raw --out /content/realdae --long-side 1024 --val 50 --seed 0
  python prep_realdae.py --src /content/realdae_raw --tree          # chỉ in cấu trúc thư mục để kiểm tra

Tự nhận cặp: tìm ảnh trong các thư mục có tên kiểu ảnh vào (input, distorted, in, ...) và ảnh chuẩn
(gt, target, clean, ...), ghép theo tên tệp (bỏ hậu tố _in/_gt/...). Tập train/test lấy theo tên thư mục
chứa 'train'/'test' nếu có, nếu không thì chia ngẫu nhiên theo --test-frac.
Nếu tự nhận sai: chỉ rõ bằng --inp-dirs và --gt-dirs (các thư mục theo cùng thứ tự, và --splits tương ứng).

Đầu ra: <out>/{train,val,test}/{inp,gt}/<tên>.png (cùng kích thước, cạnh dài <= --long-side), manifest.json.
Val tách từ train (không đụng test) để chọn checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
INP_WORDS = {"input", "inputs", "inp", "in", "distorted", "degraded", "source", "src", "origin", "original", "img", "images", "low"}
GT_WORDS = {"gt", "target", "targets", "label", "labels", "clean", "groundtruth", "ground_truth", "reference", "ref", "high"}
SUFFIX = re.compile(r"[_\-](in|inp|input|gt|target|label|clean|ref|out|distorted)$", re.I)


def role(path: Path, root: Path):
    """'inp' / 'gt' / None theo tên các thư mục cha (gần nhất trước) hoặc hậu tố tên tệp."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        p = part.lower()
        if p in GT_WORDS:
            return "gt"
        if p in INP_WORDS:
            return "inp"
    m = SUFFIX.search(path.stem)
    if m:
        return "gt" if m.group(1).lower() in {"gt", "target", "label", "clean", "ref"} else "inp"
    return None


def split_of(path: Path, root: Path):
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    if any("test" in p for p in parts):
        return "test"
    if any("train" in p for p in parts):
        return "train"
    return None


def key(path: Path):
    return SUFFIX.sub("", path.stem)


def print_tree(root: Path):
    c = Counter(str(p.parent.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in EXT)
    for d, n in sorted(c.items()):
        print(f"{n:6d}  {d}")


def resize_long(img, long_side):
    h, w = img.shape[:2]
    if not long_side or max(h, w) <= long_side:
        return img
    r = long_side / max(h, w)
    return cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--inp-dirs", type=Path, nargs="*")
    ap.add_argument("--gt-dirs", type=Path, nargs="*")
    ap.add_argument("--splits", nargs="*", help="split cho từng cặp --inp-dirs/--gt-dirs (train/test)")
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--val", type=int, default=50, help="số ảnh tách từ train làm val")
    ap.add_argument("--test-frac", type=float, default=0.25, help="chỉ dùng khi nguồn không chia sẵn train/test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tree or not args.out:
        print_tree(args.src)
        return

    # 1. tìm cặp
    pairs = defaultdict(dict)  # (split, key) -> {inp, gt}
    if args.inp_dirs:
        assert args.gt_dirs and len(args.gt_dirs) == len(args.inp_dirs), "--gt-dirs phải tương ứng --inp-dirs"
        splits = args.splits or [None] * len(args.inp_dirs)
        for di, dg, sp in zip(args.inp_dirs, args.gt_dirs, splits):
            for r, d in (("inp", di), ("gt", dg)):
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in EXT:
                        pairs[(sp, key(p))][r] = p
    else:
        for p in sorted(args.src.rglob("*")):
            if p.suffix.lower() in EXT:
                r = role(p, args.src)
                if r:
                    pairs[(split_of(p, args.src), key(p))][r] = p
    full = {k: v for k, v in pairs.items() if "inp" in v and "gt" in v}
    print(f"tìm thấy {len(full)} cặp đủ ảnh vào + ảnh chuẩn ({len(pairs) - len(full)} thiếu một nửa)")
    if not full:
        print("Không ghép được cặp nào. Cấu trúc thư mục:")
        print_tree(args.src)
        raise SystemExit("Chỉ rõ bằng --inp-dirs/--gt-dirs.")

    # 2. chia tập
    rng = np.random.default_rng(args.seed)
    keys = sorted(full, key=lambda k: (str(k[0]), k[1]))
    if all(k[0] is None for k in keys):
        idx = rng.permutation(len(keys))
        n_test = round(len(keys) * args.test_frac)
        assign = {keys[i]: ("test" if j < n_test else "train") for j, i in enumerate(idx)}
    else:
        assign = {k: (k[0] or "train") for k in keys}
    train_keys = [k for k in keys if assign[k] == "train"]
    for i in rng.permutation(len(train_keys))[: args.val]:
        assign[train_keys[i]] = "val"

    # 3. ghi
    manifest, resized = [], 0
    for k in keys:
        sp, name = assign[k], k[1]
        inp, gt = cv2.imread(str(full[k]["inp"])), cv2.imread(str(full[k]["gt"]))
        if inp is None or gt is None:
            print("bỏ (không đọc được):", full[k]); continue
        if inp.shape[:2] != gt.shape[:2]:
            inp = cv2.resize(inp, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
            resized += 1
        inp, gt = resize_long(inp, args.long_side), resize_long(gt, args.long_side)
        for r, img in (("inp", inp), ("gt", gt)):
            d = args.out / sp / r
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{name}.png"), img)
        manifest.append({"name": name, "split": sp, "inp": str(full[k]["inp"]), "gt": str(full[k]["gt"]),
                         "size": list(gt.shape[:2])})
    cnt = Counter(m["split"] for m in manifest)
    (args.out / "manifest.json").write_text(json.dumps(
        {"counts": cnt, "long_side": args.long_side, "seed": args.seed, "resized_inp_to_gt": resized,
         "items": manifest}, indent=1), encoding="utf-8")
    print("xong:", dict(cnt), f"(phải resize ảnh vào cho khớp ảnh chuẩn: {resized})")


if __name__ == "__main__":
    main()
