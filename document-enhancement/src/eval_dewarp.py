"""Đo độ phẳng sau khi làm phẳng, so với bản scan (trang phẳng chuẩn).

Ý tưởng (gần với chỉ số LD của DocUNet nhưng không cần MATLAB):
  ghép điểm SIFT giữa ảnh đã làm phẳng và scan, khớp một phép đồng dạng (xoay + dịch + co giãn đều).
  Trang càng phẳng thì phép đồng dạng càng khớp, sai lệch còn lại càng nhỏ.
  Báo sai lệch trung vị và phân vị 90, tính theo % chiều rộng trang scan.

  .venv-ocr/Scripts/python.exe -X utf8 src/eval_dewarp.py --pages 3 4 7 9 11 14 25 32 38 44 \
      --methods none corners uvdoc corners+uvdoc --out reports/results/dewarp/esp32_flatness.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from dewarp import dewarp_auto, dewarp_corners, dewarp_uvdoc  # noqa: E402

TEST_PAGES = [3, 4, 7, 9, 11, 14, 25, 32, 38, 44]


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def apply(img, method):
    out = img
    for step in method.split("+"):
        if step == "corners":
            out = dewarp_corners(out)[0]
        elif step == "uvdoc":
            out = dewarp_uvdoc(out)
        elif step == "auto":
            out = dewarp_auto(out)[0]
        elif step != "none":
            raise ValueError(step)
    return out


def flatness(img, ref, width=1200):
    """Sai lệch còn lại sau phép đồng dạng tốt nhất, theo % chiều rộng ref. Trả về dict."""
    def prep(x):
        s = width / x.shape[1]
        return cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    a, b = prep(img), prep(ref)
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 30:
        return {"matches": len(good), "median_pct": None, "p90_pct": None}
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    # lọc ghép sai bằng RANSAC đồng dạng với ngưỡng rộng (5 % chiều rộng) để không loại méo thật
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=0.05 * width,
                                         maxIters=5000)
    inl = inl.ravel().astype(bool)
    # khớp lại đồng dạng bằng bình phương nhỏ nhất trên điểm đúng
    M, _ = cv2.estimateAffinePartial2D(pa[inl], pb[inl], method=cv2.LMEDS)
    err = np.linalg.norm(pa[inl] @ M[:, :2].T + M[:, 2] - pb[inl], axis=1) / width * 100
    return {"matches": int(inl.sum()), "median_pct": round(float(np.median(err)), 3),
            "p90_pct": round(float(np.percentile(err, 90)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, nargs="*", default=TEST_PAGES)
    ap.add_argument("--methods", nargs="*", default=["none", "corners", "uvdoc", "corners+uvdoc"])
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "dewarp/esp32_flatness.json")
    ap.add_argument("--save", type=Path, help="lưu ảnh đã làm phẳng vào <save>/<method>/")
    args = ap.parse_args()
    res = {m: {} for m in args.methods}
    for p in args.pages:
        img = imread(paths.ESP32_RAW / "ảnh esp" / f"{p:06d}.jpg")
        ref = imread(paths.ESP32_PAIRS / "pages/scan" / f"{p:06d}.png")
        for m in args.methods:
            t = time.perf_counter()
            out = apply(img, m)
            dt = time.perf_counter() - t
            r = flatness(out, ref)
            r["seconds"] = round(dt, 2)
            res[m][p] = r
            if args.save:
                d = args.save / m.replace("+", "_")
                d.mkdir(parents=True, exist_ok=True)
                cv2.imencode(".png", out)[1].tofile(str(d / f"{p:06d}.png"))
            print(p, m, r, flush=True)
    summary = {}
    for m, pages in res.items():
        v = [x for x in pages.values() if x["median_pct"] is not None]
        summary[m] = {"pages": len(v),
                      "median_pct": round(float(np.mean([x["median_pct"] for x in v])), 3),
                      "p90_pct": round(float(np.mean([x["p90_pct"] for x in v])), 3),
                      "seconds": round(float(np.mean([x["seconds"] for x in pages.values()])), 2)}
        print(m, summary[m])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "pages": res}, indent=1, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
