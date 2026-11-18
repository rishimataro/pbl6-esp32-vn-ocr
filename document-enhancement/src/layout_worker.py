"""Nhận bố cục bằng model có sẵn của PaddlePaddle: PP-DocLayout-M (vùng) và SLANet_plus (cấu trúc bảng).
Chạy trong .venv-paddle, mỗi lần một ảnh, in ra một dòng JSON:
  {"regions":[{"label","score","box":[x0,y0,x1,y1]}], "tables":[{"box":[..],"cells":[[x0,y0,x1,y1]...],"tokens":[...]}]}
Dùng:  .venv-paddle/Scripts/python.exe -X utf8 src/layout_worker.py ảnh.png [--thr 0.35]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "models" / "layout_pp"
_out = sys.stdout
sys.stdout = sys.stderr

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from paddlex import create_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--thr", type=float, default=0.35)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    lay = create_model(model_name="PP-DocLayout-M", model_dir=str(M / "PP-DocLayout-M"))
    res = list(lay.predict(img, batch_size=1, threshold=a.thr))[0]
    regions = [{"label": b["label"], "score": float(b["score"]), "box": [int(v) for v in b["coordinate"]]} for b in res["boxes"]]
    tables = []
    tm = None
    for r in regions:
        if r["label"] != "table":
            continue
        x0, y0, x1, y1 = r["box"]
        crop = img[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            continue
        if tm is None:
            tm = create_model(model_name="SLANet_plus", model_dir=str(M / "SLANet_plus"))
        o = list(tm.predict(crop, batch_size=1))[0]
        cells = []
        for p in o["bbox"]:
            p = np.array(p, float).reshape(-1, 2)
            cells.append([x0 + float(p[:, 0].min()), y0 + float(p[:, 1].min()), x0 + float(p[:, 0].max()), y0 + float(p[:, 1].max())])
        tables.append({"box": r["box"], "cells": cells, "tokens": list(o["structure"])})
    print(json.dumps({"regions": regions, "tables": tables}, ensure_ascii=False), file=_out)


if __name__ == "__main__":
    main()
