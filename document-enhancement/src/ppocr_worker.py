"""Tiến trình phụ nhận dạng chữ bằng OCR-M3 (PP-OCRv4, SVTR_LCNet) của nhóm. Chạy trong .venv-paddle.

Giao thức qua stdin/stdout, mỗi dòng một JSON, dòng trả lời có tiền tố "@@":
  vào : {"images": ["<PNG base64>", ...]}
  ra  : @@{"texts": ["...", ...]}
Khi sẵn sàng in "@@READY". Nhật ký của PaddleOCR đi ra stderr hoặc không có tiền tố nên bị bỏ qua.

Chạy thử:  .venv-paddle/Scripts/python.exe -X utf8 src/ppocr_worker.py --test ảnh1.png ảnh2.png
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PADDLE_OCR = ROOT / "third_party" / "PaddleOCR"
MODELS = ROOT / "models" / "ocr_pp"

# PaddleOCR 2.7 import các thư viện chỉ dùng khi train; chặn lại bằng module giả vì chỉ cần suy luận
for _n in ("imgaug", "imgaug.augmenters", "visualdl", "wandb"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["imgaug"].augmenters = sys.modules["imgaug.augmenters"]
sys.modules["visualdl"].LogWriter = object

sys.path.insert(0, str(PADDLE_OCR))
_stdout = sys.stdout
sys.stdout = sys.stderr            # mọi print của thư viện đi ra stderr; chỉ giao thức dùng _stdout

import numpy as np  # noqa: E402
import paddle  # noqa: E402
import cv2  # noqa: E402
from ppocr.data import create_operators, transform  # noqa: E402
from ppocr.modeling.architectures import build_model  # noqa: E402
from ppocr.postprocess import build_post_process  # noqa: E402
from ppocr.utils.save_load import load_model  # noqa: E402
import tools.program as program  # noqa: E402


def load(tag: str = "M3"):
    cfg = program.load_config(str(MODELS / f"{tag}_config.yml"))
    cfg["Global"]["pretrained_model"] = str(MODELS / f"{tag}_best")
    cfg["Global"]["checkpoints"] = None
    cfg["Global"]["character_dict_path"] = str(MODELS / "vocab.txt")
    pp = build_post_process(cfg["PostProcess"], cfg["Global"])
    ch = len(pp.character)
    cfg["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": ch, "SARLabelDecode": ch + 2, "NRTRLabelDecode": ch + 3}
    model = build_model(cfg["Architecture"])
    load_model(cfg, model)
    model.eval()
    ops = [{"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
           {"RecResizeImg": {"image_shape": [3, 48, 320]}},
           {"KeepKeys": {"keep_keys": ["image"]}}]
    return model, pp, create_operators(ops, cfg["Global"])


def recognize(model, pp, ops, png_bytes: list[bytes], bs: int = 32) -> list[str]:
    out: list[str] = []
    for i in range(0, len(png_bytes), bs):
        x = np.stack([transform({"image": b}, ops)[0] for b in png_bytes[i:i + bs]])
        with paddle.no_grad():
            pred = model(paddle.to_tensor(x))
        out += [t for t, _ in pp(pred)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="M3")
    ap.add_argument("--test", nargs="*", help="đọc thử các ảnh này rồi thoát")
    args = ap.parse_args()
    paddle.set_device("cpu")
    model, pp, ops = load(args.tag)
    if args.test is not None:
        res = recognize(model, pp, ops, [Path(p).read_bytes() for p in args.test])
        for p, t in zip(args.test, res):
            print(f"{p}\t{t}", file=_stdout)
        return
    print("@@READY", file=_stdout, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            texts = recognize(model, pp, ops, [base64.b64decode(s) for s in req["images"]])
            print("@@" + json.dumps({"texts": texts}, ensure_ascii=False), file=_stdout, flush=True)
        except Exception as e:  # noqa: BLE001
            print("@@" + json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False), file=_stdout, flush=True)


if __name__ == "__main__":
    main()
