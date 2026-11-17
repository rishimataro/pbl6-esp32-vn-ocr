"""N2: (a) đối chiếu nhãn tự động với 139 dòng duyệt tay, (b) CER VietOCR pretrained trên crop test.

Chạy:
  .venv-ocr/Scripts/python.exe -X utf8 src/n2_eval.py [--split test] [--model vgg_transformer]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n2_build_data import clean_text  # noqa: E402

import paths  # noqa: E402

ROOT = paths.ROOT
REVIEWED = paths.ARCHIVE_OCR / "ocr_baseline/results/reviewed_pilot_metrics_20261103/labels_input.json"


_TONE_SHIFT = {}
for _base, _new, _old in [("oa", "àáảãạ", "òóỏõọ"), ("oe", "èéẻẽẹ", "òóỏõọ"), ("uy", "ỳýỷỹỵ", "ùúủũụ")]:
    for _a, _b in zip(_new, _old):
        _TONE_SHIFT[_base[0] + _a] = _b + _base[1]                      # "oà" → "òa"
        _TONE_SHIFT[_base[0].upper() + _a] = _b.upper() + _base[1]      # "Oà" → "Òa"
        _TONE_SHIFT[_base[0].upper() + _a.upper()] = _b.upper() + _base[1].upper()
_TONE_RE = re.compile(r"(?<![qQ])(" + "|".join(map(re.escape, _TONE_SHIFT)) + r")(?![^\W\d_])")


def normalize_tone(s: str) -> str:
    """Thống nhất kiểu đặt dấu thanh ở âm tiết mở: 'hoà'/'hòa', 'thuỷ'/'thủy' → kiểu truyền thống (như PDF)."""
    return _TONE_RE.sub(lambda m: _TONE_SHIFT[m.group(1)], unicodedata.normalize("NFC", s))


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s.replace("đ", "d").replace("Đ", "D"))


def cer_counts(ref: str, hyp: str) -> tuple[int, int]:
    return Levenshtein.distance(ref, hyp), len(ref)


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.split(), hyp.split()
    return Levenshtein.distance(r, h), len(r)


def check_labels(n1: Path, data: Path) -> dict:
    """So nhãn tự động với dòng duyệt tay: chiếu polygon tay sang khung render, ghép IoU lớn nhất."""
    align = {int(Path(r["image"]).stem): r for r in json.loads((n1 / "alignment.json").read_text("utf-8"))}
    lines = [json.loads(l) for l in (data / "lines.jsonl").read_text("utf-8").splitlines()]
    by_page: dict[int, list] = {}
    for l in lines:
        by_page.setdefault(l["page"], []).append(l)
    rows = json.loads(REVIEWED.read_text("utf-8"))["rows"]
    res = {"reviewed": len(rows), "matched": 0, "exact": 0, "edits": 0, "chars": 0, "unmatched": [], "diffs": []}
    for r in rows:
        if r.get("review_status") != "reviewed" or not r.get("reviewed_text"):
            continue
        page = int(r["page_id"].split("_")[1])
        H = np.array(align[page]["H_esp_to_render"])
        poly = cv2.perspectiveTransform(np.float32([r["polygon_xy"]]), H)[0]
        x0, y0 = poly.min(0); x1, y1 = poly.max(0)
        best, biou = None, 0.0
        for l in by_page.get(page, []):
            a0, b0, a1, b1 = l["bbox_render"]
            iw = max(0, min(x1, a1) - max(x0, a0)); ih = max(0, min(y1, b1) - max(y0, b0))
            inter = iw * ih
            u = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
            # khung detector cũ có lề rộng hơn khung PDF → dùng giao / diện tích nhỏ hơn thay cho IoU
            small = min((x1 - x0) * (y1 - y0), (a1 - a0) * (b1 - b0))
            ov = inter / small if small > 0 else 0.0
            if u > 0 and ov > biou and inter / u > 0.25:
                best, biou = l, ov
        ref = normalize_tone(clean_text(r["reviewed_text"]))
        if best is None or biou < 0.5:
            res["unmatched"].append({"id": r["id"], "text": ref, "best_iou": round(biou, 2)})
            continue
        res["matched"] += 1
        d = Levenshtein.distance(ref, best["text"])
        res["edits"] += d; res["chars"] += len(ref)
        if d == 0:
            res["exact"] += 1
        else:
            res["diffs"].append({"id": r["id"], "reviewed": ref, "auto": best["text"], "edits": d})
    res["label_cer_vs_reviewed"] = round(res["edits"] / max(res["chars"], 1), 4)
    return res


def load_model(name: str, device: str, weights=None):
    import torch
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    cfg = Cfg.load_config_from_name(name)
    cfg["cnn"]["pretrained"] = False
    cfg["device"] = device
    cfg["weights"] = str(weights or paths.OCR_WEIGHTS / f"{name}.pth")
    cfg["predictor"]["beamsearch"] = False
    torch.set_num_threads(4)
    return Predictor(cfg)


def ocr_eval(data: Path, split: str, model_name: str, variants, device="cpu", limit=None,
             weights=None, tag=None) -> dict:
    from PIL import Image
    pred = load_model(model_name, device, weights)
    tag = tag or model_name
    meta = {l["id"]: l for l in (json.loads(x) for x in (data / "lines.jsonl").read_text("utf-8").splitlines())}
    out = {}
    for v in variants:
        rows = [ln.split("\t", 1) for ln in (data / "labels" / f"{split}_{v}.txt").read_text("utf-8").splitlines() if ln]
        if limit:
            rows = rows[:limit]
        t = time.time()
        preds = []
        bs = 32
        for i in range(0, len(rows), bs):
            imgs = [Image.open(data / p).convert("RGB") for p, _ in rows[i:i + bs]]
            preds += pred.predict_batch(imgs)
        agg = {}
        per = []
        for (p, ref), hyp in zip(rows, preds):
            lid = Path(p).stem
            leg = meta[lid]["legibility"]
            ref, hyp = normalize_tone(ref), normalize_tone(hyp)
            e, n = cer_counts(ref, hyp)
            ew, nw = wer_counts(ref, hyp)
            ea, na = cer_counts(strip_accents(ref), strip_accents(hyp))
            for key in ("all", leg):
                a = agg.setdefault(key, [0, 0, 0, 0, 0, 0, 0])
                a[0] += e; a[1] += n; a[2] += ew; a[3] += nw; a[4] += ea; a[5] += na; a[6] += 1
            per.append({"id": lid, "ref": ref, "hyp": hyp, "cer": round(e / max(n, 1), 4), "legibility": leg})
        out[v] = {k: {"lines": a[6], "cer": round(a[0] / a[1], 4), "wer": round(a[2] / a[3], 4),
                      "cer_no_accent": round(a[4] / a[5], 4)} for k, a in agg.items()}
        out[v]["seconds"] = round(time.time() - t, 1)
        (data / "eval").mkdir(exist_ok=True)
        (data / "eval" / f"pred_{split}_{v}_{tag}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in per) + "\n", encoding="utf-8")
        print(v, json.dumps(out[v], ensure_ascii=False), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=Path, default=paths.ESP32_ALIGN)
    ap.add_argument("--data", type=Path, default=paths.ESP32_PAIRS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="vgg_transformer")
    ap.add_argument("--variants", nargs="*", default=["esp_raw", "esp_1a", "scan"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-labels", action="store_true")
    ap.add_argument("--weights", help="trọng số VietOCR thay cho bản pretrained (vd. checkpoint fine-tune)")
    ap.add_argument("--tag", help="tên dùng trong tệp kết quả, mặc định là --model")
    args = ap.parse_args()
    report = {}
    if not args.skip_labels:
        lab = check_labels(args.n1, args.data)
        print("labels:", {k: v for k, v in lab.items() if k not in ("diffs", "unmatched")})
        report["label_check"] = lab
    report["ocr"] = ocr_eval(args.data, args.split, args.model, args.variants, args.device, args.limit,
                             args.weights, args.tag)
    (args.data / "eval").mkdir(exist_ok=True)
    (args.data / "eval" / f"report_{args.split}_{args.tag or args.model}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
