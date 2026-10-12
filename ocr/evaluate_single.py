"""Run same Vietnamese Tesseract OCR on raw and enhanced versions.
No CER/WER is produced unless an independently checked reference text is given.
"""
import argparse
import csv
import os
from pathlib import Path
import re
import time
import unicodedata

from PIL import Image
import pytesseract
from rapidfuzz.distance import Levenshtein


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--variants", type=Path, default=Path("dataset/restored/opencv"))
    p.add_argument("--gt", type=Path, default=None)
    p.add_argument("--psm", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("ocr/outputs/one_image"))
    p.add_argument("--csv", type=Path, default=Path("evaluation/results/one_image_ocr.csv"))
    args = p.parse_args()

    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if not tesseract_cmd and os.name == "nt":
        pth = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if pth.exists():
            tesseract_cmd = str(pth)
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    languages = pytesseract.get_languages(config="")
    if "vie" not in languages:
        raise RuntimeError(f"Tesseract language 'vie' missing. Available: {languages}")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    files = [("raw", args.input)]
    if args.variants.is_dir():
        for path in sorted(args.variants.glob(f"{args.input.stem}_*.png")):
            files.append((path.stem.removeprefix(args.input.stem + "_"), path))

    reference = None
    if args.gt:
        if not args.gt.exists():
            raise FileNotFoundError(args.gt)
        reference = norm(args.gt.read_text(encoding="utf-8-sig"))
        if not reference:
            raise ValueError("Ground-truth file is empty")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, path in files:
        with Image.open(path) as im:
            t0 = time.perf_counter()
            result = pytesseract.image_to_string(im, lang="vie", config=f"--psm {args.psm}")
            elapsed = time.perf_counter() - t0
        (args.out / f"{method}.txt").write_text(result, encoding="utf-8-sig")
        pred = norm(result)
        row = {"method": method, "image": path.name, "time_s": round(elapsed, 3),
               "characters": len(pred), "cer": "", "wer": ""}
        if reference is not None:
            row["cer"] = round(Levenshtein.distance(reference, pred) / len(reference), 5)
            gt_words = reference.split()
            row["wer"] = round(Levenshtein.distance(gt_words, pred.split()) / len(gt_words), 5)
        rows.append(row)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output texts: {args.out}")
    print(f"Comparison CSV: {args.csv}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
