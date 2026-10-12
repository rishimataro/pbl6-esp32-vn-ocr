"""Test 3 OpenCV image-enhancement baselines on ONE real ESP32-CAM photo."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    buf = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Not a valid image: {path}")
    return image


def save_png(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"PNG encode failed: {path}")
    encoded.tofile(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", default="dataset/restored/opencv", type=Path)
    args = parser.parse_args()

    src = read_image(args.input)
    h, w = src.shape[:2]
    print(f"Input: {args.input} ({w}x{h})")

    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corrected = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

    smooth = cv2.GaussianBlur(corrected, (0, 0), 1.0)
    sharpened = cv2.addWeighted(corrected, 1.5, smooth, -0.5, 0)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 12,
    )

    variants = {
        "clahe": corrected,
        "clahe_sharp": sharpened,
        "adaptive": binary,
    }
    for name, image in variants.items():
        path = args.out / f"{args.input.stem}_{name}.png"
        save_png(path, image)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
