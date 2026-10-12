"""Crop a text region for lightweight pre-trained model experiments."""
import argparse
from pathlib import Path

from test_single import read_image, save_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--x", type=int, required=True)
    p.add_argument("--y", type=int, required=True)
    p.add_argument("--w", type=int, required=True)
    p.add_argument("--h", type=int, required=True)
    p.add_argument("--out", type=Path, default=Path("dataset/model_inputs/text_crop.png"))
    args = p.parse_args()
    image = read_image(args.input)
    H, W = image.shape[:2]
    x1, y1 = max(0, args.x), max(0, args.y)
    x2, y2 = min(W, args.x + args.w), min(H, args.y + args.h)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Crop outside image. Image size: {W}x{H}")
    save_png(args.out, image[y1:y2, x1:x2])
    print(f"Saved: {args.out}; crop {x2-x1}x{y2-y1}")


if __name__ == "__main__":
    main()
