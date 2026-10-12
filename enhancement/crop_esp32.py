"""Cat vung anh co toa do, hoac chon vung bang chuot (OpenCV GUI).

Vi du:
python crop_esp32.py --input P001_C01_normal.jpg --x 175 --y 374 --w 512 --h 80 --out crop.png
python crop_esp32.py --input P001_C01_normal.jpg --interactive --out crop.png
"""
from pathlib import Path
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Cat vung anh tu toa do, hoac keo chuot chon vung")
    parser.add_argument("--input", required=True, help="Duong dan anh goc")
    parser.add_argument("--out", required=True, help="Duong dan anh sau khi cat")
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--h", type=int)
    parser.add_argument("--interactive", action="store_true", help="Keo chuot chon vung, Enter xac nhan")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        parser.error(f"Khong tim thay anh: {source}")

    image = cv2.imread(str(source))
    if image is None:
        parser.error(f"Khong doc duoc anh: {source}")
    H, W = image.shape[:2]
    print(f"Kich thuoc anh goc: width={W}, height={H}")

    if args.interactive:
        title = "Keo chuot chon vung - ENTER de xac nhan, ESC de huy"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, min(W, 900), min(H, 800))
        x, y, w, h = map(int, cv2.selectROI(title, image, showCrosshair=True, fromCenter=False))
        cv2.destroyWindow(title)
    else:
        if any(val is None for val in [args.x, args.y, args.w, args.h]):
            parser.error("Can --x --y --w --h, hoac dung --interactive")
        x, y, w, h = args.x, args.y, args.w, args.h

    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
        parser.error(f"Toa do khong hop le: ({x}, {y}, {w}, {h}), anh la {W}x{H}")

    crop = image[y:y + h, x:x + w]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), crop):
        parser.error(f"Loi ghi anh: {output}")
    print(f"Da cat: x={x}, y={y}, w={w}, h={h}; (x1,y1,x2,y2)=({x},{y},{x+w},{y+h})")
    print(f"Da luu: {output.resolve()}")


if __name__ == "__main__":
    main()
