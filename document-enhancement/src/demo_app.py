"""Web demo: tải ảnh tài liệu lên, chọn phương pháp, xem trước/sau.

Chạy (trên máy, CPU):
  .venv-ocr/Scripts/python.exe -X utf8 src/demo_app.py --port 8010
rồi mở http://localhost:8010
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from dewarp import dewarp_auto, dewarp_corners, dewarp_uvdoc  # noqa: E402
from enhance import enhance_1a  # noqa: E402
from restore_model import load_model, restore_image  # noqa: E402

METHODS = {
    "m4": ("M4 NAFNet (khuyên dùng)", paths.RETRAIN_CKPT / "m4_nafnet_all/best.pt"),
    "m2": ("M2 U-Net (nhanh hơn)", paths.RETRAIN_CKPT / "m2_unet_all/best.pt"),
    "m4l": ("M4L NAFNet width 64 (lớn, chậm)", paths.RETRAIN_CKPT / "m4l_nafnet64_all/best.pt"),
    "classic": ("Cổ điển 1a (OpenCV)", None),
}

# Ảnh mẫu: (tên hiển thị, đường dẫn ảnh vào)
SAMPLES = {
    "docaligner_4_2": ("Hóa đơn giấy nhăn (DocAligner)", paths.EXTERNAL / "docaligner/inp/4_2.png"),
    "docaligner_16_1": ("Trang sách, ánh sáng lệch (DocAligner)", paths.EXTERNAL / "docaligner/inp/16_1.png"),
    "jung_rect_10": ("Bóng đổ của tay (Jung)", paths.EXTERNAL / "jung/inp/rect_10.jpg"),
    "esp32_000003": ("Trang chụp ESP32-CAM (đã nắn)", paths.ESP32_PAIRS / "pages/esp_raw/000003.png"),
    "esp32_raw_000011": ("Ảnh ESP32 gốc còn nền — thử làm phẳng", paths.ESP32_RAW / "ảnh esp/000011.jpg"),
    "docunet_55_2": ("Trang giấy cong, nhăn — thử làm phẳng (DocUNet)", paths.EXTERNAL / "docunet_raw/crop/55_2 copy.png"),
    "docaligner_26_2": ("Trang nền xanh — model làm hỏng (DocAligner)", paths.EXTERNAL / "docaligner/inp/26_2.png"),
}

DEWARP = {
    "none": "Không làm phẳng",
    "auto": "Tự động: cắt nền nếu cần + UVDoc (giấy cong)",
    "corners": "4 góc trang (chỉ sửa chụp nghiêng, cắt nền)",
    "corners+uvdoc": "4 góc + UVDoc (giấy cong, nhăn)",
    "uvdoc": "Chỉ UVDoc",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
app = FastAPI(title="PBL6 nâng chất lượng ảnh tài liệu")


@lru_cache(maxsize=4)
def get_model(key: str):
    return load_model(METHODS[key][1], DEVICE)


def resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if long_side <= 0 or max(h, w) <= long_side:
        return img
    s = long_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).with_name("demo_page.html")).read_text("utf-8")


@app.get("/api/options")
def options():
    return {
        "device": DEVICE,
        "methods": [{"key": k, "label": v[0], "ready": v[1] is None or v[1].exists()} for k, v in METHODS.items()],
        "dewarp": [{"key": k, "label": v} for k, v in DEWARP.items()],
        "samples": [{"key": k, "label": v[0]} for k, v in SAMPLES.items() if v[1].exists()],
    }


@app.get("/api/sample/{key}")
def sample(key: str):
    if key not in SAMPLES or not SAMPLES[key][1].exists():
        raise HTTPException(404, "không có ảnh mẫu này")
    return FileResponse(SAMPLES[key][1])


@app.post("/api/enhance")
async def enhance(request: Request, method: str = "m4", long_side: int = 1024, dewarp: str = "none"):
    if method not in METHODS:
        raise HTTPException(400, f"phương pháp không hỗ trợ: {method}")
    if dewarp not in DEWARP:
        raise HTTPException(400, f"cách làm phẳng không hỗ trợ: {dewarp}")
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "chưa có ảnh")
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(413, "ảnh lớn hơn 25 MB")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "không đọc được ảnh (hỗ trợ JPG, PNG, BMP, WEBP)")
    orig_hw = img.shape[:2]
    cap = 3000 if long_side <= 0 else min(long_side, 3000)
    # làm phẳng ở độ phân giải gốc (tối đa 3000 px) để trang sau khi cắt còn đủ chi tiết, rồi mới thu nhỏ
    big = resize_long(img, 3000)
    img = resize_long(img, cap)
    t = time.perf_counter()
    flat, page_found = big, None
    if dewarp == "auto":
        flat, _ = dewarp_auto(flat, DEVICE)
    if "corners" in dewarp:
        flat, page_found = dewarp_corners(flat)
    if "uvdoc" in dewarp:
        flat = dewarp_uvdoc(flat, DEVICE)
    flat = resize_long(flat if dewarp != "none" else img, cap)
    t_dewarp = time.perf_counter() - t
    if method == "classic":
        out = enhance_1a(flat)
    else:
        out = restore_image(get_model(method), flat, DEVICE)
    dt = time.perf_counter() - t
    return JSONResponse({
        "input": png_b64(img), "output": png_b64(out),
        "seconds": round(dt, 2), "seconds_dewarp": round(t_dewarp, 2), "device": DEVICE,
        "method": METHODS[method][0], "dewarp": DEWARP[dewarp], "page_found": page_found,
        "geometry_changed": dewarp != "none",
        "size": f"{out.shape[1]}×{out.shape[0]}", "orig_size": f"{orig_hw[1]}×{orig_hw[0]}",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--preload", action="store_true", help="nạp sẵn M4 khi khởi động")
    args = ap.parse_args()
    if args.preload:
        get_model("m4")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
