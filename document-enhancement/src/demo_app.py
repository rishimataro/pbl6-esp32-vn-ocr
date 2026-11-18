"""Web demo: tải ảnh tài liệu lên, chọn phương pháp, xem trước/sau và đọc chữ (phát hiện dòng + nhận dạng).

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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout_doc  # noqa: E402
import paths  # noqa: E402
from dewarp import dewarp_auto, dewarp_corners, dewarp_uvdoc  # noqa: E402
from enhance import enhance_1a  # noqa: E402
from ocr_pipeline import ENGINES as OCR_ENGINES, engine_ready, read_page  # noqa: E402
from restore_model import load_model, restore_image  # noqa: E402

METHODS = {
    "m4": ("ENH-M4 NAFNet (khuyên dùng)", paths.RETRAIN_CKPT / "m4_nafnet_all/best.pt"),
    "m2": ("ENH-M2 U-Net (nhanh hơn)", paths.RETRAIN_CKPT / "m2_unet_all/best.pt"),
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
        "engines": [{"key": k, "label": v, "ready": engine_ready(k)} for k, v in OCR_ENGINES.items()],
        "samples": [{"key": k, "label": v[0]} for k, v in SAMPLES.items() if v[1].exists()],
    }


@app.get("/api/sample/{key}")
def sample(key: str):
    if key not in SAMPLES or not SAMPLES[key][1].exists():
        raise HTTPException(404, "không có ảnh mẫu này")
    return FileResponse(SAMPLES[key][1])


NL = chr(10)


def _decode(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(400, "chưa có ảnh")
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(413, "ảnh lớn hơn 25 MB")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "không đọc được ảnh (hỗ trợ JPG, PNG, BMP, WEBP)")
    return img


def _check(method: str, dewarp: str):
    if method not in METHODS:
        raise HTTPException(400, f"phương pháp không hỗ trợ: {method}")
    if dewarp not in DEWARP:
        raise HTTPException(400, f"cách làm phẳng không hỗ trợ: {dewarp}")


def _pipeline(img: np.ndarray, method: str, long_side: int, dewarp: str):
    """Làm phẳng (tùy chọn) rồi nâng ảnh. Trả về (ảnh trước nâng, ảnh sau nâng, giây làm phẳng, giây tổng, page_found)."""
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
    return flat, out, t_dewarp, time.perf_counter() - t, page_found


@app.post("/api/enhance")
async def enhance(request: Request, method: str = "m4", long_side: int = 1024, dewarp: str = "none"):
    _check(method, dewarp)
    img = _decode(await request.body())
    orig_hw = img.shape[:2]
    shown = resize_long(img, 3000 if long_side <= 0 else min(long_side, 3000))
    _, out, t_dewarp, dt, page_found = _pipeline(img, method, long_side, dewarp)
    return JSONResponse({
        "input": png_b64(shown), "output": png_b64(out),
        "seconds": round(dt, 2), "seconds_dewarp": round(t_dewarp, 2), "device": DEVICE,
        "method": METHODS[method][0], "dewarp": DEWARP[dewarp], "page_found": page_found,
        "geometry_changed": dewarp != "none",
        "size": f"{out.shape[1]}×{out.shape[0]}", "orig_size": f"{orig_hw[1]}×{orig_hw[0]}",
    })


JOBS: dict = {}          # job id -> {"img", "lines", "layout"}; giữ 8 phiên gần nhất


def _new_job(img, lines) -> str:
    import uuid
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"img": img, "lines": lines, "layout": None}
    while len(JOBS) > 8:
        JOBS.pop(next(iter(JOBS)))
    return jid


def _layout(jid: str, use_model: bool):
    j = JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "không còn phiên này, hãy đọc chữ lại")
    key = bool(use_model)
    if j["layout"] is None or j["layout"][0] != key:
        t = time.perf_counter()
        mdl = layout_doc.run_model(j["img"]) if use_model else None
        res = layout_doc.analyse_page(j["img"], j["lines"], model=mdl)
        j["layout"] = (key, res, mdl is not None, round(time.perf_counter() - t, 2))
    return j["layout"]


@app.post("/api/layout")
def layout(job: str, model: int = 1):
    """Nhận bố cục trang đã đọc chữ: trả các khối (tiêu đề, đoạn, danh sách, bảng, hình) để vẽ lên ảnh."""
    _k, res, used, secs = _layout(job, bool(model))
    return {"blocks": layout_doc.describe(res), "columns": res.get("columns", 1), "model_used": used, "seconds": secs}


@app.get("/api/docx")
def docx(job: str, model: int = 1, font: str = "Times New Roman"):
    if font not in ("Times New Roman", "Arial", "Calibri"):
        raise HTTPException(400, "phông không hỗ trợ")
    _k, res, _u, _s = _layout(job, bool(model))
    data = layout_doc.build_docx(res, font=font)
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={"Content-Disposition": 'attachment; filename="tai_lieu.docx"'})


@app.post("/api/ocr")
async def ocr(request: Request, method: str = "m4", long_side: int = 1600, dewarp: str = "none",
              engine: str = "vietocr", compare: int = 1):
    """Nâng ảnh → phát hiện dòng → nhận dạng. compare=1: đọc cả ảnh trước nâng để so sánh."""
    _check(method, dewarp)
    if engine not in OCR_ENGINES:
        raise HTTPException(400, f"bộ nhận dạng không hỗ trợ: {engine}")
    if not engine_ready(engine):
        raise HTTPException(400, f"bộ nhận dạng chưa cài: {OCR_ENGINES[engine]}")
    img = _decode(await request.body())
    orig_hw = img.shape[:2]
    # đọc chữ cần độ phân giải đủ lớn (chữ cao ≥ ~15 px), nên không xuống dưới 1600 px
    ls = 3000 if long_side <= 0 else max(1600, min(long_side, 3000))
    before, after, t_dewarp, t_enh, page_found = _pipeline(img, method, ls, dewarp)
    t = time.perf_counter()
    after, lines = read_page(after, engine, DEVICE, color_ref=before)
    t_ocr = time.perf_counter() - t
    res = {
        "output": png_b64(after), "size": f"{after.shape[1]}×{after.shape[0]}", "orig_size": f"{orig_hw[1]}×{orig_hw[0]}",
        "w": int(after.shape[1]), "h": int(after.shape[0]),
        "method": METHODS[method][0], "engine": OCR_ENGINES[engine], "dewarp": DEWARP[dewarp], "page_found": page_found,
        "lines": lines, "text": NL.join(l["text"] for l in lines), "job": _new_job(after, lines),
        "layout_model": layout_doc.model_ready(),
        "seconds_enhance": round(t_enh, 2), "seconds_ocr": round(t_ocr, 2), "device": DEVICE,
    }
    if compare:
        t = time.perf_counter()
        before, raw_lines = read_page(before, engine, DEVICE)
        res["raw"] = {"output": png_b64(before), "w": int(before.shape[1]), "h": int(before.shape[0]),
                      "lines": raw_lines, "text": NL.join(l["text"] for l in raw_lines),
                      "seconds_ocr": round(time.perf_counter() - t, 2)}
    return JSONResponse(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--preload", action="store_true", help="nạp sẵn ENH-M4 khi khởi động")
    args = ap.parse_args()
    if args.preload:
        get_model("m4")
        from ocr_pipeline import _vietocr
        _vietocr(DEVICE)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
