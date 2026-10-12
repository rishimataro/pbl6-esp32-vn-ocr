"""FastAPI backend for ESP32-CAM JPEG upload.
Run from project root:
    python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

import csv
import hashlib
import re
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
RAW_DIR = ROOT / "dataset" / "raw" / "images"
BACKEND_CSV = BACKEND_DIR / "metadata.csv"
DATASET_CSV = ROOT / "dataset" / "metadata.csv"
MAX_BYTES = 5 * 1024 * 1024
VIETNAM_TZ = timezone(timedelta(hours=7))  # No tzdata dependency on Windows
CSV_LOCK = threading.Lock()  # Fine for one Uvicorn worker used in development

FIELDS = [
    "image_id", "saved_at", "page_id", "condition", "source",
    "original_filename", "filename", "backend_path", "dataset_path",
    "content_type", "width", "height", "file_size", "sha256", "note",
]

for folder in (UPLOAD_DIR, RAW_DIR, DATASET_CSV.parent):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ESP32-CAM OCR Dataset Receiver", version="1.1")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


def safe_token(value: str, fallback: str) -> str:
    # Used only to create filesystem-safe filenames; does not trust HTTP input.
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", (value or "").strip())[:40]
    return cleaned or fallback


def clean_note(value: str) -> str:
    return " ".join((value or "").split())[:160]


def dimensions_if_valid_jpeg(data: bytes) -> tuple[int, int]:
    if not data:
        raise HTTPException(status_code=400, detail="JPEG body is empty")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="JPEG is larger than 5 MB")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "JPEG":
                raise HTTPException(status_code=400, detail="Only JPEG images are accepted")
            size = image.size
            image.verify()  # Reject malformed images
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid or damaged JPEG image") from error
    return size


def append_metadata(row: dict) -> None:
    # Keep the same headers as the first version of app.py in this conversation.
    with CSV_LOCK:
        for csv_path in (BACKEND_CSV, DATASET_CSV):
            if csv_path.exists() and csv_path.stat().st_size:
                with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
                    existing_headers = next(csv.reader(file), [])
                if existing_headers != FIELDS:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Metadata schema mismatch: {csv_path.name}. Back up old CSV before retrying.",
                    )
            else:
                with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
                    csv.DictWriter(file, fieldnames=FIELDS).writeheader()

            with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
                csv.DictWriter(file, fieldnames=FIELDS).writerow(row)


def save_jpeg(data: bytes, page_id: str, condition: str,
              source: str, original_name: str = "", note: str = "") -> dict:
    width, height = dimensions_if_valid_jpeg(data)
    page = safe_token(page_id, "P000")
    cond = safe_token(condition, "unknown")
    now = datetime.now(VIETNAM_TZ)
    image_id = f"{page}_{cond}_{now:%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:6]}"
    filename = f"{image_id}.jpg"
    raw_path = RAW_DIR / filename
    upload_path = UPLOAD_DIR / filename
    raw_path.write_bytes(data)
    shutil.copyfile(raw_path, upload_path)
    row = {
        "image_id": image_id,
        "saved_at": now.isoformat(timespec="seconds"),
        "page_id": page,
        "condition": cond,
        "source": source,
        "original_filename": Path(original_name.replace("\\", "/")).name[:120],
        "filename": filename,
        "backend_path": upload_path.relative_to(ROOT).as_posix(),
        "dataset_path": raw_path.relative_to(ROOT).as_posix(),
        "content_type": "image/jpeg",
        "width": width,
        "height": height,
        "file_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "note": clean_note(note),
    }
    append_metadata(row)
    return row


@app.get("/")
def index():
    return {"message": "ESP32-CAM backend is running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload_raw")
async def upload_raw(request: Request,
                     page_id: str = Query("P000"),
                     condition: str = Query("C01_normal"),
                     note: str = Query("")):
    """ESP32-CAM POSTs raw JPEG bytes with Content-Type: image/jpeg."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in ("image/jpeg", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Use Content-Type: image/jpeg")
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="JPEG is larger than 5 MB")
    row = save_jpeg(bytes(payload), page_id, condition, source="esp32_raw", note=note)
    return {"ok": True, "data": row, "view_url": f"/uploads/{row['filename']}"}


@app.post("/upload")
async def upload_form(file: UploadFile = File(...),
                      page_id: str = Form("P000"),
                      condition: str = Form("C01_normal"),
                      note: str = Form("")):
    """Browser /docs POSTs multipart/form-data; use for first tests."""
    try:
        data = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    row = save_jpeg(data, page_id, condition, source="web_form",
                    original_name=file.filename or "", note=note)
    return {"ok": True, "data": row, "view_url": f"/uploads/{row['filename']}"}


@app.get("/metadata")
def metadata(limit: int = Query(10, ge=1, le=100)):
    if not DATASET_CSV.exists():
        return {"count": 0, "items": []}
    with DATASET_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    items = list(reversed(rows[-limit:]))
    return {"count": len(items), "items": items}
