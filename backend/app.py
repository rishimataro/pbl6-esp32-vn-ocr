from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
from io import BytesIO
import csv
import hashlib
import re
import shutil
import uuid

# Setup đường dẫn
ROOT_DIR = Path(__file__).resolve().parents[1]

BACKEND_DIR = ROOT_DIR / "backend"
BACKEND_UPLOADS_DIR = BACKEND_DIR / "uploads"
BACKEND_METADATA_FILE = BACKEND_DIR / "metadata.csv"

DATASET_DIR = ROOT_DIR / "dataset"
DATASET_RAW_DIR = DATASET_DIR / "raw"
DATASET_RESTORED_DIR = DATASET_DIR / "restored"
DATASET_GT_DIR = DATASET_DIR / "ground_truth"
DATASET_METADATA_FILE = DATASET_DIR / "metadata.csv"

TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

for directory in [
    BACKEND_UPLOADS_DIR,
    DATASET_RAW_DIR,
    DATASET_RESTORED_DIR,
    DATASET_GT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Khởi tạo FastAPI app
app = FastAPI(
    title="ESP32-CAM Image Receiver API",
    description="API for uploading, restoring, and managing images.",
    version="1.0.0",
)

# Cho phép CORS cho tất cả các nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/uploads",
    StaticFiles(directory=BACKEND_UPLOADS_DIR),
    name="uploads"
)

METADATA_FIELDS = [
    "image_id",
    "saved_at",
    "page_id",
    "condition",
    "source",
    "original_filename",
    "filename",
    "backend_path",
    "dataset_path",
    "content_type",
    "width",
    "height",
    "file_size",
    "sha256",
    "note",
]

# Hàm để đảm bảo rằng tệp CSV tồn tại và có tiêu đề
def ensure_csv_header(csv_path: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=METADATA_FIELDS)
            writer.writeheader()

# Hàm để thêm một hàng metadata vào tệp CSV
def append_metadata(row: dict) -> None:
    for csv_path in [BACKEND_METADATA_FILE, DATASET_METADATA_FILE]:
        ensure_csv_header(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=METADATA_FIELDS)
            writer.writerow(row)

# Hàm để làm sạch và chuẩn hóa chuỗi văn bản
def safe_text(value:str, default: str = "unknown") -> str:
    value = (value or default).strip()
    value = re.sub(r"[^\w\s-]", "", value)
    return value[:50] or default

def is_jpeg(data: bytes) -> bool:
    return data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")

def get_image_size(data: bytes) -> tuple[int, int]:
    try:
        image = Image.open(BytesIO(data))
        return image.size
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

def save_image_and_metadata(
    data: bytes,
    original_filename: str,
    content_type: str,
    page_id: str,
    condition: str,
    note: str,
    source: str
) -> dict:
    if not data:
        raise HTTPException(status_code=400, detail="No image data provided")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image size exceeds 10 MB limit")

    if not is_jpeg(data):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid JPEG image")

    width, height = get_image_size(data)

    now = datetime.now(TIMEZONE)
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")

    page_id_clean = safe_text(page_id, "P000")
    condition_clean = safe_text(condition, "unknown")

    image_id = f"{page_id_clean}_{condition_clean}_{timestamp}_{uuid.uuid4().hex[:8]}"
    filename = f"{image_id}.jpg"

    backend_path = BACKEND_UPLOADS_DIR / filename
    dataset_path = DATASET_RAW_DIR / filename

    # Lưu tệp ảnh vào thư mục backend/uploads
    with backend_path.open("wb") as f:
        f.write(data)

    # Sao chép tệp ảnh vào thư mục dataset/raw
    shutil.copy2(backend_path, dataset_path)

    sha256_hash = hashlib.sha256(data).hexdigest()
    row = {
        "image_id": image_id,
        "saved_at": now.isoformat(),
        "page_id": page_id_clean,
        "condition": condition_clean,
        "source": source,
        "original_filename": original_filename or "",
        "filename": filename,
        "backend_path": str(backend_path.relative_to(ROOT_DIR)),
        "dataset_path": str(dataset_path.relative_to(ROOT_DIR)),
        "content_type": content_type or "image/jpeg",
        "width": width,
        "height": height,
        "file_size": len(data),
        "sha256": sha256_hash,
        "note": note or "",
    }

    append_metadata(row)
    return row


# Xây dưng các endpoint API
@app.get("/")
def home():
    return {
        "message": "ESP32-CAM Image Receiver API is running.",
        "endpoints": {
            "health": "health",
            "multipart_upload": "upload",
            "raw_upload": "upload_raw",
            "metadata": "/metadata",
            "view_uploaded_images": "uploads/<filename>",
        },
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_image_form(
    file: UploadFile = File(...),
    page_id: str = Form("P000"),
    condition: str = Form("unknown"),
    note: str = Form("")
):
    if file.content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Only JPEG images are allowed")

    data = await file.read()

    row = save_image_and_metadata(
        data=data,
        original_filename=file.filename,
        content_type=file.content_type,
        page_id=page_id,
        condition=condition,
        note=note,
        source="multipart_form",
    )

    return {
        "ok": True,
        "message": "Image uploaded successfully",
        "metadata": row,
        "view_url": f"/uploads/{row['filename']}"
    }

@app.post("/upload_raw")
async def upload_image_raw(
    request: Request,
    page_id: str = Query("P000"),
    condition: str = Query("unknown"),
    note: str = Query("")
):
    content_type = request.headers.get("content-type", "")
    if content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Only JPEG images are allowed")

    data = await request.body()

    row = save_image_and_metadata(
        data=data,
        original_filename="raw_upload.jpg",
        content_type=content_type,
        page_id=page_id,
        condition=condition,
        note=note,
        source="raw_body",
    )

    return {
        "ok": True,
        "message": "Image uploaded successfully",
        "metadata": row,
        "view_url": f"/uploads/{row['filename']}"
    }

@app.get("/metadata")
def read_metadata(limit: int = 20):
    if not BACKEND_METADATA_FILE.exists():
        raise {"items": []}

    with DATASET_METADATA_FILE.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    rows = rows[-limit:]
    rows.reverse()
    return {
        "count": len(rows),
        "items": rows
    }