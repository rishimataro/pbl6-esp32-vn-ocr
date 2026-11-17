"""Đường dẫn mặc định trong dự án. Mọi script trong src/ lấy vị trí dữ liệu, model, kết quả từ đây.

Trên Colab, code được giải nén vào /content/pbl6_pipeline/, nên ROOT là /content và các đường dẫn
dưới đây không tồn tại: luôn truyền đường dẫn qua tham số dòng lệnh (--data, --out, ...).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# dữ liệu
DATA = ROOT / "data"
ESP32_RAW = DATA / "raw" / "esp32_scan_pairs"          # "ảnh esp/", "ảnh scan/", image_page_order.csv
PAGE_ORDER_CSV = ESP32_RAW / "image_page_order.csv"    # ảnh thứ i ↔ trang PDF
SOURCE_PDF = DATA / "raw" / "source_pdf" / "P17878411dab3c0b918424157e400ddf99c.pdf"
ESP32_ALIGN = DATA / "interim" / "esp32_alignment"     # kết quả căn chỉnh ESP32 ↔ scan ↔ PDF (n1_align.py)
ESP32_PAIRS = DATA / "processed" / "esp32_pairs"       # trang đã nắn, patch, crop dòng, nhãn (n2_build_data.py)
EXTERNAL = DATA / "external"                           # RealDAE, tập tổng hợp: nằm trên Google Drive

# model
MODELS = ROOT / "models"
RETRAIN_CKPT = MODELS / "retrain"                      # <run>/best.pt
OCR_WEIGHTS = ROOT / ".cache" / "ocr" / "models"       # trọng số VietOCR gốc (vgg_transformer.pth, ...)

# kết quả
REPORTS = ROOT / "reports"
RESULTS = REPORTS / "results"
FIGURES = REPORTS / "figures"

# đề tài OCR cũ (chỉ để tra cứu)
ARCHIVE_OCR = ROOT / "archive" / "ocr_topic"
