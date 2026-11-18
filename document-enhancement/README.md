# PBL6: Nâng chất lượng ảnh tài liệu chụp trong điều kiện xấu

Model học sâu nhận ảnh chụp trang tài liệu bị tối, nhiễu, ám màu, sáng không đều, nhòe hoặc nén JPEG, và trả về ảnh trang sạch, giống ảnh scan.

- **Kiến trúc chính:** NAFNet.
- **Dữ liệu train:**
  - RealDAE (ảnh chụp thật);
  - cặp ảnh ESP32-CAM ↔ scan;
  - ảnh suy biến tổng hợp.

Kế hoạch: [docs/KE_HOACH_NANG_ANH.md](docs/KE_HOACH_NANG_ANH.md).
Cách train lại từ đầu: [docs/HUONG_DAN_TRAIN_LAI.md](docs/HUONG_DAN_TRAIN_LAI.md).
Nhật ký train: [docs/NHAT_KY_TRAIN_LAI.md](docs/NHAT_KY_TRAIN_LAI.md).

## Kết quả chính (tập test)

PSNR dB / SSIM / LPIPS (LPIPS càng thấp càng tốt). Bảng đầy đủ: `reports/results/retrain/summary_main.csv`.

| Model | Tham số | RealDAE (150 ảnh thật) | Tổng hợp (1.260 ảnh) |
|---|---|---|---|
| Ảnh vào, không xử lý | — | 12,40 / 0,858 / 0,216 | 17,68 / 0,726 / 0,261 |
| Cổ điển (cân bằng trắng, CLAHE, ...) | — | 17,77 / 0,886 / 0,177 | 14,97 / 0,753 / 0,292 |
| U-Net (ENH-M2) | 7,8M | 25,77 / 0,965 / 0,061 | 23,42 / 0,896 / 0,153 |
| NAFNet (ENH-M4), model chạy CPU | 6,8M | 26,48 / 0,968 / 0,050 | 24,75 / 0,912 / 0,125 |
| NAFNet width 64 (M4L) | 27,1M | **27,31 / 0,971 / 0,045** | **27,83 / 0,942 / 0,082** |

## Cấu trúc thư mục

```
PBL6/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     dữ liệu gốc, không sửa
│   │   ├── esp32_scan_pairs/    45 ảnh ESP32 ("ảnh esp"), 45 ảnh scan ("ảnh scan"), image_page_order.csv
│   │   └── source_pdf/          PDF gốc của tài liệu (lớp text dùng sinh nhãn)
│   ├── interim/
│   │   └── esp32_alignment/     căn chỉnh ESP32 ↔ scan ↔ PDF (src/n1_align.py)
│   ├── processed/
│   │   └── esp32_pairs/         trang đã nắn, cặp patch, crop dòng, nhãn OCR (src/n2_build_data.py)
│   └── external/                RealDAE và tập test tổng hợp: nằm trên Google Drive (xem README trong đó)
├── src/                         toàn bộ code đang dùng
│   ├── paths.py                 đường dẫn mặc định của dự án
│   ├── n1_align.py              [dữ liệu] căn chỉnh ảnh ESP32 với scan và PDF
│   ├── n2_build_data.py         [dữ liệu] sinh cặp patch ESP32 → scan, crop dòng, nhãn
│   ├── prep_realdae.py          [dữ liệu] chuẩn hóa RealDAE về {train,val,test}/{inp,gt}
│   ├── degrade.py               [dữ liệu] sinh ảnh suy biến tổng hợp (6 loại lỗi, 3 mức)
│   ├── restore_model.py         [model]   NAFNet, U-Net, nạp checkpoint, suy luận chia ô
│   ├── enhance.py               [model]   phương pháp cổ điển (đối chứng)
│   ├── dewarp.py, uvdoc_model.py [model]  làm phẳng: 4 góc trang (OpenCV) và UVDoc đã train sẵn (MIT)
│   ├── train_restore.py         [train]   train từ đầu, nhiều nguồn dữ liệu, --resume; ENH-M5: --edge-w, --paper-color, --identity-p
│   ├── eval_sets.py             [đánh giá] PSNR / SSIM / LPIPS trên tập {inp,gt}
│   ├── restore_eval.py          [đánh giá] trang ESP32 so với scan, sinh crop dòng để đo CER
│   ├── n2_eval.py               [đánh giá] CER / WER bằng VietOCR
│   ├── eval_dewarp.py           [đánh giá] độ phẳng so với scan (SIFT + phép đồng dạng)
│   └── demo_app.py, demo_page.html  [demo] web tải ảnh lên, xem trước/sau
├── notebooks/
│   ├── PBL6_Colab_Retrain.ipynb notebook Colab: dữ liệu → train → đánh giá
│   ├── PBL6_Ket_qua_retrain.ipynb  notebook KẾT QUẢ đợt 2 (ENH-M5), có đầu ra: đường cong train, bảng, biểu đồ (+ bản .html mở không cần Jupyter)
│   ├── make_results_notebook.py sinh và chạy notebook kết quả từ models/retrain2 và reports/results/retrain2
│   ├── make_retrain_notebook.py sinh notebook + gói code
│   └── pbl6_pipeline_code.zip   gói src/ để đưa lên Drive (MyDrive/PBL6/code/)
├── models/
│   ├── retrain/<run>/           best.pt, log.jsonl, config.json của 6 run: r0, m2, m3, m4, m4_s1, m4l
│   ├── retrain2/<run>/          đợt 2 (ENH-M5): sẽ có khi Colab chạy xong
│   ├── dewarp/                  trọng số UVDoc + giấy phép MIT
│   ├── legacy/restore_v1|v2/    NAFNet bản đầu (chỉ train trên ESP32)
│   └── ocr/                     VietOCR fine-tune trên ESP32 (từ đề tài cũ)
├── reports/
│   ├── results/retrain/         JSON đánh giá từng run × từng tập, summary.csv, meta/ (log, môi trường)
│   └── figures/                 hình cho báo cáo
├── docs/                        kế hoạch, hướng dẫn, nhật ký, manifest cấu trúc lại thư mục
├── third_party/UVDoc/           repo UVDoc gốc (chỉ để tham khảo, không import trực tiếp)
├── hardware/                    firmware ESP32-CAM và script chụp ảnh
└── archive/ocr_topic/           đề tài OCR cũ (chỉ để tra cứu, xem README trong đó)
```

Thư mục ẩn giữ nguyên chỗ cũ:

- `.venv-ocr/`, `.venv-paddle/`: môi trường Python, không chuyển được vì chứa đường dẫn tuyệt đối;
- `.cache/ocr/models/`: trọng số VietOCR gốc;
- `.tools/`, `.claude/`, `.mcp.json`: cấu hình công cụ.

## Chạy

Web demo (tải ảnh lên, chọn ENH-M4 / ENH-M2 / M4L / cổ điển, kéo thanh để so trước/sau, tải ảnh ra):

```bash
.venv-ocr/Scripts/python.exe -X utf8 src/demo_app.py --port 8010
```

Mở http://localhost:8010. Trang cạnh dài 1024 px mất khoảng 2–3 giây với ENH-M4 trên CPU. Thêm `--host 0.0.0.0` để máy khác trong mạng LAN mở được.

Chấm một tập ảnh trên máy (Windows, CPU):

```bash
.venv-ocr/Scripts/python.exe -X utf8 src/eval_sets.py --set <thư mục có inp/ và gt/> --method model --ckpt models/retrain/m4_nafnet_all/best.pt --out reports/results/thu.json
```

Nâng một ảnh bằng model ENH-M4:

```python
import sys, cv2
sys.path.insert(0, "src")
from restore_model import load_model, restore_image
model = load_model("models/retrain/m4_nafnet_all/best.pt")          # CPU
out = restore_image(model, cv2.imread("anh_vao.jpg"))               # khoảng 2–3 giây cho ảnh cạnh dài 1024 px trên CPU
cv2.imwrite("anh_ra.png", out)
```

Train lại trên Colab:

1. Chạy `python notebooks/make_retrain_notebook.py` để sinh gói code.
2. Đưa `notebooks/pbl6_pipeline_code.zip` lên `MyDrive/PBL6/code/`.
3. Mở `notebooks/PBL6_Colab_Retrain.ipynb` trên Colab và chạy từ trên xuống.

Chi tiết: [docs/HUONG_DAN_TRAIN_LAI.md](docs/HUONG_DAN_TRAIN_LAI.md).

## Hoàn tác việc cấu trúc lại thư mục (16/11/2026)

- Mọi lần chuyển được ghi trong `docs/restructure_manifest_20261116.json`.
- Chạy `python docs/undo_restructure_20261116.py` để trả file về chỗ cũ.
- Code trong `src/` đã được sửa đường dẫn theo cấu trúc mới, nên sau khi hoàn tác cần sửa đường dẫn lại.
