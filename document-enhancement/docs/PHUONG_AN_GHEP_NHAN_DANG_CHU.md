# Phương án ghép phần nhận dạng chữ (text recognition) vào đề tài nâng chất lượng ảnh

Ngày 17/11/2026. Tài liệu nghiên cứu, chưa thay đổi gì trong mã hay báo cáo. Bản sao lưu trước khi ghép: `archive/backups/PBL6_truoc_ghep_OCR_20261117.zip` (181 tệp, 45 MB: mã, tài liệu, notebook, bảng kết quả, checkpoint ENH-M4) và tag git `backup-truoc-ghep-ocr-20261117` trong repo nhóm. File Word `BAO_CAO_PBL6.docx` không có trong zip vì lúc đó đang bị Word khóa; dựng lại được từ `BAO_CAO_PBL6.md`.

## 1. Phần nhận dạng chữ của nhóm gồm những gì

Nguồn: thư mục Drive `PBL6` (chủ sở hữu tvmthtr2), notebook `ocr pbl.ipynb`, gói dữ liệu `dataset_pack.zip` (1,09 GB, **chưa tải**).

| Hạng mục | Nội dung |
|---|---|
| Bài toán | Nhận dạng **một dòng chữ tiếng Việt** từ ảnh cắt sẵn (text recognition, không gồm phát hiện dòng) |
| Model | PP-OCRv4 recognizer của PaddleOCR (SVTR-LCNet, backbone PPLCNetV3, đầu ra kép CTC + NRTR), đầu vào 3×48×320, tối đa 64 ký tự, từ điển 282 ký tự tiếng Việt |
| Huấn luyện | PaddleOCR `release/2.7`, Adam lr 5e-4, cosine, warmup 5 epoch, batch 64, 15 hoặc 20 epoch, chạy trên Colab |
| Dữ liệu train | 60.000 dòng sinh tổng hợp: bản "clean" (100 % sạch) và bản "multi-tier" (sạch 25 %, kiểu scan 45 %, kiểu ESP32 30 %) |
| Dữ liệu test | Scan: 155 dòng đã chấm (165 trong gói), ESP32-CAM: 44 dòng đã chấm (46 trong gói) |
| 3 model so sánh | OCR-M1: từ đầu, chỉ dữ liệu clean. OCR-M2: từ đầu, dữ liệu multi-tier. OCR-M3: tinh chỉnh từ trọng số PP-OCRv4 chính thức (tiếng Trung), dữ liệu multi-tier. OCR-M4: trọng số gốc, không train |
| Đã xuất | Model dạng Paddle Inference (`inference.pdiparams`, `inference.json`) cho OCR-M1, OCR-M2, OCR-M3 (`ppocrv4_m1/m2/m3_inference`, bản `m3_legacy`) |
| Cơ chế phụ | Sao lưu checkpoint tự động sang Drive, tự động tiếp tục khi bị ngắt |

### Kết quả nhóm đã đo

Hai tệp kết quả **không khớp nhau** nên cần hỏi lại bạn phụ trách phần này tệp nào là bản cuối.

`benchmark_results.json` (cũng là kết quả in ra trong notebook): độ chính xác theo cả dòng (acc) và 1 − NED (độ giống chuỗi, càng cao càng tốt).

| Model | Scan 155 dòng: acc / 1−NED | ESP32 44 dòng: acc / 1−NED |
|---|---|---|
| OCR-M1 từ đầu, clean | 16,77 / 50,32 | 0,00 / 4,38 |
| OCR-M2 từ đầu, multi-tier | **26,45** / 55,47 | **18,18** / 56,83 |
| OCR-M3 tinh chỉnh, multi-tier | 25,16 / 44,33 | 15,91 / 52,13 |

`empirical_benchmark_all_models.json` (có thêm OCR-M4 và tốc độ): 1−NED cao hơn rõ (OCR-M2: scan 83,06, ESP32 68,44; OCR-M3: 82,66 và 68,57; OCR-M4 gốc: 67,68 và 51,24), acc gần như cũ (OCR-M2: 22,58 và 18,18). Có thể hai lần đo dùng cách chuẩn hóa, giải mã hoặc tập lọc khác nhau.

Nhận xét của mình:
- Dữ liệu đa tầng (có kiểu ESP32) quan trọng: OCR-M1 chỉ đạt 0 % trên ảnh ESP32, OCR-M2 đạt 18 %. Kết luận này đồng nhất với kết luận của đề tài này (học từ nhiều nguồn tốt hơn một nguồn).
- Độ chính xác cả dòng còn thấp (tối đa khoảng 26 % scan, 18 % ESP32). Trên ESP32 mới chỉ có 44 dòng nên số liệu nhiễu.

## 2. So với phần hiện tại, hai bên chưa so trực tiếp được

| | Đề tài nâng ảnh (của mình) | Phần nhận dạng của nhóm |
|---|---|---|
| Model đọc chữ | VietOCR vgg_transformer, bản gốc, không tinh chỉnh | PP-OCRv4 SVTR-LCNet, tự train hoặc tinh chỉnh |
| Chỉ số | CER (tỉ lệ ký tự sai), CER bỏ dấu | acc cả dòng, 1−NED |
| Test ESP32 | 317 dòng của 10 trang (trang 3, 4, 7, 9, 11, 14, 25, 32, 38, 44) | 44 dòng, chưa rõ nguồn trang |
| Ảnh vào | ảnh ESP32 gốc, hoặc đã qua ENH-M4 | ảnh ESP32 gốc |
| Kết quả trên ESP32 | CER 21,0 % ảnh gốc, 13,7 % sau ENH-M4 (tương đương 1−NED khoảng 79 % và 86 %) | 1−NED khoảng 57–69 % (tùy tệp), acc 18 % |

1−NED và 1−CER gần nhau nhưng không bằng nhau: NED chuẩn hóa theo độ dài lớn hơn trong hai chuỗi, CER chuẩn hóa theo độ dài nhãn. Muốn kết luận phải chấm cả hai model trên **cùng tập dòng, cùng công thức**.

Hai điểm trùng tên cần đổi khi ghép: nhóm dùng OCR-M1–M4 cho các model nhận dạng; đề tài này dùng ENH-M0–M5 cho các model nâng ảnh. Đề xuất đặt tiền tố `OCR-` và `ENH-`.

## 3. Các phương án ghép

**A. Nối tiếp: nâng ảnh rồi mới nhận dạng (khuyên dùng).** Cho ảnh dòng chữ đi qua ENH-M4 trước khi vào PP-OCRv4. Đo trên cùng 317 dòng ESP32: {ảnh gốc, cổ điển, ENH-M4, scan} × {VietOCR, PP-OCRv4 M2, PP-OCRv4 M3}. Trả lời câu hỏi "nâng ảnh có giúp model của nhóm không", và model đọc nào tốt hơn. Cần: model suy luận của nhóm (nhỏ, OCR-M2 và OCR-M3 vài chục MB), PaddlePaddle để chạy.
- Giá trị: cao nhất, ghép đúng ý đề tài (nâng ảnh phục vụ nhận dạng), ít việc nhất.
- Rủi ro: model PP-OCRv4 đã học với dữ liệu "ESP32 tổng hợp", ảnh đã nâng là miền khác (nền trắng như scan); có thể nó đọc ảnh ESP32 gốc tốt hơn ảnh đã nâng. Kết quả này vẫn là kết quả báo cáo được.

**B. Dùng bộ nâng ảnh làm nguồn dữ liệu cho bộ nhận dạng.** Dùng `degrade.py` của đề tài để sinh tầng "ESP32 tổng hợp" thật hơn (thiếu sáng, ám màu, nhiễu, mờ, JPEG, có tham số), hoặc đưa ảnh đã nâng vào tập train. Train lại PP-OCRv4 trên Colab.
- Giá trị: có thể tăng acc ESP32 nhiều nhất.
- Rủi ro: tốn GPU nhiều giờ, cần tải `dataset_pack.zip` (1,09 GB, cần bạn đồng ý), kết quả phụ thuộc chất lượng nhãn và sinh dữ liệu của nhóm.

**C. So sánh hai bộ nhận dạng trên một thước đo chung.** Chạy VietOCR (CER) và PP-OCRv4 OCR-M2, OCR-M3 trên đúng 317 dòng, ghi cả CER và 1−NED. Đây là bước chuẩn bị của A, và cũng cho một bảng riêng "chọn model nhận dạng nào".

**D. Pipeline đầu-cuối trong demo web.** Thêm nút "đọc chữ" trong `demo_app.py`: ảnh trang, nâng ảnh, cắt dòng (cần bộ phát hiện dòng, chưa có trong hai phần), nhận dạng, hiện văn bản. Phần cắt dòng là việc mới.
- Giá trị: demo ấn tượng; Rủi ro: cần phát hiện dòng, chưa có; làm sau cùng.

## 4. Đề xuất thực hiện

1. **C rồi A** (trong khoảng 1 ngày):
   - lấy model suy luận OCR-M2 và OCR-M3 từ Drive của nhóm (vài chục MB);
   - chạy trên Colab với PaddlePaddle (cùng cách nhóm đã chạy, tránh cài Paddle trên Python 3.13 của máy mình);
   - chấm trên 317 dòng ESP32 với các phiên bản ảnh {gốc, cổ điển, ENH-M4, scan};
   - ghi CER và 1−NED; ghép bảng vào mục mới "Ghép với bộ nhận dạng" trong báo cáo.
2. Nếu kết quả A cho thấy lợi ích (hoặc cho thấy khoảng cách miền), mới làm B với `degrade.py` làm nguồn dữ liệu.
3. D để sau cùng, nếu còn thời gian.

Trong báo cáo ghi rõ phân công: phần nhận dạng (PP-OCRv4, dữ liệu 60.000 dòng, OCR-M1–OCR-M4) là của nhóm; đề tài này đóng góp bộ nâng ảnh, đánh giá trên ảnh ESP32 thật và phép ghép.

## 5. Việc cần bạn quyết định hoặc xác nhận

1. Phương án nào làm trước (khuyến nghị C rồi A).
2. Cho phép tải: model suy luận OCR-M2, OCR-M3 của nhóm từ Drive (ước tính vài chục MB, `inference.pdiparams` OCR-M1 là 7,7 MB), và, chỉ nếu làm B, `dataset_pack.zip` 1,09 GB.
3. Hỏi bạn phụ trách phần nhận dạng: tệp kết quả nào là bản cuối (hai tệp lệch nhau), 44 dòng ESP32 lấy từ trang nào, và có đồng ý đưa kết quả vào báo cáo chung không.
4. Cách đặt tên model khi ghép (đề xuất `OCR-M1…M4` và `ENH-M0…M5`).
