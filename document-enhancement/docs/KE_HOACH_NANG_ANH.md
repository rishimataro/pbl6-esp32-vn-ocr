# Kế hoạch đề tài mới: nâng chất lượng ảnh tài liệu (6 ngày)

Ngày lập: 11/11/2026. Thay cho hướng OCR đầu-cuối trong `archive/ocr_topic/docs/KE_HOACH_HOAN_THIEN.md`.

## 1. Đề tài

**Nâng chất lượng ảnh tài liệu chụp bằng camera trong điều kiện xấu: thiếu sáng, nhiễu, ám màu, sáng không đều và nhòe.**

- **Đầu vào:** ảnh chụp trang tài liệu từ bất kỳ camera nào: điện thoại, webcam, ESP32-CAM.
- **Đầu ra:** ảnh trang sáng đều, sạch nhiễu, nền giấy trắng, chữ rõ, giống ảnh scan.
- **Đánh giá:**
  - so với ảnh chuẩn: PSNR, SSIM, LPIPS;
  - ảnh có dễ đọc hơn không: CER của OCR, dùng VietOCR cho tiếng Việt.

## 2. Dữ liệu

| Bộ | Nội dung | Vai trò |
|---|---|---|
| **RealDAE** | 600 cặp ảnh chụp thật ↔ ảnh chuẩn, căn khớp từng pixel. Chia 450 train / 150 test. MIT, tải từ OneDrive trong repo GCDRNet. | Benchmark chính, có sẵn kết quả đã công bố để so |
| **Tổng hợp** | Ảnh sạch (ảnh chuẩn của RealDAE, 45 trang scan, trang PDF render) bị làm hỏng có điều khiển bằng `degrade.py` | Tăng dữ liệu train. Đo riêng từng loại lỗi. |
| **ESP32-CAM** | 45 cặp ESP32 ↔ scan đã căn (N1). Dùng chia train/val/test cũ. | Test thật với camera rất rẻ, thiếu sáng nặng, tài liệu tiếng Việt (đo được CER) |

Các loại suy biến tổng hợp:

| Loại | Cách tạo |
|---|---|
| Thiếu sáng | hệ số sáng 0,15–0,5, gamma 1,5–2,5 |
| Sáng không đều / bóng | nhân với bản đồ ánh sáng mượt (gradient, vignette, vệt bóng mềm) |
| Nhiễu cảm biến | Poisson + Gaussian, nhiễu mạnh hơn ở vùng tối, σ 5–25 |
| Ám màu | nhân hệ số riêng cho R, G, B |
| Nhòe | Gaussian blur hoặc motion blur 0,5–2 px |
| Nén JPEG | quality 30–80 |

Mỗi loại có một tập test riêng: 150 ảnh, 3 mức nhẹ/vừa/nặng.
Tập train trộn ngẫu nhiên nhiều loại cùng lúc.

## 3. Kiến trúc chính: NAFNet

- **Giữ nguyên:** NAFNet bản nhỏ trong `src/restore_model.py`, 6,82M tham số, chạy được trên CPU.
- **Cấu trúc:** U-Net 4 tầng.
  - Mỗi khối là NAFBlock: LayerNorm, conv 1×1, conv depthwise 3×3, SimpleGate (nhân hai nửa kênh thay cho hàm kích hoạt), Simplified Channel Attention.
  - Học phần dư: đầu ra = ảnh vào + ảnh dự đoán.
- **Loss:** L1 + 0,2·(1 − SSIM).
- **Train:** patch 256, ảnh lớn xử lý theo ô (`restore_image(..., tile=...)` đã có).
- **Vì sao chọn NAFNet:**
  - kiến trúc chuẩn cho khử nhiễu và khử nhòe (SOTA trên SIDD và GoPro năm 2022);
  - nhẹ hơn Restormer nhiều;
  - code và pipeline train đã chạy được trên Colab, mỗi lần chạy khoảng 75 phút.

## 4. So sánh

| Mã | Phương pháp | Trạng thái |
|---|---|---|
| ENH-M0 | Ảnh vào, không xử lý | — |
| ENH-M1 | Cổ điển: cân bằng trắng, chia nền, NLM, CLAHE, unsharp (`enhance.py`) | có |
| ENH-M2 | U-Net thường, cùng loss và số vòng lặp | cần train |
| ENH-M3 | NAFNet, chỉ RealDAE train | cần train |
| ENH-M4 | NAFNet, RealDAE + tổng hợp + ESP32 train (**model đề xuất**) | cần train |
| GCDRNet | GCDRNet có trọng số pretrained (MIT), chạy zero-shot | chỉ chạy suy luận |

Kết quả đã công bố trên RealDAE test để đối chiếu (theo repo Recommendations-Document-Image-Processing):

| Phương pháp | PSNR | SSIM |
|---|---|---|
| Ảnh vào | 12,26 | 0,826 |
| DocProj (TOG'19) | 19,35 | 0,868 |
| GCDRNet (TAI'23) | 24,42 | 0,942 |
| DocRes (CVPR'24) | 24,65 | 0,922 |

Mục tiêu thực tế: ENH-M4 vượt DocProj, tiến gần GCDRNet.
Với 450 cặp train và 1 tuần, không hứa vượt SOTA.

Bảng kết quả cuối có 3 phần:

1. RealDAE test: PSNR / SSIM / LPIPS;
2. từng loại suy biến tổng hợp: PSNR theo mức;
3. ESP32 test: PSNR / SSIM / CER.

## 5. Lịch 6 ngày

| Ngày | Việc | Kết quả |
|---|---|---|
| 1 | Tải RealDAE và trọng số GCDRNet lên Drive. Viết `degrade.py` và tạo các tập test tổng hợp. Sửa `train_restore.py`: đọc nhiều nguồn dữ liệu, sinh ảnh hỏng tổng hợp khi train, thêm `--arch unet`. | dữ liệu + script |
| 2 | Train trên Colab: ENH-M3, ENH-M4, ENH-M2 (khoảng 4 giờ GPU). Chạy GCDRNet trên các tập test. | checkpoint |
| 3 | Đánh giá đủ 3 bảng. Hình trước/sau theo từng loại lỗi. Đo tốc độ CPU. | bảng + hình |
| 4 | Demo web: tải lên ảnh bất kỳ (điện thoại hoặc ESP32), hiện trước/sau, tải ảnh đã nâng về, OCR là tùy chọn. Sửa từ `n9_demo.py`. | demo chạy |
| 5 | Viết lại báo cáo. Giữ phần căn chỉnh ESP32 làm cách tạo dữ liệu cặp. Bỏ layout, bảng, DOCX. Làm slide. | báo cáo + slide |
| 6 | Dự phòng, tập thuyết trình. | |

## 6. Cắt bỏ khỏi phạm vi

- Layout, nhận bảng, ghép DOCX, fine-tune VietOCR: chỉ nhắc ở mục ứng dụng và hướng phát triển.
- Nắn trang (dewarping) và khử bóng mạnh: ngoài phạm vi. Ảnh vào giả định đã thấy đủ trang. Bóng mềm được xử lý như sáng không đều.

## 7. Rủi ro

- **Link OneDrive của RealDAE không tải được bằng script.** Tải tay bằng trình duyệt rồi đưa lên Drive. Nếu không tải được: dùng dữ liệu tổng hợp + ESP32, và chỉ so với GCDRNet trên tập tổng hợp.
- **Hết quota Colab.** Thứ tự ưu tiên là ENH-M4 > ENH-M3 > ENH-M2. Rút ENH-M2 xuống 4.000 vòng.
- **ENH-M4 thua GCDRNet.** Đây vẫn là kết quả báo cáo được. Nhấn mạnh model nhỏ hơn và chạy CPU được, có đo số tham số và thời gian chạy của cả hai.
