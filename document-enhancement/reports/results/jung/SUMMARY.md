# Thử trên tập Jung et al. (ACCV 2018)

Ngày 16/11/2026.

**Dữ liệu:**

- 87 ảnh tài liệu chụp bằng điện thoại, có bóng đổ (tay, vật che), kích thước khoảng 700×500. Mỗi ảnh có một ảnh chuẩn chụp dưới ánh sáng đều.
- Nguồn: tài liệu bổ sung của bài báo trên Springer (`484511_1_En_25_MOESM1_ESM.zip`). Bản gốc ở `data/external/jung_raw/`, bản chuẩn hóa ở `data/external/jung/{inp,gt}`.
- Gói tải về có sẵn kết quả của các phương pháp đã công bố trên đúng 87 ảnh này. Các phương pháp đó được chấm lại bằng cùng script (`eval_sets.py --method pred`), nên so sánh là công bằng.

**Cách đo:** PSNR trên RGB, SSIM trên ảnh xám. Không đo LPIPS vì máy chưa cài gói `lpips`. Ảnh ra của model ở `pred/<run>/`.

## Kết quả (87 ảnh)

| Phương pháp | PSNR | SSIM |
|---|---|---|
| Water (công bố, K 0,85, KS 5) | **23,38** | 0,904 |
| ImageJ rolling ball 50 (công bố) | 20,76 | **0,910** |
| Inpaint (công bố) | 20,14 | 0,886 |
| Jung CVPR (công bố) | 19,04 | 0,830 |
| Cổ điển 1a (`enhance.py`) | 18,36 | 0,806 |
| M4 NAFNet (của mình) | 16,89 | 0,851 |
| M4 seed 1 | 16,81 | 0,847 |
| M2 U-Net | 16,80 | 0,846 |
| M3 NAFNet chỉ RealDAE | 16,67 | 0,840 |
| M4L NAFNet width 64 | 16,50 | 0,827 |
| R0 NAFNet chỉ ESP32 | 15,29 | 0,759 |
| Ảnh vào, không xử lý | 14,87 | 0,884 |

## Vì sao model của mình thấp: lệch tông màu, không phải khử bóng kém

Xem `reports/figures/jung_m4_vs_water.png`.

- M4 khử bóng tốt và cho ảnh kiểu "scan": nền trắng tinh (phân vị 90 của độ sáng là 255) và chữ đậm.
- Ảnh chuẩn của Jung là ảnh chụp dưới ánh sáng đều: nền giấy xám nhạt (phân vị 90 khoảng 233) và độ tương phản thấp.
- PSNR phạt nặng phần lệch tông màu trên toàn ảnh này.

**Kiểm tra:** học một đường cong tông màu chung (bảng tra 256 mức cho mỗi kênh) trên 20 ảnh, rồi áp cho 67 ảnh còn lại. Mỗi phương pháp có đường cong riêng nhưng dùng chung cho mọi ảnh, nên không dùng ảnh chuẩn của 67 ảnh test.

| Phương pháp (67 ảnh) | PSNR gốc | PSNR sau chỉnh tông chung |
|---|---|---|
| Water | 23,40 | **24,95** |
| M4 | 16,92 | 24,44 |
| M2 | 16,83 | 24,35 |
| M4L | 16,53 | 24,03 |
| ImageJ | 20,80 | 23,75 |
| Ảnh vào | 14,93 | 23,19 |

**Kết luận:**

1. Sau khi bỏ phần lệch tông màu, M4 hơn ảnh vào 1,25 dB và hơn ImageJ. M4 vẫn kém Water khoảng 0,5 dB; Water là phương pháp tác giả thiết kế cho chính loại ảnh này.
2. Riêng ảnh vào đã tăng từ 14,9 lên 23,2 dB chỉ nhờ chỉnh tông màu. Nghĩa là PSNR trên Jung đo **tông màu** là chính. Bóng đổ chỉ chiếm một phần nhỏ diện tích ảnh nên ảnh hưởng ít.
3. Model lớn (M4L) không hơn M4 trên tập này. Cải thiện của M4L trên RealDAE và tập tổng hợp không chuyển sang được kiểu ảnh khác.
4. Khi viết báo cáo:
   - Nêu rõ PSNR phụ thuộc vào "phong cách" ảnh chuẩn. Model mình được dạy cho ra ảnh kiểu scan, nền trắng.
   - Muốn hợp với kiểu ảnh chuẩn như Jung thì cần thêm một bước chỉnh tông nhẹ, hoặc fine-tune trên dữ liệu cùng kiểu.
   - Bóng đổ cứng vẫn là điểm yếu nên thử thêm, ví dụ trên RDD hoặc SD7K.
