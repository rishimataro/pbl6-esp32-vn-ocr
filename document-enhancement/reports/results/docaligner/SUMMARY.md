# Thử trên tập DocUNet-DocAligner

Ngày 16/11/2026.

**Dữ liệu:**

- 130 ảnh tài liệu chụp bằng điện thoại: báo, tạp chí, hóa đơn, tài liệu in màu.
- Mỗi ảnh đã được DocAligner căn khớp từng pixel với ảnh scan chuẩn. Ảnh chuẩn có nền trắng kiểu scan, cùng kiểu với cách model được dạy.
- Bản gốc ở `data/external/docaligner_raw/` (`<id>_docaligner.png` là ảnh vào, `<id>_target.png` là ảnh chuẩn).
- Đã thu nhỏ để cạnh dài là 1024 px, giống RealDAE. Bản dùng để chấm ở `data/external/docaligner/{inp,gt}`.

**Cách đo:** PSNR trên RGB, SSIM trên ảnh xám, chạy trên CPU máy. Không có kết quả công bố nào chấm theo đúng cách này để so.

## Kết quả (130 ảnh)

| Phương pháp | PSNR | SSIM | giây/ảnh (CPU) |
|---|---|---|---|
| **M4 NAFNet** | **20,16** | **0,863** | 1,7 |
| M4L NAFNet width 64 | 20,02 | 0,860 | 4,9 |
| M2 U-Net | 19,98 | 0,858 | 1,0 |
| M3 NAFNet chỉ RealDAE | 19,94 | 0,859 | 1,8 |
| Cổ điển 1a | 17,35 | 0,810 | 0,8 |
| R0 NAFNet chỉ ESP32 | 16,78 | 0,739 | 1,7 |
| Ảnh vào, không xử lý | 13,96 | 0,790 | — |

**Nhận xét:**

- **M4 hơn rõ.** So với ảnh vào, M4 tăng 6,2 dB. M4 hơn ảnh vào ở 122/130 ảnh và hơn cổ điển ở 122/130 ảnh.
- **Các model train trên nhiều nguồn ngang nhau.** M4, M4L, M2, M3 chỉ cách nhau dưới 0,25 dB, cỡ bằng độ dao động giữa hai seed (~0,2 dB). Model lớn M4L chỉ hơn M4 ở 45/130 ảnh, nên trên ảnh thật ngoài RealDAE, bản nhỏ là đủ.
- **Chỉ train trên ESP32 thì không tổng quát được.** R0 đạt 16,8 dB, kém cả cổ điển.

## Trường hợp model làm hỏng

Xem `reports/figures/docaligner_m4.png`. Ba hàng là ảnh trung vị, ảnh tốt nhất và ảnh tệ nhất xét theo mức tăng so với ảnh vào.

- **Tốt nhất:** hóa đơn giấy nhăn, nền xám xanh. PSNR tăng từ 10,3 lên 29,1 dB.
- **Tệ nhất:** trang in **nền màu có chủ ý** (nền xanh lá hoặc xám), ví dụ ảnh 26_1, 26_2, 8_2. PSNR giảm tới 6 dB.
  - Model coi màu nền in sẵn là ám màu do ánh sáng và tẩy về trắng.
  - Đây là giới hạn cần nêu trong báo cáo: model chưa phân biệt được "giấy màu hoặc nền in màu" với "ánh sáng sai màu".
  - Hướng sửa: thêm dữ liệu trang có nền màu vào ảnh chuẩn khi train, hoặc chỉ cân bằng màu theo vùng giấy trắng.
