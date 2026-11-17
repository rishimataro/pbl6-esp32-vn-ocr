# Ảnh nâng chất lượng có giúp OCR đọc tốt hơn không? (trang ESP32-CAM)

Ngày 16/11/2026.

**Dữ liệu:** 10 trang test chụp bằng ESP32-CAM (trang 3, 4, 7, 9, 11, 14, 25, 32, 38, 44), gồm 317 dòng chữ.

- Các trang đã được nắn khớp với bản scan.
- Nhãn đúng của từng dòng lấy từ lớp text của PDF gốc.

**Cách đo:**

- Cho từng model xử lý cả trang trên CPU (`src/restore_eval.py`).
- Cắt từng dòng ở cùng một vị trí trên mọi phiên bản trang.
- Đọc bằng **VietOCR vgg_transformer bản pretrained, không fine-tune** (`src/n2_eval.py`).
- CER là tỉ lệ ký tự sai. "CER bỏ dấu" là CER sau khi xóa hết dấu thanh và dấu mũ ở cả nhãn lẫn kết quả đọc. Hiệu giữa hai số là phần lỗi chỉ do dấu.

## Kết quả

| Phiên bản trang | CER | CER bỏ dấu | Lỗi chỉ do dấu | Số dòng tốt hơn / kém hơn ảnh vào | PSNR so với scan |
|---|---|---|---|---|---|
| Ảnh vào ESP32 | 21,0 % | 17,2 % | 3,9 | — | 5,8 |
| Cổ điển 1a | 19,8 % | 15,9 % | 3,9 | 119 / 128 | — |
| **M4 NAFNet (model chung)** | **13,7 %** | 8,7 % | 5,1 | 195 / 74 | 17,1 |
| M2 U-Net | 14,4 % | 10,3 % | 4,1 | 193 / 79 | 17,4 |
| R0 NAFNet chỉ ESP32 | 12,4 % | 7,1 % | 5,3 | 203 / 56 | 17,5 |
| (cũ) restore_v1, chỉ ESP32 | 12,2 % | 7,0 % | 5,3 | 207 / 64 | 17,1 |
| Bản scan (trần trên) | 2,8 % | 2,6 % | 0,2 | 289 / 6 | — |

## Nhận xét

1. **Nâng ảnh giúp OCR rõ rệt.** M4 giảm CER từ 21,0 % xuống 13,7 % (giảm 35 % tương đối). M4 đọc tốt hơn ảnh vào ở 195/317 dòng. Phương pháp cổ điển gần như không giúp: số dòng tốt hơn và kém hơn ngang nhau.
2. **Lỗi chữ cái giảm một nửa, nhưng lỗi dấu lại tăng.**
   - CER bỏ dấu giảm từ 17,2 % xuống 8,7 %.
   - Phần lỗi chỉ do dấu tăng từ 3,9 lên 5,1 điểm. Với M4, khoảng 37 % số lỗi còn lại là lỗi dấu.
   - Có 25 dòng M4 đọc đúng hết chữ cái nhưng sai dấu, ví dụ "Giới" → "Giói", "trùm" → "trúm", "với" → "vỏi".
   - Lý do: model làm nét chữ sắc và đậm hơn, nhưng dấu nhỏ (móc ơ/ư, huyền/sắc) bị làm mờ hoặc biến dạng. Đây là điểm yếu chính cần nêu trong báo cáo.
   - Hướng sửa: thêm loss có trọng số cao ở vùng nét mảnh (ví dụ loss theo gradient hoặc loss OCR), hoặc tăng độ phân giải xử lý.
3. **Model riêng cho ESP32 (R0) chỉ hơn model chung M4 1,3 điểm CER.** Nhưng R0 kém hẳn trên ảnh điện thoại: 16,8 dB so với 20,2 dB trên DocAligner. Vì vậy M4 là lựa chọn cân bằng nhất.
4. **PSNR không phản ánh đúng độ dễ đọc.** M2 có PSNR cao hơn M4 (17,4 so với 17,1) nhưng CER lại kém hơn. Báo cáo nên dùng cả chỉ số ảnh lẫn CER.
5. **Vẫn còn cách xa bản scan (2,8 %).** Ảnh ESP32 độ phân giải thấp và mờ nên mất thông tin, model chỉ phục hồi được một phần.

**Tệp trong thư mục này:**

- `summary.json`: bảng trên;
- `pred_test_<phiên bản>_vgg_transformer.jsonl`: nhãn đúng và kết quả đọc của từng dòng;
- `restore_<run>.json`: PSNR/SSIM theo trang;
- `run.log`: log chạy.
