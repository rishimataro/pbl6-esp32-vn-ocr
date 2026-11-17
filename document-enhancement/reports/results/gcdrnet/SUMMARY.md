# So sánh với GCDRNet (trọng số tác giả, chạy zero-shot)

Ngày 17/11/2026. GCDRNet (Zhang và cộng sự, IEEE TAI 2023, MIT) chạy bằng `src/run_gcdrnet.py` với trọng số `gcnet` và `drnet` do tác giả công bố, trên cùng ảnh đầu vào và cùng script chấm `src/eval_sets.py` (PSNR RGB, SSIM xám) như các model của đề tài. Không chạy lại huấn luyện GCDRNet. Chạy trên CPU, khoảng 0,7 giây mỗi ảnh 1024 px.

**Kiểm tra giao thức.** Chạy lại M4 trên 150 ảnh RealDAE test bằng cùng script cho 26,48 dB / 0,968, trùng kết quả trước đó.

| Tập | Ảnh vào | Cổ điển | GCDRNet | M4 (6,8 M) | M4-L (27,1 M) |
|---|---|---|---|---|---|
| RealDAE test (150) | 12,40 | 17,77 | 25,01 / 0,949 | 26,48 / 0,968 | **27,31** / 0,971 |
| Tổng hợp (1.260) | 17,68 | 14,97 | 19,19 / 0,813 | 24,75 / 0,912 | **27,83** / 0,942 |
| DocAligner (130) | 13,96 | 17,35 | 19,27 / 0,834 | **20,16** / 0,863 | 20,02 / 0,860 |
| Jung (87) | 14,87 | 18,36 | 16,57 / 0,829 | **16,89** / 0,851 | 16,50 / 0,827 |

Số công bố trên RealDAE cho GCDRNet là 24,42 dB; ta đo 25,01 dB bằng giao thức của mình (cạnh dài 1024 px), nên hai số cùng cỡ.

## Nhận xét

1. **M4 hơn GCDRNet trên cả bốn tập**, khoảng 1,5 dB trên RealDAE, 5,6 dB trên tập tổng hợp, 0,9 dB trên DocAligner, 0,3 dB trên Jung (mức Jung nằm trong nhiễu). M4 chỉ có 6,8 triệu tham số.
2. **Điều kiện không ngang nhau.** M4 train trên RealDAE train, ESP32 và dữ liệu tổng hợp do đề tài sinh; GCDRNet dùng trọng số của tác giả, dữ liệu huấn luyện của nó chưa được kiểm tra ở đây. Tập tổng hợp có các loại suy biến do chính đề tài thiết kế, nên chênh 5,6 dB ở đó thiên về M4. RealDAE và DocAligner là so sánh có ý nghĩa nhất.
3. **GCDRNet yếu nhất ở thiếu sáng nặng** (15,1 dB trên `lowlight`, M4: 23,9 dB) và hỗn hợp nhiều lỗi (`mix` 14,4 dB so với 20,9 dB). Tốt hơn ảnh vào ở bóng đổ, ám màu, nhưng kém ảnh vào ở nhòe (20,1 so với 22,3) và JPEG (22,4 so với 29,7), giống các model của đề tài.
4. Chưa train lại GCDRNet với dữ liệu của đề tài nên chưa tách được ảnh hưởng của dữ liệu và của kiến trúc. Chưa đo CER của GCDRNet trên ESP32.
