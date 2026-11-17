# Làm phẳng ảnh tài liệu (ESP32 và DocUNet)

Ngày 16/11/2026.

**Hai cách làm phẳng** (đều trong `src/dewarp.py`):

| Cách | Làm gì | Sửa được | Tốc độ (CPU) |
|---|---|---|---|
| `corners` | tìm 4 góc trang (OpenCV), biến đổi phối cảnh | chụp nghiêng; cắt bỏ nền | 0,01 s |
| `uvdoc` | UVDoc (Verhoeven et al., SIGGRAPH Asia 2023), model đã train sẵn, giấy phép MIT, trọng số ở `models/dewarp/` | phối cảnh, giấy cong, giấy nhăn | 0,3 s |
| `corners+uvdoc` | cắt trang bằng 4 góc, rồi UVDoc sửa phần cong còn lại | cả hai | 0,3–0,5 s |

Không train lại UVDoc.

**Cách đo độ phẳng** (`src/eval_dewarp.py`):

- Ghép điểm SIFT giữa ảnh đã làm phẳng và bản scan.
- Khớp một phép đồng dạng: xoay, dịch, co giãn đều.
- Báo sai lệch còn lại theo % chiều rộng trang. Trang càng phẳng thì sai lệch càng nhỏ.
- Số này gần với ý tưởng của chỉ số LD trên DocUNet nhưng **không so được với số công bố**.

## Kết quả trên 10 trang test ESP32 (ảnh gốc 1600×1200, còn nền)

| Cách | Sai lệch trung vị | Phân vị 90 |
|---|---|---|
| Không làm phẳng | 0,13 % | 0,54 % |
| 4 góc | 0,27 % | 0,69 % |
| UVDoc | 1,10 % | 5,24 % |
| 4 góc + UVDoc | 0,71 % | 1,83 % |

Hình: `reports/figures/dewarp_esp32.jpg`. Mỗi hàng gồm ảnh gốc, ảnh làm phẳng bằng 4 góc và ảnh làm phẳng bằng UVDoc.

**Nhận xét:**

1. **Trang ESP32 đã gần như phẳng và chụp thẳng.** Máy được đặt cố định phía trên trang giấy. Vì vậy làm phẳng không cải thiện hình học, sai lệch vốn đã chỉ khoảng 0,1 %.
2. **Cách 4 góc** cắt đúng trang ở cả 10 ảnh và bỏ được nền đen. Cách này có ích khi trình bày kết quả, nhưng làm sai lệch tăng nhẹ vì mép trang tìm được lệch vài pixel.
3. **UVDoc trên ảnh còn nhiều nền thì kém.** Nó giữ nền đen và làm mép trang lượn sóng. UVDoc được train trên ảnh mà trang chiếm gần hết khung hình, nên phải cắt trang trước rồi mới đưa vào.
4. **Trên trang vốn đã phẳng, UVDoc thêm méo nhẹ** (khoảng 0,7 %) và phóng to vào phần chữ. Chỉ nên bật UVDoc khi giấy cong thật.
5. Tập ESP32 không có trang cong, nên phần giấy cong được đo trên DocUNet ở dưới.

Chế độ `auto` (thêm sau khi đo DocUNet) trên ESP32 có sai lệch trung vị 0,76 %, phân vị 90 là 1,85 %. Mức này ngang "4 góc + UVDoc".

## Kết quả trên DocUNet (130 ảnh giấy cong, gập, nhăn)

**Dữ liệu:**

- Ảnh vào: `crop.zip` của DocUNet benchmark, tải từ trang Stony Brook ngày 16/11/2026. Nằm ở `data/external/docunet_raw/crop/`.
- Ảnh chuẩn: bản scan phẳng `data/external/docaligner_raw/<id>_target.png`.

**Chỉ số** (`src/eval_docunet.py`), cài đặt lại bằng Python nên **không so trực tiếp với số trong các bài báo**:

- **MS-SSIM:** ảnh xám, diện tích 598 400 px, 5 mức. Cao hơn là tốt hơn.
- **LD:** độ dời trung bình, tính bằng pixel, của luồng quang học DIS từ scan sang ảnh ra. Bản gốc dùng SIFT Flow trên MATLAB. Thấp hơn là tốt hơn.
- **Kiểm tra:** scan so với chính nó cho MS-SSIM 1,0 và LD 0.

| Cách | MS-SSIM ↑ | LD ↓ | giây/ảnh (CPU) |
|---|---|---|---|
| Không làm phẳng | 0,215 | 45,3 | — |
| 4 góc | 0,281 | 35,4 | 0,02 |
| 4 góc + UVDoc | 0,318 | 30,6 | 0,31 |
| **UVDoc** | **0,447** | **15,3** | 0,31 |
| auto (cắt 4 góc khi trang < 60 % khung, rồi UVDoc) | 0,440 | 16,9 | 0,29 |
| M4 không làm phẳng | 0,192 | 44,0 | 1,9 |
| **UVDoc + M4 (pipeline đầy đủ)** | **0,451** | **12,7** | 2,4 |

Hình: `reports/figures/dewarp_docunet.jpg`. Ba hàng là ảnh trung vị, tốt nhất và tệ nhất xét theo LD. Bốn cột là ảnh chụp cong, UVDoc, UVDoc + M4 và scan.

**Nhận xét:**

1. **UVDoc làm phẳng rất hiệu quả trên giấy cong.** LD giảm từ 45,3 xuống 15,3, tức khoảng 3 lần. UVDoc tốt hơn ảnh vào ở 128/130 ảnh theo LD và 129/130 ảnh theo MS-SSIM.
2. **Pipeline đầy đủ UVDoc + M4 cho kết quả tốt nhất.** Chỉ chạy M4 thì không giúp được ảnh cong, vì M4 không sửa hình học.
   - Phần LD giảm thêm nhờ M4 (15,3 → 12,7) một phần đến từ việc ảnh sạch hơn giúp luồng quang học ghép điểm dễ hơn, chứ không phải hình học tốt hơn.
3. **Cắt 4 góc trước UVDoc chỉ có lợi khi ảnh còn nhiều nền.**
   - Ảnh DocUNet đã cắt sát trang. Trên trang cong, cách 4 góc dễ bắt nhầm mép, nên "4 góc + UVDoc" kém hẳn UVDoc một mình.
   - Ảnh ESP32 thì ngược lại: phải cắt trước, nếu không UVDoc sẽ giữ cả nền đen.
   - Chế độ `auto` dung hòa được cả hai và được dùng làm lựa chọn làm phẳng chính trong demo.
4. **Giới hạn:**
   - Giấy vò nhàu nặng (ví dụ 46_2) vẫn còn dòng chữ lượn sóng. Tỉ lệ khung cũng khác bản scan nên LD còn cao.
   - Ảnh hóa đơn nền xanh (5_1) được M4 tẩy trắng. Đây là giới hạn đã ghi ở phần DocAligner.

## Tích hợp

- **Web demo:** thêm ô "Làm phẳng" (không / tự động / 4 góc / 4 góc + UVDoc / chỉ UVDoc). Bước làm phẳng chạy ở độ phân giải gốc, sau đó mới thu nhỏ và nâng chất lượng.
- **Dòng lệnh:**

```bash
.venv-ocr/Scripts/python.exe -X utf8 src/dewarp.py --inp anh.jpg --out ra/ --method auto --enhance models/retrain/m4_nafnet_all/best.pt
```
