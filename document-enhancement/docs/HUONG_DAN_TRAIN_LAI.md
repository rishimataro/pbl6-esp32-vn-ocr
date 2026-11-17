# Train lại model nâng ảnh từ đầu

Ngày 11/11/2026. Đi kèm `KE_HOACH_NANG_ANH.md`.
"Từ đầu" nghĩa là:

- khởi tạo trọng số ngẫu nhiên, không nạp checkpoint cũ ở `models/legacy/`;
- dựng lại dữ liệu từ ảnh gốc;
- mỗi lần chạy có thư mục output riêng và seed cố định.

**Cập nhật 11/11:**

- Code ở mục 6 đã viết xong, nên mọi lệnh 🛠 bên dưới giờ đều chạy được.
- Đã chạy thử trên CPU với dữ liệu ESP32 và một bộ giả lập RealDAE, chưa chạy trên GPU.
- Notebook làm đủ mục 1–5: [`notebooks/PBL6_Colab_Retrain.ipynb`](../notebooks/PBL6_Colab_Retrain.ipynb).
- Gói code: `notebooks/pbl6_pipeline_code.zip`. Sinh lại cả hai bằng `notebooks/make_retrain_notebook.py`.

Ký hiệu (giữ để biết phần nào là mới):

- ✅ có từ trước;
- 🛠 viết mới ngày 11/11.

## 0. Bố cục trên Google Drive

```
MyDrive/PBL6/
  code/pbl6_pipeline_code.zip  gói code (notebooks/pbl6_pipeline_code.zip): cần tải lên lại sau mỗi lần sửa code
  data/n2_patches.zip          ESP32: cặp patch (đã có trên Drive từ N2)
  data/n2_pages_layout.zip     ESP32: trang đã nắn (đã có)
  data/n2_ocr_lines.zip        ESP32: crop dòng, nhãn, lines.jsonl (đã có)
  data/realdae_raw.zip         tải tay từ OneDrive (mục 1.2)
  checkpoints/<tên_run>/       best.pt, last.pt, log.jsonl, config.json
  results/                     JSON đánh giá, hình
```

Trên Colab, giải nén dữ liệu vào `/content/` (đĩa cục bộ, đọc nhanh). Chỉ checkpoint và kết quả ghi vào Drive.

## 1. Dữ liệu

### 1.1. ESP32 (✅, chạy trên máy, khoảng 10 phút CPU)

Chỉ cần nếu muốn dựng lại từ ảnh gốc. Nếu không, dùng luôn `data/processed/esp32_pairs`.

```
.venv-ocr/Scripts/python.exe -X utf8 src/n1_align.py --out data/interim/esp32_alignment
.venv-ocr/Scripts/python.exe -X utf8 src/n2_build_data.py
```

- Kết quả: `data/processed/esp32_pairs/patches/{train,val}/{esp,scan}/*.png`, gồm 2.374 cặp train và 381 cặp val, mỗi cặp 256×256.
- Kết quả còn có các trang đã nắn trong `pages/` để đánh giá cả trang.
- Chia train/val/test theo trang, giữ nguyên như cũ:
  - val: trang 6, 17, 21, 29, 45;
  - test: trang 3, 4, 7, 9, 11, 14, 25, 32, 38, 44.
- Ba zip `n2_patches.zip`, `n2_pages_layout.zip`, `n2_ocr_lines.zip` đã có trên `MyDrive/PBL6/data/`. Notebook giải nén cả ba vào `/content/esp32`. Chỉ cần làm lại nếu dựng lại dữ liệu.

### 1.2. RealDAE (tải tay + 🛠 `prep_realdae.py`)

1. Mở link OneDrive trong README của https://github.com/ZZZHANG-jx/GCDRNet, tải bằng trình duyệt, đưa lên `MyDrive/PBL6/data/realdae_raw.zip`.
2. Xem cấu trúc thư mục thật sau khi giải nén: tên thư mục ảnh vào và ảnh chuẩn, danh sách train/test.
3. Chạy:
   ```
   python prep_realdae.py --src /content/realdae_raw --out /content/realdae --long-side 1024 --val 50 --seed 0
   ```
   - Thu nhỏ để cạnh dài là 1024 px.
   - Tách 50 ảnh từ 450 ảnh train làm val. **Không dùng 150 ảnh test để chọn checkpoint.**
   - Cắt patch 256 cho train/val, cùng cấu trúc `patches/{train,val}/{esp,scan}` như ESP32 để `train_restore.py` đọc chung được.
   - Giữ ảnh test nguyên trang ở `pages/test/{inp,gt}`.

### 1.3. Dữ liệu tổng hợp (🛠 `degrade.py`)

- **Khi train:** sinh ngay trong vòng lặp từ patch ảnh chuẩn, với xác suất `--synth`. Không cần lưu ra đĩa.
- **Khi test:** tạo một lần với seed cố định, để mọi model được chấm trên cùng một bộ ảnh:
  ```
  python degrade.py --clean /content/realdae/pages/test/gt /content/esp32/pages/scan --pages-split val,test \
      --out /content/synth_test --types lowlight shading noise color blur jpeg --levels 1 2 3 --seed 123
  ```
  Kết quả: `synth_test/<loại>/<mức>/{inp,gt}/*.png`.

## 2. Môi trường Colab (✅)

```
!nvidia-smi                                   # chọn A100 hoặc H100 (Runtime → Change runtime type)
!pip -q install lpips
from google.colab import drive; drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/PBL6/code/pbl6_pipeline /content/
!unzip -q /content/drive/MyDrive/PBL6/data/esp32_n2.zip -d /content/esp32
```

**Chọn GPU:**

| GPU | Dùng thế nào |
|---|---|
| **A100 (40/80 GB) hoặc H100: chọn cái này** | bf16 chạy tốt. Đủ bộ nhớ để chạy 3 run cùng lúc trên một GPU (mục 4.1). |
| L4 | Chạy được, mỗi run 75–80 phút, chạy lần lượt. |
| T4 | Tránh dùng. Không hỗ trợ bf16 bằng phần cứng, cần 🛠 `--amp fp16`. |

- A100/H100 tốn compute unit nhiều hơn L4 cho mỗi giờ, nhưng xong sớm hơn nhiều.
- **Tắt runtime ngay khi train xong** để không bị trừ thêm.

**Điểm nghẽn mới là CPU, không còn là GPU.**

- Hiện `batch()` cắt patch bằng numpy ngay trên luồng chính. Khi thêm `degrade()` (nhiễu, nhòe, JPEG) sinh ngay lúc train, CPU sẽ không theo kịp A100.
- Code Ngày 1 phải chuyển sang `torch.utils.data.DataLoader` với `num_workers` khoảng 6–8.
- Kiểm tra bằng `nvidia-smi`: GPU-Util dưới 70% kéo dài là đang chờ CPU.

## 3. Kiểm tra nhanh trước khi train thật (✅, khoảng 3 phút)

Mục đích: chắc chắn môi trường, dữ liệu và code đều chạy.

```
!python /content/pbl6_pipeline/train_restore.py --data /content/esp32 \
    --out /content/smoke --iters 300 --eval-every 100 --seed 0
```

- **Đạt:** `val_psnr` tăng rõ so với `val_psnr_input`. Ảnh vào khoảng 6 dB, sau vài trăm vòng đã lên trên 12.
- **Không đạt:** dừng lại sửa trước khi chạy dài.

## 4. Các lần train (đều từ trọng số ngẫu nhiên)

Thiết lập chung (giữ như v1 để so sánh công bằng):

- 8.000 vòng, batch 16, crop 192;
- AdamW lr 1e-3, warmup 500 vòng + cosine;
- loss L1 + 0,2·(1−SSIM), seed 0;
- đánh giá mỗi 500 vòng.

Bản v1 đạt best ở vòng 6.500, nên 8.000 vòng là đủ.

Thứ tự ưu tiên (nếu hết quota thì bỏ từ dưới lên):

| Thứ tự | Run | Lệnh | Thời gian (L4) |
|---|---|---|---|
| 0 | **R0: tái lập v1** (chỉ ESP32). Kiểm tra pipeline từ đầu cho lại best val PSNR khoảng 18,6. | ✅ `train_restore.py --data /content/esp32 --out $CK/r0_esp_only --iters 8000 --eval-every 500 --seed 0` | ~75 phút |
| 1 | **M4: model đề xuất** (RealDAE + ESP32 + tổng hợp) | 🛠 `train_restore.py --data /content/realdae /content/esp32 --synth 0.5 --out $CK/m4_nafnet_all --iters 8000 --eval-every 500 --seed 0` | ~80 phút |
| 2 | **M3:** NAFNet chỉ RealDAE | 🛠 `train_restore.py --data /content/realdae --out $CK/m3_nafnet_realdae ...` | ~75 phút |
| 3 | **M2:** U-Net thường, dữ liệu giống M4 | 🛠 `train_restore.py --arch unet --data /content/realdae /content/esp32 --synth 0.5 --out $CK/m2_unet_all ...` | ~60 phút |

`$CK = /content/drive/MyDrive/PBL6/checkpoints`

- **R0 chỉ cần nếu muốn chứng minh tái lập được.** Nếu thiếu thời gian, bỏ R0 và dùng checkpoint v1 cũ làm M3' (chỉ ESP32).
- **Colab ngắt giữa chừng:** chạy lại đúng lệnh đó, thêm `--resume`. Script nạp `last.pt` của chính run đó, không phải checkpoint cũ của đề tài trước, nên vẫn tính là train từ đầu.
- **Chọn checkpoint:** lấy `best.pt` theo PSNR trên val. Val gồm cả patch val của RealDAE và ESP32. Không nhìn tập test.
- **Hai model này mới có seed 0.** Nếu dư GPU, chạy thêm seed 1 cho M4 để thấy độ dao động.

### 4.1. Trên A100/H100: chạy song song và thêm model lớn

NAFNet width 32, batch 16, crop 192 chỉ dùng vài GB bộ nhớ và không lấp đầy A100. Vì vậy chạy nhiều run cùng lúc trên một GPU, mỗi run là một tiến trình nền:

```
!nohup python train_restore.py ... --out $CK/m4_nafnet_all  > $CK/m4.log 2>&1 &
!nohup python train_restore.py ... --out $CK/m3_nafnet_realdae > $CK/m3.log 2>&1 &
!nohup python train_restore.py ... --out $CK/m2_unet_all > $CK/m2.log 2>&1 &
!tail -n 3 $CK/*.log; nvidia-smi
```

**Đợt 1** (khoảng 1,5–2 giờ cho cả ba): M4, M3, M2, giữ đúng thiết lập ở bảng trên để so sánh công bằng.

**Đợt 2** (tùy chọn, khoảng 2 giờ), dùng thêm GPU có sẵn:

| Run | Thay đổi so với M4 | Để trả lời câu hỏi |
|---|---|---|
| M4-L | `--width 64` (khoảng 27M tham số), batch 32, crop 256, 16.000 vòng | Model lớn hơn và train lâu hơn có tăng PSNR không? |
| M4-s1 | `--seed 1` | Kết quả dao động bao nhiêu giữa các lần chạy? |

- **Model chính cho demo vẫn là M4 width 32**, vì chạy được trên CPU.
- M4-L là cận trên khi có GPU. Báo cáo ghi cả PSNR, số tham số và thời gian/ảnh của hai bản.
- Đặt `--num-workers` khoảng 3 cho mỗi tiến trình khi chạy song song, để tổng không vượt số vCPU của runtime (xem bằng `!nproc`).

## 5. Đánh giá

| Tập | Lệnh | Chỉ số |
|---|---|---|
| ESP32 test, cả trang | ✅ `restore_eval.py --data /content/esp32 --ckpt <best.pt> --variant esp_<run>` rồi ✅ `n2_eval.py --data /content/esp32 --skip-labels --variants esp_<run> --device cuda` | PSNR, SSIM, CER, WER |
| RealDAE test 150 ảnh | 🛠 `eval_sets.py --ckpt <best.pt> --set /content/realdae/pages/test --tile 512 --out results/<run>_realdae.json` | PSNR, SSIM, LPIPS |
| Tổng hợp theo loại và mức | 🛠 `eval_sets.py --ckpt <best.pt> --set /content/synth_test --out results/<run>_synth.json` | PSNR, SSIM theo từng ô (loại × mức) |
| M1 cổ điển, M5 GCDRNet | 🛠 `eval_sets.py --method classic` và `--method gcdrnet --gcdr-weights ...` | như trên |

Cách đo trên RealDAE phải khớp với bài báo GCDRNet (cùng độ phân giải và cùng cách tính PSNR/SSIM) thì mới đặt cạnh số đã công bố được. Nếu không khớp, ghi rõ trong báo cáo và tự chạy GCDRNet (M5) bằng cùng script để so cho công bằng.

## 6. Code mới (đã viết 11/11)

**Kết quả chạy thử trên CPU:**

- `degrade.py`: ảnh hỏng nhẹ ở mức 1, nặng rõ ở mức 3.
- `prep_realdae.py`: tự ghép cặp đúng.
- `train_restore.py`: chạy được với 2 nguồn + `--synth`, với `--resume`, với `--arch unet`.
- `eval_sets.py`: chạy được cả 4 cách (model, identity, classic, pred). Checkpoint v1 cũ nạp bình thường.

**Phát hiện khi chạy thử:**

- **v1 cũ không xử lý được ảnh thiếu sáng nặng.** Trên tập tổng hợp lowlight mức 3, PSNR chỉ 6,6 dB, trong khi các loại khác đạt 17–23 dB. Đây là lý do cần train với dữ liệu tổng hợp.
- **Chia ô khi suy luận làm lệch kết quả**, vì NAFNet có pooling toàn cục. Ô 512 lệch nhỏ (PSNR giữa hai bản 41,8 dB), ô 256 lệch rõ (29 dB). Trên GPU luôn chạy cả ảnh. Chỉ chia ô ≥ 512 khi thiếu bộ nhớ.
- **Trên Windows, DataLoader nhiều worker + dữ liệu lớn bị crash** (access violation khi chấm val trên CPU). Script mặc định `--num-workers 0` trên Windows. Colab (Linux) dùng fork nên không gặp, nhưng vẫn nên xem kết quả bước kiểm tra nhanh.

| Tệp | Thay đổi |
|---|---|
| `degrade.py` (mới) | Hàm `degrade(img, types, level, rng)` cho 6 loại suy biến. CLI tạo tập test tổng hợp. |
| `prep_realdae.py` (mới) | Thu nhỏ, tách train/val/test, cắt patch theo cấu trúc chung. |
| `train_restore.py` | `--data` nhận nhiều thư mục (lấy mẫu cân bằng giữa các nguồn). `--synth p`: với xác suất p, thay ảnh vào bằng `degrade(scan_patch)`. `--arch {nafnet,unet}`. `--amp {bf16,fp16,none}`. Chuyển sang `DataLoader` có `--num-workers`. Lưu `arch` và `width` vào checkpoint. |
| `restore_model.py` | Thêm `UNet` thường cùng giao diện. `load_model(ckpt)` đọc `arch`. |
| `eval_sets.py` (mới) | Chạy model hoặc phương pháp lên một tập `{inp,gt}`, chia ô ảnh lớn, tính PSNR/SSIM/LPIPS, ghi JSON theo ảnh và trung bình. |
| `make_retrain_notebook.py` (mới) | Sinh notebook Colab gồm mục 2–5, mỗi run một ô. |

## 7. Tổng thời gian GPU

| Việc | L4, chạy lần lượt | A100/H100, chạy song song |
|---|---|---|
| Kiểm tra nhanh | 3 phút | 3 phút |
| R0 (tùy chọn) | 75 phút | chạy chung đợt 1 |
| M4 + M3 + M2 | khoảng 3,5 giờ | khoảng 1,5–2 giờ |
| M4-L + M4-s1 (tùy chọn) | không đủ thời gian | khoảng 2 giờ |
| Đánh giá tất cả | khoảng 30 phút | khoảng 15 phút |

**Tổng:**

- L4: khoảng 4–5,5 giờ.
- A100/H100: khoảng 2–2,5 giờ cho phần bắt buộc, khoảng 4,5 giờ nếu chạy cả đợt 2.

Thời gian trên A100/H100 là ước tính. Đo thực tế ở bước kiểm tra nhanh: lấy số giây sau 300 vòng nhân lên.
