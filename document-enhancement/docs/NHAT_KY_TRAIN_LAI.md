# Nhật ký train lại model nâng ảnh (11/11/2026)

Đi kèm `KE_HOACH_NANG_ANH.md` và `HUONG_DAN_TRAIN_LAI.md`. Ghi lại thực tế đã chạy: môi trường, dữ liệu, lệnh, số liệu, sự cố.
Mọi run khởi tạo trọng số ngẫu nhiên (không nạp checkpoint cũ).

## 1. Nơi lưu kết quả

| Gì | Ở đâu (Google Drive, `MyDrive/PBL6/`) |
|---|---|
| Checkpoint mỗi run (`best.pt`, `last.pt`), `log.jsonl` (mỗi 500 vòng), `config.json`, `stdout.log` | `checkpoints/retrain/<run>/` |
| JSON đánh giá theo ảnh và trung bình, `summary.csv` | `results/retrain/` |
| Thông tin quá trình: môi trường (`env.json`, `pip_freeze.txt`), cách chia RealDAE (`realdae_manifest.json`), script chạy chuỗi + log, log kiểm tra nhanh, bản sao notebook kèm output | `results/retrain/meta/` |
| Tập test tổng hợp cố định (1.260 ảnh) | `data/synth_test_v2.zip` |
| Code đúng phiên bản đã chạy | `code/pbl6_pipeline_code.zip` (bản sinh 11/11 09:31) |

Bản sao trên máy: checkpoint ở `models/retrain/`, kết quả ở `reports/results/retrain/` (chép từ Drive sau khi xong).

## 2. Môi trường

- Colab, GPU **NVIDIA L4** 23 GB, 12 vCPU. Đã thử chuyển A100/H100 nhưng runtime vẫn là L4 → chạy trên L4.
- bf16 AMP, `DataLoader` 8–10 worker. Chi tiết phiên bản: `results/retrain/meta/env.json`, `pip_freeze.txt`.
- Trên L4 một run đã dùng 99–100% GPU nên chạy song song không nhanh hơn → các run chạy **lần lượt** bằng `run_all.sh`.

## 3. Dữ liệu

**ESP32** (từ N2, không dựng lại): 2.374 cặp patch train, 381 val; test 10 trang 3, 4, 7, 9, 11, 14, 25, 32, 38, 44.

**RealDAE** (tải từ OneDrive trong README của GCDRNet, thư mục `RealDAE`):

- Cấu trúc gốc: `task_{bleed,color,shadow}_{train,test}/`, ảnh vào `*_in.jpg`, ảnh chuẩn `*_gt.jpg`, cùng thư mục. Ảnh gốc lớn (đến 3040×2448).
- 600 cặp: 450 train (150/nhóm), 150 test (50/nhóm). Không thiếu cặp, không cần resize ảnh vào cho khớp ảnh chuẩn.
- `prep_realdae.py --long-side 1024 --val 50 --seed 0` → train 400, val 50 (bleed 20, shadow 16, color 14), test 150.
- Val tách từ train; tập test 150 ảnh không dùng để chọn checkpoint.

**Test tổng hợp** (`degrade.py --seed 123 --long-side 1024 --max-images 60`): 10 trang test ESP32 (ảnh scan) + 50 ảnh chuẩn đầu tiên của RealDAE test, × 7 loại (blur, color, jpeg, lowlight, mix, noise, shading) × 3 mức = **1.260 ảnh**.

- Lần chạy r0 buổi sáng dùng bộ cũ chỉ có 10 trang ESP32 (210 ảnh); bộ đó đã bị thay. Mọi số "synth" trong `summary.csv` cuối cùng là trên bộ 1.260 ảnh.

## 4. Thiết lập train chung

8.000 vòng, batch 16, crop 192, AdamW lr 1e-3, warmup 500 + cosine, loss L1 + 0,2·(1−SSIM), seed 0, đánh giá val mỗi 500 vòng, chọn `best.pt` theo PSNR val (trung bình các nguồn).

| Run | Kiến trúc | Dữ liệu | Ghi chú |
|---|---|---|---|
| r0_nafnet_esp | NAFNet w32 (6,82M) | ESP32 | tái lập v1 |
| m4_nafnet_all | NAFNet w32 | RealDAE + ESP32 + tổng hợp (p=0,5) | model đề xuất |
| m3_nafnet_realdae | NAFNet w32 | RealDAE | |
| m2_unet_all | U-Net | như ENH-M4 | đối chứng kiến trúc |

## 5. Kiểm tra nhanh

| Lần | Dữ liệu | Kết quả | Tốc độ |
|---|---|---|---|
| 1 (sáng) | ESP32 + tổng hợp, 300 vòng | PSNR val ESP32 6,16 → 13,38; tổng hợp 9,27 → 15,28 | 3,2 vòng/s, data_wait 0,001 |
| 2 (chiều) | RealDAE + ESP32 + tổng hợp, 200 vòng | RealDAE 12,08 → 15,24; ESP32 6,16 → 12,99; tổng hợp 8,94 → 14,29 | 2,8 vòng/s, data_wait 0,001 |

Log lần 2: `results/retrain/meta/smoke2/log.jsonl`. Log lần 1 không còn (chỉ nằm trên runtime đã tắt); số liệu ở bảng trên.

## 6. Kết quả train

| Run | Bắt đầu (giờ Colab, UTC) | Thời gian | it/s | Best val PSNR | Ghi chú |
|---|---|---|---|---|---|
| r0_nafnet_esp | 03:52 | ~35 phút | 3,95 | **18,54** (SSIM 0,817; ảnh vào 6,16) | khớp v1 (~18,6) |
| m4_nafnet_all | 08:13 | ~37 phút | 3,76 | **22,42** ở vòng 7.500 (RealDAE 26,67 / ESP32 18,17 / tổng hợp 25,06; ảnh vào 12,08 / 6,16 / 8,94) | ESP32 val thấp hơn r0 0,4 dB |
| m3_nafnet_realdae | ~08:51 | ~34 phút | 3,97 | **26,61** ở vòng 8.000 (chỉ val RealDAE) | |
| m2_unet_all | ~09:26 | ~8 phút | 18,3 | **22,03** ở vòng 7.500 (RealDAE 26,11 / ESP32 17,95 / tổng hợp 23,60) | U-Net 7,76M tham số, nhanh gấp ~5 lần NAFNet |

**Đợt 2** (runtime G4: NVIDIA RTX PRO 6000 Blackwell 96 GB, 48 vCPU; hai run chạy song song từ 10:17 UTC, 16 worker mỗi run; lệnh trong `meta/cmd_<run>.txt`, đo tốc độ trong `meta/timing_g4_*`):

| Run | Thay đổi so với ENH-M4 | Thời gian | it/s | Best val | Ghi chú |
|---|---|---|---|---|---|
| m4_nafnet_all_s1 | `--seed 1` | ~14 phút | 9,3 (chạy song song) | **22,32** ở vòng 8.000 (RealDAE 26,54 / ESP32 18,10 / tổng hợp 24,98) | seed 0: 22,42 → chênh 0,1 dB |
| m4l_nafnet64_all | width 64 (27,1M), batch 32, crop 256, 16.000 vòng | ~1 giờ 43 phút (6.146 s) | 1,4 khi chạy chung, 2,8 khi chạy riêng | **22,93** ở vòng 14.500 (RealDAE 27,86 / ESP32 17,99 / tổng hợp 27,64) | +0,5 dB so với ENH-M4; vòng cuối 22,91 |

Đo riêng trên G4 (200 vòng): ENH-M4 chuẩn ~15 vòng/s (gấp ~4 lần L4), ENH-M4-L ~2,6 vòng/s.

Các run đợt 1 kết thúc lúc ~09:34 UTC; đợt 2 kết thúc ~12:00 UTC. Bản sao `best.pt` + log + config trên máy: `models/retrain/<run>/`.

## 7. Đánh giá (tập test, không dùng để chọn checkpoint)

Bảng chính (`results/retrain/summary_main.csv`; chi tiết từng ô loại × mức trong `summary.csv`). PSNR dB / SSIM / LPIPS (thấp là tốt):

| Model | RealDAE test (150 ảnh) | Tổng hợp (1.260 ảnh, TB 21 ô) | ESP32 10 trang test |
|---|---|---|---|
| ENH-M0 ảnh vào | 12,40 / 0,858 / 0,216 | 17,68 / 0,726 / 0,261 | 5,79 / 0,571 |
| ENH-M1 cổ điển | 17,77 / 0,886 / 0,177 | 14,97 / 0,753 / 0,292 | – |
| R0 NAFNet, ESP32 | 16,77 / 0,817 / 0,306 | 14,96 / 0,710 / 0,443 | **17,51 / 0,818** |
| ENH-M3 NAFNet, RealDAE | **26,63** / 0,966 / **0,045** | 19,62 / 0,791 / 0,211 | 16,31 / 0,760 |
| ENH-M2 U-Net, tất cả | 25,77 / 0,965 / 0,061 | 23,42 / 0,896 / 0,153 | 17,38 / 0,813 |
| ENH-M4 NAFNet, tất cả | 26,48 / 0,968 / 0,050 | 24,75 / 0,912 / 0,125 | 17,10 / 0,812 |
| ENH-M4 seed 1 (đợt 2) | 26,49 / 0,969 / 0,049 | 24,60 / 0,908 / 0,127 | 17,29 / 0,814 |
| **ENH-M4-L NAFNet w64 (đợt 2)** | **27,31 / 0,971 / 0,045** | **27,83 / 0,942 / 0,082** | 17,05 / 0,812 |

PSNR trên tập tổng hợp theo loại (trung bình 3 mức):

| Loại | ENH-M0 | ENH-M1 | R0 | ENH-M3 | ENH-M2 | ENH-M4 | ENH-M4 s1 | ENH-M4-L |
|---|---|---|---|---|---|---|---|---|
| blur | 22,32 | 14,64 | 14,77 | 20,81 | 21,68 | 21,83 | 21,94 | **23,59** |
| color | 20,52 | 15,76 | 15,87 | 21,83 | 25,19 | 27,07 | 26,68 | **30,31** |
| jpeg | **29,65** | 15,46 | 15,73 | 23,12 | 25,98 | 26,74 | 26,74 | 28,58 |
| lowlight | 4,51 | 16,01 | 14,43 | 15,20 | 22,01 | 23,90 | 23,81 | **29,46** |
| mix | 9,50 | 12,89 | 13,84 | 13,36 | 19,79 | 20,88 | 20,76 | **23,13** |
| noise | 23,54 | 14,24 | 15,78 | 21,75 | 24,23 | 27,79 | 27,42 | **30,40** |
| shading | 13,72 | 15,79 | 14,29 | 21,26 | 25,05 | 25,05 | 24,83 | **29,30** |

Theo mức (L1/L2/L3): ENH-M4-L 29,29 / 28,01 / 26,18; ENH-M4 25,99 / 25,02 / 23,24; ENH-M4 s1 25,93 / 24,90 / 22,97; ENH-M2 24,66 / 23,63 / 21,97; ENH-M0 20,05 / 17,60 / 15,39.

Thời gian/ảnh (ảnh ≤1024 px, bf16): trên L4 NAFNet w32 0,11 s, U-Net 0,04 s, cổ điển 0,43 s (CPU). Trên G4 (đợt 2): NAFNet w32 0,041 s, ENH-M4-L 0,044 s — hai GPU khác nhau nên không so trực tiếp với số L4. ENH-M4-L có 27,1M tham số (gấp 4 lần ENH-M4), checkpoint `best.pt` 104 MB.

**Nhận xét:**

- ENH-M4 tốt nhất trên tập tổng hợp ở mọi mức, và ngang ENH-M3 trên RealDAE (26,48 so với 26,63 dB): thêm ESP32 + tổng hợp gần như không làm giảm điểm RealDAE.
- Model chỉ train một nguồn không tổng quát: R0 kém hơn cả ảnh vào trên RealDAE; ENH-M3 tụt xuống 16,3 dB trên ESP32.
- Trên ESP32, ENH-M4 thấp hơn R0 0,4 dB (17,10 so với 17,51). Đổi lại ENH-M4 xử lý được thiếu sáng (23,9 dB so với 14,4 của R0).
- NAFNet hơn U-Net cùng dữ liệu: +0,7 dB RealDAE, +1,3 dB tổng hợp; U-Net nhanh hơn ~3 lần khi suy luận.
- Với blur và jpeg, mọi model đợt 1 vẫn kém hơn ảnh vào (21,8 so với 22,3; 26,7 so với 29,7): model làm thay đổi ảnh gần như sạch. ENH-M4-L đã vượt ảnh vào ở blur (23,6) nhưng jpeg vẫn kém (28,6 so với 29,7). Cần cải thiện thêm (ví dụ mẫu "ảnh sạch → ảnh sạch" khi train).
- **Seed:** ENH-M4 seed 0 và seed 1 lệch ≤0,15 dB trên cả ba tập test (RealDAE 26,48 / 26,49; tổng hợp 24,75 / 24,60; ESP32 17,10 / 17,29) → chênh lệch giữa các model lớn hơn 0,3 dB là có ý nghĩa; nhỏ hơn thì coi như ngang nhau.
- **Model lớn (ENH-M4-L):** tăng width 32→64, crop 256, batch 32, gấp đôi số vòng → +0,8 dB RealDAE, +3,1 dB tổng hợp (lowlight +5,6, shading +4,3), LPIPS tổng hợp 0,125 → 0,082. ESP32 không đổi (17,05, trong khoảng nhiễu seed). Đổi lại gấp 4 lần tham số và ~10 lần thời gian train; trên G4 suy luận vẫn nhanh (0,044 s/ảnh). Chưa tách được phần lợi do model to hơn với phần do crop/batch/số vòng lớn hơn.
- ESP32 test kẹt quanh 17,0–17,5 dB với mọi model train trên nhiều nguồn: 10 trang quá ít và ảnh ESP32 khác hẳn RealDAE; cần thêm dữ liệu ESP32 hoặc đo CER thay vì PSNR.
- Số RealDAE ở đây đo ở cạnh dài 1.024 px, chưa chắc khớp cách đo trong bài báo GCDRNet; chưa chạy GCDRNet bằng cùng script.

## 8. Sự cố và cách xử lý

- Runtime vẫn là L4 sau khi đổi sang A100/H100 → người dùng chọn chạy trên L4.
- Kết nối Claude ↔ tab Colab bị ngắt khi một ô chạy quá 10 phút; train vẫn chạy tiếp vì là tiến trình nền ghi log thẳng vào Drive. Từ đó chỉ theo dõi qua log trên Drive, không dùng ô chờ dài.
- Sau khi mở lại tab, runtime đã khởi động lại (mất biến): chạy lại ô thiết lập trước khi đánh giá.
- Buổi chiều, sau một lần kết nối lại, tab Colab chỉ còn hiện các ô mới; lần lưu notebook sau đó ghi đè bản sao 9 ô đầu (thiết lập, RealDAE, tập tổng hợp, kiểm tra nhanh, khởi chạy). Bản hiện có: `meta/PBL6_Colab_Retrain_run_part2_eval.ipynb` (phần đánh giá). Bản 9 ô nằm trong lịch sử phiên bản của chính tệp này (đã đổi tên sau khi ghi đè): Drive → chuột phải tệp → Thông tin tệp → Quản lý phiên bản. Toàn bộ lệnh của phần 1 vẫn còn trong `run_all.sh`, `base_eval.sh`, `model_eval.sh`, `smoke2/` và nhật ký này.
- Runtime đã tắt (`runtime.unassign()`) sau khi đánh giá xong.
- Đợt 2: runtime mới lúc đầu chỉ có CPU → người dùng đổi sang G4. Khi kết nối lại giữa chừng, tab notebook lại chỉ hiện ô mới (các ô đợt 2 không còn trong notebook); biến trong kernel vẫn còn. Đã lưu toàn bộ code đã chạy của đợt 2 từ lịch sử kernel vào `meta/batch2_kernel_history.py`, notebook (1 ô) vào `meta/PBL6_Colab_Retrain_run_part3_batch2.ipynb` (không ghi đè part2). Script và log đánh giá: `meta/eval_m4_s1.{sh,log}`, `meta/eval_m4l.{sh,log}`; GPU: `meta/nvidia_smi_batch2.txt`.
- Runtime G4 đã tắt sau khi đánh giá đợt 2 xong. `summary.csv`/`summary_main.csv` đã dựng lại gồm cả 2 run đợt 2.

## 9. Đợt 3 (ENH-M5): ba cải tiến cho ba lỗi còn lại (17/11/2026)

Mục tiêu: kiểm tra từng cải tiến của ENH-M5 có sửa được lỗi tương ứng không: **dấu tiếng Việt** (`--edge-w 1.0`, thêm loss cạnh), **nền giấy màu** (`--paper-color 0.2`), **ảnh vốn đã sạch bị model làm hỏng** (`--identity-p 0.1`). Notebook: `notebooks/PBL6_Colab_Retrain.ipynb` (38 ô). Mọi run khởi tạo ngẫu nhiên, seed 0, cùng thiết lập chung như mục 4 (8.000 vòng, batch 16, crop 192, bf16).

**Nơi lưu.** Drive `MyDrive/PBL6/`: `checkpoints/retrain2/<run>/`, `results/retrain2/` (JSON từng tập, `summary*.csv`, `ablation_delta.csv`, `cer_esp32.csv`, `figures/`, `meta/`). Bản trên máy: `models/retrain2/<run>/` (best.pt, log.jsonl, config.json, stdout.log), `reports/results/retrain2/`.

### 9.1 Môi trường

- Colab **NVIDIA A100-SXM4-40GB**, 12 vCPU, Python 3.13 (`meta/nvidia_smi.txt`, `env.json`, `pip_freeze.txt`).
- Kiểm tra nhanh (300 vòng, đủ ba cải tiến) đạt: val PSNR RealDAE 12,08 → 16,85; ESP32 6,16 → 13,52; tổng hợp 8,94 → 15,38; `data_wait_frac` 0,002; 7,5 vòng/giây.

### 9.2 Train

| Run | Cải tiến | it/s | Best val PSNR (vòng) | RealDAE / ESP32 / tổng hợp (val) |
|---|---|---|---|---|
| m4r_baseline | không (chạy lại ENH-M4) | 11,77 | 22,40 (8.000) | 26,66 / 18,13 / 25,08 |
| m5_all | cạnh + nền màu + ảnh sạch | 11,42 | 22,51 (8.000) | 26,73 / 18,29 / 24,75 |
| m5a_edge | chỉ loss cạnh | 11,53 | 22,61 (8.000) | 26,91 / 18,30 / 25,37 |
| m5b_color | chỉ nền màu | 11,78 | 22,42 (8.000) | 26,67 / 18,18 / 24,67 |
| m5c_identity | chỉ ảnh sạch | 6,87 (chạy chung với run khác một phần) | 22,51 (7.500) | 26,84 / 18,17 / 24,94 |

Mỗi run chạy riêng mất khoảng 12 phút. Run lớn `m5l_nafnet64` **không chạy** (`RUN_LARGE=False`, theo dặn). Chênh PSNR val giữa 5 run ≤ 0,21 dB.

### 9.3 Kết quả trên tập test

PSNR (dB), cao hơn là tốt hơn. `m4_prev` là ENH-M4 đợt 1 (`models/retrain/m4_nafnet_all`), chấm cùng lúc làm mốc:

| Model | RealDAE test | Tổng hợp (TB 21 ô) | DocAligner | Jung | ESP32 10 trang |
|---|---|---|---|---|---|
| ENH-M0 ảnh vào | 12,40 | 17,68 | 13,96 | 14,87 | – |
| ENH-M1 cổ điển | 17,77 | 14,97 | 17,35 | 18,36 | – |
| m4_prev (ENH-M4 đợt 1) | 26,48 | 24,75 | 20,17 | 16,91 | 17,10 |
| m4r_baseline | 26,49 | 24,54 | 20,26 | 16,90 | 17,08 |
| m5a_edge | 26,63 | 24,71 | 20,27 | 17,03 | 17,23 |
| m5b_color | 26,49 | 24,08 | 20,23 | 16,96 | 17,08 |
| m5c_identity | 26,60 | 24,97 | 20,31 | 16,92 | 16,97 |
| m5_all | 26,44 | 24,59 | 20,12 | 16,99 | 17,33 |

LPIPS (thấp hơn là tốt hơn), RealDAE / tổng hợp / DocAligner / Jung: m4r_baseline 0,050 / 0,127 / 0,129 / 0,110; m5_all 0,048 / 0,126 / 0,133 / 0,107; m5a_edge 0,050 / 0,124 / 0,129 / 0,104; m5b_color 0,050 / 0,131 / 0,133 / 0,108; m5c_identity 0,049 / 0,127 / 0,129 / 0,107. SSIM: mọi run 0,967–0,969 trên RealDAE, 0,906–0,913 trên tập tổng hợp (bảng đầy đủ: `summary.csv`).

**Ablation** (chênh PSNR so với `m4r_baseline`, `ablation_delta.csv`):

| Run | RealDAE | Tổng hợp | DocAligner | Jung | ESP32 |
|---|---|---|---|---|---|
| m5a_edge | +0,13 | +0,17 | +0,01 | +0,13 | +0,15 |
| m5b_color | 0,00 | **−0,46** | −0,03 | +0,06 | 0,00 |
| m5c_identity | +0,11 | **+0,44** | +0,04 | +0,02 | −0,11 |
| m5_all | −0,05 | +0,05 | −0,14 | +0,09 | +0,24 |

Độ ngẫu nhiên giữa hai seed của cùng cấu hình ở đợt 2 là ≤ 0,2 dB (mục 7), nên chênh dưới ~0,3 dB coi là nhiễu. Chỉ có hai ô vượt ngưỡng: `m5b_color` thấp hơn 0,46 dB và `m5c_identity` cao hơn 0,44 dB trên tập tổng hợp (một seed mỗi run, chưa lặp).

Tập tổng hợp theo loại suy biến (PSNR, trung bình 3 mức), cột: baseline / edge / color / identity / all:
blur 21,89 / 21,93 / 21,83 / 21,96 / 22,09; color 26,66 / 26,69 / 26,02 / 27,47 / 26,36; jpeg 26,43 / 26,52 / 25,99 / 27,63 / 27,34; lowlight 23,82 / 24,23 / 23,19 / 23,91 / 23,35; mix 20,83 / 20,84 / 20,40 / 20,78 / 20,39; noise 27,36 / 27,70 / 26,86 / 27,89 / 28,06; shading 24,77 / 25,03 / 24,25 / 25,17 / 24,55. Ảnh vào (ENH-M0) trên blur 22,32 và jpeg 29,65: mọi model vẫn thấp hơn ảnh vào ở hai loại này (jpeg: tốt nhất 27,63).

### 9.4 Ba phép thử riêng cho ba lỗi

1. **Ảnh vốn đã sạch (`clean_identity`, 60 ảnh, ảnh vào = ảnh chuẩn; PSNR ảnh ra so với ảnh sạch, dB):** m4_prev 28,40; m4r_baseline 28,26; m5a_edge 28,27; m5b_color 27,79; **m5c_identity 30,62; m5_all 30,03**. Hai run có `--identity-p` cao hơn baseline 1,8–2,4 dB, tức làm hỏng ảnh sạch ít hơn. Đây là cải thiện rõ nhất của ENH-M5, nhưng model vẫn chưa trả ảnh nguyên vẹn (PSNR hữu hạn).
2. **Trang nền màu (3 ảnh; PSNR, dB):** ảnh vào 16,69; baseline 11,42; m5b_color 11,63; m5c_identity 11,44; m5a_edge 11,18; **m5_all 13,72**; m4_prev 11,20; cổ điển 9,14. Với 20 ảnh tệ nhất của ENH-M4: ảnh vào 16,27; baseline 16,44; m5_all 16,88. Mới có 3 ảnh nền màu nên số này chỉ mang tính tham khảo. Cải tiến nền màu riêng lẻ (`m5b_color`) không giúp; chỉ run kết hợp (`m5_all`) khá hơn (+2,3 dB trên 3 ảnh) và mọi model vẫn kém ảnh vào trên 3 ảnh này.
3. **Dấu tiếng Việt, đo bằng CER của VietOCR gốc trên 317 dòng của 10 trang ESP32 test** (`cer_esp32.csv`):

| Phiên bản | CER | CER bỏ dấu | Lỗi riêng do dấu |
|---|---|---|---|
| ảnh ESP32 gốc (raw) | 0,209 | 0,170 | 0,038 |
| bản scan (giới hạn trên) | 0,028 | 0,026 | 0,002 |
| m4_prev | 0,137 | 0,087 | 0,050 |
| m4r_baseline | 0,136 | 0,087 | 0,049 |
| m5a_edge | 0,136 | 0,085 | 0,051 |
| m5b_color | 0,135 | 0,087 | 0,048 |
| m5c_identity | 0,136 | 0,086 | 0,050 |
| m5_all | 0,137 | 0,086 | 0,051 |

Mọi model giảm CER từ 0,209 xuống ~0,136, nhưng **phần lỗi riêng do dấu (0,048–0,051) cao hơn cả ảnh gốc (0,038)**: model làm sạch nền nhưng không sửa được dấu. Loss cạnh (`m5a_edge`) không cải thiện (0,0505 so với 0,0485 của baseline). Cả sáu phiên bản chỉ lệch nhau ≤ 0,002 CER, tức không khác nhau.

### 9.5 Kết luận (chỉ dựa trên số đo ở trên, một seed mỗi cấu hình)

- Run tốt nhất theo từng tập (ô tự động ở mục 8 của notebook): RealDAE `m5a_edge` (26,63), tổng hợp `m5c_identity` (24,97), DocAligner `m5c_identity` (20,31), Jung `m5a_edge` (17,03), ESP32 10 trang `m5_all` (17,33). Mọi chênh lệch so với `m5_all` trên năm tập đều dưới 0,3 dB (nhiễu seed) nên không có một run "thắng" ở các tập này.
- **ENH-M5 (`m5_all`) so với baseline: không khác biệt đo được** trên RealDAE (−0,05), tổng hợp (+0,05), DocAligner (−0,14), Jung (+0,09), ESP32 (+0,24).
- Về ba lỗi: **ảnh sạch bị hỏng** được cải thiện rõ (+1,8 đến +2,4 dB, do `--identity-p`); **nền màu** cải thiện chỉ khi gộp ba cải tiến và chỉ trên 3 ảnh (cần thêm dữ liệu nền màu để kết luận); **dấu tiếng Việt** không cải thiện (CER do dấu 0,051 so với 0,049).
- `--paper-color` dùng riêng làm giảm PSNR tập tổng hợp 0,46 dB; nên cân nhắc trước khi dùng riêng.
- Chưa so với model lớn (`m5l_nafnet64` không chạy). ENH-M4-L ở mục 7 (PSNR tổng hợp 27,83) vẫn cao hơn mọi biến thể ENH-M5 cỡ nhỏ (≤ 24,97) ở tập tổng hợp.

### 9.6 Sự cố và cách xử lý

- **Cầu nối Colab không mở được notebook người dùng đã mở:** mỗi lần gọi `open_colab_browser_connection` nó mở một tab scratchpad riêng. Vì vậy 38 ô của `PBL6_Colab_Retrain.ipynb` được đọc từ Drive và chạy lần lượt trong tab đó (đúng nội dung từng ô, đúng thứ tự). Hệ quả: **không tạo được `PBL6_Colab_Retrain_da_chay.ipynb`** (ô 37 không chạy, vì sẽ lưu nhầm notebook scratchpad). Thay bằng: `meta/PBL6_Colab_Retrain_goc_da_dung.ipynb` (notebook gốc đã dùng), `meta/PBL6_Colab_Retrain_runner_scratchpad.ipynb` (notebook scratchpad), `meta/kernel_history_retrain2.py` (toàn bộ code đã chạy), cùng `figures/` và các bảng CSV. Đầu ra bảng/biểu đồ không nằm trong notebook mà trong `results/retrain2/`.
- **Lần chạy đầu của runtime là CPU** (không có `nvidia-smi`): dừng, người dùng đổi sang A100.
- **Ô 7 của notebook lỗi cuối ô:** bảng thống kê khai báo 3 cột nhưng hàng RealDAE có 4 giá trị (`ValueError: 3 columns passed, passed data had 4 columns`). Dữ liệu trong ô đã xử lý xong trước dòng lỗi, mình dựng lại bảng riêng. **Chưa sửa notebook trên Drive**; cần sửa trong file gốc.
- **`plt.show()` làm cầu nối kẹt >10 phút** (ô 9, ảnh lớn). Runtime không treo; ảnh đã lưu vào `figures/` trước đó. Sau đó chặn `plt.show` cho cả phiên.
- **5 run song song hết bộ nhớ GPU (CUDA OOM)** trên A100 40 GB: 4 run (`m4r_baseline`, `m5_all`, `m5a_edge`, `m5b_color`) chết ở khoảng vòng 500, chỉ `m5c_identity` sống sót (vòng 8.000, một phần chạy chung nên 6,87 it/s). Chạy lại 4 run từ đầu, lần lượt, bằng `meta/redo_seq.sh`, dùng `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 8 worker, mọi hạt giống và tham số khác giữ như ban đầu. Log của lần chạy hỏng giữ ở `models/retrain2/_oom_aborted/` (log nghèo: dừng ở vòng 500; `m5b_color` không có log).
- **Lỗi của mình khi chấm điểm:** xóa các khóa `*_oom_aborted` khỏi dict `CKPTS` đúng lúc vòng lặp đang duyệt dict, nên luồng chấm điểm nền dừng sau 3 file kết quả. Chạy lại luồng với danh sách đã loại các checkpoint hỏng, bỏ qua JSON đã có (`meta/eval_progress.log`). Các JSON đã có trước đó (identity/cổ điển/`m4r_baseline` tập tổng hợp) được dùng lại, không tính lại.
- **Ô CER (ô 34) lỗi ở bước cài thư viện:** `vietocr==0.3.13` ghim `pillow==10.2.0`, không có wheel cho Python 3.13 và không build được. Cài `vietocr --no-deps` rồi cài các thư viện cần dùng bản mới (`einops 0.8.2`, `pillow 11.3.0`, `gdown`, `prefetch_generator`, `lmdb`, `rapidfuzz`, `pypdfium2`); import thành công và đo CER chạy được. Sai khác phiên bản so với ràng buộc của vietocr chưa được kiểm tra riêng; CER của `raw` (0,209) và `scan` (0,028) hợp lý nên coi kết quả dùng được. Cần ghi nhận khi so sánh với các lần đo CER trước.
- **Runtime đã tắt** (`runtime.unassign()`) sau khi lưu kết quả và thông tin quá trình. Run lớn `m5l_nafnet64` không chạy.
