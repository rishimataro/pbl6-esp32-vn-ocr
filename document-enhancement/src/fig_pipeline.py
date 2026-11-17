"""Vẽ sơ đồ pipeline cho báo cáo: reports/figures/pipeline.png"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")

def box(x, y, w, h, title, sub, fc, ec="#444"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.8, color="#333", linespacing=1.35)

def arrow(x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.5, color="#444", ls=ls))

# hàng chính (suy luận)
box(0.2, 2.9, 2.0, 1.7, "Ảnh chụp", "camera bất kỳ:\nđiện thoại, webcam,\nESP32-CAM", "#f3e9d2")
box(2.9, 2.9, 2.4, 1.7, "Làm phẳng\n(tùy chọn)", "4 góc trang (OpenCV)\n+ UVDoc (giấy cong)", "#dfe8f3")
box(6.0, 2.9, 2.7, 1.7, "NAFNet", "6,82 triệu tham số\nảnh cạnh dài ≤1024 px\nảnh ra = ảnh vào + phần dư", "#d9ecd9")
box(9.4, 2.9, 2.0, 1.7, "Ảnh sạch", "sáng đều, nền trắng,\nchữ rõ, kiểu scan", "#f3e9d2")
box(11.7, 2.9, 1.2, 1.7, "OCR", "tùy chọn\n(VietOCR)", "#eeeeee", ec="#999")
for x0, x1 in [(2.2, 2.9), (5.3, 6.0), (8.7, 9.4)]:
    arrow(x0, 3.75, x1, 3.75)
arrow(11.4, 3.75, 11.7, 3.75, ls="--")

# hàng dưới (huấn luyện)
ax.text(0.2, 2.35, "Huấn luyện (từ trọng số ngẫu nhiên, Colab GPU)", fontsize=11, fontweight="bold", color="#9b4a1a")
box(0.2, 0.3, 3.0, 1.7, "Dữ liệu thật", "RealDAE: 400 cặp\nESP32 ↔ scan: 2.374 patch", "#f6e1d3")
box(3.8, 0.3, 3.2, 1.7, "Dữ liệu tổng hợp", "ảnh sạch + suy biến ngẫu nhiên:\nthiếu sáng, bóng, ám màu,\nnhiễu, mờ, JPEG (degrade.py)", "#f6e1d3")
box(7.6, 0.3, 2.7, 1.7, "Loss", "L1 + 0,2·(1 − SSIM)\nAdamW, 8.000 vòng\nbatch 16, crop 192", "#f6e1d3")
box(10.9, 0.3, 2.0, 1.7, "Đánh giá", "PSNR, SSIM, LPIPS\nCER (VietOCR)", "#f6e1d3")
for x0, x1 in [(3.2, 3.8), (7.0, 7.6), (10.3, 10.9)]:
    arrow(x0, 1.15, x1, 1.15)
arrow(7.35, 2.0, 7.35, 2.9, ls=":")
ax.text(7.5, 2.45, "trọng số", fontsize=8.5, color="#555")
plt.tight_layout(); plt.savefig("reports/figures/pipeline.png", dpi=130); print("ok")
