"""Model nâng chất lượng ảnh tài liệu: NAFNet bản nhỏ (Chen et al., 2022) và U-Net thường để đối chứng.

Dùng chung cho train (Colab GPU) và suy luận (CPU/GPU). Nạp checkpoint: load_model(path).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.w[:, None, None] + self.b[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    """Đầu vào/ra RGB [0,1]. Kích thước ảnh vào phải chia hết cho 2**len(enc_blocks) (tự pad khi suy luận)."""

    # mặc định ~6,8M tham số: khoảng 10 giây/trang 1179x1616 trên CPU 4 luồng, dưới 0,1 giây trên GPU
    def __init__(self, width=32, enc_blocks=(1, 1, 1, 2), middle_blocks=2, dec_blocks=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.decoders, self.ups, self.downs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad = 2 ** len(enc_blocks)

    def forward(self, inp):
        _, _, h, w = inp.shape
        ph, pw = (-h) % self.pad, (-w) % self.pad
        x0 = F.pad(inp, (0, pw, 0, ph), mode="replicate")
        x = self.intro(x0)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = dec(up(x) + skip)
        x = self.ending(x) + x0
        return x[:, :, :h, :w]


class UNet(nn.Module):
    """U-Net thường để đối chứng: 2 conv 3×3 + ReLU mỗi tầng, maxpool, transposed conv, học phần dư như NAFNet.
    width 32 → khoảng 7,8M tham số, cỡ tương đương NAFNet width 32."""

    def __init__(self, width=32, depth=4):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(co, co, 3, padding=1), nn.ReLU(inplace=True))

        chs = [width * 2 ** i for i in range(depth + 1)]
        self.encs = nn.ModuleList([block(3 if i == 0 else chs[i - 1], chs[i]) for i in range(depth)])
        self.mid = block(chs[depth - 1], chs[depth])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, 2) for i in reversed(range(depth))])
        self.decs = nn.ModuleList([block(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.ending = nn.Conv2d(width, 3, 1)
        self.pad = 2 ** depth

    def forward(self, inp):
        _, _, h, w = inp.shape
        x0 = F.pad(inp, (0, (-w) % self.pad, 0, (-h) % self.pad), mode="replicate")
        x, skips = x0, []
        for enc in self.encs:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = dec(torch.cat([up(x), skip], 1))
        return (self.ending(x) + x0)[:, :, :h, :w]


def build_model(arch="nafnet", width=32) -> nn.Module:
    if arch == "nafnet":
        return NAFNet(width=width)
    if arch == "unet":
        return UNet(width=width)
    raise ValueError(f"arch không hỗ trợ: {arch}")


def load_model(ckpt_path, device="cpu") -> nn.Module:
    """Đọc best.pt/last.pt. Checkpoint cũ (trước khi có 'arch') mặc định là NAFNet."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck.get("arch", "nafnet"), ck.get("width", 32))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    """SSIM trên ảnh [0,1], cửa sổ Gauss 11, tính trên kênh xám."""
    def gray(t):
        return (0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3])
    x, y = gray(x), gray(y)
    g = torch.exp(-((torch.arange(11, device=x.device) - 5) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None, :])[None, None]
    mu_x, mu_y = F.conv2d(x, k, padding=5), F.conv2d(y, k, padding=5)
    sxx = F.conv2d(x * x, k, padding=5) - mu_x ** 2
    syy = F.conv2d(y * y, k, padding=5) - mu_y ** 2
    sxy = F.conv2d(x * y, k, padding=5) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return m.mean()


def psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)


@torch.no_grad()
def restore_tensor(model, x, tile=None, overlap=32, amp=None):
    """x: tensor 1x3xHxW [0,1]. tile: cạnh ô (None = cả ảnh một lần). Các ô chồng nhau `overlap` px,
    ghép bằng trọng số giảm dần ở mép để không lộ đường nối. amp: dtype autocast trên GPU (None = fp32)."""
    def run(t):
        with torch.autocast("cuda", dtype=amp or torch.bfloat16, enabled=amp is not None and t.is_cuda):
            return model(t).float()

    _, _, h, w = x.shape
    if not tile or (h <= tile and w <= tile):
        return run(x).clamp(0, 1)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    th, tw = min(tile, h), min(tile, w)
    ramp_y = torch.ones(th, device=x.device)
    ramp_x = torch.ones(tw, device=x.device)
    r = torch.linspace(0.05, 1, overlap, device=x.device)
    if th > 2 * overlap:
        ramp_y[:overlap], ramp_y[-overlap:] = r, r.flip(0)
    if tw > 2 * overlap:
        ramp_x[:overlap], ramp_x[-overlap:] = r, r.flip(0)
    wt = (ramp_y[:, None] * ramp_x[None, :])[None, None]
    out = torch.zeros_like(x)
    acc = torch.zeros(1, 1, h, w, device=x.device)
    for y0 in ys:
        for x0 in xs:
            p = run(x[:, :, y0:y0 + th, x0:x0 + tw])
            out[:, :, y0:y0 + th, x0:x0 + tw] += p * wt
            acc[:, :, y0:y0 + th, x0:x0 + tw] += wt
    return (out / acc).clamp(0, 1)


@torch.no_grad()
def restore_image(model, img_bgr, device="cpu", tile=None, amp=None):
    """img_bgr: numpy uint8 HxWx3 (OpenCV). Trả về numpy uint8 BGR."""
    import numpy as np
    x = torch.from_numpy(img_bgr[..., ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(device)
    y = restore_tensor(model, x, tile=tile, amp=amp)[0].permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray((y[..., ::-1] * 255 + 0.5).astype(np.uint8))
