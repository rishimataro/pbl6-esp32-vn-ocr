"""Đọc chữ từ ảnh trang tài liệu: phát hiện dòng chữ rồi nhận dạng từng dòng.

- detect_lines: tìm dòng chữ bằng nhị phân hóa thích nghi + chiếu ngang (không cần model tải thêm).
  Hợp trang in, chữ nằm ngang, ít nghiêng. Trang nhiều cột được tách khi hai cột cách nhau xa.
- recognize: nhận dạng từng dòng bằng VietOCR (vgg_transformer, trọng số có sẵn trong dự án).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

ENGINES = {
    "vietocr": "VietOCR (vgg_transformer)",
    "ppocr_m3": "OCR-M3: PP-OCRv4 tinh chỉnh (của nhóm)",
}
PP_PY = paths.ROOT / ".venv-paddle" / "Scripts" / "python.exe"
PP_FILES = [paths.ROOT / "models" / "ocr_pp" / n for n in ("M3_best.pdparams", "M3_config.yml", "vocab.txt")] + [
    paths.ROOT / "third_party" / "PaddleOCR" / "ppocr", PP_PY]


def engine_ready(key: str) -> bool:
    return key == "vietocr" or (key == "ppocr_m3" and all(f.exists() for f in PP_FILES))


def _ink(gray: np.ndarray) -> np.ndarray:
    """Ảnh nhị phân: 1 = mực. Thích nghi theo vùng nên chịu được nền sáng không đều."""
    h, w = gray.shape
    k = max(15, (min(h, w) // 40) | 1)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    b = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, k, 12)
    # bỏ chấm nhiễu rất nhỏ
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    small = np.where(st[:, cv2.CC_STAT_AREA] < max(4, h * w // 200000))[0]
    if len(small):
        b[np.isin(lab, small)] = 0
    return b


def _runs(mask: np.ndarray):
    """Các đoạn liên tiếp True: [(bắt đầu, kết thúc)] với kết thúc không tính."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def estimate_skew(ink: np.ndarray, max_deg: float = 10.0) -> float:
    """Góc nghiêng (độ) làm các hàng chữ sắc nhất: cực đại hóa phương sai hồ sơ chiếu ngang."""
    h, w = ink.shape
    sc = 800 / max(h, w)
    small = cv2.resize(ink.astype(np.float32), (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    c = (small.shape[1] / 2, small.shape[0] / 2)

    def score(deg):
        m = cv2.getRotationMatrix2D(c, deg, 1.0)
        r = cv2.warpAffine(small, m, (small.shape[1], small.shape[0]), flags=cv2.INTER_LINEAR)
        return float(np.var(r.sum(1)))
    best = max(np.arange(-max_deg, max_deg + 0.01, 1.0), key=score)
    fine = np.arange(best - 1.0, best + 1.01, 0.2)
    return float(max(fine, key=score))


def deskew(img: np.ndarray, deg: float):
    if abs(deg) < 0.15:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _pitch(prof: np.ndarray, lo: int = 8, hi: int = 150) -> int:
    """Chu kỳ dòng (khoảng cách giữa hai dòng) qua tự tương quan của hồ sơ chiếu ngang.
    Lấy đỉnh đầu tiên đủ mạnh, không phải đỉnh cao nhất: đỉnh cao nhất hay rơi vào bội số 2x, 3x của chu kỳ thật."""
    x = prof - prof.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    hi = min(hi, len(ac) - 2)
    if hi <= lo + 1:
        return lo
    seg = ac[lo:hi + 1]
    top = float(seg.max())
    for i in range(1, len(seg) - 1):
        if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] >= 0.6 * top:
            return int(lo + i)
    return int(lo + np.argmax(seg))


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    k = int(max(1, round(sigma * 3)))
    t = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (t / max(sigma, 0.5)) ** 2)
    return np.convolve(x, g / g.sum(), mode="same")


def detect_lines(img: np.ndarray, min_h: int = 8, color_ref: np.ndarray | None = None):
    """Trả về danh sách khung (x0, y0, x1, y1) theo thứ tự đọc. Ảnh nên đã được nắn thẳng (deskew)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = gray.shape
    ink = _ink(gray)
    ref = color_ref if color_ref is not None else img
    if ref.ndim == 3 and ref.shape[:2] == (H, W):
        b, g_, r = [ref[..., i].astype(np.int16) for i in range(3)]
        red = (r - np.maximum(g_, b) > 14)                    # dấu triện đỏ không phải chữ cần đọc; lấy màu từ ảnh chưa nâng vì nâng ảnh làm phai màu
        if red.mean() < 0.03:                                 # nếu "đỏ" phủ cả trang thì là ánh sáng ám màu, không phải dấu triện: bỏ qua
            red = cv2.dilate(red.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            ink[red] = 0
    m = max(2, int(0.01 * min(H, W)))
    ink[:m] = 0; ink[-m:] = 0; ink[:, :m] = 0; ink[:, -m:] = 0
    raw = ink.sum(1).astype(float)
    if raw.max() < 3:
        return []
    pitch = _pitch(_smooth(raw, 1.5))
    # bỏ các mảng lớn không phải chữ (ảnh minh họa, khung, băng tiêu đề đậm): thành phần liên thông cao và rộng hơn 3 chu kỳ dòng
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    lv = max(int(3 * pitch), 20)
    vr = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, lv)))
    nvr, _l, vst, _c = cv2.connectedComponentsWithStats(vr, connectivity=8)
    ruls = [(int(x + w / 2), int(y), int(y + h)) for x, y, w, h, _a in vst[1:]]                      # đường kẻ dọc (khung bảng): luôn là ranh giới giữa hai ô
    big = np.where((st[:, cv2.CC_STAT_HEIGHT] > 3 * pitch) & (st[:, cv2.CC_STAT_WIDTH] > 3 * pitch))[0]
    big = big[big != 0]
    if len(big):
        ink[np.isin(lab, big)] = 0
        raw = ink.sum(1).astype(float)
        if raw.max() < 3:
            return []
    prof = _smooth(raw, max(1.0, pitch / 10))
    # mỗi dòng chữ là một đỉnh của hồ sơ; các dòng sát nhau tách ở chỗ lõm giữa hai đỉnh
    nz = prof[prof > 0.15 * prof.max()]
    floor = max(2.0, 0.1 * np.median(nz)) if len(nz) else 1.0
    peaks = []
    for i in range(1, H - 1):
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > floor:
            if peaks and i - peaks[-1] < 0.6 * pitch:
                if prof[i] > prof[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    if not peaks:
        return []
    cuts = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        cuts.append(a + int(np.argmin(prof[a:b + 1])))
    cuts.append(H)
    boxes = []
    for pk, a, b in zip(peaks, cuts[:-1], cuts[1:]):
        band = ink[a:b]
        rows = band.sum(1)
        # cắt phần trống đầu/cuối dải
        thr = max(1.0, 0.08 * rows.max())
        c = min(max(pk - a, 0), len(rows) - 1)
        lo_i = c
        while lo_i > 0 and rows[lo_i - 1] > thr:
            lo_i -= 1
        hi_i = c
        while hi_i < len(rows) - 1 and rows[hi_i + 1] > thr:
            hi_i += 1
        # cho phép bắc qua khe nhỏ (dấu mũ, dấu thanh tách khỏi thân chữ)
        gap_ok = max(2, int(0.2 * pitch))
        while lo_i > 0 and rows[max(0, lo_i - gap_ok):lo_i].max() > thr and (hi_i - lo_i) < 1.6 * pitch:
            lo_i = max(0, lo_i - gap_ok) + int(np.where(rows[max(0, lo_i - gap_ok):lo_i] > thr)[0][0])
        while hi_i < len(rows) - 1 and rows[hi_i + 1:hi_i + 1 + gap_ok].max(initial=0) > thr and (hi_i - lo_i) < 1.6 * pitch:
            hi_i = hi_i + 1 + int(np.where(rows[hi_i + 1:hi_i + 1 + gap_ok] > thr)[0][-1])
        ys = np.array([lo_i, hi_i])
        y0, y1 = a + int(ys[0]), a + int(ys[-1]) + 1
        if y1 - y0 < min_h:
            continue
        hh = y1 - y0
        cols = band[ys[0]:ys[-1] + 1].sum(0) > 0
        gap = max(int(1.25 * pitch), 24)          # khoảng trống ngang lớn hơn 1,25 chu kỳ dòng = cột khác, số mục, nhãn
        segs = _runs(cols)
        if not segs:
            continue
        rul = np.array([rx for rx, ra, rb in ruls if ra <= y0 + 0.25 * hh and rb >= y1 - 0.25 * hh], int)
        cur = list(segs[0]); parts = []
        for s_, e_ in segs[1:]:
            if s_ - cur[1] > gap or (len(rul) and np.any((rul > cur[1]) & (rul < s_))):
                parts.append(cur); cur = [s_, e_]
            else:
                cur[1] = e_
        parts.append(cur)
        for x0, x1 in parts:
            if x1 - x0 < 1.2 * hh and x1 - x0 < 3 * min_h and not (x1 - x0 >= 0.25 * hh and hh >= 0.35 * pitch):
                continue                          # đốm nhỏ (ký tự đơn như số mục "1", "2" thì giữ)
            dens = float(band[ys[0]:ys[-1] + 1, x0:x1].mean())
            if dens > 0.6:
                continue                          # đặc quá: ảnh hoặc mảng đen, không phải dòng chữ (chữ in thường 0,15–0,45)
            empty = float((band[ys[0]:ys[-1] + 1, x0:x1].sum(0) == 0).mean())
            if empty < 0.04 and (x1 - x0) > 8 * hh:
                continue                          # mực liền một dải, không có khe giữa chữ: ảnh minh họa, không phải dòng chữ
            pad = max(2, int(0.12 * hh))
            boxes.append((max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)))
    boxes.sort(key=lambda t: ((t[1] + t[3]) // 2 // max(8, pitch // 2), t[0]))
    return boxes


@lru_cache(maxsize=1)
def _vietocr(device: str = "cpu"):
    from n2_eval import load_model
    return load_model("vgg_transformer", device)


MAX_RATIO = 13.0   # VietOCR thu ảnh về cao 32, rộng tối đa 512: tỉ lệ rộng/cao lớn hơn ~16 bị nén ngang và model đọc sai


def _chunks(crop: np.ndarray, max_ratio: float = MAX_RATIO):
    """Cắt dòng quá dài thành các đoạn có tỉ lệ rộng/cao <= max_ratio, cắt ở cột ít mực nhất gần chỗ chia đều."""
    h, w = crop.shape[:2]
    k = int(np.ceil(w / (max_ratio * h)))
    if k <= 1:
        return [crop]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    empty = b.sum(0) == 0
    proj = np.convolve(b.sum(0).astype(float), np.ones(3) / 3, mode="same")
    cuts = [0]
    for i in range(1, k):
        c = int(w * i / k); r = int(0.2 * w / k)
        lo, hi = max(c - r, cuts[-1] + 5), min(c + r, w - 5)
        if hi <= lo:
            cuts.append(c); continue
        # ưu tiên khe trống dài nhất (khoảng trắng giữa hai từ), cắt ở giữa khe; nếu không có thì cột ít mực nhất
        runs = [(s_, e_) for s_, e_ in _runs(empty[lo:hi])]
        if runs:
            s_, e_ = max(runs, key=lambda t: t[1] - t[0])
            cuts.append(lo + (s_ + e_) // 2)
        else:
            cuts.append(lo + int(np.argmin(proj[lo:hi])))
    cuts.append(w)
    return [crop[:, a:b_] for a, b_ in zip(cuts[:-1], cuts[1:]) if b_ - a > 5]


MAX_RATIO_PP = 6.0   # PP-OCRv4 đưa ảnh về 48x320: tỉ lệ rộng/cao lớn hơn ~6,7 bị nén; chạy từng đoạn có tỉ lệ <= 6


class _PPClient:
    """Tiến trình phụ chạy src/ppocr_worker.py trong .venv-paddle; nạp model một lần, dùng lại cho mọi yêu cầu."""
    _p = None

    @classmethod
    def _start(cls):
        import subprocess
        cls._p = subprocess.Popen([str(PP_PY), "-X", "utf8", str(Path(__file__).with_name("ppocr_worker.py"))],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding="utf-8", bufsize=1)
        while True:
            line = cls._p.stdout.readline()
            if not line:
                raise RuntimeError("tiến trình OCR-M3 thoát sớm (kiểm tra .venv-paddle, third_party/PaddleOCR, models/ocr_pp)")
            if line.startswith("@@READY"):
                return

    @classmethod
    def read(cls, crops_bgr: list[np.ndarray], batch: int = 32) -> list[str]:
        import base64
        import json
        if cls._p is None or cls._p.poll() is not None:
            cls._start()
        out: list[str] = []
        for i in range(0, len(crops_bgr), batch):
            imgs = [base64.b64encode(cv2.imencode(".png", c)[1].tobytes()).decode() for c in crops_bgr[i:i + batch]]
            cls._p.stdin.write(json.dumps({"images": imgs}) + chr(10)); cls._p.stdin.flush()
            while True:
                line = cls._p.stdout.readline()
                if not line:
                    raise RuntimeError("tiến trình OCR-M3 đã dừng")
                if line.startswith("@@"):
                    r = json.loads(line[2:])
                    if "error" in r:
                        raise RuntimeError(r["error"])
                    out += r["texts"]
                    break
        return out


def recognize(img: np.ndarray, boxes, engine: str = "vietocr", device: str = "cpu", batch: int = 16):
    """Nhận dạng từng khung; trả về list chuỗi cùng thứ tự với boxes. Dòng quá dài được cắt đoạn rồi nối lại."""
    if engine == "ppocr_m3":
        if not engine_ready(engine):
            raise RuntimeError("chưa cài OCR-M3: thiếu .venv-paddle, third_party/PaddleOCR hoặc models/ocr_pp")
        pieces, owner = [], []
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            for c in _chunks(img[y0:y1, x0:x1], MAX_RATIO_PP):
                pieces.append(c if c.ndim == 3 else cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)); owner.append(i)
        outs = _PPClient.read(pieces) if pieces else []
        res = [[] for _ in boxes]
        for o, t in zip(owner, outs):
            res[o].append(t.strip())
        return [" ".join(r) for r in res]
    if engine != "vietocr":
        raise ValueError(f"engine không hỗ trợ: {engine}")
    from PIL import Image
    pred = _vietocr(device)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pieces, owner = [], []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        for c in _chunks(rgb[y0:y1, x0:x1]):
            pieces.append(Image.fromarray(c)); owner.append(i)
    outs = []
    for i in range(0, len(pieces), batch):
        outs += pred.predict_batch(pieces[i:i + batch])
    res = [[] for _ in boxes]
    for o, t in zip(owner, outs):
        res[o].append(t.strip())
    return [" ".join(r) for r in res]


def read_page(img: np.ndarray, engine: str = "vietocr", device: str = "cpu", straighten: bool = True,
              color_ref: np.ndarray | None = None):
    """Trả về (ảnh đã nắn thẳng, danh sách dòng). Tọa độ khung tính trên ảnh trả về.
    color_ref: ảnh cùng cỡ còn màu gốc (trước khi nâng) để nhận ra và bỏ dấu triện đỏ."""
    if straighten:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        deg = estimate_skew(_ink(gray))
        img = deskew(img, deg)
        if color_ref is not None:
            color_ref = deskew(color_ref, deg)
    boxes = detect_lines(img, color_ref=color_ref)
    texts = recognize(img, boxes, engine, device) if boxes else []
    lines = [{"box": list(map(int, b)), "text": t} for b, t in zip(boxes, texts)]
    lines = [l for l in lines if any(ch.isalnum() for ch in l["text"])]   # bỏ khung chỉ đọc ra ký hiệu (thường là nhiễu, ảnh)
    return img, lines


if __name__ == "__main__":
    import json
    import time
    p = Path(sys.argv[1])
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    t = time.perf_counter()
    _, res = read_page(im)
    print(f"{len(res)} dòng, {time.perf_counter() - t:.1f} s")
    for r in res:
        print(r["box"], r["text"])
