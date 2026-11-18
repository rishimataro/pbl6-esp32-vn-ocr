"""Bảng gộp: VietOCR vs PP-OCRv4 (OCR-M2, OCR-M3) trên cùng 317 dòng, tách dòng ngắn (<=30 ký tự) / dài."""
import json, sys
from pathlib import Path
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, str(Path(__file__).resolve().parent))
from n2_eval import normalize_tone
R = Path(__file__).resolve().parents[1] / "reports/results"
V = {"esp_raw": "esp_raw", "esp_1a": "esp_1a", "m4_nafnet_all": "m4_nafnet_all", "m2_unet_all": "m2_unet_all", "r0_nafnet_esp": "r0_nafnet_esp", "scan": "scan"}
def stat(pairs):
    E = L = 0; ned = 0.0
    for ref, hyp in pairs:
        ref, hyp = normalize_tone(ref.strip()), normalize_tone(hyp.strip()); d = Levenshtein.distance(ref, hyp)
        E += d; L += len(ref); ned += 1 - d / max(len(ref), len(hyp), 1)
    return 100 * E / L, 100 * ned / len(pairs), len(pairs)
def load_pp(m, v):
    return {Path(json.loads(l)["path"]).stem: json.loads(l) for l in open(R / f"ocr_pp_chunk/{m}__ours_{v}.jsonl", encoding="utf-8")}
res = {}
print(f"{'ảnh vào':16s}| " + " | ".join(f"{m:^21s}" for m in ["VietOCR", "OCR-M2", "OCR-M3"]) + "   (CER % / 1-NED %)")
for sub, (lo, hi) in {"tất cả": (0, 999), "ngắn<=30": (1, 30), "dài>30": (31, 999)}.items():
    print(f"--- {sub}")
    for v in V:
        viet = {r["id"]: r for r in (json.loads(l) for l in open(R / f"esp32_cer/pred_test_{v}_vgg_transformer.jsonl", encoding="utf-8"))}
        row = {}
        for name, get in [("VietOCR", lambda i: viet[i]["hyp"]), ("OCR-M2", None), ("OCR-M3", None)]:
            if name == "VietOCR": ids = list(viet); ref = {i: viet[i]["ref"] for i in ids}; hyp = {i: viet[i]["hyp"] for i in ids}
            else:
                pp = load_pp(name[-2:], v); ids = list(pp); ref = {i: pp[i]["ref"] for i in ids}; hyp = {i: pp[i]["hyp"] for i in ids}
            pairs = [(ref[i], hyp[i]) for i in ids if lo <= len(ref[i].strip()) <= hi]
            row[name] = stat(pairs)
        res[f"{sub}|{v}"] = row
        print(f"{v:16s}| " + " | ".join(f"{row[m][0]:7.1f} / {row[m][1]:6.1f} (n={row[m][2]})" for m in row))
(R / "ocr_pp_chunk/vs_vietocr.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
