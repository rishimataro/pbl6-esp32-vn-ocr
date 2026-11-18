"""Chấm kết quả PP-OCRv4 (OCR-M2, OCR-M3) với cùng cách chuẩn hóa như n2_eval.py (CER, CER bỏ dấu, 1-NED, acc cả dòng)."""
import json, sys
from pathlib import Path
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, str(Path(__file__).resolve().parent))
from n2_eval import normalize_tone, strip_accents
D = Path(__file__).resolve().parents[1] / "reports/results/ocr_pp"
def score(f):
    E = L = Es = Ls = 0; ned = 0.0; acc = 0; n = 0
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); ref = normalize_tone(r["ref"]); hyp = normalize_tone(r["hyp"])
        d = Levenshtein.distance(ref, hyp); E += d; L += len(ref)
        rs, hs = strip_accents(ref), strip_accents(hyp); Es += Levenshtein.distance(rs, hs); Ls += len(rs)
        ned += 1 - d / max(len(ref), len(hyp), 1); acc += ref == hyp; n += 1
    return dict(n=n, cer=100*E/L, cer_noacc=100*Es/Ls, one_minus_ned=100*ned/n, acc=100*acc/n)
rows = {}
for f in sorted(D.glob("*.jsonl")):
    m, s = f.stem.split("__"); rows[f"{m} {s}"] = score(f)
print(f"{'model/tập':32s}{'n':>5}{'CER':>8}{'CERbd':>8}{'1-NED':>8}{'acc':>8}")
for k, v in rows.items(): print(f"{k:32s}{v['n']:5d}{v['cer']:8.1f}{v['cer_noacc']:8.1f}{v['one_minus_ned']:8.1f}{v['acc']:8.1f}")
(D / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")

# --- theo độ dài dòng (PP-OCRv4 nén ảnh về 48x320 nên dòng dài bị mất chữ) ---
import collections
def score_sub(f, lo, hi):
    rr = [json.loads(l) for l in open(f, encoding="utf-8")]
    rr = [r for r in rr if lo <= len(r["ref"].strip()) <= hi]
    if not rr: return None
    E = L = 0; ned = 0.0; acc = 0
    for r in rr:
        ref = normalize_tone(r["ref"].strip()); hyp = normalize_tone(r["hyp"].strip())
        d = Levenshtein.distance(ref, hyp); E += d; L += len(ref); ned += 1 - d / max(len(ref), len(hyp), 1); acc += ref == hyp
    return dict(n=len(rr), cer=100*E/L, one_minus_ned=100*ned/len(rr), acc=100*acc/len(rr))
bins = [(1, 30), (31, 200)]
out = {}
print("\nTheo độ dài nhãn (ký tự):")
for f in sorted(D.glob("*ours_*.jsonl")):
    m, s = f.stem.split("__")
    out[f.stem] = {f"{lo}-{hi}": score_sub(f, lo, hi) for lo, hi in bins}
    a, b = out[f.stem]["1-30"], out[f.stem]["31-200"]
    print(f"{m} {s:22s} ngắn n={a['n']:3d} CER {a['cer']:5.1f} 1-NED {a['one_minus_ned']:5.1f} acc {a['acc']:5.1f} | dài n={b['n']:3d} CER {b['cer']:5.1f} 1-NED {b['one_minus_ned']:5.1f} acc {b['acc']:5.1f}")
(D / "by_length.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
