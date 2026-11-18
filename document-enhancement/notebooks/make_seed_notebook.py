"""Sinh notebooks/PBL6_Colab_Seeds.ipynb: đợt 3 = thêm seed cho ENH-M4 và ENH-M5 + thử tăng cường dữ liệu nền màu.

Dùng lại các ô dữ liệu/train/đánh giá của make_retrain_notebook.py (cùng giao thức), chỉ thay danh sách run và phần kết quả.
  .venv-ocr/Scripts/python.exe -X utf8 notebooks/make_seed_notebook.py
Cần chạy make_retrain_notebook.py trước hoặc sau đều được; zip code được sinh lại ở đây.
"""
import json
import re
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
src = (ROOT / "make_retrain_notebook.py").read_text("utf-8")
ns = {"__file__": str(ROOT / "make_retrain_notebook.py")}
exec(compile(src[:src.index("nb = {")], "make_retrain_notebook.py", "exec"), ns)
old = ns["cells"]


def text(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def find(substr, kind=None, start=0):
    for i in range(start, len(old)):
        if substr in text(old[i]) and (kind is None or old[i]["cell_type"] == kind):
            return i
    raise KeyError(substr)


def md(t):
    return dict(cell_type="markdown", metadata={}, source=textwrap.dedent(t).strip() + "\n")


def code(t):
    return dict(cell_type="code", metadata={}, source=textwrap.dedent(t).strip() + "\n", execution_count=None, outputs=[])


def clone(i, **repl):
    c = json.loads(json.dumps(old[i]))
    s = text(c)
    for a, b in repl.items():
        pass
    return c, s


def sub(i, pairs):
    c = json.loads(json.dumps(old[i]))
    s = text(c)
    for a, b in pairs:
        assert a in s, (a, s[:80])
        s = s.replace(a, b)
    c["source"] = s
    return c


cells = []
cells.append(md("""
# PBL6: đợt 3 — thêm seed và tăng cường dữ liệu nền màu

**Mục đích.** (1) Đo nhiễu giữa các seed để biết chênh lệch nào giữa ENH-M4 và ENH-M5 là thật; (2) thử dữ liệu nhuộm màu giấy mạnh hơn để sửa lỗi model tẩy trắng trang nền màu.

| Run | Cấu hình | Seed |
|---|---|---|
| `m4_s1`, `m4_s2` | như ENH-M4 chạy lại (không cải tiến) | 1, 2 |
| `m5_all_s1`, `m5_all_s2` | `--edge-w 1.0 --paper-color 0.2 --identity-p 0.1` | 1, 2 |
| `m5d_color50` | như ENH-M5 nhưng `--paper-color 0.5` | 0 |
| `m5e_color50_noedge` | `--paper-color 0.5 --identity-p 0.1` (không loss gradient) | 0 |

Seed 0 của `m4r_baseline` và `m5_all` lấy từ đợt 2 (`checkpoints/retrain2`), nên mỗi cấu hình có đủ 3 seed.

**Cần có trên Drive:** như đợt 2, cộng `checkpoints/retrain2/` (checkpoint đợt 2) và `code/pbl6_pipeline_code.zip` bản mới.
Chạy từ trên xuống; Runtime → A100/H100.
"""))

i_smoke = find("## 2. Kiểm tra nhanh", "markdown")
head = [sub(0 if False else find("import json, os, subprocess", "code"), [
    ("CK = D / 'checkpoints' / 'retrain2'          # đợt 2: mỗi run một thư mục con", "CK = D / 'checkpoints' / 'retrain3'          # đợt 3: mỗi run một thư mục con\nCK2 = D / 'checkpoints' / 'retrain2'         # đợt 2 (m4r_baseline, m5_all seed 0), chỉ đọc"),
    ("RES = D / 'results' / 'retrain2'", "RES = D / 'results' / 'retrain3'"),
])]
start = find("import json, os, subprocess", "code")
cells += [old[j] if j != start else head[0] for j in range(start - 0, i_smoke) if j > find("# PBL6: train lại", "markdown")]

# --- train
i_run = find("COMMON = ['--iters'", "code")
s = text(old[i_run])
s = re.sub(r"RUNS = \{.*?\n\}\n", """RUNS = {
    'm4_s1':              BASE + ['--seed', '1'],
    'm4_s2':              BASE + ['--seed', '2'],
    'm5_all_s1':          BASE + ['--edge-w', '1.0', '--paper-color', '0.2', '--identity-p', '0.1', '--seed', '1'],
    'm5_all_s2':          BASE + ['--edge-w', '1.0', '--paper-color', '0.2', '--identity-p', '0.1', '--seed', '2'],
    'm5d_color50':        BASE + ['--edge-w', '1.0', '--paper-color', '0.5', '--identity-p', '0.1'],
    'm5e_color50_noedge': BASE + ['--paper-color', '0.5', '--identity-p', '0.1'],
}
""", s, flags=re.S)
cells.append(md("## 3. Train (tối đa 3 run song song, 6 run, khoảng 30–40 phút trên A100)"))
cells.append(dict(cell_type="code", metadata={}, source=s, execution_count=None, outputs=[]))
cells.append(old[find("# Theo dõi + đường cong train", "code")])
cells.append(old[find("# Nếu runtime bị ngắt", "code")])

# --- đánh giá
cells.append(md("## 4. Đánh giá (cùng giao thức đợt 2: RealDAE test, tổng hợp, DocAligner, Jung, ảnh sạch, ESP32)"))
i_ev = find("def ev(tag, setdir", "code")
s = text(old[i_ev])
a = "CKPTS = {p.parent.name: p for p in sorted(CK.glob('*/best.pt'))}\nif (CK_OLD / 'm4_nafnet_all' / 'best.pt').exists():\n    CKPTS['m4_prev'] = CK_OLD / 'm4_nafnet_all' / 'best.pt'"
assert a in s
s = s.replace(a, """CKPTS = {'m4_s0': CK2 / 'm4r_baseline' / 'best.pt', 'm5_all_s0': CK2 / 'm5_all' / 'best.pt'}
CKPTS.update({p.parent.name: p for p in sorted(CK.glob('*/best.pt'))})
CKPTS = {k: v for k, v in CKPTS.items() if v.exists()}""")
cells.append(dict(cell_type="code", metadata={}, source=s, execution_count=None, outputs=[]))
cells.append(md("Trang ESP32 (cả trang):"))
cells.append(old[find("restore_eval.py", "code")])

cells.append(md("## 5. Kết quả"))
cells.append(code("""
import numpy as np, matplotlib.pyplot as plt
rows = []
for f in sorted(RES.glob('*__*.json')):
    run, sname = f.stem.split('__')
    r = json.loads(f.read_text())
    if 'groups' in r:
        for g, v in r['groups'].items():
            rows.append({'run': run, 'set': sname, 'group': g, **v['mean']})
    else:
        rows.append({'run': run, 'set': sname, 'group': 'test', **r['mean']})
df = pd.DataFrame(rows); df.to_csv(RES / 'summary.csv', index=False)
NAMES = {'realdae': 'RealDAE', 'synth': 'Tổng hợp', 'docaligner': 'DocAligner', 'jung': 'Jung', 'esp32_pages': 'ESP32'}
def overall(sname, metric='psnr'):
    d = df[df['set'] == sname]
    if sname == 'synth': d = d[d['group'] != '.']
    return d.groupby('run')[metric].mean()
main = pd.DataFrame({NAMES[s]: overall(s) for s in NAMES if s in set(df['set'])})
display(main.round(2))
main.round(3).to_csv(RES / 'summary_main_psnr.csv')

# nhóm cấu hình: mean ± std qua seed (std mẫu, n-1)
def cfg(r):
    return 'ENH-M4' if r.startswith('m4_s') else 'ENH-M5 (cả ba)' if r.startswith('m5_all_s') else r
g = main.drop(index=[r for r in main.index if r in ('m0_identity', 'm1_classic')], errors='ignore').copy()
g['cfg'] = [cfg(r) for r in g.index]
agg = g.groupby('cfg').agg(['mean', 'std', 'count'])
out = pd.DataFrame({c: [f"{agg[(c,'mean')][k]:.2f} ± {agg[(c,'std')][k]:.2f}" if agg[(c,'count')][k] > 1 else f"{agg[(c,'mean')][k]:.2f}" for k in agg.index] for c in main.columns}, index=agg.index)
out['số seed'] = [int(agg[(main.columns[0], 'count')][k]) for k in agg.index]
print('PSNR theo cấu hình, trung bình ± độ lệch chuẩn qua các seed (dB)'); display(out)
out.to_csv(RES / 'summary_seeds.csv')
"""))
cells.append(code("""
# Phép thử ảnh sạch và trang nền màu
ci = overall('clean_identity').to_frame('PSNR ảnh ra so với ảnh sạch (dB)').drop(index='m0_identity', errors='ignore')
print('Ảnh sạch → ảnh sạch (cao hơn là tốt hơn)'); display(ci.round(2))
def per_image(run, sname):
    g = json.loads((RES / f'{run}__{sname}.json').read_text())['groups']
    return {im['name']: im['psnr'] for v in g.values() for im in v['images']}
dc = {r: per_image(r, 'docaligner') for r in main.index if (RES / f'{r}__docaligner.json').exists()}
bad = ['26_1.png', '26_2.png', '8_2.png']
t2 = pd.DataFrame({r: {'3 ảnh nền màu (PSNR dB)': np.mean([v[k] for k in bad])} for r, v in dc.items()}).T
print('Trang nền màu của DocAligner (26_1, 26_2, 8_2)'); display(t2.round(2))
t2.round(3).to_csv(RES / 'mau_nen.csv'); ci.round(3).to_csv(RES / 'anh_sach.csv')
"""))
cells.append(md("## 6. CER (VietOCR gốc, 317 dòng ESP32)"))
i_cer = find("vietocr==0.3.13", "code")
s = text(old[i_cer]).replace("'--tag', 'retrain2'", "'--tag', 'retrain3'")
cells.append(dict(cell_type="code", metadata={}, source=s, execution_count=None, outputs=[]))
cells.append(code("""
try:
    from google.colab import _message
    nb = _message.blocking_request('get_ipynb', request='', timeout_sec=120)['ipynb']
    (RES / 'PBL6_Colab_Seeds_da_chay.ipynb').write_text(json.dumps(nb, ensure_ascii=False), encoding='utf-8')
except Exception as e:
    print('Không tự lưu được, dùng File → Download .ipynb:', e)
print(sorted(p.name for p in RES.iterdir()))
"""))

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "PBL6_Colab_Seeds.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
with zipfile.ZipFile(ROOT / "pbl6_pipeline_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((PROJ / "src").glob("*.py")):
        z.write(p, f"pbl6_pipeline/{p.name}")
print("→", out, len(cells), "ô")
