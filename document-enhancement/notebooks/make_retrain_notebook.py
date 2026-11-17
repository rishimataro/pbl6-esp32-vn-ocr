"""Sinh notebook Colab train lại model nâng ảnh từ đầu: notebooks/PBL6_Colab_Retrain.ipynb.
Đồng thời đóng gói code trong src/ vào notebooks/pbl6_pipeline_code.zip để đưa lên MyDrive/PBL6/code/.

  .venv-ocr/Scripts/python.exe notebooks/make_retrain_notebook.py

Trong zip, code nằm ở thư mục pbl6_pipeline/ để notebook trên Colab giữ nguyên đường dẫn /content/pbl6_pipeline.
"""
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=textwrap.dedent(text).strip() + "\n"))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, source=textwrap.dedent(text).strip() + "\n",
                      execution_count=None, outputs=[]))


# ---------------------------------------------------------------------------------------------- 0. mô tả
md("""
# PBL6: train lại model nâng chất lượng ảnh tài liệu (đợt 2: M5)

**Bài toán.** Ảnh tài liệu chụp bằng camera bất kỳ (ESP32-CAM, điện thoại) thường thiếu sáng, nhiễu, ám màu, có bóng đổ, mờ, nén JPEG.
Model nhận ảnh xấu và trả về ảnh sạch kiểu scan, cùng kích thước.

**Model.** NAFNet bản nhỏ (6,8 triệu tham số), train từ trọng số ngẫu nhiên. Loss gốc: L1 + 0,2·(1 − SSIM).

**Vì sao train lại.** Đợt 1 (M4) cho ba điểm yếu, nên đợt này thử ba cải tiến, mỗi cái nhắm vào một lỗi:

| Lỗi thấy ở M4 | Cải tiến thử | Cờ trong `train_restore.py` |
|---|---|---|
| Dấu tiếng Việt bị biến dạng ("Giới" → "Giói"), lỗi dấu tăng sau khi nâng ảnh | thêm loss gradient Sobel để giữ nét mảnh | `--edge-w 1.0` |
| Trang nền màu (xanh lá, xám) bị tẩy thành trắng; PSNR giảm tới 6 dB | nhuộm màu giấy nhạt cho ảnh chuẩn khi sinh dữ liệu tổng hợp | `--paper-color 0.2` |
| Ảnh vốn đã tốt (chỉ mờ hoặc chỉ nén JPEG) bị model làm xấu đi | thêm mẫu "ảnh sạch → ảnh sạch" | `--identity-p 0.1` |

**Các run (mỗi run train từ đầu, 8.000 vòng, batch 16, crop 192, seed 0):**

| Run | Mô tả |
|---|---|
| `m4r_baseline` | cấu hình M4 của đợt 1, chạy lại để làm mốc cùng điều kiện |
| `m5_all` | M4 + cả ba cải tiến (model đề xuất) |
| `m5a_edge` | M4 + chỉ loss gradient |
| `m5b_color` | M4 + chỉ nhuộm màu giấy |
| `m5c_identity` | M4 + chỉ mẫu ảnh sạch |
| `m5l_nafnet64` (tùy chọn) | M5 với width 64, batch 32, crop 256, 16.000 vòng |

Dữ liệu train: RealDAE (400 cặp ảnh chụp thật ↔ ảnh chuẩn), 2.374 patch ESP32 ↔ scan, và ảnh suy biến tổng hợp (xác suất 0,5).

**Đánh giá (tập test, không dùng để chọn checkpoint):** RealDAE test, tập tổng hợp 1.260 ảnh, DocAligner 130 ảnh chụp điện thoại,
Jung 87 ảnh bóng đổ, 10 trang ESP32. Chỉ số: PSNR (RGB), SSIM (xám), LPIPS. Ngoài ra có hai phép thử riêng cho từng lỗi ở trên.

**Cách chạy.** Chạy lần lượt từ trên xuống. Runtime → Change runtime type → **A100 / H100** (L4 cũng chạy, lâu hơn).
Checkpoint ghi thẳng vào Drive nên runtime bị ngắt vẫn chạy tiếp được bằng `--resume`.

**Cần có trên Drive (`MyDrive/PBL6/`):** `code/pbl6_pipeline_code.zip`; `data/n2_patches.zip`, `n2_pages_layout.zip`, `n2_ocr_lines.zip`;
`data/realdae_raw.zip`; `data/eval_docaligner.zip`, `data/eval_jung.zip`; `data/vgg_transformer.pth` (chỉ cho mục đo CER).
""")

code("""
import json, os, subprocess, sys, glob, shutil, time
from pathlib import Path
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
NPROC = os.cpu_count(); print('vCPU:', NPROC)

from google.colab import drive
drive.mount('/content/drive')
D = Path('/content/drive/MyDrive/PBL6')
CK_OLD = D / 'checkpoints' / 'retrain'       # đợt 1 (M4...), chỉ đọc
CK = D / 'checkpoints' / 'retrain2'          # đợt 2: mỗi run một thư mục con
RES = D / 'results' / 'retrain2'
FIG = RES / 'figures'
for p in (CK, RES, FIG): p.mkdir(parents=True, exist_ok=True)
C = Path('/content/pbl6_pipeline')
subprocess.run(['unzip', '-oq', str(D / 'code' / 'pbl6_pipeline_code.zip'), '-d', '/content'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lpips'], check=True)
print(sorted(p.name for p in C.glob('*.py')))
""")

# ---------------------------------------------------------------------------------------------- 1. dữ liệu
md("""
## 1. Dữ liệu

ESP32: giải nén ba zip của bước N2 vào `/content/esp32`.
""")

code("""
E = Path('/content/esp32'); E.mkdir(exist_ok=True)
for z in ['n2_patches.zip', 'n2_pages_layout.zip', 'n2_ocr_lines.zip']:
    subprocess.run(['unzip', '-oq', str(D / 'data' / z), '-d', str(E)], check=True)

def lift(name):
    # zip có thể chứa thêm một thư mục bọc ngoài (vd n2_data/patches): đưa về E/<name>
    if (E / name).exists():
        return
    hits = [p for p in E.rglob(name) if p.parent != E]
    if hits:
        shutil.move(str(hits[0]), str(E / name))
for n in ['patches', 'pages', 'crops', 'labels', 'lines.jsonl']:
    lift(n)
for n in ['patches/train/esp', 'patches/val/esp', 'pages/esp_raw', 'pages/scan', 'lines.jsonl']:
    print(f'{n:22s}', 'OK' if (E / n).exists() else 'THIẾU')
""")

md("RealDAE: giải nén rồi chuẩn hóa về `/content/realdae/{train,val,test}/{inp,gt}` (cạnh dài 1024 px; 50 ảnh train tách làm val).")

code("""
R_RAW, R = Path('/content/realdae_raw'), Path('/content/realdae')
HAVE_REALDAE = (D / 'data' / 'realdae_raw.zip').exists()
if HAVE_REALDAE:
    subprocess.run(['unzip', '-oq', str(D / 'data' / 'realdae_raw.zip'), '-d', str(R_RAW)], check=True)
    subprocess.run([sys.executable, C / 'prep_realdae.py', '--src', R_RAW, '--tree'])
    subprocess.run([sys.executable, C / 'prep_realdae.py', '--src', R_RAW, '--out', R,
                    '--long-side', '1024', '--val', '50', '--seed', '0'], check=True)
    print(json.loads((R / 'manifest.json').read_text())['counts'])
else:
    print('Chưa có realdae_raw.zip trên Drive → chỉ dùng ESP32 + tổng hợp')
""")

md("""
Tập test ngoài (không dùng khi train): **DocAligner** (130 ảnh chụp điện thoại, ảnh chuẩn là bản scan) và **Jung** (87 ảnh bóng đổ).
Tập test tổng hợp cố định (6 loại + mix, 3 mức, seed 123) từ ảnh chuẩn RealDAE test và 10 trang scan ESP32 test.
""")

code("""
EXT = Path('/content/ext'); EXT.mkdir(exist_ok=True)
for z in ['eval_docaligner.zip', 'eval_jung.zip']:
    subprocess.run(['unzip', '-oq', str(D / 'data' / z), '-d', str(EXT)], check=True)

TEST_PAGES = [3, 4, 7, 9, 11, 14, 25, 32, 38, 44]
EC = Path('/content/esp32_test_clean'); EC.mkdir(exist_ok=True)
for p in TEST_PAGES:
    shutil.copy(E / 'pages' / 'scan' / f'{p:06d}.png', EC / f'esp{p:02d}.png')
clean = [str(EC)] + ([str(R / 'test' / 'gt')] if HAVE_REALDAE else [])
SYN = Path('/content/synth_test')
subprocess.run([sys.executable, C / 'degrade.py', '--clean', *clean, '--out', SYN,
                '--long-side', '1024', '--max-images', '60', '--seed', '123'], check=True)

# tập "ảnh sạch → ảnh sạch" để đo model có làm hỏng ảnh đã tốt không (phép thử cho --identity-p)
CLEAN = Path('/content/clean_identity'); (CLEAN / 'inp').mkdir(parents=True, exist_ok=True); (CLEAN / 'gt').mkdir(exist_ok=True)
src = sorted((R / 'test' / 'gt').glob('*.png'))[:60] if HAVE_REALDAE else sorted(EC.glob('*.png'))
for f in src:
    shutil.copy(f, CLEAN / 'inp' / f.name); shutil.copy(f, CLEAN / 'gt' / f.name)

import pandas as pd
def count(d): return len(list(Path(d).glob('*.png'))) + len(list(Path(d).glob('*.jpg')))
def trio(*v): return ' / '.join(str(x) for x in v)
tab = pd.DataFrame([
    ['ESP32 patch (train / val)', trio(len(list((E/'patches/train/esp').glob('*.png'))), len(list((E/'patches/val/esp').glob('*.png'))))],
    ['RealDAE (train / val / test)', trio(*(count(R/s/'gt') for s in ('train', 'val', 'test'))) if HAVE_REALDAE else 'không có'],
    ['Tổng hợp test (ảnh)', len(list(SYN.glob('*/L*/inp/*.png')))],
    ['DocAligner test', count(EXT/'docaligner/gt')],
    ['Jung test', count(EXT/'jung/gt')],
], columns=['Tập', 'Số lượng'])
tab
""")

md("""
**Ví dụ dữ liệu huấn luyện tổng hợp.** Một ảnh chuẩn, các loại suy biến (cường độ tăng dần) và ảnh nhuộm màu giấy (cải tiến `--paper-color`).
""")

code("""
sys.path.insert(0, str(C))
import numpy as np, cv2, matplotlib.pyplot as plt
from degrade import degrade, level_config
from train_restore import tint_paper

def rd(p): return cv2.imread(str(p))[..., ::-1]
ref = rd(sorted((R / 'test' / 'gt' if HAVE_REALDAE else EC).glob('*.png'))[3])
ref = cv2.resize(ref, None, fx=640 / max(ref.shape[:2]), fy=640 / max(ref.shape[:2]), interpolation=cv2.INTER_AREA)
rng = np.random.default_rng(1)
types = ['shading', 'color', 'lowlight', 'blur', 'noise', 'jpeg']
fig, ax = plt.subplots(3, 4, figsize=(16, 11))
panels = [('Ảnh chuẩn', ref)] + [(f'{t} mức 3', degrade(ref, level_config(t, 3, rng), rng)) for t in types] + \
         [('mix mức 2', degrade(ref, level_config('mix', 2, rng), rng))] + \
         [(f'nhuộm giấy #{i}', tint_paper(ref, rng)) for i in range(4)]
for a, (t, im) in zip(ax.ravel(), panels):
    a.imshow(im); a.set_title(t); a.axis('off')
plt.tight_layout(); plt.savefig(FIG / 'du_lieu_tong_hop.png', dpi=110); plt.show()
""")

# ---------------------------------------------------------------------------------------------- 2. kiểm tra nhanh
md("""
## 2. Kiểm tra nhanh (khoảng 3 phút)

Chạy 300 vòng với đủ ba cải tiến để chắc không có lỗi. Đạt khi `val_..._psnr` cao hơn `val_..._psnr_input` và `data_wait_frac` nhỏ
(CPU kịp cấp dữ liệu).
""")

code("""
DATA_ALL = ([str(R)] if HAVE_REALDAE else []) + [str(E)]
subprocess.run([sys.executable, C / 'train_restore.py', '--data', *DATA_ALL, '--synth', '0.5',
                '--edge-w', '1.0', '--paper-color', '0.2', '--identity-p', '0.1',
                '--out', '/content/smoke', '--iters', '300', '--eval-every', '100'], check=True)
""")

# ---------------------------------------------------------------------------------------------- 3. train
md("""
## 3. Train: các run chạy song song trên một GPU

Mỗi run là một tiến trình nền, log ghi vào Drive (`log.jsonl`). Ô sau khởi chạy rồi trả về ngay.
Dùng ô theo dõi bên dưới để xem tiến độ và vẽ đường cong; ô đó chạy lại được bất cứ lúc nào.
""")

code("""
COMMON = ['--iters', '8000', '--bs', '16', '--crop', '192', '--eval-every', '500', '--seed', '0']
BASE = ['--data', *DATA_ALL, '--synth', '0.5']
RUNS = {
    'm4r_baseline': BASE,
    'm5_all':       BASE + ['--edge-w', '1.0', '--paper-color', '0.2', '--identity-p', '0.1'],
    'm5a_edge':     BASE + ['--edge-w', '1.0'],
    'm5b_color':    BASE + ['--paper-color', '0.2'],
    'm5c_identity': BASE + ['--identity-p', '0.1'],
}
WORKERS = max(2, (NPROC - 1) // len(RUNS))
procs = {}
PARALLEL = 3      # 5 run cùng lúc từng hết bộ nhớ GPU trên A100 40 GB (CUDA OOM): chỉ chạy tối đa 3 run một lúc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
pending = []

def launch(name, extra, resume=False):
    out = CK / name
    out.mkdir(parents=True, exist_ok=True)
    # COMMON trước, extra sau: đối số trùng thì giá trị sau (của run) thắng
    cmd = [sys.executable, str(C / 'train_restore.py'), *COMMON, *extra, '--out', str(out),
           '--num-workers', str(WORKERS)] + (['--resume'] if resume else [])
    log = open(out / 'stdout.log', 'a')
    procs[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    print('chạy', name, 'pid', procs[name].pid)

def tick():
    # khởi chạy các run đang chờ khi số run đang chạy < PARALLEL; ô theo dõi bên dưới gọi hàm này
    running = [n for n, pr in procs.items() if pr.poll() is None]
    while pending and len(running) < PARALLEL:
        n = pending.pop(0)
        launch(n, RUNS[n]); running.append(n)

pending = list(RUNS)
tick()
""")

code("""
# Theo dõi + đường cong train. Chạy lại ô này bất cứ lúc nào.
import matplotlib.pyplot as plt, pandas as pd

def read_log(name):
    f = CK / name / 'log.jsonl'
    if not f.exists() or not f.read_text().strip():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()])

tick()      # đủ chỗ trống thì chạy tiếp các run đang chờ
if pending: print('đang chờ:', pending)
status = []
for name in RUNS:
    df = read_log(name)
    rc = procs[name].poll() if name in procs else None
    chua = name in pending
    last = df.iloc[-1] if len(df) else {}
    status.append({'run': name, 'trạng thái': 'chờ' if chua else 'đang chạy' if rc is None and name in procs else f'xong (mã {rc})' if name in procs else '?',
                   'vòng': int(last['iter']) if len(df) else 0, 'val PSNR TB': last.get('val_score'),
                   'vòng/giây': last.get('it_per_s'), 'chờ dữ liệu': last.get('data_wait_frac')})
display(pd.DataFrame(status))

logs = {n: read_log(n) for n in RUNS}
fig, ax = plt.subplots(1, 4, figsize=(21, 4.2))
for n, df in logs.items():
    if not len(df): continue
    ax[0].plot(df['iter'], df['train_l1'], label=n)
    ax[1].plot(df['iter'], df['val_score'], label=n)
    sy = [c for c in df.columns if c == 'val_synth_psnr']
    if sy: ax[2].plot(df['iter'], df['val_synth_psnr'], label=n)
    ax[3].plot(df['iter'], df['it_per_s'], label=n)
for a, t in zip(ax, ['Loss L1 train', 'PSNR val trên ảnh thật (TB các nguồn)', 'PSNR val trên ảnh tổng hợp', 'Tốc độ (vòng/giây)']):
    a.set_title(t); a.set_xlabel('vòng'); a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / 'duong_cong_train.png', dpi=110); plt.show()
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv'], capture_output=True, text=True).stdout)
""")

code("""
# Nếu runtime bị ngắt: chạy lại các ô mục 0-1 (mount, dữ liệu) và ô định nghĩa RUNS/launch (bỏ vòng for cuối), rồi chạy ô này
def target_iters(extra):
    return int(extra[extra.index('--iters') + 1]) if '--iters' in extra else 8000

pending = [n for n, extra in RUNS.items()
           if not len(read_log(n)) or read_log(n)['iter'].iloc[-1] < target_iters(extra)]
RUNS_RESUME = True
def tick():
    running = [n for n, pr in procs.items() if pr.poll() is None]
    while pending and len(running) < PARALLEL:
        n = pending.pop(0)
        launch(n, RUNS[n], resume=True); running.append(n)
tick()
""")

md("""
### Run tùy chọn: model lớn `m5l_nafnet64`

Khi còn GPU: width 64 (27,1 triệu tham số), batch 32, crop 256, 16.000 vòng, kèm cả ba cải tiến. Mất khoảng 1,5–2 giờ trên GPU cỡ L4/G4.
Bỏ qua ô này nếu thiếu thời gian.
""")

code("""
RUNS['m5l_nafnet64'] = BASE + ['--edge-w', '1.0', '--paper-color', '0.2', '--identity-p', '0.1',
                               '--width', '64', '--bs', '32', '--crop', '256', '--iters', '16000']
RUN_LARGE = False      # đổi thành True để chạy
if RUN_LARGE:
    pending.append('m5l_nafnet64'); tick()
else:
    RUNS.pop('m5l_nafnet64')
""")

# ---------------------------------------------------------------------------------------------- 4. đánh giá
md("""
## 4. Đánh giá

Chấm mọi `best.pt` (checkpoint có PSNR val cao nhất, chọn trên tập val, không phải tập test), cùng M4 đợt 1 (`m4_prev`) và hai đối chứng không học
(ảnh vào, cổ điển). Kết quả JSON ghi vào `MyDrive/PBL6/results/retrain2/`.
""")

code("""
def ev(tag, setdir, *method):
    out = RES / f'{tag}.json'
    subprocess.run([sys.executable, C / 'eval_sets.py', '--set', setdir, *method, '--lpips', '--out', out], check=True)

SETS = {'synth': SYN, 'docaligner': EXT / 'docaligner', 'jung': EXT / 'jung', 'clean_identity': CLEAN}
if HAVE_REALDAE:
    SETS['realdae'] = R / 'test'

CKPTS = {p.parent.name: p for p in sorted(CK.glob('*/best.pt'))}
if (CK_OLD / 'm4_nafnet_all' / 'best.pt').exists():
    CKPTS['m4_prev'] = CK_OLD / 'm4_nafnet_all' / 'best.pt'
print('Sẽ chấm:', list(CKPTS))

for sname, sdir in SETS.items():
    ev(f'm0_identity__{sname}', sdir, '--method', 'identity')
    if sname != 'clean_identity':
        ev(f'm1_classic__{sname}', sdir, '--method', 'classic')
    for run, ck in CKPTS.items():
        ev(f'{run}__{sname}', sdir, '--method', 'model', '--ckpt', ck, '--amp', 'bf16')
""")

md("Trang ESP32 (cả trang, so với bản scan trong vùng hợp lệ):")

code("""
for run, ck in CKPTS.items():
    v = 'esp_' + run
    subprocess.run([sys.executable, C / 'restore_eval.py', '--data', E, '--ckpt', ck, '--variant', v, '--splits', 'test'], check=True)
    shutil.copy(E / 'eval' / f'restore_{v}.json', RES / f'{run}__esp32_pages.json')
""")

# ---------------------------------------------------------------------------------------------- 5. bảng + biểu đồ
md("## 5. Kết quả: bảng và biểu đồ")

code("""
rows = []
for f in sorted(RES.glob('*__*.json')):
    run, sname = f.stem.split('__')
    r = json.loads(f.read_text())
    if 'groups' in r:
        for g, v in r['groups'].items():
            rows.append({'run': run, 'set': sname, 'group': g, **v['mean'], 'sec': v['sec_per_image'], 'params': r.get('params')})
    else:   # restore_eval (ESP32 cả trang)
        rows.append({'run': run, 'set': sname, 'group': 'test', **r['mean']})
df = pd.DataFrame(rows)
df.to_csv(RES / 'summary.csv', index=False)

ORDER = ['m0_identity', 'm1_classic', 'm4_prev', 'm4r_baseline', 'm5a_edge', 'm5b_color', 'm5c_identity', 'm5_all', 'm5l_nafnet64']
ORDER = [o for o in ORDER if o in set(df['run'])]
NAMES = {'realdae': 'RealDAE test', 'synth': 'Tổng hợp (TB các ô)', 'docaligner': 'DocAligner', 'jung': 'Jung', 'esp32_pages': 'ESP32 10 trang'}

def overall(sname, metric):
    d = df[df['set'] == sname]
    if sname == 'synth':   # trung bình các ô loại × mức, bỏ gộp '.'
        d = d[d['group'] != '.']
    return d.groupby('run')[metric].mean()

main = pd.DataFrame({NAMES[s]: overall(s, 'psnr') for s in NAMES if s in set(df['set'])}).reindex(ORDER)
print('PSNR (dB), cao hơn là tốt hơn')
display(main.round(2).style.highlight_max(axis=0, color='#c6efce'))
main.round(3).to_csv(RES / 'summary_main_psnr.csv')
""")

code("""
extra = {}
for metric, better in [('ssim', 'cao'), ('lpips', 'thấp')]:
    if metric in df.columns:
        t = pd.DataFrame({NAMES[s]: overall(s, metric) for s in NAMES if s in set(df['set']) and df[df['set'] == s][metric].notna().any()}).reindex(ORDER)
        print(f'{metric.upper()} ({better} hơn là tốt hơn)')
        display(t.round(3))
        extra[metric] = t
sec = overall('docaligner', 'sec').reindex(ORDER)
print('Thời gian xử lý trung bình trên DocAligner (giây/ảnh, GPU hiện tại):'); display(sec.round(3).to_frame('giây/ảnh'))
""")

code("""
# Biểu đồ cột: PSNR từng tập
cols = list(main.columns)
fig, ax = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4.5), sharey=False)
for a, c in zip(np.atleast_1d(ax), cols):
    v = main[c]
    bars = a.bar(range(len(v)), v.values, color=['#bbbbbb' if r in ('m0_identity', 'm1_classic') else '#7aa6c2' if r in ('m4_prev', 'm4r_baseline')
                                                  else '#d9822b' if r == 'm5_all' else '#9dc183' for r in v.index])
    a.set_xticks(range(len(v))); a.set_xticklabels(v.index, rotation=70, ha='right', fontsize=8)
    a.set_title(c); a.set_ylabel('PSNR (dB)'); a.set_ylim(max(0, np.nanmin(v.values) - 3), np.nanmax(v.values) + 1.5)
    for b, x in zip(bars, v.values): a.text(b.get_x() + b.get_width() / 2, x + 0.1, f'{x:.1f}', ha='center', fontsize=7)
plt.tight_layout(); plt.savefig(FIG / 'psnr_theo_tap.png', dpi=110); plt.show()
""")

code("""
# Tập tổng hợp: PSNR theo loại suy biến (TB 3 mức), kèm biểu đồ nhiệt mức cải thiện so với ảnh vào
s = df[(df['set'] == 'synth') & (df['group'] != '.')].copy()
s['loai'] = s['group'].str.split('/').str[0]
by = s.pivot_table(index='loai', columns='run', values='psnr').reindex(columns=ORDER)
print('PSNR theo loại suy biến (dB)'); display(by.round(2).style.highlight_max(axis=1, color='#c6efce'))
gain = by.sub(by['m0_identity'], axis=0).drop(columns=['m0_identity'])
fig, a = plt.subplots(figsize=(1.3 * len(gain.columns) + 2, 4.2))
im = a.imshow(gain.values, cmap='RdYlGn', vmin=-8, vmax=8, aspect='auto')
a.set_xticks(range(gain.shape[1])); a.set_xticklabels(gain.columns, rotation=40, ha='right'); a.set_yticks(range(gain.shape[0])); a.set_yticklabels(gain.index)
for i in range(gain.shape[0]):
    for j in range(gain.shape[1]): a.text(j, i, f'{gain.values[i, j]:+.1f}', ha='center', va='center', fontsize=8)
a.set_title('Mức tăng PSNR so với ảnh vào (dB), xanh = tốt hơn ảnh vào'); plt.colorbar(im, ax=a); plt.tight_layout()
plt.savefig(FIG / 'synth_theo_loai.png', dpi=110); plt.show()
""")

md("""
### Phép thử riêng cho từng cải tiến

1. **Ảnh vốn đã sạch (`--identity-p`).** Đưa ảnh scan sạch vào, đo ảnh ra so với chính nó. PSNR càng cao thì model càng ít làm hỏng ảnh tốt.
2. **Trang nền màu (`--paper-color`).** Trên các ảnh DocAligner có nền màu chủ ý (26_1, 26_2, 8_2) và 20 ảnh tệ nhất của M4 trước đó, so PSNR của từng run.
3. **Giữ nét mảnh (`--edge-w`).** Chỉ số gián tiếp: SSIM và LPIPS ở bảng trên; kết luận cuối cùng dựa vào CER OCR (xem mục 7).
""")

code("""
ci = overall('clean_identity', 'psnr').reindex([r for r in ORDER if r != 'm0_identity']).to_frame('PSNR ảnh ra so với ảnh sạch (dB)')
print('Phép thử 1: đưa ảnh sạch vào, model có làm hỏng không (càng cao càng tốt)'); display(ci.round(2))

def per_image(run, sname):
    f = RES / f'{run}__{sname}.json'
    g = json.loads(f.read_text())['groups']
    return {im['name']: im['psnr'] for v in g.values() for im in v['images']}

dc = {r: per_image(r, 'docaligner') for r in ORDER if (RES / f'{r}__docaligner.json').exists()}
bad = ['26_1.png', '26_2.png', '8_2.png']
worst = sorted(dc['m4_prev' if 'm4_prev' in dc else 'm4r_baseline'], key=lambda k: dc['m4_prev' if 'm4_prev' in dc else 'm4r_baseline'][k] - dc['m0_identity'][k])[:20]
t2 = pd.DataFrame({r: {'3 ảnh nền màu': np.mean([v[k] for k in bad]), '20 ảnh tệ nhất của M4': np.mean([v[k] for k in worst])} for r, v in dc.items()}).T
print('Phép thử 2: trang nền màu (PSNR dB)'); display(t2.round(2))
""")

md("""
### Tác động của từng cải tiến (ablation)

Mức chênh PSNR so với `m4r_baseline` (cùng cấu hình, cùng seed). Hai lần chạy M4 ở đợt 1 lệch nhau tới 0,15 dB, nên chênh dưới ~0,3 dB coi như ngang nhau.
""")

code("""
abl = main.loc[[r for r in ['m4r_baseline', 'm5a_edge', 'm5b_color', 'm5c_identity', 'm5_all', 'm5l_nafnet64'] if r in main.index]]
d = abl.sub(abl.loc['m4r_baseline'], axis=1).drop(index='m4r_baseline')
print('Chênh PSNR so với m4r_baseline (dB)'); display(d.round(2).style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))
d.round(3).to_csv(RES / 'ablation_delta.csv')
""")

# ---------------------------------------------------------------------------------------------- 6. ảnh trước/sau
md("""
## 6. Ảnh trước / sau

Mỗi hàng một ảnh test, cột: ảnh vào, M4 đợt 1, M5 (model đề xuất), ảnh chuẩn. Chọn ảnh có ý nghĩa cho từng lỗi: trang nền màu (DocAligner 26_2), bóng đổ (Jung), trang ESP32, và ảnh chỉ mờ.
""")

code("""
sys.path.insert(0, str(C))
import torch
from restore_model import load_model, restore_image
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
show_runs = [r for r in ['m4_prev', 'm4r_baseline', 'm5_all'] if r in CKPTS][:1] + (['m5_all'] if 'm5_all' in CKPTS else [])
mods = {r: load_model(CKPTS[r], dev) for r in show_runs}
blur_img = sorted((SYN / 'blur' / 'L2' / 'inp').glob('*.png'))[1]
cases = [('DocAligner 26_2 (nền màu)', EXT / 'docaligner/inp/26_2.png', EXT / 'docaligner/gt/26_2.png'),
         ('DocAligner 16_1 (điển hình)', EXT / 'docaligner/inp/16_1.png', EXT / 'docaligner/gt/16_1.png'),
         ('Jung rect_10 (bóng đổ)', EXT / 'jung/inp/rect_10.jpg', EXT / 'jung/gt/rect_10.jpg'),
         ('Tổng hợp: chỉ mờ mức 2', blur_img, blur_img.parent.parent / 'gt' / blur_img.name),
         ('ESP32 trang 3', E / 'pages/esp_raw/000003.png', E / 'pages/scan/000003.png')]
fig, ax = plt.subplots(len(cases), 2 + len(mods), figsize=(4.4 * (2 + len(mods)), 5.2 * len(cases)))
for i, (title, pi, pg) in enumerate(cases):
    x = cv2.imread(str(pi)); g = cv2.imread(str(pg))
    outs = [('Ảnh vào', x)] + [(r, restore_image(m, x, dev, amp=torch.bfloat16 if dev == 'cuda' else None)) for r, m in mods.items()] + [('Ảnh chuẩn', g)]
    for j, (t, im) in enumerate(outs):
        ax[i, j].imshow(im[..., ::-1]); ax[i, j].axis('off'); ax[i, j].set_title(f'{title}\\n{t}' if j == 0 else t, fontsize=10)
plt.tight_layout(); plt.savefig(FIG / 'truoc_sau.png', dpi=100); plt.show()
""")

# ---------------------------------------------------------------------------------------------- 7. CER
md("""
## 7. Đo CER bằng VietOCR (tùy chọn)

Cắt từng dòng chữ của 10 trang ESP32 test từ ảnh đã nâng, cho VietOCR (bản gốc, không fine-tune) đọc, tính tỉ lệ ký tự sai (CER).
Đo thêm CER sau khi bỏ dấu để tách lỗi dấu khỏi lỗi chữ cái, vì cải tiến loss gradient nhắm vào lỗi dấu.
Trọng số VietOCR gốc (145 MB) lấy từ `MyDrive/PBL6/data/vgg_transformer.pth`.
""")

code("""
# vietocr==0.3.13 ghim pillow==10.2.0 (không có wheel cho Python 3.13): cài --no-deps rồi các gói cần dùng
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'vietocr==0.3.13'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'einops', 'gdown', 'prefetch_generator', 'lmdb', 'rapidfuzz', 'pypdfium2', 'pyyaml', 'requests', 'tqdm'], check=True)
WEIGHTS = '/content/vgg_transformer.pth'
shutil.copy(D / 'data' / 'vgg_transformer.pth', WEIGHTS)     # VietOCR gốc, chưa fine-tune
cer_rows = []
variants = ['esp_raw', 'scan'] + ['esp_' + r for r in CKPTS]
for v in variants:
    r = subprocess.run([sys.executable, C / 'n2_eval.py', '--data', E, '--skip-labels', '--variants', v, '--split', 'test',
                        '--device', 'cuda', '--weights', WEIGHTS, '--tag', 'retrain2'], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith(v + ' ')]
    if line:
        o = json.loads(line[0][len(v) + 1:])['all']
        cer_rows.append({'phiên bản': v.replace('esp_', ''), 'CER': o['cer'], 'CER bỏ dấu': o['cer_no_accent'], 'lỗi riêng do dấu': round(o['cer'] - o['cer_no_accent'], 4)})
    else:
        print(v, 'lỗi:', r.stderr[-300:])
cer = pd.DataFrame(cer_rows).set_index('phiên bản')
display(cer.style.format('{:.3f}').highlight_min(axis=0, color='#c6efce'))
cer.to_csv(RES / 'cer_esp32.csv')
ax = cer[['CER', 'CER bỏ dấu']].plot.bar(figsize=(9, 4), rot=30); ax.set_ylabel('CER'); ax.set_title('CER trên 317 dòng ESP32 test (thấp hơn là tốt hơn)')
plt.tight_layout(); plt.savefig(FIG / 'cer_esp32.png', dpi=110); plt.show()
""")

# ---------------------------------------------------------------------------------------------- 8. kết luận
md("## 8. Kết luận tự động")

code("""
best = {s: main[NAMES[s]].drop(['m0_identity', 'm1_classic'], errors='ignore').idxmax() for s in NAMES if s in set(df['set'])}
print('Run tốt nhất theo từng tập:', best)
if 'm5_all' in main.index and 'm4r_baseline' in main.index:
    dd = (main.loc['m5_all'] - main.loc['m4r_baseline']).round(2)
    print('\\nM5 so với M4 chạy lại (PSNR dB):'); print(dd.to_string())
    for c, v in dd.items():
        print(f'- {c}:', 'tốt hơn' if v > 0.3 else 'kém hơn' if v < -0.3 else 'ngang nhau (trong nhiễu seed)', f'({v:+.2f} dB)')
print('\\nTệp đã lưu vào', RES)
print(sorted(p.name for p in RES.iterdir()))
""")

code("""
# Lưu bản sao notebook này (kèm đầu ra) vào Drive để nộp báo cáo: File → Save a copy in Drive, hoặc dùng ô dưới nếu chạy trong Colab
try:
    from google.colab import _message
    nb = _message.blocking_request('get_ipynb', request='', timeout_sec=120)['ipynb']
    (RES / 'PBL6_Colab_Retrain_da_chay.ipynb').write_text(json.dumps(nb, ensure_ascii=False), encoding='utf-8')
    print('đã lưu', RES / 'PBL6_Colab_Retrain_da_chay.ipynb')
except Exception as e:
    print('Không tự lưu được, hãy dùng File → Download .ipynb:', e)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "PBL6_Colab_Retrain.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

zpath = ROOT / "pbl6_pipeline_code.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((PROJ / "src").glob("*.py")):
        z.write(p, f"pbl6_pipeline/{p.name}")
print("→", out, "\n→", zpath)
