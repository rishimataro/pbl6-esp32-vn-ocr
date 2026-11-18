
import sys, os, json, glob, time, math
sys.path.insert(0, '/content/ocr')
from ppocr_run import *
import cv2, numpy as np, paddle
print('device', paddle.device.get_device(), flush=True)

def split(img):
    h, w = img.shape[:2]
    wp = w * 48 / h
    if wp <= 330: return [img]
    k = math.ceil(wp / 290)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.convolve(b.sum(0).astype(float), np.ones(3) / 3, mode='same')
    cuts = [0]
    for i in range(1, k):
        c = int(w * i / k); r = int(0.12 * w / k)
        lo, hi = max(c - r, cuts[-1] + 5), min(c + r, w - 5)
        cuts.append(lo + int(np.argmin(proj[lo:hi])) if hi > lo else c)
    cuts.append(w)
    return [img[:, a:b_] for a, b_ in zip(cuts[:-1], cuts[1:]) if b_ - a > 5]

def run_chunks(model, pp, ops, paths, bs=64):
    pieces, owner = [], []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        for c in split(img):
            pieces.append(cv2.imencode('.png', c)[1].tobytes()); owner.append(i)
    outs = []
    for j in range(0, len(pieces), bs):
        x = paddle.to_tensor(np.stack([transform({'image': b}, ops)[0] for b in pieces[j:j+bs]]))
        with paddle.no_grad(): pred = model(x)
        outs += [t for t, s in pp(pred)]
    hyp = [[] for _ in paths]
    for o, t in zip(owner, outs): hyp[o].append(t.strip())
    return [' '.join(h) for h in hyp], len(pieces)

cb = '/content/ocr/crops_tmp'
os.makedirs('/content/ocr/res_chunk', exist_ok=True)
for tag in ['M2', 'M3']:
    model, pp, ops = load(tag)
    for lf in sorted(glob.glob(f'{cb}/labels/test_*.txt')):
        v = os.path.basename(lf)[5:-4]
        rows = [l.rstrip('\n').split('\t') for l in open(lf, encoding='utf-8')]
        t0 = time.time()
        hyp, npc = run_chunks(model, pp, ops, [f'{cb}/{a}' for a, _ in rows])
        with open(f'/content/ocr/res_chunk/{tag}__ours_{v}.jsonl', 'w', encoding='utf-8') as f:
            for (a, ref), h in zip(rows, hyp):
                f.write(json.dumps({'path': a, 'ref': ref, 'hyp': h, 'score': 0}, ensure_ascii=False) + '\n')
        print(tag, v, len(rows), 'pieces', npc, f'{time.time()-t0:.1f}s', flush=True)
