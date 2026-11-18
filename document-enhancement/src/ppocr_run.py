import sys, types
for _n in ['imgaug', 'imgaug.augmenters', 'visualdl', 'wandb']:
    _m = types.ModuleType(_n); sys.modules[_n] = _m
sys.modules['visualdl'].LogWriter = object
sys.modules['imgaug'].augmenters = sys.modules['imgaug.augmenters']

import os, sys, json, glob, time
sys.path.insert(0, '/content/ocr/PaddleOCR')
os.chdir('/content/ocr/PaddleOCR')
import numpy as np, paddle
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program

def load(tag):
    cfg = program.load_config(f'/content/ocr/{tag}_config.yml')
    cfg['Global']['pretrained_model'] = f'/content/ocr/{tag}_best'
    cfg['Global']['checkpoints'] = None
    pp = build_post_process(cfg['PostProcess'], cfg['Global'])
    ch = len(pp.character)
    cfg['Architecture']['Head']['out_channels_list'] = {'CTCLabelDecode': ch, 'SARLabelDecode': ch + 2, 'NRTRLabelDecode': ch + 3}
    model = build_model(cfg['Architecture'])
    load_model(cfg, model)
    model.eval()
    ops = [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
           {'RecResizeImg': {'image_shape': [3, 48, 320]}},
           {'KeepKeys': {'keep_keys': ['image']}}]
    return model, pp, create_operators(ops, cfg['Global'])

def read(path):
    with open(path, 'rb') as f: return f.read()

def run(model, pp, ops, paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i+bs]:
            d = transform({'image': read(p)}, ops)
            batch.append(d[0])
        x = paddle.to_tensor(np.stack(batch))
        with paddle.no_grad():
            pred = model(x)
        res = pp(pred)
        out += [(t, float(s)) for t, s in res]
    return out

if __name__ == '__main__':
    import shutil
    for tag in ['M2', 'M3']:
        if not os.path.exists(f'/content/ocr/{tag}_best.pdparams'): raise SystemExit('no ckpt')
    os.makedirs('/content/ocr/res', exist_ok=True)
    # 1) team's 44 ESP32 + 155 scan
    # 2) our crops
    sets = {}
    base = '/content/ocr/pack/dataset_multitier'
    for nm in ['test_esp', 'test_scan']:
        rows = [l.rstrip('\n').split('\t') for l in open(f'{base}/{nm}.txt', encoding='utf-8')]
        sets['team_' + nm] = [(f'{base}/{a}', b) for a, b in rows]
    cb = '/content/ocr/crops_tmp'
    for lf in sorted(glob.glob(f'{cb}/labels/test_*.txt')):
        v = os.path.basename(lf)[5:-4]
        rows = [l.rstrip('\n').split('\t') for l in open(lf, encoding='utf-8')]
        sets['ours_' + v] = [(f'{cb}/{a}', b) for a, b in rows]
    for tag in ['M2', 'M3']:
        model, pp, ops = load(tag)
        for nm, rows in sets.items():
            t0 = time.time()
            pr = run(model, pp, ops, [p for p, _ in rows])
            with open(f'/content/ocr/res/{tag}__{nm}.jsonl', 'w', encoding='utf-8') as f:
                for (p, ref), (h, s) in zip(rows, pr):
                    f.write(json.dumps({'path': p, 'ref': ref, 'hyp': h, 'score': s}, ensure_ascii=False) + '\n')
            print(tag, nm, len(rows), f'{time.time()-t0:.1f}s', flush=True)
