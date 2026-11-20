# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["torch", "numpy", "pandas", "scipy", "scikit-learn"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Local server of the CoAD web demo: serves the static demo AND trains CoAD on a CSV uploaded in the page.

One command (installs Python 3.11, CPU PyTorch and the other packages in a private cache, then opens the browser):
    uv run demo/serve_coad.py

A CoAD checkpoint only knows the series it was trained on, so a new CSV needs its own training run. This server
does exactly what precompute.py does for a benchmark series, on the uploaded values:
  1. min-max scaling on the training part (the first n_train points, assumed normal), period by TSB-AD find_length,
     window = 4 periods (rounded to the patch size), the authors' training stride rule;
  2. trains CoAD in code mode (coad_core.py, seed 46) for the chosen number of epochs;
  3. writes data/series_<group>.json/.bin in the demo format (group >= 9001), so every screen works on it.
Labels, when the CSV has them, are only used for evaluation and display, never for training or thresholds.

usage (from demo/):  .venv/Scripts/python serve_coad.py [--port 8765] [--repo <authors' CoAD repo, for VUS-PR>]
API: GET /api/health · POST /api/train {name, values, labels|null, n_train, epochs} -> {job} · GET /api/job/<job>
"""
import argparse
import contextlib
import io
import json
import re
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coad_core as cc  # noqa: E402
import precompute as pc  # noqa: E402  (train_scores, patch_diagnostics, threshold_policies, clean)

DATA = HERE / 'data'
JOBS, LOCK = {}, threading.Lock()
EVALUATE = None
MAX_POINTS = 200_000


def find_length(x):
    """TSB-AD find_length: argmax of the ACF local maxima for lags 3..400 on the first 20 000 points; 125 if none in [3, 300]."""
    x = np.asarray(x[:20000], dtype=float)
    x = x - x.mean()
    c0 = float((x * x).sum())
    if len(x) < 50 or c0 == 0:
        return 125
    lags = min(400, len(x) - 1)
    acf = np.array([1.0] + [float((x[:-k] * x[k:]).sum()) / c0 for k in range(1, lags + 1)])[3:]
    peaks = [i for i in range(1, len(acf) - 1) if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]]
    if not peaks:
        return 125
    best = max(peaks, key=lambda i: acf[i])
    return 125 if best < 3 or best > 300 else best + 3


def make_series(values, labels, n_train, max_windows=None):
    """Series dict in the format of coad_core.load_series (authors' window and stride rules)."""
    values = np.asarray(values, dtype=float)
    train_raw = values[:n_train]
    lo, hi = train_raw.min(), train_raw.max()
    scale = (hi - lo) if hi > lo else 1.0
    test = (values - lo) / scale
    train = test[:n_train].copy()
    period = find_length(train)
    multiple = cc.WINDOW_MULTIPLE if len(train) // period > cc.WINDOW_MULTIPLE else len(train) // period
    if multiple < 1:
        raise ValueError(f'phần học ({n_train} điểm) ngắn hơn một chu kỳ ({period} điểm)')
    window = (period * multiple) // cc.PATCH * cc.PATCH
    if window < 2 * cc.PATCH:
        raise ValueError('cửa sổ quá ngắn; cần phần học dài hơn')
    period = min(period, window)
    n = len(train)
    stride = 2 if n < 1000 else 6 if n < 10000 else 10 if n < 60000 else cc.PATCH if n < 100000 else period
    if max_windows and (n - window) // stride + 1 > max_windows:
        # fast CPU mode: same model, loss and scoring, fewer training windows per epoch (wider training stride)
        stride = int(np.ceil((n - window) / (max_windows - 1)))
    label = np.asarray(labels if labels is not None else np.zeros(len(values)), dtype=float)
    return dict(group=None, train=train, test=test, label=label, period=period, window=window, train_stride=stride)


def next_group():
    used = [int(m.group(1)) for p in DATA.glob('series_*.json') if (m := re.match(r'series_(\d+)\.json', p.name))]
    return max([9000] + [g for g in used if g >= 9001]) + 1


class Progress(io.TextIOBase):
    """Captures the 'epoch e/E: loss ...' lines printed by coad_core.train_model."""
    def __init__(self, job):
        self.job = job

    def write(self, s):
        m = re.search(r'epoch (\d+)/(\d+): loss ([\d.]+)', s)
        if m:
            j = JOBS[self.job]
            j.update(epoch=int(m.group(1)), epochs=int(m.group(2)), loss=float(m.group(3)))
            done = time.time() - j['t_train']
            j['eta'] = round(done / j['epoch'] * (j['epochs'] - j['epoch']), 1)
        return len(s)


def run_job(job, req):
    j = JOBS[job]
    try:
        values, labels = req['values'], req.get('labels')
        fast = bool(req.get('fast'))
        series = make_series(values, labels, int(req['n_train']), max_windows=128 if fast else None)
        epochs = int(req.get('epochs', 300))
        group = next_group()
        series['group'] = group
        j.update(status='training', group=group, period=series['period'], window=series['window'], epochs=epochs, epoch=0)
        t0 = j['t_train'] = time.time()
        with contextlib.redirect_stdout(Progress(job)):
            model, history = cc.train_model(series, faithful=True, seed=46, epochs=epochs, log_every=1)
        train_seconds = time.time() - t0
        j['status'] = 'scoring'
        t1 = time.time()
        test, train, scales = pc.train_scores(model, series)
        infer_seconds = time.time() - t1
        diag, starts, _ = pc.patch_diagnostics(model, series['test'], series['window'], series['period'])
        arrays, fields = [], {}

        def add(name, v):
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            fields[name] = {'offset': int(sum(a.size for a in arrays)), 'length': int(v.size)}
            arrays.append(v)
        add('input', series['test'])
        add('label', series['label'])
        for k, v in test.items():
            add(k, v)
        for k, v in diag.items():
            add('patch_' + k, v)
        (DATA / f'series_{group}.bin').write_bytes(np.concatenate(arrays).astype('<f4').tobytes())

        has_label = labels is not None and bool(np.any(series['label'] > 0))
        def metrics(score):
            if not (has_label and EVALUATE):
                return {'VUS_PR': None, 'auc_prc': None}
            return {k: round(v, 6) for k, v in cc.score_metrics(score, series).items()}
        stage_train = {k: pc.threshold_policies(v) for k, v in train.items()}
        period = series['period']
        meta = {
            'group': group, 'file': req.get('name', 'upload.csv'), 'family': 'CSV', 'length': len(series['test']),
            'n_train': len(series['train']), 'period': period, 'window': series['window'], 'patch': cc.PATCH,
            'train_stride': series['train_stride'], 'inference_stride': period, 'has_label': has_label, 'valid_from': period,
            'run': {'checkpoint_id': f'upload-g{group}-s46-e{epochs}-{cc.DEVICE.type}{"-fast" if fast else ""}', 'seed': 46, 'epochs': epochs,
                    'training_mode': 'nhanh: tối đa 128 cửa sổ/epoch (stride train giãn ra)' if fast else 'đầy đủ: stride train như code tác giả',
                    'device': str(cc.DEVICE), 'train_seconds': round(train_seconds, 1), 'source': 'trained by serve_coad.py on the uploaded CSV',
                    'result_mode': 'trained on upload', 'inference_device': str(cc.DEVICE), 'inference_seconds': round(infer_seconds, 2),
                    'inference_scope': 'test + training part, all signals of both methods (no patch diagnostics)',
                    'final_loss': {k: round(float(v), 5) for k, v in history.iloc[-1].items()} if len(history) else None,
                    'code': 'demo/coad_core.py (code mode)', 'created': time.strftime('%Y-%m-%d %H:%M')},
            'calibration': {'source': 'training part (label-free)', 'sd_reconstruction': scales['s_r'], 'sd_classification': scales['s_c']},
            'fields': fields, 'windows': {'starts': starts, 'n_patches': series['window'] // cc.PATCH},
            'stage_thresholds': stage_train,
            'methods': {
                'coad': {'label': 'CoAD gốc', 'score': 'coad_final', 'pre': 'coad_pre', 'thresholds': stage_train['coad_final'], 'metrics': metrics(test['coad_final'])},
                'calibrated': {'label': 'Calibrated', 'score': 'cal_final', 'pre': 'cal_pre', 'thresholds': stage_train['cal_final'], 'metrics': metrics(test['cal_final'])},
            },
        }
        (DATA / f'series_{group}.json').write_text(json.dumps(pc.clean(meta), indent=1, allow_nan=False), encoding='utf-8')
        j.update(status='done', seconds=round(time.time() - t0, 1))
    except Exception as e:  # report the reason to the page instead of failing silently
        j.update(status='error', error=f'{type(e).__name__}: {e}')
        traceback.print_exc()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def end_headers(self):  # pages always revalidate, so a changed index.html (new ?v= asset links) is picked up
        if self.path.split('?')[0].endswith(('/', '.html')):
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/api/health'):
            return self._json({'ok': True, 'device': str(cc.DEVICE), 'evaluator': EVALUATE is not None,
                               'busy': any(j['status'] in ('queued', 'training', 'scoring') for j in JOBS.values())})
        if self.path.startswith('/api/uploads'):
            out = []
            for p in sorted(DATA.glob('series_9*.json')):
                try:
                    m = json.loads(p.read_text(encoding='utf-8'))
                    out.append({'group': m['group'], 'name': m['file'], 'length': m['length'], 'epochs': m['run']['epochs']})
                except Exception:
                    pass
            return self._json(out)
        if self.path.startswith('/api/job/'):
            j = JOBS.get(self.path.rsplit('/', 1)[-1])
            return self._json({k: v for k, v in j.items() if k != 't_train'} if j else {'status': 'unknown'}, 200 if j else 404)
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith('/api/train'):
            return self._json({'error': 'not found'}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            vals = req['values']
            if not 0 < len(vals) <= MAX_POINTS or any(v is None for v in vals):
                raise ValueError(f'cần 1–{MAX_POINTS} giá trị số, không có ô trống')
            if req.get('labels') is not None and len(req['labels']) != len(vals):
                raise ValueError('cột nhãn khác độ dài cột giá trị')
            if not 64 <= int(req['n_train']) < len(vals):
                raise ValueError('phần học phải dài ít nhất 64 điểm và ngắn hơn cả chuỗi')
        except Exception as e:
            return self._json({'error': f'{e}'}, 400)
        with LOCK:
            if any(j['status'] in ('queued', 'training', 'scoring') for j in JOBS.values()):
                return self._json({'error': 'đang có một lần train khác chạy'}, 409)
            job = f'j{int(time.time() * 1000)}'
            JOBS[job] = {'status': 'queued', 'epoch': 0, 'epochs': int(req.get('epochs', 300))}
        threading.Thread(target=run_job, args=(job, req), daemon=True).start()
        return self._json({'job': job})

    def log_message(self, fmt, *args):
        if '/api/' in str(args[0] if args else ''):
            return
        super().log_message(fmt, *args)


def main():
    global EVALUATE
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--repo', help="authors' CoAD repository (enables VUS-PR / AUC-PR with their evaluator)")
    ap.add_argument('--no-browser', action='store_true', help='do not open the browser')
    a = ap.parse_args()
    if a.repo:
        try:
            sys.path[:0] = [a.repo, str(Path(a.repo) / 'evaluation')]
            from evaluation.evaluator import evaluate
            cc.evaluate = EVALUATE = evaluate
        except Exception as e:
            print('evaluator not available, VUS-PR will be N/A:', e)
    url = f'http://localhost:{a.port}'
    server = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    print(f'CoAD demo on {url}  (device {cc.DEVICE}, torch threads {cc.torch.get_num_threads()})  Ctrl+C to stop', flush=True)
    if not a.no_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
