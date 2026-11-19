"""Precomputes the data of the CoAD web demo (label-free scoring, labels only for evaluation and display).

For every demo series: trains CoAD (code mode, seed 46) with coad_core.py, then writes
  data/series_<group>.json   metadata, metrics, threshold policies, stage definitions, patch-window index
  data/series_<group>.bin    float32 arrays (timestamp-level signals and patch-window diagnostics)
and data/benchmark.json (per-series results of the experiments: CoAD code mode vs calibrated, Kmeans).

usage (from the repository root of the authors' code, which provides the loader and the evaluator):
  python precompute.py --repo <CoAD repo> --groups 234 237 --epochs 300
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coad_core as cc  # noqa: E402

FAMILY_SPLIT_SEED = 2026


def clean(o):
    """JSON-safe copy: NaN/inf -> null (JSON has no NaN), numpy scalars -> Python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, 'item'):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def train_scores(model, series):
    """Timestamp-level signals of one trained model on the test series and on the training part."""
    period, window = series['period'], series['window']
    parts = {}
    for name, values in [('test', series['test']), ('train', series['train'])]:
        parts[name] = predict_components(model, values, window, period)
    t, tr = parts['test'], parts['train']
    err_t = np.abs(series['test'] - t['reconstruction'])
    err_tr = np.abs(series['train'] - tr['reconstruction'])

    def processed(raw_test, raw_train, smooth=True):
        """Authors' code-mode processing of one source (identical to cell 82 `processed`)."""
        mean = raw_train.mean()
        out = []
        for raw in (raw_test, raw_train):
            value = np.abs(raw - mean)
            value = cc.moving_average(value, period) if smooth else value.copy()
            value[:period] = 0
            out.append(value)
        return out

    r, r_tr = processed(err_t, err_tr)
    c, c_tr = processed(t['combined'], tr['combined'])
    r0, r0_tr = processed(err_t, err_tr, smooth=False)
    c0, c0_tr = processed(t['combined'], tr['combined'], smooth=False)
    s_r, s_c = np.std(r_tr) + 1e-12, np.std(c_tr) + 1e-12
    test = dict(oe=t['oe'], residual=t['residual'], combined=t['combined'], reconstruction=t['reconstruction'],
                rec_error=err_t, c_raw=c0, r_raw=r0, c_smooth=c, r_smooth=r,
                coad_pre=(r0 + c0) / 2, coad_final=(r + c) / 2,
                cal_c=c / s_c, cal_r=r / s_r, cal_pre=(r0 / s_r + c0 / s_c) / 2, cal_final=(r / s_r + c / s_c) / 2)
    train = dict(oe=tr['oe'], residual=tr['residual'], combined=tr['combined'], rec_error=err_tr, c_raw=c0_tr,
                 r_raw=r0_tr, c_smooth=c_tr, r_smooth=r_tr, coad_pre=(r0_tr + c0_tr) / 2, coad_final=(r_tr + c_tr) / 2,
                 cal_c=c_tr / s_c, cal_r=r_tr / s_r, cal_pre=(r0_tr / s_r + c0_tr / s_c) / 2,
                 cal_final=(r_tr / s_r + c_tr / s_c) / 2)
    return test, train, dict(s_r=float(s_r), s_c=float(s_c))


@torch.no_grad()
def predict_components(model, values, window, stride):
    """Same as notebook cell 82: per-point A (OE), A^r, A_c and the final reconstruction."""
    windows, pad = cc.make_windows(values, window, stride)
    model.eval()
    parts = {k: [] for k in ['oe', 'residual', 'combined', 'reconstruction']}
    for batch in torch.from_numpy(windows).split(2 * cc.BATCH_SIZE):
        x = batch.to(cc.DEVICE)
        oe = torch.sigmoid(model.classifier(x))
        first = model.mae(x, oe)
        residual = torch.sigmoid(model.classifier.residual_logit(x, first))
        combined = (oe + residual) / 2
        final = model.mae(x, combined) if model.faithful else first
        for key, value in [('oe', oe), ('residual', residual), ('combined', combined)]:
            parts[key].append(value.repeat_interleave(cc.PATCH, dim=1).squeeze(-1).cpu())
        parts['reconstruction'].append(final.cpu())
    return {k: cc.windows_to_series(torch.cat(v).numpy(), len(values), stride, pad) for k, v in parts.items()}


@torch.no_grad()
def patch_diagnostics(model, values, window, stride):
    """Per test window and patch: time / frequency probabilities, max-fused A (mask of the first pass),
    residual time / frequency probabilities, fused A^r, A_c (mask of the second pass, code mode)."""
    windows, pad = cc.make_windows(values, window, stride)
    model.eval()
    clf = model.classifier
    out = {k: [] for k in ['p_time', 'p_freq', 'p_mask', 'r_time', 'r_freq', 'r_fused', 'p_comb']}
    for batch in torch.from_numpy(windows).split(2 * cc.BATCH_SIZE):
        x = batch.to(cc.DEVICE)
        h_f, h_t = clf.features(x)
        p_f, p_t = torch.sigmoid(clf.classifier_freq(h_f)), torch.sigmoid(clf.classifier_time(h_t))
        p_mask = torch.maximum(p_f, p_t)
        first = model.mae(x, p_mask)
        hf, ht = clf.features(torch.cat([x, first]))
        (fi, fr), (ti, tr) = hf.chunk(2), ht.chunk(2)
        r_f, r_t = torch.sigmoid(clf.classifier_residual(fi - fr)), torch.sigmoid(clf.classifier_residual(ti - tr))
        r_fused = torch.maximum(r_f, r_t)
        for k, v in [('p_time', p_t), ('p_freq', p_f), ('p_mask', p_mask), ('r_time', r_t), ('r_freq', r_f),
                     ('r_fused', r_fused), ('p_comb', (p_mask + r_fused) / 2)]:
            out[k].append(v.squeeze(-1).cpu())
    starts = (np.arange(len(windows)) * stride - pad).tolist()       # window start index in the series (may be < 0)
    return {k: torch.cat(v).numpy() for k, v in out.items()}, starts, pad


def threshold_policies(train_signal):
    """Label-free policies from the training part (normal reference): mean + 3 SD and maximum."""
    valid = train_signal[np.isfinite(train_signal)]
    return {'train_mu3sd': float(valid.mean() + 3 * valid.std()), 'train_max': float(valid.max())}


def kmeans_score(repo, group):
    """KMeansAD of the authors' baseline_compare.py (score computed on the test series itself)."""
    path = Path(repo) / 'baseline_result' / 'TSB-AD' / 'Kmeans' / 'anomaly_score' / f'group_{str(group).zfill(3)}.npy'
    if not path.exists():
        import subprocess
        subprocess.run([sys.executable, 'baseline_compare.py', '--model_name', 'Kmeans', '--dataset_name', 'TSB-AD',
                        '--group_name', str(group)], cwd=repo, check=True, capture_output=True)
    return np.load(path).astype(float)


def metrics(score, series):
    return {k: round(v, 6) for k, v in cc.score_metrics(score, series).items()}


def find_checkpoint(ckpt_root, group):
    """Folder with model.pt / config.json / scores.npy of the benchmark run (runs/reimpl/<run_id> on Drive)."""
    if not ckpt_root:
        return None
    for d in sorted(Path(ckpt_root).iterdir()):
        cfg = d / 'config.json'
        if cfg.exists() and json.loads(cfg.read_text())['group'] == group and (d / 'model.pt').exists():
            return d
    return None


def build_series(repo, group, epochs, out_dir, raw_files, ckpt_root=None):
    t0 = time.time()
    series = cc.load_series(group)
    print(f'group {group}: train {len(series["train"])}, test {len(series["test"])}, period {series["period"]}, '
          f'window {series["window"]}', flush=True)
    ckpt = find_checkpoint(ckpt_root, group)
    if ckpt is not None:                     # the GPU model of the benchmark (no retraining)
        cfg = json.loads((ckpt / 'config.json').read_text())
        assert cfg['mode'] == 'code', 'demo uses code-mode checkpoints'
        model = cc.CoADReimpl(faithful=True).to(cc.DEVICE)
        state = torch.load(ckpt / 'model.pt', map_location=cc.DEVICE)
        model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
        history = pd.read_csv(ckpt / 'loss.csv') if (ckpt / 'loss.csv').exists() else None
        run_info = {'checkpoint_id': cfg['run_id'], 'seed': cfg['seed'], 'epochs': cfg.get('epochs'),
                    'device': cfg.get('device', 'cuda') + ' (training)', 'train_seconds': cfg.get('seconds'),
                    'source': 'runs/reimpl/' + cfg['run_id'] + ' (benchmark model)', 'commit': cfg.get('commit')}
    else:
        model, history = cc.train_model(series, faithful=True, seed=46, epochs=epochs, log_every=50)
        run_info = {'checkpoint_id': f'coad-code-s46-g{group}-{cc.DEVICE.type}-retrain', 'seed': 46, 'epochs': epochs,
                    'device': str(cc.DEVICE), 'train_seconds': round(time.time() - t0, 1), 'source': 'retrained by precompute.py'}
    train_seconds = run_info['train_seconds']
    t1 = time.time()
    test, train, scales = train_scores(model, series)
    infer_seconds = time.time() - t1
    check = np.abs(test['coad_final'] - cc.anomaly_score(model, series)).max()
    assert check < 1e-6, f'coad_final differs from anomaly_score by {check}'
    saved_check = None
    if ckpt is not None and (ckpt / 'scores.npy').exists():
        saved = np.load(ckpt / 'scores.npy')
        saved_check = float(np.abs(saved - test['coad_final']).max()) if len(saved) == len(test['coad_final']) else None
        print(f'  max |score - saved benchmark score| = {saved_check}', flush=True)
    km = kmeans_score(repo, group)
    if len(km) != len(series['test']):
        km = np.resize(km, len(series['test']))
    diag, starts, _ = patch_diagnostics(model, series['test'], series['window'], series['period'])

    arrays, fields = [], {}
    def add(name, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        fields[name] = {'offset': int(sum(a.size for a in arrays)), 'length': int(values.size)}
        arrays.append(values)
    add('input', series['test'])
    add('label', np.asarray(series['label'], dtype=np.float32))
    for k, v in test.items():
        add(k, v)
    add('kmeans_final', km)
    for k, v in diag.items():
        add('patch_' + k, v)
    (out_dir / f'series_{group}.bin').write_bytes(np.concatenate(arrays).astype('<f4').tobytes())

    period, n_train = series['period'], len(series['train'])
    name = raw_files[group - 1]
    stage_train = {k: threshold_policies(v) for k, v in train.items() if k in
                   ['oe', 'residual', 'combined', 'rec_error', 'c_raw', 'r_raw', 'c_smooth', 'r_smooth',
                    'coad_pre', 'coad_final', 'cal_c', 'cal_r', 'cal_pre', 'cal_final']}
    km_valid = km[period:]
    meta = {
        'group': group, 'file': name, 'family': name.split('_')[1], 'length': len(series['test']),
        'n_train': n_train, 'period': period, 'window': series['window'], 'patch': cc.PATCH,
        'train_stride': series['train_stride'], 'inference_stride': period, 'has_label': True,
        'valid_from': period,
        'run': dict(run_info, result_mode='precomputed', inference_device=str(cc.DEVICE),
                    inference_seconds=round(infer_seconds, 2),
                    inference_scope='test + training part, all signals and both methods (no patch diagnostics)',
                    max_abs_diff_to_saved_benchmark_score=saved_check,
                    final_loss=({k: round(float(v), 5) for k, v in history.iloc[-1].items() if k in ('loss', 'classify', 'recon')}
                                if history is not None and len(history) else None),
                    code='demo/coad_core.py (CoAD_reimplementation.ipynb, code mode)',
                    created=time.strftime('%Y-%m-%d %H:%M')),
        'calibration': {'source': 'training part (label-free)', 'sd_reconstruction': scales['s_r'],
                        'sd_classification': scales['s_c']},
        'fields': fields,
        'windows': {'starts': starts, 'n_patches': series['window'] // cc.PATCH},
        'stage_thresholds': stage_train,
        'methods': {
            'coad': {'label': 'CoAD gốc', 'score': 'coad_final', 'pre': 'coad_pre',
                     'thresholds': stage_train['coad_final'], 'metrics': metrics(test['coad_final'], series)},
            'calibrated': {'label': 'Calibrated', 'score': 'cal_final', 'pre': 'cal_pre',
                           'thresholds': stage_train['cal_final'], 'metrics': metrics(test['cal_final'], series)},
            'kmeans': {'label': 'KMeansAD', 'score': 'kmeans_final', 'pre': None,
                       'thresholds': {'self_mu3sd': float(km_valid.mean() + 3 * km_valid.std()),
                                      'self_max': None},
                       'metrics': metrics(km, series)},
        },
    }
    (out_dir / f'series_{group}.json').write_text(json.dumps(clean(meta), indent=1, allow_nan=False), encoding='utf-8')
    print(f'group {group} done: {train_seconds:.0f}s train | VUS-PR CoAD {meta["methods"]["coad"]["metrics"]["VUS_PR"]:.3f}'
          f' calibrated {meta["methods"]["calibrated"]["metrics"]["VUS_PR"]:.3f}', flush=True)


def build_benchmark(results_dir, m11_dir, raw_files, out_dir, demo_groups):
    """Per-series experiment results: CoAD (code mode) vs calibrated (M11 B1), seeds 46-48; design/confirmation."""
    r = pd.concat([pd.read_csv(f) for f in glob.glob(str(Path(results_dir) / 'reimpl_results_*.csv'))])
    r = r[(r['mode'] == 'code') & (r.status == 'success')].drop_duplicates(['group', 'seed'], keep='last')
    m = pd.concat([pd.read_csv(f) for f in glob.glob(str(Path(m11_dir) / 'm11_*_gpu-*.csv')) if 'spike' not in f])
    m = m[m.variant == 'B1 calibrated'].drop_duplicates(['group', 'seed'], keep='last')
    locked = json.loads((Path(m11_dir) / 'm11_locked.json').read_text())
    design = set(locked['development_series'])
    keys = ['f1_score', 'auc_prc', 'R_AUC_PR', 'VUS_PR']
    rows = []
    for g, rg in r.groupby('group'):
        mg = m[m.group == g]
        if mg.empty:
            continue
        both = rg.set_index('seed')[keys].join(mg.set_index('seed')[keys], lsuffix='_coad', rsuffix='_cal', how='inner')
        if both.empty:
            continue
        name = raw_files[int(g) - 1]
        row = {'group': int(g), 'file': name, 'family': name.split('_')[1],
               'split': 'design' if g in design else 'confirmation', 'seeds': sorted(both.index.tolist()),
               'demo': int(g) in demo_groups}
        for k in keys:
            row[f'{k}_coad'] = float(both[f'{k}_coad'].mean())
            row[f'{k}_cal'] = float(both[f'{k}_cal'].mean())
            row[f'{k}_coad_sd'] = float(both[f'{k}_coad'].std(ddof=1)) if len(both) > 1 else None
            row[f'{k}_cal_sd'] = float(both[f'{k}_cal'].std(ddof=1)) if len(both) > 1 else None
        rows.append(row)
    k = pd.concat([pd.read_csv(f) for f in glob.glob(str(Path(results_dir) / 'tsbad64_results*.csv'))])
    k = k[(k.method == 'Kmeans') & (k.status == 'success')].drop_duplicates('group', keep='last').set_index('group')
    for row in rows:
        row['VUS_PR_kmeans'] = float(k.VUS_PR.get(row['group'], np.nan))
    bench = {'source': {'coad': 'results/reimpl_results_*.csv (code mode, seeds 46-48, GPU)',
                        'calibrated': 'm11 B1 calibrated (same saved models, rescored)',
                        'kmeans': 'results/tsbad64_results*.csv (authors baseline_compare.py, deterministic)',
                        'split': 'design = 30 series fixed with default_rng(2026); others = confirmation'},
             'aggregation': 'per series: mean over the seeds listed; families: mean over series',
             'rows': rows}
    (out_dir / 'benchmark.json').write_text(json.dumps(clean(bench), indent=1, allow_nan=False), encoding='utf-8')
    print('benchmark rows:', len(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--groups', type=int, nargs='*', default=[])
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--results', help='folder with reimpl_results_*.csv and tsbad64_results*.csv')
    ap.add_argument('--m11', help='folder with m11_*.csv and m11_locked.json')
    ap.add_argument('--demo_groups', type=int, nargs='*', default=[])
    ap.add_argument('--ckpt', help='folder of runs/reimpl checkpoints (model.pt, config.json, scores.npy); retrain if absent')
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out_dir = HERE / 'data'
    out_dir.mkdir(exist_ok=True)
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / 'evaluation')]
    from toolkit.utils import load_dataset
    from evaluation.evaluator import evaluate
    cc.load_dataset, cc.evaluate = load_dataset, evaluate
    raw_files = sorted(os.listdir(repo / 'dataset' / 'TSB-AD' / 'raw'))
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    for g in args.groups:
        build_series(repo, g, args.epochs, out_dir, raw_files, args.ckpt)
    if args.results and args.m11:
        build_benchmark(args.results, args.m11, raw_files, out_dir, set(args.demo_groups or args.groups))
