"""Deep-learning baselines of Table 1 (CoAD, KDD 2026), one (model, dataset, group) per call.

Copied into the root of the authors' repository (it imports their loaders, wrappers and evaluator) and run
from there:  python t12_baselines.py --model couta --dataset TSB-AD --group 225 --protocol script

Why this file instead of the authors' deep_baseline_compare.py
  * In the published commit that script cannot run most models: the imports of CutAddPaste, FCVAE, CATCH and
    TFAD are commented out, it calls fit(..., epochs=1) / detect_fit(..., epochs=1) on classes whose methods
    have no `epochs` argument, TFMAE never sets eval_result, TranAD and MAUT (listed in
    runners/run_deep_baseline.sh) have no branch, and args.patch_size does not exist.
  * Every model there is trained for 1 epoch (efficiency-measurement settings).

Two protocols, which differ ONLY in the number of training epochs (data, windows, scoring and evaluation are
the same code as the authors' script):
  script   the settings of the published script: 1 training epoch for every trainable model, TFMAE untrained
           (its fit call is commented out). This is the most likely source of the Table 1 baseline numbers.
  default  each wrapper's own default number of epochs (the original papers' settings shipped in the repo).

Models outside the authors' runner (DCdetector, AnomalyTrans, FITS) use their library defaults under both
protocols. MMA, AnomalyBERT and TriAD have no code in the repository and are not run.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

# --- compatibility with the current Colab stack (the wrappers were written for NumPy 1.x / torch 1.x) ---
np.Inf, np.NaN = np.inf, np.nan          # removed in NumPy 2.0; used by the EarlyStopping classes of the wrappers
import thop                               # noqa: E402
thop.profile = lambda *a, **k: (0, 0)     # FLOP counting only (not used for detection); thop breaks on dict inputs

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, 'evaluation'),
                os.path.join(REPO, 'baseline', 'ts_benchmark'), os.path.join(REPO, 'baseline', 'ts_benchmark', 'baselines')]

from evaluation.evaluator import evaluate               # noqa: E402  (authors' evaluator)
from toolkit.utils import load_dataset, get_dataloader  # noqa: E402
from toolkit.get_anomaly_score import AnomalyScoreCalculator  # noqa: E402

MODELS = ['cutaddpaste', 'couta', 'fcvae', 'catch', 'tfad', 'dada', 'moment', 'TFMAE', 'm2n2',
          'TranAD', 'MAUT', 'DCdetector', 'AnomalyTrans', 'FITS']
TABLE1_NAME = {'cutaddpaste': 'CutAddPaste', 'couta': 'COUTA', 'fcvae': 'FCVAE', 'catch': 'CATCH', 'tfad': 'TFAD',
               'dada': 'DADA', 'moment': 'MOMENT', 'TFMAE': 'TFMAE', 'm2n2': 'M2N2', 'TranAD': 'TranAD',
               'MAUT': 'MAUT', 'DCdetector': 'DCdetector', 'AnomalyTrans': 'AnomalyTrans', 'FITS': 'FITS'}

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=MODELS)
parser.add_argument('--dataset', required=True, choices=['TSB-AD', 'UCR'])
parser.add_argument('--group', required=True, type=int)
parser.add_argument('--protocol', default='script', choices=['script', 'default'])
parser.add_argument('--seed', type=int, default=42)          # default of deep_baseline_compare.py
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--out', default='t12_results')          # <out>/<protocol>/<dataset>/<model>/<seed>/...
parser.add_argument('--smoke', action='store_true', help='CPU self-test: 1 epoch, short series')


class EpochCap:
    """Caps tqdm.trange/range loops of a wrapper module whose epoch count is hard-coded inside fit()."""

    def __init__(self, module, cap):
        self.module, self.cap = module, cap

    def __enter__(self):
        import tqdm as _tqdm
        self.saved = self.module.tqdm
        cap = self.cap

        class _T:   # behaves like the tqdm module for the attributes the wrappers use
            def __getattr__(self, name):
                return getattr(_tqdm, name)

            @staticmethod
            def trange(n, *a, **k):
                return range(min(n, cap))

            tqdm = _tqdm.tqdm
        self.module.tqdm = _T()

    def __exit__(self, *exc):
        self.module.tqdm = self.saved


def train_stride_for(n_train, period):
    """Same rule as deep_baseline_compare.py (args.patch_size there is undefined; the CoAD value 8 is used)."""
    if n_train < 1000:
        return 1
    if n_train < 10000:
        return 2
    if n_train < 60000:
        return 5
    if n_train < 100000:
        return 8
    return period


def fit_window(config, attr, n_train, info):
    """CATCH and TFAD validate on the last 20% of the training part with windows of 192 points. On short training
    parts (e.g. TODS: 500 points) that split holds no window and the wrapper crashes on an empty DataLoader.
    Shrink the window to the validation length (multiple of 16 for CATCH's patches); recorded in run_info."""
    default = getattr(config, attr)
    limit = int(0.2 * n_train) // 16 * 16
    if limit < default:
        setattr(config, attr, max(limit, 16))
        info['window_reduced'] = f'{attr} {default} -> {getattr(config, attr)}'


def recon_score(train, test, recon_train, recon_test, smoother):
    """Reconstruction-error score of the authors' train_other_model.py (TranAD, MAUT)."""
    calc = AnomalyScoreCalculator(mode='error', average_window=smoother)
    return calc.calculate_anomaly_score(raw_train_data=train, raw_test_data=test,
                                        recon_train_data=recon_train, recon_test_data=recon_test).test_score_all


def run(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():   # CPU test of wrappers that hard-code 'cuda'
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.nn.Module.cuda = lambda self, *a, **k: self
        def _cpu(fn):
            def wrapped(self, *a, **k):
                a = tuple('cpu' if isinstance(x, str) and x.startswith('cuda') else x for x in a)
                if isinstance(k.get('device'), str) and k['device'].startswith('cuda'):
                    k['device'] = 'cpu'
                return fn(self, *a, **k)
            return wrapped
        torch.Tensor.to = _cpu(torch.Tensor.to)
        torch.nn.Module.to = _cpu(torch.nn.Module.to)
    use_ucr = args.dataset == 'UCR'
    script = args.protocol == 'script'

    tr, te, lb, per = load_dataset(data_name=args.dataset, group_name=[str(args.group)])
    train, test, label, period = tr[0], te[0], lb[0], int(per[0])
    if args.smoke:
        train, test, label = train[:3000], test[:4000], label[:4000]
        if label.sum() == 0:
            label = label.copy(); label[2000:2050] = 1
    stride = train_stride_for(len(train), period)
    if train.ndim == 1:
        train, test = train[:, None], test[:, None]
    epochs = (lambda default: 1 if (script or args.smoke) else default)
    info = {}

    m = args.model
    if m == 'cutaddpaste':
        import baseline.CutAddPaste as mod
        model = mod.CutAddPaste(device=device).to(device)
        with EpochCap(mod, epochs(300)):                      # fit() hard-codes 300 epochs
            model.fit(train)
        loader, _ = get_dataloader(data=test, batch_size=256, window_length=64, mode='test', test_stride=1)
        score = model.predict(loader)
        label = label[:len(score)]; score = score[:len(label)]
        info['epochs'] = epochs(300)

    elif m == 'couta':
        from baseline.COUTA import COUTA
        model = COUTA(sequence_length=64, stride=stride, num_epochs=epochs(40), device=str(device))
        model.fit(pd.DataFrame(train))
        score = model.predict(pd.DataFrame(test))['score_t']
        info['epochs'] = epochs(40)

    elif m == 'fcvae':
        from baseline.FCVAE import MyVAE
        model = MyVAE().to(device)
        model.hp.max_epoch = epochs(model.hp.max_epoch)       # 30 by default
        model.fit(train, device=device)
        score, label = model.predict(test, label, device=device)
        info['epochs'] = model.hp.max_epoch

    elif m == 'catch':
        from baseline.ts_benchmark.baselines.catch.CATCH import CATCH
        model = CATCH()
        model.config.train_stride = stride
        fit_window(model.config, 'seq_len', len(train), info)
        model.config.num_epochs = epochs(model.config.num_epochs)   # 20 by default
        tr2, te2 = pd.DataFrame(np.concatenate([train, train], 1)), pd.DataFrame(np.concatenate([test, test], 1))
        model.detect_fit(tr2, te2)
        score = np.asarray(model.detect_score(te2)).reshape(-1)
        label = label[:len(score)]
        info['epochs'] = model.config.num_epochs

    elif m == 'tfad':
        from baseline.ts_benchmark.baselines.self_impl.TFAD.TFAD import TFAD
        model = TFAD()
        model.config.train_stride = stride
        fit_window(model.config, 'n_window', len(train), info)
        model.config.num_epochs = epochs(model.config.num_epochs)   # 20 by default
        model.detect_fit(pd.DataFrame(train))
        score = np.asarray(model.detect_score(pd.DataFrame(test))).reshape(-1)
        label = label[:len(score)]
        info['epochs'] = model.config.num_epochs

    elif m == 'dada':                                          # zero-shot, weights shipped in baseline/DADA
        # Same computation as baseline/DADA.py (zero_shot -> model.infer(x, norm) -> anomaly score), but the network
        # is built from config.json + pytorch_model.bin directly: AutoModel(trust_remote_code) of current
        # transformers rejects the class (it declares `configs_class` instead of `config_class`).
        import importlib, types
        from baseline.DADA import parser as dada_parser       # the wrapper's own defaults (batch 512, seq_len 100, norm 0)
        dargs = dada_parser.parse_args([])
        ddir = os.path.join(REPO, 'baseline', 'DADA')
        pkg = types.ModuleType('dada_hf'); pkg.__path__ = [ddir]; sys.modules['dada_hf'] = pkg
        net_mod = importlib.import_module('dada_hf.modeling_DADA')
        cfg = json.load(open(os.path.join(ddir, 'config.json')))
        net = net_mod.Model(**{k: cfg[k] for k in ['seq_len', 'hidden_dim', 'd_model', 'bn_dims', 'k', 'patch_len',
                                                    'mask_mode', 'depth', 'max_iters']})
        state = torch.load(os.path.join(ddir, 'pytorch_model.bin'), map_location='cpu')
        net.load_state_dict({k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')})
        net = net.to(device).eval()
        loader, conv = get_dataloader(data=test, batch_size=dargs.batch_size,
                                      window_length=dargs.seq_len, test_stride=dargs.seq_len)
        parts = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.float().to(device)
                out = net.infer(bx, norm=dargs.norm)
                parts.append(net.cal_anomaly_score(batch_x=bx, batch_out_copies=out, anomaly_criterion=None,
                                                   L=dargs.L).detach().cpu().numpy())
        s = np.concatenate(parts, axis=0)[:, :, None]
        score = conv.windows_to_sequence(s).squeeze()
        info['epochs'] = 0

    elif m == 'moment':                                        # zero-shot, weights in baseline/momentfm
        from baseline.MOMENT import MOMENT
        model = MOMENT(device=device)
        score, _, _ = model.zero_shot(test)
        info['epochs'] = 0

    elif m == 'm2n2':
        from baseline.M2N2 import M2N2
        model = M2N2(device=device, epochs=epochs(10))
        model.fit(train)
        score, _, _ = model.decision_function(test)
        info['epochs'] = epochs(10)

    elif m == 'TFMAE':
        from baseline.MTFAE import MTFA
        model = MTFA(win_size=100, seq_size=10, c_in=1, c_out=1, d_model=128, e_layers=3, fr=0.4, tr=0.25,
                     dev=device)
        n_ep = 0 if (script and not args.smoke) else epochs(100)   # script: fit() commented out
        if n_ep:
            loader, _ = get_dataloader(data=train, batch_size=256, window_length=100, train_stride=stride, mode='train')
            model.fit(loader, torch.optim.Adam(model.parameters(), lr=1e-4), epochs=n_ep)
        loader, conv = get_dataloader(data=test, batch_size=256, window_length=100, test_stride=100, mode='test')
        s, _, _ = model.predict(loader)
        score = conv.windows_to_sequence(s).squeeze()
        info['epochs'] = n_ep

    elif m == 'TranAD':                                        # settings of train_other_model.py
        import inspect
        import baseline.TranAD.tran_ad as ta
        for cls in (ta.TransformerEncoderLayer, ta.TransformerDecoderLayer):
            # torch >= 2.0 passes is_causal / *_is_causal to custom layers written for torch 1.x: drop what they lack
            if not getattr(cls.forward, '_t12', False):
                accepted = set(inspect.signature(cls.forward).parameters)
                def fwd(self, *a, _f=cls.forward, _ok=accepted, **k):
                    return _f(self, *a, **{key: v for key, v in k.items() if key in _ok})
                fwd._t12 = True
                cls.forward = fwd
        TranAD = ta.TranAD
        model = TranAD(feats=1, window_length=10).to(device)
        pad = np.ones([model.n_window - 1, 1])
        tr_p, te_p = np.concatenate([pad, train]), np.concatenate([pad, test])
        tl, _ = get_dataloader(data=tr_p, batch_size=64, window_length=10, train_stride=1, mode='train')
        vl, _ = get_dataloader(data=tr_p, batch_size=64, window_length=10, test_stride=1, mode='test')
        sl, _ = get_dataloader(data=te_p, batch_size=64, window_length=10, test_stride=1, mode='test')
        opt = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9)
        model.fit(train_loader=tl, epochs=epochs(30), optimizer=opt, scheduler=sch, device=device)
        rec_tr, rec_te = model.predict(vl, device=device), model.predict(sl, device=device)
        score = recon_score(train, test, rec_tr, rec_te, period if use_ucr else None)
        label = label[:len(score)]
        info['epochs'] = epochs(30)

    elif m == 'MAUT':                                          # settings of train_other_model.py
        import importlib, types
        pkg = types.ModuleType('other_models'); sub = types.ModuleType('other_models.MemAugUTransAD_main')
        sys.modules['other_models'], sys.modules['other_models.MemAugUTransAD_main'] = pkg, sub
        for name in ['modules', 'transformer', 'memory']:      # the wrapper imports them under another path
            sys.modules[f'other_models.MemAugUTransAD_main.{name}'] = importlib.import_module(f'baseline.MAUT.{name}')
        from baseline.MAUT.model_pyramid_trans_mem import PYRAMID_TRANS_MEM
        model = PYRAMID_TRANS_MEM(n_features=1, window_size=100, out_dim=1).to(device)
        tl, _ = get_dataloader(data=train, batch_size=64, window_length=100, train_stride=None, mode='train')
        vl, vc = get_dataloader(data=train, batch_size=64, window_length=100, test_stride=100, mode='test')
        sl, sc = get_dataloader(data=test, batch_size=64, window_length=100, test_stride=100, mode='test')
        model.fit(train_loader=tl, epochs=epochs(100), optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                  device=device)
        rec_tr = vc.windows_to_sequence(model.predict(vl, device=device))
        rec_te = sc.windows_to_sequence(model.predict(sl, device=device))
        score = recon_score(train, test, rec_tr, rec_te, period if use_ucr else None)
        info['epochs'] = epochs(100)

    elif m == 'DCdetector':                                    # TFB implementation shipped in the repo
        from baseline.ts_benchmark.baselines.self_impl.DCdetector.DCdetector import DCdetector
        model = DCdetector()
        if args.smoke:
            model.config.num_epochs = 1
        model.detect_fit(pd.DataFrame(train), pd.DataFrame(test))
        score = np.asarray(model.detect_score(pd.DataFrame(test))[0]).reshape(-1)
        label = label[:len(score)]
        info['epochs'] = model.config.num_epochs

    elif m in ['AnomalyTrans', 'FITS']:                        # TSB-AD implementations, univariate HPs of HP_list
        from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
        hp = dict(Optimal_Uni_algo_HP_dict['AnomalyTransformer' if m == 'AnomalyTrans' else 'FITS'])
        if m == 'AnomalyTrans':
            from TSB_AD.models.AnomalyTransformer import AnomalyTransformer as Cls
        else:
            from TSB_AD.models.FITS import FITS as Cls
        model = Cls(input_c=1, batch_size=128, **hp)
        attr = 'num_epochs' if m == 'AnomalyTrans' else 'epochs'   # defaults: 10 (AnomalyTransformer), 50 (FITS)
        if args.smoke:
            setattr(model, attr, 1)
        model.fit(train)
        score = np.asarray(model.decision_function(test)).reshape(-1)
        info.update(hp, epochs=getattr(model, attr))

    score = np.nan_to_num(np.asarray(score, dtype=float).reshape(-1))
    label = np.asarray(label).reshape(-1)[:len(score)].astype(int)
    score = score[:len(label)]
    result = evaluate(ground_truth=label, anomaly_scores=score, subsequence=period, use_ucr=use_ucr, mode='test')
    return score, result, info


if __name__ == '__main__':
    args = parser.parse_args()
    t0 = time.time()
    score, result, info = run(args)
    out = os.path.join(args.out, args.protocol, args.dataset, args.model, str(args.seed))
    os.makedirs(os.path.join(out, 'data'), exist_ok=True)
    os.makedirs(os.path.join(out, 'anomaly_score'), exist_ok=True)
    g3 = str(args.group).zfill(3)
    np.save(os.path.join(out, 'anomaly_score', f'group_{g3}.npy'), score)
    payload = asdict(result)
    payload['run_info'] = dict(info, model=args.model, dataset=args.dataset, group=args.group,
                               protocol=args.protocol, seed=args.seed, seconds=round(time.time() - t0, 1),
                               smoke=args.smoke, n_scored=int(len(score)))
    with open(os.path.join(out, 'data', f'group_{g3}.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    pw, vus = payload['point_wise'], payload['vus']
    print(f"RESULT {args.model} {args.dataset} {args.group} {args.protocol}: F1={pw['f1_score']:.4f} "
          f"AUC-PR={pw['auc_prc']:.4f} R-AUC-PR={vus['R_AUC_PR']:.4f} VUS-PR={vus['VUS_PR']:.4f}")
