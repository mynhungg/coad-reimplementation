"""Builds CoAD_table12.ipynb (Colab or Kaggle). The three t12_*.py scripts are embedded with %%writefile so the
notebook is self-contained.  python build_nb.py"""
import json
from pathlib import Path

HERE = Path(__file__).parent
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': text.strip('\n').splitlines(keepends=True)})


md(r'''
# CoAD — reproduce every number of Table 1 and Table 2 (KDD21 + TSB-AD)

**Already done in earlier notebooks (read back, not re-run):** TSB-AD 64 series — CoAD (`tsbad64_results*.csv`),
the 7 data-mining baselines (`tsbad64_results*.csv`) and the 9 ablations of the published code (`ablation_results*.csv`).

**This notebook runs what is missing** (task names used in step 4):

| Task | What | Runs | Device |
|---|---|---|---|
| `coad_kdd21` | CoAD on the 250 KDD21 series (flags of `runners/run_all_CoAD.sh`) | 250 | GPU |
| `dm_kdd21` | 7 data-mining baselines on KDD21 (`runners/run_dm_baseline.sh`) | 1 750 | CPU (MP: GPU) |
| `deep` | 14 deep baselines on TSB-AD (64) and KDD21 (250), `t12_baselines.py` | 4 396 per protocol | GPU |
| `abl_kdd21` | 9 ablations of the published code on KDD21 (`runners/run_ablation_CoAD.sh`) | 2 250 | GPU |
| `abl_extra` | 5 rows of Table 2 the code lacks (OE alone ×2, MAE alone ×2, Guide w/o Score; Appendix G), TSB-AD + KDD21 | 1 570 | GPU |

**Deep baselines, two protocols** (only the number of training epochs differs, see `t12_baselines.py`):
`script` = settings of the authors' `deep_baseline_compare.py` (1 epoch; TFMAE untrained) — the likely source of
Table 1; `default` = each wrapper's own epochs. Run `script` first.
Not runnable: **MMA, AnomalyBERT, TriAD** (no code in the repository; the paper only says it used each method's official repository with early stopping and 5 seeds).

**KDD21 data:** the authors' preprocessed files are not published. Step 2 rebuilds them from the UCR Anomaly Archive
2021 and checks every series against the authors' `dataset/UCR/all_length.csv` (min-max scaling on the training part
is our assumption).

**How to run:** run steps 1–3 in every new runtime, choose the work in step 4, run step 5 (re-run to continue after a
disconnect; finished runs are skipped). Several copies can run at the same time if each has its own `SESSION_TAG` and a
different `SHARD`. Step 6 compares with the paper at any time; step 7 packs results on Kaggle.
''')

md('## 1. Session setup (Colab: Google Drive; Kaggle: attached dataset)')
code(r'''
import json, os, re, shutil, subprocess, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

PINNED_COMMIT = '3286813619091eeebec04b3c9d73c837bdc4469e'   # env/commit.txt of the project (M0)
TSBAD_URL = 'https://www.thedatum.org/datasets/TSB-AD-U.zip'

# Kaggle sets KAGGLE_KERNEL_RUN_TYPE; recent Colab images also contain /kaggle/input, so the folder is not a test
ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ and 'COLAB_RELEASE_TAG' not in os.environ
if ON_KAGGLE:
    # Optional: the Drive folder CoAD_project attached as a private Kaggle dataset (read-only) -> copy the small
    # folders and reuse its data and earlier results. Without it the notebook downloads the data and starts empty.
    source = next((p.parents[1] for p in Path('/kaggle/input').rglob('env/commit.txt')), None)
    STORE = Path('/kaggle/working/CoAD_project')
    if source is None:
        print('No CoAD_project dataset attached: pinned commit, TSB-AD-U downloaded; earlier results not available.')
        for sub in ['env', 'data_source', 'results']:
            (STORE / sub).mkdir(parents=True, exist_ok=True)
        (STORE / 'env' / 'commit.txt').write_text(PINNED_COMMIT, encoding='utf-8')
    elif not STORE.exists():
        for sub in ['env', 'eda', 'results']:
            if (source / sub).exists():
                shutil.copytree(source / sub, STORE / sub)
        (STORE / 'data_source').mkdir(parents=True)
        for f in (source / 'data_source').iterdir():
            (STORE / 'data_source' / f.name).symlink_to(f)
        latest = {}   # result files of earlier Kaggle versions attached as inputs: keep the largest copy
        for csv in Path('/kaggle/input').rglob('results/*.csv'):
            if csv.name not in latest or csv.stat().st_size > latest[csv.name].stat().st_size:
                latest[csv.name] = csv
        (STORE / 'results').mkdir(exist_ok=True)
        for name, csv in latest.items():
            shutil.copy2(csv, STORE / 'results' / name)
    REPO = Path('/kaggle/working/CoAD')
else:
    from google.colab import drive
    drive.mount('/content/drive')
    STORE = Path('/content/drive/MyDrive/CoAD_project')
    REPO = Path('/content/CoAD')

commit = (STORE / 'env' / 'commit.txt').read_text(encoding='utf-8').strip()
if not (STORE / 'data_source' / 'TSB-AD-U.zip').exists():
    subprocess.run(['wget', '-q', '-O', str(STORE / 'data_source' / 'TSB-AD-U.zip'), TSBAD_URL], check=True)
if not REPO.exists():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/richard-tang199/CoAD.git', str(REPO)], check=True)
subprocess.run(['git', 'checkout', '-q', commit], cwd=REPO, check=True)
RESULTS_DIR = STORE / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print('Store:', STORE, '| commit:', subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip())
''')

md('''## 2. Dependencies, data, patches

Installs `requirements.txt` (as in the earlier notebooks) plus what the deep baselines need: `momentfm`
(MOMENT wrapper), `TSB-AD` (FITS and AnomalyTransformer implementations), the MOMENT-1-base weights (the repository
ships only the config). Then: TSB-AD-U from the archive on Drive, KDD21 rebuilt from the UCR archive (downloaded once to
`data_source/`), the compatibility patches of the earlier notebooks, and the three `t12_*.py` files.''')

for name in ['t12_baselines.py', 't12_prepare_kdd21.py', 't12_ablation_patch.py']:
    code(f'%%writefile /tmp/{name}\n' + (HERE / name).read_text(encoding='utf-8'))

code(r'''
CHECK_IMPORTS = ('import arch, stumpy, tslearn, sktime, torch_geometric, statsmodels, numba, momentfm, TSB_AD, torchinfo, '
                 'thop, torchvision, pytorch_lightning, transformers')
def deps_ok():
    return subprocess.run([sys.executable, '-c', CHECK_IMPORTS], capture_output=True).returncode == 0

if not deps_ok():
    log = []
    for cmd in [['-r', str(REPO / 'requirements.txt')],
                ['--no-deps', 'momentfm', 'TSB-AD'],          # their pinned deps would downgrade torch/transformers
                ['torchinfo', 'thop', 'h5py', 'huggingface_hub', 'safetensors', 'pytorch_lightning']]:  # TFAD: lightning
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *cmd], capture_output=True, text=True)
        log.append(' '.join(cmd) + '\n' + r.stdout + r.stderr)
        assert r.returncode == 0, r.stderr[-2000:]
    (STORE / 'env' / 'install_t12.log').write_text('\n'.join(log), encoding='utf-8')
tsl = subprocess.check_output([sys.executable, '-c', 'import tslearn; print(tslearn.__version__)'], text=True).strip()
if tsl.startswith('0.6.'):   # same as the earlier notebooks: current tslearn, SAND patch below
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', '--no-deps', 'tslearn'], check=True)
assert deps_ok(), 'missing packages: inspect env/install_t12.log'

# MOMENT-1-base weights next to the repository's config.json
moment_dir = REPO / 'baseline' / 'momentfm'
if not (moment_dir / 'model.safetensors').exists():
    from huggingface_hub import hf_hub_download
    hf_hub_download('AutonLab/MOMENT-1-base', 'model.safetensors', local_dir=str(moment_dir))

# TSB-AD-U (870 files)
RAW = REPO / 'dataset' / 'TSB-AD' / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(STORE / 'data_source' / 'TSB-AD-U.zip') as z:
    for m in z.namelist():
        if m.startswith('TSB-AD-U/') and m.endswith('.csv') and '..' not in m and not (RAW / Path(m).name).exists():
            with z.open(m) as src, (RAW / Path(m).name).open('wb') as dst:
                shutil.copyfileobj(src, dst)
assert len(list(RAW.iterdir())) == 870
RAW_FILES = sorted(p.name for p in RAW.iterdir())

# KDD21 (250 series) from the UCR Anomaly Archive 2021
ucr_zip = STORE / 'data_source' / 'UCR_TimeSeriesAnomalyDatasets2021.zip'
if not ucr_zip.exists():
    # local disk of this runtime (not Drive): the copies may download at the same time
    target = Path('/kaggle/working' if ON_KAGGLE else '/content') / ucr_zip.name
    subprocess.run(['wget', '-q', '-O', str(target),
                    'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip'], check=True)
    ucr_zip = target
if len(list((REPO / 'dataset' / 'UCR' / 'processed' / 'train').glob('*.npy'))) != 250:
    print(subprocess.run([sys.executable, '/tmp/t12_prepare_kdd21.py', str(ucr_zip), str(REPO)],
                         capture_output=True, text=True, check=True).stdout)

# Compatibility patches of the earlier notebooks (imports of baseline_compare.py, NumPy 2 aliases, SAND last batch)
bc = REPO / 'baseline_compare.py'
head = 'import os\nimport numpy as np\nimport torch\n'
if not bc.read_text(encoding='utf-8').startswith(head):
    bc.write_text(head + bc.read_text(encoding='utf-8'), encoding='utf-8')
for rel in ['baseline/sand.py', 'baseline/feature.py']:
    p = REPO / rel
    p.write_text(p.read_text(encoding='utf-8').replace('np.Inf', 'np.inf').replace('np.NaN', 'np.nan'), encoding='utf-8')
sand = REPO / 'baseline' / 'sand.py'
sand.write_text(sand.read_text(encoding='utf-8').replace(
    'ks = KShape(n_clusters=self.k, verbose=False)',
    'ks = KShape(n_clusters=min(self.k, len(all_subsequences)), verbose=False)'), encoding='utf-8')
# NumPy 2 rejects np.sum(generator) (Series2Graph scoring); the builtin sum gives the same value
s2g = REPO / 'baseline' / 'series2graph.py'
s2g.write_text(s2g.read_text(encoding='utf-8').replace('np.sum(dict_edge[str(edge)]', 'sum(dict_edge[str(edge)]'),
               encoding='utf-8')
# MP calls stumpy.gpu_stump; without a usable CUDA driver (CPU runtime) numba fails. stumpy.stump computes the same
# matrix profile on the CPU, so fall back to it instead of failing.
MP_OLD = ('        all_gpu_devices = [device.id for device in cuda.list_devices()]\n'
          '        test_data = np.squeeze(test_data)\n'
          '        test_anomaly_score = stumpy.gpu_stump(test_data, m=subsequence_length, device_id=all_gpu_devices)[:, 0]\n')
MP_NEW = ('        test_data = np.squeeze(test_data)\n'
          '        try:\n'
          '            all_gpu_devices = [device.id for device in cuda.list_devices()]\n'
          '            test_anomaly_score = stumpy.gpu_stump(test_data, m=subsequence_length, device_id=all_gpu_devices)[:, 0]\n'
          '        except Exception as e:  # no usable CUDA driver: same matrix profile on the CPU\n'
          '            print("MP: CUDA unavailable (%s), using stumpy.stump on CPU" % type(e).__name__)\n'
          '            test_anomaly_score = stumpy.stump(test_data, m=subsequence_length)[:, 0]\n'
          '        test_anomaly_score = np.asarray(test_anomaly_score, dtype=np.float64)\n')
src = bc.read_text(encoding='utf-8')
if MP_OLD in src:
    bc.write_text(src.replace(MP_OLD, MP_NEW), encoding='utf-8')

shutil.copy('/tmp/t12_baselines.py', REPO / 't12_baselines.py')
subprocess.run([sys.executable, '/tmp/t12_ablation_patch.py', str(REPO)], check=True)
(STORE / 'env' / 't12_repo.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=REPO, text=True), encoding='utf-8')
gpu = subprocess.run([sys.executable, '-c', 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'],
                     capture_output=True, text=True).stdout.strip()
print('Ready | GPU:', gpu or 'NONE')
if not gpu:
    print('WARNING: no GPU in this runtime (Runtime > Change runtime type > T4 GPU). dm_kdd21 runs on CPU (MP falls back to stumpy.stump, slower).')
''')

md('## 3. Series, jobs and the one-run helper')
code(r'''
def tsbad_groups():
    text = (REPO / 'runners' / 'run_all_CoAD.sh').read_text(encoding='utf-8')
    groups = []
    for block in text.split('for g in $(seq ')[1:]:
        a, b = map(int, re.match(r'(\d+) (\d+)\)', block).groups())
        if '--dataset_name TSB-AD' in block.split('done')[0]:
            groups += list(range(a, b + 1))
    return groups

TSBAD = tsbad_groups()
assert len(TSBAD) == 64
LENGTHS = pd.read_csv(REPO / 'dataset' / 'UCR' / 'all_length.csv').sort_values('file_name').reset_index(drop=True)
LENGTHS['group'] = np.arange(1, 251)
KDD21 = LENGTHS.assign(n=LENGTHS.train_length + LENGTHS.test_length).sort_values('n').group.tolist()  # short first
TSBAD_LEN = {g: sum(1 for _ in open(RAW / RAW_FILES[g - 1])) for g in TSBAD}
TSBAD = sorted(TSBAD, key=TSBAD_LEN.get)

DM = ['Kmeans', 'KshapeAD', 'SAND', 'Series2Graph', 'MP', 'PCA', 'damp']
DEEP = ['cutaddpaste', 'couta', 'fcvae', 'catch', 'tfad', 'dada', 'moment', 'TFMAE', 'm2n2',
        'TranAD', 'MAUT', 'DCdetector', 'AnomalyTrans', 'FITS']
ABL = {'freq_hard': [], 'freq_gating': [], 'guide_hard': ['--window_multiple', '4'], 'time_soft': [],
       'freq_time_point': ['--window_multiple', '4'], 'freq_time_window': ['--window_multiple', '1'],
       'freq_time_add': [], 'freq_time_gate': [], 'freq_time_mean': []}
ABL_EXTRA = ['oe_time', 'oe_freq_time', 'mae_random', 'mae_gating', 'guide_wo_score']

def coad_cmd(model, dataset, group, ts, batch, extra=()):
    script = 'train_classify_mask_t12.py' if model in ABL_EXTRA else 'train_classify_mask.py'
    return [sys.executable, script, '--dataset_name', dataset, '--group_name', str(group), '--model_name', model,
            '--timestamp', ts, '--device', 'cuda:0', '--plot', 'False', '--num_epochs', '300',
            '--batch_size', str(batch), '--window_multiple', '4', '--lr', '2e-3', '--patch_size', '8',
            '--omega', '10', '--random_seed', '46', *extra]

def jobs_for(task, protocol='script'):
    """(task, method, dataset, group, protocol) tuples of one task, shortest series first."""
    if task == 'coad_kdd21':
        return [(task, 'CoAD', 'UCR', g, '-') for g in KDD21]
    if task == 'dm_kdd21':
        return [(task, m, 'UCR', g, '-') for g in KDD21 for m in DM]
    if task == 'deep':
        return ([(task, m, 'TSB-AD', g, protocol) for g in TSBAD for m in DEEP] +
                [(task, m, 'UCR', g, protocol) for g in KDD21 for m in DEEP])
    if task == 'abl_kdd21':
        return [(task, v, 'UCR', g, '-') for g in KDD21 for v in ABL]
    if task == 'abl_extra':
        return ([(task, v, 'TSB-AD', g, '-') for g in TSBAD for v in ABL_EXTRA] +
                [(task, v, 'UCR', g, '-') for g in KDD21 for v in ABL_EXTRA])
    raise ValueError(task)

METRIC_KEYS = ['f1_score', 'auc_prc', 'auc_roc', 'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR']
COLUMNS = ['task', 'method', 'dataset', 'group', 'protocol', 'status', 'seconds', 'run_id', 'commit', 'error'] + METRIC_KEYS
KEY = ['task', 'method', 'dataset', 'group', 'protocol']
RUNS = STORE / 'runs' / 't12'

def read_t12():
    files = sorted(RESULTS_DIR.glob('t12_results_*.csv'))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values('run_id', kind='stable')

def summarize(metric_file):
    m = json.loads(Path(metric_file).read_text())
    row = {k: m['point_wise'][k] for k in ['f1_score', 'auc_prc', 'auc_roc']}
    row.update({k: m['vus'][k] for k in ['R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR']})
    return row

def run_job(job, results_file, timeout_min):
    task, method, dataset, group, protocol = job
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f'{ts}_{task}_{method}_{dataset}_{group}_{protocol}'
    out = RUNS / run_id
    out.mkdir(parents=True)
    g3 = str(group).zfill(3)
    if task == 'deep':
        cmd = [sys.executable, 't12_baselines.py', '--model', method, '--dataset', dataset, '--group', str(group),
               '--protocol', protocol, '--out', 't12_results']
        metric = REPO / 't12_results' / protocol / dataset / method / '42' / 'data' / f'group_{g3}.json'
    elif task == 'dm_kdd21':
        cmd = [sys.executable, 'baseline_compare.py', '--model_name', method, '--dataset_name', dataset, '--group_name', str(group)]
        metric = REPO / 'baseline_result' / dataset / method / 'data' / f'group_{g3}.json'
    else:   # CoAD and ablations (CoAD on KDD21: batch 64 as run_all_CoAD.sh; ablations: 128 as run_ablation_CoAD.sh)
        cmd = coad_cmd(method if task != 'coad_kdd21' else 'freq_tf_v2', dataset, group, ts,
                       64 if task == 'coad_kdd21' else 128, ABL.get(method, []))
        metric = None
    (out / 'command.json').write_text(json.dumps(cmd), encoding='utf-8')
    row = dict(zip(KEY, job), run_id=run_id, commit=commit)
    started = time.perf_counter()
    try:
        with (out / 'console.log').open('w', encoding='utf-8') as log:
            res = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_min * 60)
        assert res.returncode == 0, f'exit code {res.returncode}'
        if metric is None:
            name = 'freq_tf_v2' if task == 'coad_kdd21' else method
            src = REPO / 'mask_results' / dataset / f'{name}_max_{ts}'
            shutil.copytree(src, out / 'artifacts', ignore=shutil.ignore_patterns('*.pth', '*.png'))
            shutil.rmtree(src)
            metric = next((out / 'artifacts' / 'all' / 'data').glob('group_*.json'))
        else:
            (out / 'artifacts').mkdir()
            shutil.copy2(metric, out / 'artifacts' / metric.name)
            score = metric.parent.parent / 'anomaly_score' / f'group_{g3}.npy'
            if score.exists():
                shutil.copy2(score, out / 'artifacts' / score.name)
        row.update(summarize(metric), status='success', error='')
    except Exception as exc:
        tail = (out / 'console.log').read_text(encoding='utf-8', errors='replace').strip().splitlines()[-3:]
        row.update(status='failed', error=(f'{type(exc).__name__}: {exc} | ' + ' / '.join(tail))[:500])
    row['seconds'] = round(time.perf_counter() - started, 1)
    save_row(row, results_file)
    return row

def save_row(row, results_file):
    """Rewrites this copy's result file with the new row and fsyncs it, so the row reaches Drive at once
    (rows only appended may stay in Colab's write cache and are lost when the runtime is killed)."""
    old = results_file.read_text(encoding='utf-8') if results_file.exists() else ''
    new = pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(index=False, header=not old)
    with open(results_file, 'w', encoding='utf-8', newline='') as f:
        f.write(old + new)
        f.flush()
        os.fsync(f.fileno())

print('TSB-AD:', len(TSBAD), '| KDD21:', len(KDD21), '| jobs per task:',
      {t: len(jobs_for(t)) for t in ['coad_kdd21', 'dm_kdd21', 'deep', 'abl_kdd21', 'abl_extra']})
''')

md('''## 4. Choose the work for this session

Suggested order (cheapest and most informative first): `deep` with `PROTOCOL = 'script'` on TSB-AD → `coad_kdd21`
→ `deep` script on KDD21 → `dm_kdd21` (CPU runtime is enough except MP) → `abl_extra` → `abl_kdd21` →
`deep` with `PROTOCOL = 'default'`. Split one task over several runtimes with `SHARD = (i, n)`: copy i of n takes
every n-th job.''')
code(r'''
SESSION_TAG = 'gpu-a'               # unique per notebook copy running at the same time
TASKS = ['deep', 'coad_kdd21', 'dm_kdd21']   # = everything Table 1 still needs, run in this order
                                    # Table 2 afterwards: ['abl_extra', 'abl_kdd21']
PROTOCOL = 'script'                 # deep baselines only: 'script' first, then 'default'
DATASETS = ['TSB-AD', 'UCR']        # 'UCR' = KDD21
METHODS = None                      # None = all methods of the tasks, or e.g. ['couta', 'm2n2']
SHARD = (0, 1)                      # (i, n): this copy runs jobs i, i+n, i+2n, ...
TIME_BUDGET_MIN = 600 if ON_KAGGLE else 300   # no new run starts after this
RUN_TIMEOUT_MIN = 180               # one run longer than this is stopped and recorded as failed
RETRY_FAILED = True                 # retry runs whose latest attempt failed ...
MAX_ATTEMPTS = 6                    # ... until they have failed this many times (then they are left alone)
REVERSE = False                     # True: take the jobs of SHARD from the END (help a copy that is still running
                                    # the same SHARD from the start; each job is re-checked just before it runs)
HELP_SHARDS = None                  # helper mode for a copy whose own shard is done, e.g. [(0, 4), (1, 4), (2, 4)]:
                                    # each run takes the LAST pending job of the shard with the most jobs left
                                    # (SHARD and REVERSE are then ignored)
AUTO_HELP = True                    # when this copy's own SHARD is finished, go on with the other shards
                                    # (last jobs of the shard with the most jobs left), no setting needed

RESULTS = RESULTS_DIR / f't12_results_{SESSION_TAG}.csv'

def recorded():
    """(finished, blocked) job keys over the result files of ALL copies, read now. blocked = latest attempt failed
    and no retry is due (RETRY_FAILED off, or MAX_ATTEMPTS failures reached)."""
    prev = read_t12()
    if not len(prev):
        return set(), set()
    prev = prev.astype({'group': int})
    latest = prev.drop_duplicates(KEY, keep='last')
    done = set(map(tuple, latest.loc[latest.status == 'success', KEY].values))
    failed = latest[latest.status == 'failed']
    if RETRY_FAILED and len(failed):
        n_fail = prev[prev.status == 'failed'].groupby(KEY).size()
        failed = failed[[n_fail.get(tuple(k), 0) >= MAX_ATTEMPTS for k in failed[KEY].values]]
    return done, set(map(tuple, failed[KEY].values))

def selected_jobs():
    return [j for t in TASKS for j in jobs_for(t, PROTOCOL)
            if j[2] in DATASETS and (METHODS is None or j[1] in METHODS)]

def remaining(shard, done, failed):
    return [j for j in selected_jobs()[shard[0]::shard[1]] if j not in done and j not in failed]

def pending():
    done, failed = recorded()
    jobs = remaining(SHARD, done, failed)
    return jobs[::-1] if REVERSE else jobs

CLAIMS = STORE / 'claims' / 't12'   # one small file per job being run: other copies skip it meanwhile

def claim_file(job):
    return CLAIMS / ('__'.join(map(str, job)) + '.txt')

def claimed_by_other(job):
    p = claim_file(job)
    try:
        tag, t = p.read_text(encoding='utf-8').split('|')
    except Exception:
        return False
    return tag != SESSION_TAG and time.time() - float(t) < (RUN_TIMEOUT_MIN + 5) * 60

def claim(job):
    CLAIMS.mkdir(parents=True, exist_ok=True)
    with open(claim_file(job), 'w', encoding='utf-8') as f:
        f.write(f'{SESSION_TAG}|{time.time()}'); f.flush(); os.fsync(f.fileno())

def _pick(jobs, from_end):
    for j in (reversed(jobs) if from_end else jobs):
        if not claimed_by_other(j):
            return j
    return None

def next_job(taken):
    """Next job to run, decided from the result files and claims as they are NOW (other copies keep writing).
    The chosen job is claimed at once, so a copy helping the same shard picks another one."""
    done, failed = recorded()
    def free(s):
        return [j for j in remaining(s, done, failed) if j not in taken]
    def helper(shards, label):
        left = {s: free(s) for s in shards}
        for s in sorted(left, key=lambda s: -len(left[s])):     # shard with the most jobs left first
            j = _pick(left[s], from_end=True)
            if j is not None:
                return j, f'{label}{s}, {len(left[s]) - 1} left there'
        return None, ''
    if HELP_SHARDS:
        job, note = helper(HELP_SHARDS, 'helps ')
    else:
        jobs = free(SHARD)
        job, note = (_pick(jobs, from_end=REVERSE), f'{len(jobs) - 1} left') if jobs else (None, '')
        if job is None and AUTO_HELP and SHARD[1] > 1:     # own shard finished: help the other shards
            job, note = helper([(k, SHARD[1]) for k in range(SHARD[1]) if k != SHARD[0]], 'own shard done, helps ')
    if job is not None:
        claim(job)
    return job, note

if HELP_SHARDS:
    _d, _f = recorded()
    print('helper mode | jobs left per shard:', {s: len(remaining(s, _d, _f)) for s in HELP_SHARDS},
          '| writes to', RESULTS.name)
else:
    print(len(pending()), 'runs to do | writes to', RESULTS.name)
''')

md('## 5. Run (re-run to continue)')
code(r'''
# Each job is chosen just before it runs from the result files of all copies: re-running this cell never repeats
# finished jobs, and copies sharing a shard (REVERSE / HELP_SHARDS) do not collide.
start, i, taken = time.perf_counter(), 0, set()
while True:
    if (time.perf_counter() - start) / 60 > TIME_BUDGET_MIN:
        print('Time budget reached: re-run this cell (or step 4-5 in a new runtime) to continue.')
        break
    job, note = next_job(taken)
    if job is None:
        print('Nothing left to run for these settings.')
        break
    taken.add(job)
    i += 1
    row = run_job(job, RESULTS, RUN_TIMEOUT_MIN)
    print(f"[{i}] {job[1]:13s} {job[2]:6s} {job[3]:3d} {job[4]:7s} {row['status']:7s} {row['seconds']:6.0f}s "
          f"VUS-PR={100 * row.get('VUS_PR', float('nan')):.2f} ({note})"
          + ('' if row['status'] == 'success' else '  ' + row['error'][:160]), flush=True)
''')

md('''## 5b. Recover results from the run folders (run once if rows went missing)

Every successful run also keeps its metric file in `runs/t12/<run_id>/artifacts/`. This cell adds a row for each such
run that no result file lists (e.g. rows lost when a runtime was killed) to `results/t12_results_recovered.csv`.
Safe to run any time and from any copy; then re-run step 4.''')
code(r'''
TASK_NAMES = ['coad_kdd21', 'dm_kdd21', 'deep', 'abl_kdd21', 'abl_extra']

def parse_run_id(run_id):
    """<UTC time>_<task>_<method>_<dataset>_<group>_<protocol>; task and method may contain '_'."""
    _, rest = run_id.split('_', 1)
    task = next(t for t in sorted(TASK_NAMES, key=len, reverse=True) if rest.startswith(t + '_'))
    method, dataset, group, protocol = rest[len(task) + 1:].rsplit('_', 3)
    return task, method, dataset, int(group), protocol

def recover_from_runs():
    known = set(read_t12()['run_id']) if len(read_t12()) else set()
    rec_file = RESULTS_DIR / 't12_results_recovered.csv'
    rows = []
    for d in sorted(RUNS.glob('*')) if RUNS.exists() else []:
        if d.name in known or not (d / 'artifacts').exists():
            continue
        files = sorted((d / 'artifacts').rglob('group_*.json'))
        main = [f for f in files if f.parent.name == 'data' and f.parent.parent.name == 'all'] or \
               [f for f in files if f.parent == d / 'artifacts']
        if not main:
            continue
        try:
            task, method, dataset, group, protocol = parse_run_id(d.name)
            row = dict(task=task, method=method, dataset=dataset, group=group, protocol=protocol, run_id=d.name,
                       commit=commit, status='success', error='recovered from run folder', seconds=np.nan,
                       **summarize(main[0]))
        except Exception as exc:
            print('skip', d.name, exc)
            continue
        rows.append(row)
    if rows:
        old = rec_file.read_text(encoding='utf-8') if rec_file.exists() else ''
        new = pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(index=False, header=not old)
        with open(rec_file, 'w', encoding='utf-8', newline='') as f:
            f.write(old + new); f.flush(); os.fsync(f.fileno())
    print(len(rows), 'rows recovered into', rec_file.name)

recover_from_runs()
''')

md('''## 6. Comparison with Table 1 and Table 2 (any time)

Means over the series that finished (%). `n` = series finished out of 64 (TSB-AD) or 250 (KDD21); compare with the
paper only when `n` is complete. CoAD and the data-mining baselines on TSB-AD, and the 9 published ablations on TSB-AD,
come from the earlier notebooks.''')
code(r'''
M4 = {'f1_score': 'F1', 'auc_prc': 'AUC-PR', 'R_AUC_PR': 'R-AUC-PR', 'VUS_PR': 'VUS-PR'}
P1 = {  # Table 1: KDD21 then TSB-AD, (F1, AUC-PR, R-AUC-PR, VUS-PR); None = N/A in the paper
 'DADA': [(3.49, 1.39, 2.04, 2.05), (17.36, 12.30, 14.92, 14.53)], 'MOMENT': [(11.06, 7.71, 9.20, 9.13), (25.29, 18.80, 25.90, 24.67)],
 'MMA': [(44.47, 39.24, 37.97, 37.38), (43.20, 38.65, 37.33, 36.94)], 'TFMAE': [(2.53, 0.99, 1.75, 1.75), (7.29, 3.45, 5.83, 5.77)],
 'AnomalyBERT': [(23.42, 15.92, 14.16, 13.79), (19.14, 9.85, 13.21, 13.85)], 'CutAddPaste': [(21.75, 15.51, 19.00, 18.20), (26.22, 21.34, 25.45, 25.08)],
 'TriAD': [(16.78, 22.37, 28.09, 27.56), None], 'COUTA': [(6.58, 3.65, 3.79, 3.82), (18.78, 13.12, 11.07, 11.01)],
 'FCVAE': [(11.23, 7.18, 6.25, 6.38), (28.47, 22.12, 20.61, 20.58)], 'TFAD': [(1.85, 0.86, 1.59, 1.58), (6.17, 3.34, 6.21, 5.99)],
 'CATCH': [(13.29, 9.08, 9.26, 9.19), (24.46, 20.02, 22.32, 21.67)], 'TranAD': [(11.23, 7.78, 7.94, 7.89), (18.07, 13.06, 12.23, 12.05)],
 'MAUT': [(30.20, 23.84, 23.94, 23.65), (19.82, 14.67, 14.91, 14.75)], 'M2N2': [(5.57, 2.70, 3.15, 3.18), (16.98, 10.38, 9.41, 9.25)],
 'FITS': [(5.34, 2.63, 3.46, 3.45), (13.23, 7.14, 10.63, 10.37)], 'DCdetector': [(2.66, 1.09, 1.88, 1.85), (6.47, 3.12, 5.79, 5.69)],
 'AnomalyTrans': [(2.37, 0.97, 1.87, 1.80), (7.01, 3.36, 6.27, 6.13)],
 'KShapeAD': [(43.27, 39.37, 39.48, 39.02), (35.51, 32.67, 31.87, 31.27)], 'SAND': [(39.43, 34.72, 34.23, 33.78), (34.74, 31.52, 31.85, 31.05)],
 'Sub-PCA': [(15.45, 11.32, 14.06, 13.17), (32.49, 27.59, 23.88, 23.85)], 'Series2Graph': [(28.11, 22.61, 25.63, 24.81), (33.48, 30.11, 30.13, 29.57)],
 'KMeansAD': [(37.97, 34.33, 33.49, 33.20), (41.42, 37.26, 37.64, 36.81)], 'Matrix Profile': [(28.00, 18.50, 25.37, 24.13), (35.10, 27.93, 29.88, 28.94)],
 'DAMP': [(29.31, 18.99, 25.09, 24.18), (17.44, 11.59, 12.98, 12.60)], 'CoAD': [(52.82, 48.10, 46.35, 45.67), (49.13, 43.66, 40.50, 39.83)],
}
P2 = {  # Table 2, same layout
 'oe_time': [(27.84, 24.12, 24.81, 24.38), (29.79, 22.36, 27.14, 26.28)], 'oe_freq_time': [(47.95, 42.75, 41.06, 40.41), (37.68, 31.21, 30.48, 29.90)],
 'mae_random': [(35.65, 30.23, 30.96, 30.36), (32.30, 25.74, 23.55, 23.35)], 'mae_gating': [(38.99, 32.31, 32.76, 31.99), (35.01, 29.08, 28.51, 28.03)],
 'freq_hard': [(44.88, 41.26, 39.55, 39.05), (45.01, 39.31, 35.79, 35.20)], 'freq_gating': [(46.16, 41.95, 40.86, 40.19), (47.07, 41.48, 37.39, 37.08)],
 'guide_hard': [(49.03, 45.27, 43.87, 43.49), (44.06, 38.57, 35.31, 34.76)], 'time_soft': [(48.45, 43.21, 42.87, 42.35), (43.10, 36.87, 37.91, 36.92)],
 'freq_time_point': [(43.09, 36.62, 35.43, 35.00), (42.52, 36.06, 34.02, 33.37)], 'freq_time_window': [(15.26, 10.21, 11.29, 11.02), (22.43, 17.08, 15.71, 15.53)],
 'freq_time_add': [(50.76, 45.93, 44.76, 44.25), (40.11, 35.15, 33.40, 32.94)], 'freq_time_gate': [(50.52, 45.58, 45.53, 44.71), (42.23, 37.20, 34.87, 34.40)],
 'freq_time_mean': [(51.50, 45.98, 45.53, 44.71), (42.24, 36.23, 33.71, 33.18)], 'guide_wo_score': [(48.13, 42.92, 42.93, 42.09), (37.07, 30.54, 28.04, 27.73)],
 'CoAD': [(52.82, 48.10, 46.35, 45.67), (49.13, 43.66, 40.50, 39.83)],
}
T1_NAME = {'cutaddpaste': 'CutAddPaste', 'couta': 'COUTA', 'fcvae': 'FCVAE', 'catch': 'CATCH', 'tfad': 'TFAD', 'dada': 'DADA',
           'moment': 'MOMENT', 'm2n2': 'M2N2', 'Kmeans': 'KMeansAD', 'KshapeAD': 'KShapeAD', 'MP': 'Matrix Profile',
           'PCA': 'Sub-PCA', 'damp': 'DAMP'}
N_SERIES = {'UCR': 250, 'TSB-AD': 64}

rows = []   # one row per (source, method, dataset, group): latest successful run
t12 = read_t12().query("status == 'success'")
for _, r in t12.drop_duplicates(KEY, keep='last').iterrows():
    src = 'deep:' + r.protocol if r.task == 'deep' else ('ablation' if r.task.startswith('abl') else 'main')
    rows.append(dict(source=src, method=r.method, dataset=r.dataset, group=int(r.group), **{k: r[k] for k in M4}))
old = sorted(RESULTS_DIR.glob('tsbad64_results*.csv'))   # CoAD (seed 46, authors' code) and DM baselines on TSB-AD
if old:
    o = pd.concat([pd.read_csv(f) for f in old]).query("status == 'success'").sort_values('run_id').drop_duplicates(['method', 'group'], keep='last')
    rows += [dict(source='main', method=r.method, dataset='TSB-AD', group=int(r.group), **{k: r[k] for k in M4}) for _, r in o.iterrows()]
old = sorted(RESULTS_DIR.glob('ablation_results*.csv'))  # 9 published ablations on TSB-AD (M6-B)
if old:
    o = pd.concat([pd.read_csv(f) for f in old]).query("status == 'success'").sort_values('run_id').drop_duplicates(['variant', 'group'], keep='last')
    rows += [dict(source='ablation', method=r.variant, dataset='TSB-AD', group=int(r.group), **{k: r[k] for k in M4}) for _, r in o.iterrows()]
ALL = pd.DataFrame(rows)

def compare(source_filter, paper):
    out = []
    sub = ALL[ALL.source.str.startswith(source_filter)]
    for (src, method, ds), g in sub.groupby(['source', 'method', 'dataset']):
        name = T1_NAME.get(method, method)
        ref = paper.get(name, paper.get(method, [None, None]))[0 if ds == 'UCR' else 1]
        rec = {'source': src, 'method': name, 'dataset': 'KDD21' if ds == 'UCR' else ds,
               'n': f"{g.group.nunique()}/{N_SERIES[ds]}"}
        for k, lab in M4.items():
            rec[lab] = 100 * g[k].mean()
            rec[lab + ' paper'] = ref[list(M4).index(k)] if ref else np.nan
        rec['VUS-PR diff'] = rec['VUS-PR'] - rec['VUS-PR paper']
        out.append(rec)
    return pd.DataFrame(out).round(2) if out else pd.DataFrame()

pd.set_option('display.width', 250)
TABLE1 = pd.concat([compare('main', P1), compare('deep', P1)], ignore_index=True)
TABLE2 = compare('ablation', P2)
(STORE / 'reports').mkdir(exist_ok=True)
TABLE1.to_csv(STORE / 'reports' / 't12_table1.csv', index=False)
TABLE2.to_csv(STORE / 'reports' / 't12_table2.csv', index=False)
display(TABLE1.sort_values(['dataset', 'source', 'method']))
display(TABLE2.sort_values(['dataset', 'method']))
lat = read_t12().drop_duplicates(KEY, keep='last')
if len(lat):
    display(lat.groupby(['task', 'method', 'dataset', 'protocol']).agg(
        success=('status', lambda s: (s == 'success').sum()), failed=('status', lambda s: (s == 'failed').sum()),
        median_s=('seconds', 'median')).reset_index())
    bad = lat.query("status == 'failed'")
    if len(bad):
        display(bad.groupby(['method', 'dataset'])['error'].first().str[:220].to_frame())
''')

md('## 7. Kaggle only: pack the results for Drive (download `to_drive.zip` from the Output tab)')
code(r'''
OUT = Path('/kaggle/working/to_drive') if ON_KAGGLE else None
if OUT:
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / 'results').mkdir(parents=True)
    for csv in RESULTS_DIR.glob(f't12_results_{SESSION_TAG}.csv'):
        shutil.copy2(csv, OUT / 'results' / csv.name)
    for sub in ['reports', 'runs/t12']:
        if (STORE / sub).exists():
            shutil.copytree(STORE / sub, OUT / sub)
    shutil.make_archive(str(OUT), 'zip', OUT)
    print('to_drive.zip ready')
''')

COPIES = [('gpu-a', 0), ('gpu-b', 1), ('gpu-c', 2), ('gpu-d', 3)]   # 4 notebooks at the same time, a quarter of the jobs each
for tag, i in COPIES:
    out = []
    for c in cells:
        c = dict(c)
        text = ''.join(c['source'])
        if text.startswith('SESSION_TAG'):
            text = (text.replace("SESSION_TAG = 'gpu-a'", f"SESSION_TAG = '{tag}'")
                        .replace("SHARD = (0, 1)", f"SHARD = ({i}, {len(COPIES)})"))
            c['source'] = text.splitlines(keepends=True)
        if text.startswith('# CoAD'):
            c['source'] = (text + f"\n\n**This copy: `{tag}`, shard {i} of {len(COPIES)}.** Run the 4 copies at the same time "
                           "with the SAME `TASKS`, `PROTOCOL`, `DATASETS`, `METHODS` (only `SESSION_TAG` and `SHARD` differ).").splitlines(keepends=True)
        out.append(c)
    nb = {'cells': out, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                                     'colab': {'gpuType': 'T4', 'provenance': []}},
          'nbformat': 4, 'nbformat_minor': 5}
    path = HERE / f'CoAD_table12_{tag}.ipynb'
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('written', path.name)
