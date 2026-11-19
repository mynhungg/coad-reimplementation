"""Builds dataset/UCR/processed/{train,test,label}/*.npy for the authors' load_dataset('UCR', ...).

The authors' KDD21 preprocessing is not published; only dataset/UCR/all_length.csv (train_length, test_length,
anomaly_length, period per series) is. This script rebuilds the files from the original UCR Anomaly Archive 2021
(file name = <id>_UCR_Anomaly_<name>_<train_end>_<anomaly_begin>_<anomaly_end>.txt):

  train = x[:train_end], test = x[train_end:], label = 1 on [anomaly_begin-1, anomaly_end-1] (1-based, inclusive)
  scaling: min-max with the training part (the rule the authors' loader applies to TSB-AD; an assumption here)

and checks every series against all_length.csv (train/test/anomaly lengths must match exactly). Two deviations
of the authors' files are reproduced because all_length.csv shows them:
  * 239-241 (taichidbS0715Master, 900 000 points): all lengths are 1/10 of the archive -> downsampled by 10
    (every 10th point; indices divided by 10). Point decimation vs block mean is not recoverable: assumption.
  * 079, 108, 187 (resperation2, a 1-point anomaly): anomaly_length 3 -> the point plus one neighbour per side.

usage: python t12_prepare_kdd21.py <UCR_TimeSeriesAnomalyDatasets2021.zip> <repo root>
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN = re.compile(r'^(\d{3})_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$')


def read_values(raw: bytes) -> np.ndarray:
    # most files have one value per line; a few store all values on one line separated by spaces
    return np.array(raw.decode('ascii', errors='ignore').split(), dtype=float)


def main(archive: str, repo: str):
    repo = Path(repo)
    ref = pd.read_csv(repo / 'dataset' / 'UCR' / 'all_length.csv').set_index('file_name')
    out = repo / 'dataset' / 'UCR' / 'processed'
    for sub in ['train', 'test', 'label']:
        (out / sub).mkdir(parents=True, exist_ok=True)

    found = {}
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            m = PATTERN.match(Path(member).name)
            if m and '..' not in member:
                found.setdefault(m.group(1), (member, m))
        assert len(found) == 250, f'expected 250 KDD21 series in the archive, found {len(found)}'
        rows = []
        for idx in sorted(found):
            member, m = found[idx]
            name = f'{idx}_{m.group(2)}'
            train_end, a_begin, a_end = (int(m.group(k)) for k in (3, 4, 5))
            x = read_values(z.read(member))
            r = ref.loc[name] if name in ref.index else None
            factor = 1
            if r is not None and int(r.train_length) != train_end and train_end // int(r.train_length) == 10:
                factor = 10                                    # authors' downsampling (239-241)
                x = x[:len(x) // factor * factor:factor]
                train_end, a_begin, a_end = train_end // factor, a_begin // factor, a_end // factor
            train, test = x[:train_end], x[train_end:]
            label = np.zeros(len(test), dtype=int)
            lo_idx, hi_idx = a_begin - 1 - train_end, a_end - train_end          # 0-based, end exclusive
            if r is not None and factor == 1 and hi_idx - lo_idx < int(r.anomaly_length):
                pad = (int(r.anomaly_length) - (hi_idx - lo_idx)) // 2        # widen symmetrically (079/108/187)
                lo_idx, hi_idx = lo_idx - pad, hi_idx + pad
            label[lo_idx:hi_idx] = 1
            lo, hi = train.min(), train.max()
            scale = (hi - lo) if hi > lo else 1.0
            np.save(out / 'train' / f'{name}.npy', (train - lo) / scale)
            np.save(out / 'test' / f'{name}.npy', (test - lo) / scale)
            np.save(out / 'label' / f'{name}.npy', label)
            rows.append(dict(file_name=name, train=len(train), test=len(test), anomaly=int(label.sum()),
                             ok=(r is not None and int(r.train_length) == len(train) and int(r.test_length) == len(test)
                                 and int(r.anomaly_length) == int(label.sum()))))
    check = pd.DataFrame(rows)
    bad = check[~check['ok']]
    print(f'{len(check)} series written to {out}; {len(bad)} differ from all_length.csv')
    if len(bad):
        print(bad.merge(ref, left_on='file_name', right_index=True, how='left').to_string())
    # load_dataset indexes the sorted file lists; they must follow the order of all_length.csv
    assert sorted(p.stem for p in (out / 'train').glob('*.npy')) == sorted(ref.index), 'file names differ from all_length.csv'
    return check


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
