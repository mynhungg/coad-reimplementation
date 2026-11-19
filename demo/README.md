# CoAD Lab — web demo

Research workspace for the CoAD reimplementation: series analysis, error analysis (CoAD vs `calibrated`) and the
experiment results. Spec: `../CoAD_Web_Demo_Error_Analysis_Spec.md`; design: `../CoAD_Web_UI_Design.md`.

## Run

```bash
cd demo
python -m http.server 8765
```

Open http://localhost:8765. A web server is needed (the page loads `data/*.json` and `data/*.bin` with `fetch`).
Everything is local (Plotly is in `vendor/`), so the demo works offline; only the font comes from Google Fonts.

## Run with CoAD training on your own CSV

CoAD only works on the series it was trained on, so a new CSV needs its own training run. `serve_coad.py` serves the demo
and trains CoAD (code mode, seed 46) on an uploaded file, then writes `data/series_<9001+>.json/.bin` so every screen
works on it. The CSV journey is: Data (insights) → Train CoAD → CoAD step by step (Inspector) → results.

```bash
cd demo
uv venv --python 3.11 .venv311
uv pip install --python .venv311/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv311/Scripts/python.exe numpy pandas scipy scikit-learn
.venv311/Scripts/python serve_coad.py --port 8765            # add --repo <CoAD repo> for VUS-PR
```

Notes: the Anaconda Python of this machine ships an old MSVC runtime (14.27) and recent torch fails to load there
(WinError 1114), hence the standalone Python 3.11. On a CPU, the fast mode (default) keeps the model, loss and scoring
and trains on at most 128 windows per epoch (about 8 s per epoch, 50 epochs about 7 min for 10k points); the full
mode uses the authors' training stride (about 65 s per epoch on CPU). Labels are only used for evaluation.

## Files

| File | Role |
|---|---|
| `index.html`, `style.css`, `app.js` | UI: four screens, Model Inspector, upload analysis and in-browser KMeansAD, export, guided tour |
| `illus.js` | Schematic SVG illustrations for the explanation mode |
| `rules.js` | The only implementation of the event rules (hit, coverage, false alarms, recovered/lost) |
| `precompute.py` | Builds `data/` from the benchmark checkpoints (label-free scoring) |
| `coad_core.py` | Model and scoring code extracted from `CoAD_reimplementation.ipynb` (code mode) |
| `data/series_<g>.json/.bin` | Metadata + float32 signals of one series |
| `data/benchmark.json` | Per-series results: CoAD (code mode, seeds 46–48) vs calibrated (M11 B1), KMeansAD |
| `data/datasets.json` | "Data" screen: metadata of the 64 paper series, the 60 external series, KDD21 summary |
| `samples/` | CSV for the upload check: two NAB series (Numenta Anomaly Benchmark, `realKnownCause/nyc_taxi.csv` and `machine_temperature_system_failure.csv`, MIT licence in `NAB_LICENSE.txt`) with a `label` column built from NAB `labels/combined_windows.json`; a TSB-AD SED 234 excerpt; a file with deliberate errors |
| `tools/build_datasets.js` | Builds `datasets.json`: `node tools/build_datasets.js <TSB-AD raw dir> <UCR all_length.csv> <m13 csv dir>` |

## Rebuild the data

1. Download the checkpoint folders `runs/reimpl/<run_id>/` (model.pt, config.json, loss.csv, scores.npy) of the
   wanted series from `CoAD_project` on Drive into one folder, e.g. `ckpt/`.
2. Have the authors' repository with `dataset/TSB-AD/raw` (870 TSB-AD-U files).
3. Run:

```bash
python precompute.py --repo <CoAD repo> --ckpt ckpt --groups 234 237 269 276 295 --results <folder with reimpl_results_*.csv and tsbad64_results*.csv> --m11 <folder with m11_*.csv and m11_locked.json>
```

Without `--ckpt` the script retrains the model (300 epochs). With checkpoints it reports the maximum difference
between the recomputed score and the saved benchmark score (4e-6 or less for the five demo series).

## Rules and scope

- Thresholds: training-part μ + 3σ (default) or training max, per method on its own scale; KMeansAD has no
  training part, so its threshold is μ + 3σ of its own test score. Interactive thresholds are marked and never
  overwrite the benchmark. No threshold uses test labels.
- Threshold-free metrics (VUS-PR, AUC-PR, R-AUC-PR, F1) are computed by the authors' evaluator in `precompute.py`;
  threshold-dependent counts come from `rules.js`.
- Upload: the CSV is validated and described on the "Data" screen (sampling, labels, ACF period as TSB-AD find_length).
  CoAD cannot run on a new series (each checkpoint is trained on one series). The page instead runs a KMeans
  subsequence detector in the browser (KMeansAD-style: windows of one period, 20 centres, threshold μ+3σ of its own
  score; labels only for evaluation). It is a baseline, not CoAD, and no benchmark metric (VUS-PR) is computed for it.
- Novice mode: "💡 Explanations" shows short captions with schematic SVG illustrations (illus.js); "❓ Guide"
  starts a 7-step guided tour (shown once on the first visit).
- Charts draw at most ~4 000 points of the visible range (min/max per bucket); metrics always use the full arrays.
