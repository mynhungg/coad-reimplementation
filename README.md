# CoAD reimplementation, reproduction and error-analysis demo

Course project (Data Mining) around **CoAD** — *Bridging Classification and Reconstruction: Cooperative Time Series
Anomaly Detection* (KDD 2026, arXiv 2605.26193). It contains:

- a from-scratch PyTorch **reimplementation** of CoAD, verified against the authors' code (outputs within 1e-5);
- **reproduction** of the paper's Table 1 (CoAD and 21 baselines on TSB-AD-U and KDD21) and Table 2 (ablation);
- **experiments**: error analysis of 8 262 anomaly events, label-free scoring improvements, external checks;
- an interactive **web demo** to explore the results and to run CoAD on your own CSV.

## Run the demo in one command

Requires only [uv](https://docs.astral.sh/uv/) (the scripts below install it if missing). uv downloads Python 3.11,
CPU PyTorch and the other packages into its own cache, starts the server and opens http://localhost:8765.

```bash
uv run demo/serve_coad.py
```

or, without uv installed:

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1      # Windows
```

```bash
sh run.sh                                             # macOS / Linux
```

The first start takes about a minute (package download). In the demo:

1. **Dữ liệu** (data): the datasets used, and analysis of an uploaded CSV (sampling, period, labels, insights);
2. **Phân tích chuỗi** (series analysis): input, ground truth, alarms, anomaly score, reconstruction, a step-by-step
   view of how CoAD processes a window, and the signal trace of each event;
3. **Phân tích lỗi** (error analysis): CoAD vs the improved `calibrated` scoring, event by event;
4. **Kết quả thực nghiệm** (results): per-series and per-family deltas.

Upload a CSV (`timestamp, value[, label]`, samples from NAB are included) to analyse it and train CoAD on it locally.
On a CPU the default fast mode takes a few minutes (see `demo/README.md`). Labels are used only for evaluation.

## Repository layout

| Path | Content |
|---|---|
| `demo/` | Web demo (static HTML/JS + Plotly), `serve_coad.py` (server + CoAD training on uploads), `coad_core.py` (model), `precompute.py`, precomputed data of 5 series, sample CSVs |
| `notebooks/CoAD_reimplementation.ipynb` | Reimplementation: paper mode and code mode (21 code-only details D1–D21), equivalence tests, experiments M1–M21 |
| `notebooks/CoAD_reproduction.ipynb` | Reproduction of the authors' public code on the 64 TSB-AD-U series |
| `notebooks/CoAD_experiments.ipynb` | Additional experiments |
| `notebooks/verify_reported_numbers.ipynb` | Recomputes every reported number from the raw result files |
| `notebooks/kaggle/` | Notebooks of individual experiments (M6–M11) prepared for Kaggle GPUs |
| `notebooks/table12/` | Notebooks and scripts reproducing Table 1 and Table 2 (4 parallel GPU sessions), result snapshots |
| `tools/`, `experiment_results/` | Audit scripts and raw experiment outputs |
| `docs/` | Report (`report.md`, `summary.md`, Word versions), demo specification and UI design |

## Main findings (64 TSB-AD-U series, 3 seeds)

- The reimplementation following the authors' code matches it (paired Wilcoxon p ≥ 0.30). Following the paper text
  alone reaches only 18.44 VUS-PR on MGAB vs 41.71 for the authors' code: 21 training/architecture details are not in
  the paper.
- The authors' code itself is ~3 points below the paper on F1/AUC-PR and ~2 above on R-AUC-PR/VUS-PR for TSB-AD
  (2.6–4.6 seed SDs), while it matches the KDD21 column within 0.3 points.
- Most missed events still carry signal in the raw reconstruction error; the fixed moving-average smoothing removes it.

Data: TSB-AD-U (Liu & Paparrizos, NeurIPS 2024), UCR Anomaly Archive 2021 (KDD21), NAB samples (MIT licence, see
`demo/samples/NAB_LICENSE.txt`). The paper PDF and the authors' repository are not redistributed here.
