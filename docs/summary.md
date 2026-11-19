# CoAD: Reproduction, Error Analysis and Pre-registered Tests of Improvements

**Paper.** *Bridging Classification and Reconstruction: Cooperative Time Series Anomaly Detection* (CoAD, KDD 2026). **Data.** The 64 TSB-AD-U series used in the paper (MGAB, SED, SVDB, IOPS, TODS) and 60 series from 12 other TSB-AD-U families for external checks. The KDD21 column of Table 1 is checked with the authors' code on 246/250 series. **Protocol.** VUS-PR ×100, 3 seeds unless stated, and Wilcoxon tests (sidedness stated) with Holm correction. Predictions were written before each run, and from M11 on the 64 series were split into 30 design and 34 confirmation series.

**1. Reproduction.** We reimplemented CoAD in a *paper* mode (as described) and a *code* mode (21 code-only details). Loading the authors' weights gives max output error < 1e-5, and code mode matches the authors' public code (41.69 ± 0.96 vs 41.70, p = 0.56). **The TSB-AD figures of Table 1 are not obtained with that code**: F1/AUC-PR are 3.2–3.8 lower and R-AUC-PR/VUS-PR about 2 higher (1.9–5.1 seed SDs). The gap is not explained by our evaluation pipeline (SAND, PCA, MP and DAMP match Table 1) or by score smoothing. The paper omits loss and batching details (MGAB: 18.44 in paper mode vs 44.73 in code mode). It also reports early stopping, which the code lacks; adding it lowers all four metrics on MGAB (VUS-PR −8.57, p = 0.025), so it cannot explain the gap alone. On KDD21 the authors' code, run on 246/250 series, matches Table 1 on all four metrics (gap < 0.3; e.g. VUS-PR 45.94 vs 45.67), so the mismatch is specific to the TSB-AD part. The paper reports only means, so the gap is measured in seed SDs; a paired test against the paper is not possible.

**2. Baselines.** On the 64 series, CoAD has the best mean VUS-PR but ranks 3rd of 8 (2nd on F1) and does not beat KMeans (Holm p = 0.46). On 58 external series it trails KMeans (52.4 vs 59.0, two-sided p = 0.04; KMeans fits on the test series itself). Of the authors' ablations, mean fusion moves in the paper's direction (n.s.), hard masking is within seed noise, and removing the frequency branch shows no decrease (+1.91, n.s.).

**3. Error analysis (8,262 events; label-based thresholds, diagnosis only).**
- **Core mechanisms.** We found no evidence for the hypothesised core failures: no boundary shortcut, soft masking is selective, and there is no gradient conflict. One masking test is inconclusive.
- **Signal lost mainly at smoothing.** For events the classifier misses, raw reconstruction error reaches the 0.99 threshold for 53% and the final score for 7%. An order-free Shapley attribution assigns 54–100% of the loss to the one-period moving average and 15–46% to branch fusion (highest on SVDB). Controlled spikes confirm this: 0.66 → 0.03, with dilution −0.36 and spill-over −0.29.
- **Remaining misses.** No branch reaches the threshold for 52% of TODS and 17% of MGAB events. Repairing contaminated TODS training points changes nothing (+0.07).
- **Robustness.** A context-splice test detected even our control, so it measures sensitivity to dynamics changes rather than to context. An apparent residual-branch blind spot (11% recall) did not recur (55%) under another reference set and population.

**4. Improvements.**

| Change (label-free) | Confirmation (34 series) | External (60 series) | Verdict |
|---|---|---|---|
| Per-branch SD calibration (found post hoc) | +3.42 vs CoAD (Holm 0.078) | +1.44 (p = 0.44) | Not general |
| Calibration with reconstruction weight 2 (locked on design set) | +0.80 vs calibration (Holm 0.004); +4.22 vs CoAD (p = 0.001) | +0.81 vs calibration (p < 0.001, 9/12 families); +2.25 vs CoAD (p = 0.17) | Beats calibration; not general vs CoAD |
| Instant + mean smoothing | +4.79 vs CoAD (Holm 0.001) | +1.62 (p = 0.16) | Short anomalies only |
| Rarity fusion, MAD scaling, OE-independent residual heads | Below calibration | — | Negative |

The calibration mechanism was confirmed on unseen series: gains grow when calibration up-weights the reconstruction branch (ρ = −0.46, p = 0.003). External gains concentrate in a few families. Without the four families the paper excludes for data quality, calibration gains +0.20 over CoAD.

**5. Conclusions.** The public code reproduces, but the TSB-AD figures of Table 1 do not follow from it (KDD21: no detectable gap). Most missed events still carry signal in the raw reconstruction error, and fixed smoothing removes most of it. Up-weighting reconstruction is a robust gain over calibration, but no score change beats CoAD reliably outside the paper's families, where KMeans leads. **Limitations:** label-based diagnostics, synthetic stress tests, univariate data, few seeds, a transductive KMeans baseline, internal (not time-stamped) pre-registration, and two design errors in our own decision rules, which we report.
