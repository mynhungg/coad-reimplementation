# Reimplementing CoAD

## 1. Goal and Scope

This part covers our own reimplementation of CoAD (*Bridging Classification and Reconstruction: Cooperative Time Series Anomaly Detection*, KDD 2026). It describes the implementation process and the results. It analyses the results and the model's errors, and compares them with the numbers the paper reports in Table 1 (main results) and Table 2 (ablations).

Rather than only rerunning the authors' code, we rewrote the model in PyTorch, for two reasons:

- to test whether the method can be rebuilt from the paper's text alone;
- to have a code base we understand line by line, for error analysis and for trying improvements.

From one code base we built two versions:

- **The main build** follows the paper's description. Where the paper is unclear, it takes the simplest reading, and we record the choice. This build gives the results **closest to Table 1**, so we report it as the main result.
- **The extended build** follows the authors' code at all 21 points where the paper is silent, ambiguous or different from the code (17 of them differ from the main build). Examples: each batch mixes clean and corrupted windows; the classification loss uses a class weight and averages the two classifiers; a StepLR learning-rate schedule with fp16; a 2-layer GRU (the paper says 3) with an MLP classification head (the paper says linear); two-pass inference (the paper describes one pass). The full list is in Appendix A. None of these were tuned by us: every detail is taken as is from the authors' code. It verifies that our code is correct and explains the gaps. Because it matches the authors' model, it is also the basis for the error analysis.

## 2. Implementation Process

### 2.1. Steps

1. **Read the paper and split it into components.** CoAD has five blocks:
   - a synthetic-anomaly generator;
   - time- and frequency-domain embeddings;
   - an anomaly classifier (OE classifier);
   - a masked autoencoder (MAE) with a soft mask;
   - a residual classifier.

   There are also the loss (Eq. 10) and the inference-time score. For each block we recorded its inputs, outputs, tensor shapes and the corresponding equations.
2. **Compare with the authors' code** (commit `3286813`). We read the paper and the code side by side and recorded every point the paper leaves unclear or describes differently: 21 in total. Most concern the training procedure, which the paper barely describes.
3. **Write and test each module separately**, in dependency order: synthetic anomalies → STFT → encoders → MAE → residual → loss → training loop → score. Each module was tested before being combined (8 tests, Section 3).
4. **Run on the full data.**
   - Two builds × 64 series × 3 seeds = 384 training runs.
   - Evaluation uses the TSB-AD evaluator, as the authors do.
   - The notebook runs in parallel on several GPUs and resumes automatically when Colab/Kaggle disconnects.
5. **Compare and analyse.** We compared the results with Table 1. We then ran extended experiments to explain the gaps (Section 6), an error analysis (Section 7), a baseline comparison (Section 8) and improvement tests (Section 9).
6. **Rerun all of Table 1 with the authors' code** (Section 6.5): CoAD on all 250 KDD21 series, the 14 deep-learning baselines on both benchmarks, and the 7 data-mining baselines. The notebooks run in parallel on 4 GPU sessions; they split the work, help each other and never run a job twice.
7. **Build a web demo** (Section 10) to present the results, the error analysis and the comparison of CoAD with the improved variant on real data.

### 2.2. Model (main build)

**Input.**
- Each series is min–max scaled on its training part, as in the TSB-AD loader.
- The model works on windows of `T = 4 ×` the dominant period (fewer periods if the training part is too short), rounded down to a multiple of 8. Each window is split into patches of 8 points.
- The frequency view is an STFT with `n_fft = 4`.
- At inference, windows slide with a stride of one period. A point's score is the mean over the windows that cover it.

**Synthetic anomalies.** Each training window receives one distorted segment of 10–100% of a period, of one of four types:
- *uniform*: the segment is replaced by a random constant in [min, max] of the distorted segment;
- *length scaling*: the segment is stretched or shrunk by a factor of 2–4;
- *jitter*: noise N(0, 0.1) is added, i.e. σ ≈ 0.32 on data scaled to [0, 1];
- *mirror*: the segment is flipped along the time axis or the value axis.

Each patch is labelled as distorted or not.

**OE classifier.**
- Patches of the time and frequency views are linearly projected to embeddings and passed through a 3-layer Bi-GRU shared by both views.
- A linear head (hidden size 24) outputs a per-patch anomaly probability `A`.

**MAE with a soft mask.**
- Each patch embedding is blended with a learned mask vector in proportion to `A`, so the more likely a patch is anomalous, the more it is masked. The model then reconstructs the window.
- The idea is that the model must rebuild suspicious parts from the normal context around them, so real anomalies produce large reconstruction errors.

**Residual classifier.**
- Both `x` and its reconstruction `x̂` go through the OE encoder. The difference of the two feature sets is classified into `A^r`.
- Inference uses `A_c = (A + A^r)/2` (Eq. 9).

**Loss.** BCE between `A_c` and the patch labels, plus `λ = 10` × the reconstruction loss of Eq. (10). That loss is the squared error summed within each patch and averaged over patches.

**Score.** Two parts:
- the reconstruction score `|x̂ − x|`;
- the classification score `A_c`, repeated over the 8 points of each patch.

The two are summed and smoothed by a moving average of one period.

**Training configuration.**
- Adam, learning rate 0.002, 300 epochs, batch size 128.
- `P = 8`, `T = 4 ×` period, `K = 4`, `λ = 10`, the learning rate and the number of epochs come from Appendix C.1 of the paper.
- The paper does not state the batch size, so we took 128 from the authors' run script.

**Environment.** T4 GPUs on Colab and Kaggle. One training run takes about 70 s on average for the main build and 180 s for the extended build.

**Where the paper lacks information.**
- The paper does not describe batch composition, loss balancing, class weights in the BCE or the learning-rate schedule, nor whether fp16 is used.
- It also contradicts itself in two places:
  - the equations use two separate GRUs while Fig. 2 says "Shared GRU";
  - the equations have no activation while Fig. 2 says "MLP".

  The main build follows Fig. 2 for the first and the equations for the second.
- The paper says the model is selected by early stopping, but the authors' code has none (Section 6.4).

## 3. Verification

Before running experiments we checked the code with 8 tests:

| Test | Check | Result |
|---|---|---|
| T1 | Synthetic anomalies | Extended build identical to the authors' 4 functions (20 seeds × 4 types); main build checked visually (labels at the right positions) |
| T2 | STFT | Identical to the authors' `time_to_timefreq` |
| T3 | Tensor shapes | A dummy batch passes through the whole model with correct shapes in both builds |
| T4 | Authors' trained weights loaded into the extended build, same input | Same outputs as the authors' model, difference < 1e-5 |
| T5 | Overfitting one fixed batch | Loss falls to 5.8% of its initial value in 500 steps |
| T6 | Training group 236 (3-epoch check on CPU; then 384 runs of 300 epochs on GPU) | Stable loss, no NaN |
| T7 | Score and evaluator | Correct score length, no NaN/Inf, metrics in [0, 1] |
| T8 | Comparison with the authors' code on 64 series | Section 4.1 |

T4 is the strongest test. With the authors' details switched on, our code produces **the same outputs** as the authors' model on the same weights. The architecture and inference path are therefore correct, and any difference between the main build and the authors' code comes from details the paper does not describe, not from implementation bugs.

## 4. Results

**Setup.**
- **Data:** all 64 series of the five TSB-AD-U families used in the paper: MGAB (9), SED (3), SVDB (20), IOPS (17), TODS (15).
- **Seeds:** 46, 47, 48.
- **Metrics:** F1, AUC-PR, R-AUC-PR and VUS-PR (×100) from the TSB-AD evaluator. VUS-PR is the primary metric because it needs no threshold and is robust to small boundary shifts.
- **Statistics:** comparisons use paired Wilcoxon tests over series.
- **Runs:** 384/384 completed.

### 4.1. Comparison with Table 1

| Metric | Table 1 | **Main build** (± seed SD) | Gap to Table 1 | Extended build | Authors' code (1 run) |
|---|---|---|---|---|---|
| F1 | 49.13 | **45.99 ± 0.46** | −3.14 (−6.8 SD) | 45.62 ± 0.69 | 45.99 |
| AUC-PR | 43.66 | **39.81 ± 0.62** | −3.85 (−6.2 SD) | 39.87 ± 1.24 | 40.43 |
| R-AUC-PR | 40.50 | **39.63 ± 1.06** | −0.87 (−0.8 SD) | 42.80 ± 0.91 | 42.73 |
| VUS-PR | 39.83 | **38.72 ± 1.00** | −1.11 (−1.1 SD) | 41.69 ± 0.96 | 41.70 |

**The main build is closest to Table 1.** Its total absolute gap over the four metrics is 8.97, against 11.46 for the extended build.
- On the two range-based metrics (R-AUC-PR, VUS-PR) the gap is about 1 point, within normal seed variation.
- On the two point-based metrics (F1, AUC-PR) the main build is 3–4 points lower, more than 6 seed SDs. This gap is real, **not random**: the three seeds vary by only about ±0.5 points, so no choice of seed would lift F1 from 46 to 49.
- The paper reports only means, not per-series scores, so **a Wilcoxon test against the paper is not possible**. Every Wilcoxon test in this report compares two sets of results for which we have per-series scores (e.g. the extended build against the authors' code). The gap to Table 1 is therefore measured in seed SDs.
- The authors' own code deviates from Table 1 in the same way (last column and Section 5.1).

**The extended build matches the authors' code.**
- No metric differs significantly (p = 0.30, 0.30, 0.75, 0.56).
- On 32/64 series, the authors' single run falls inside our 3-seed min–max range; about 50% is expected if both are the same model.
- So our code reproduces the authors' code both in its outputs (T4) and in its final results.

### 4.2. By seed (main build)

| Seed | F1 | AUC-PR | R-AUC-PR | VUS-PR |
|---|---|---|---|---|
| 46 | 45.48 | 39.22 | 38.45 | 37.60 |
| 47 | 46.37 | 40.46 | 39.95 | 39.06 |
| 48 | 46.12 | 39.77 | 40.51 | 39.51 |

**The 64-series mean is stable; single series are not.**
- The mean varies little: VUS-PR moves by 1.9 points across the three seeds.
- At the level of single series, the VUS-PR range across the three seeds has a median of 6.8 points and a maximum of 45.7. 19/64 series vary by more than 10 points.
- GPU computation is also not fully deterministic: rerunning **the same seed** can change one series by up to 23.7 points.

**Consequence:** conclusions about single series or small families (SED has only 3 series) need several seeds. A single run, such as the authors' code, is only a reference for trends. Table 1 averages 5 seeds.

### 4.3. By family (mean of 3 seeds)

| Family | F1 | AUC-PR | R-AUC-PR | VUS-PR | VUS-PR, extended | VUS-PR, authors' code |
|---|---|---|---|---|---|---|
| MGAB (9) | 39.53 | 25.15 | 19.22 | **18.44** | 44.73 | 41.71 |
| SED (3) | 73.60 | 75.01 | 91.38 | 88.30 | 91.97 | 93.19 |
| SVDB (20) | 60.48 | 57.35 | 60.11 | **58.97** | 54.74 | 56.09 |
| IOPS (17) | 36.83 | 30.37 | 29.30 | 28.62 | 31.48 | 32.50 |
| TODS (15) | 35.41 | 28.90 | 25.95 | 25.44 | 23.99 | 22.62 |

The overall mean is close to Table 1, but the families are not:
- **MGAB:** the main build reaches only 18.44 VUS-PR, 23 points below the authors' code.
- **SVDB and TODS:** the main build is above the authors' code (+2.9 and +2.8).
- **SED and IOPS:** slightly below.

The overall mean is close to Table 1 partly because these differences **offset each other**. A matching mean is therefore not enough to say the main build "reproduces" CoAD.

## 5. Analysis of the Results

### 5.1. Why F1 and AUC-PR are below Table 1

All three results (main build, extended build, authors' code) are 3–4 points below Table 1 on F1 and AUC-PR. Since the authors' own code shows the same gap, **this gap is not caused by our implementation**.

The table separates three sources: the numbers published in the paper, the authors' code rerun by us (commit `3286813`, same 64 series, same evaluator, one run, seed 46), and our builds.

| Metric | Paper (Table 1) | Authors' code | Authors' code − paper | Main build (paper) | Extended build (code) | p (extended vs authors' code) |
|---|---|---|---|---|---|---|
| F1 | 49.13 | 45.99 | **−3.14** | 45.99 ± 0.46 | 45.62 ± 0.69 | 0.30 |
| AUC-PR | 43.66 | 40.43 | **−3.23** | 39.81 ± 0.62 | 39.87 ± 1.24 | 0.30 |
| R-AUC-PR | 40.50 | 42.73 | +2.23 | 39.63 ± 1.06 | 42.80 ± 0.91 | 0.75 |
| VUS-PR | 39.83 | 41.70 | +1.87 | 38.72 ± 1.00 | 41.69 ± 0.96 | 0.56 |

How to read it:
- **Paper vs authors' code:** the authors' own code is about 3 points below the paper on F1 and AUC-PR, but about 2 points above it on R-AUC-PR and VUS-PR.
- **Extended build vs authors' code:** no difference on any of the four metrics (paired Wilcoxon over 64 series). This build reproduces the code.
- **Main build vs paper:** its closeness on VUS-PR and R-AUC-PR (about 1 SD) is partly a coincidence. The main build is about 3 points below the authors' code on these metrics (it lacks the training details, Section 5.3), while the authors' code is about 2 points above the paper.
- **KDD21 as a control:** the same authors' code, run on 246/250 KDD21 series, matches Table 1 on all four metrics (gap < 0.3, Section 6.5). The gap is therefore confined to the TSB-AD part of Table 1.

We also ruled out our evaluation pipeline. Rerunning four Table 1 baselines through the same pipeline reproduces the paper:

| Baseline | VUS-PR gap to Table 1 |
|---|---|
| SAND | 0.01 |
| PCA | 0.07 |
| MP | 0.54 |
| DAMP | 0.39 |

For the authors' code the gap even **changes sign** between metric types: point-based metrics are below Table 1, range-based metrics above. A model that is simply better or worse usually moves all four metrics in the same direction. Table 1 was therefore most likely produced by a procedure (training, model selection or post-processing) that differs from the public code. Section 6 tests two concrete hypotheses, score smoothing and early stopping. Neither explains the gap.

### 5.2. Why MGAB is low and SVDB high

To understand the family differences, we computed VUS-PR using each branch alone (seed 46):

| Build | Scope | Full score | Classifier only | Reconstruction only |
|---|---|---|---|---|
| Main | 64 series | 37.60 | 33.98 | **47.61** |
| Extended | 64 series | 40.61 | 36.99 | 40.88 |
| Main | MGAB | 13.1 | 12.2 | **43.4** |
| Extended | MGAB | 41.3 | 41.2 | 22.0 |

**In the main build, reconstruction alone beats the full model by about 10 points.** The two scores are combined without putting them on a common scale, so the weak classifier pulls the combined score down. The extended build's scoring (subtracting the training mean, averaging the two branches) does not change the scale either. Adding only that scoring to the main build gains just +0.5 on MGAB. The difference lies in the quality of the classifier, not in the fusion.

A pre-registered test confirms this. On seeds 47–48, the main build using reconstruction only scores 44.76, above the extended build (42.23): 41/64 series improve, p = 0.019. The prediction held.

**MGAB.**
- **Main build.** On MGAB the classifier is very weak (12.2), while reconstruction alone reaches 43.4. The final score almost coincides with the classifier branch, so it drops to 13.1 (seed 46).
- **Extended build.** The classifier is strong (41.2, close to the final 41.3) and carries MGAB. It owes this to the code's training details (batch composition, loss balancing, class weights, learning-rate schedule, fp16) together with the architecture details (Section 5.3).

**Other families** (extended build, seed 46):
- SVDB and IOPS rely mainly on reconstruction. SVDB: full 53.4, classifier 51.0, reconstruction 63.1. IOPS: 31.5 / 22.5 / 33.4.
- On TODS both branches are weak: 23.2 / 21.2 / 23.4.
- The main build has a stronger reconstruction branch, so it scores above the authors' code on SVDB.

### 5.3. Which differences matter (MGAB, 9 series × 3 seeds)

To find which code details create the 26-point MGAB gap, we split the 21 differences into four groups. We then removed one group at a time from the extended build, or added one group to the main build:

| Configuration | VUS-PR | Change | p |
|---|---|---|---|
| Extended build | 44.73 | — | — |
| − training details (clean + distorted batches, loss balancing, class weight, LR schedule, fp16, loss scale) | 19.00 | −25.73 (0/9 series improve) | 0.004 |
| − architecture details (GRU layers, dropout, MLP head, hidden size, SELU) | 30.68 | −14.04 (0/9) | 0.004 |
| − synthetic-anomaly details | 38.76 | −5.96 | 0.164 |
| − scoring details (two-pass inference, relative score) | 43.59 | −1.13 | 0.074 |
| Main build | 18.44 | — | — |
| Main + training details | 29.64 | +11.20 (8/9) | 0.012 |

**The training details are necessary.** Removing them from the extended build drops MGAB from 44.73 to 19.00, the level of the main build, and all 9 series fall. These are exactly the details the paper leaves out.

**They are not sufficient.**
- Adding them alone to the main build recovers only 11.2 of the 26.3-point gap.
- The architecture details also matter: removing them costs 14 points. Yet adding them alone to the main build *lowers* MGAB by 7.84 (to 10.60).
- The two groups interact, and both are needed to reach the code's result.

**Consequence:** a reader with only the paper reaches less than half of the code's score on MGAB.

## 6. Comparison with the Paper: Extended Experiments

Experiments 6.1–6.4 each have a **prediction written before running**, with a pass/fail rule. Except for Sections 6.2 and 6.5, which run the authors' code, they use the extended build. Because that build matches the authors' code, gaps to Table 1 are not mixed up with implementation differences.

### 6.1. Does score smoothing explain the gap? (64 series × 3 seeds, 192 models)

The authors' repository added score smoothing for TSB-AD in a later commit, so Table 1 may have been computed without it. We rescored the same models with and without smoothing.

| Metric | Table 1 | With smoothing | Without | p |
|---|---|---|---|---|
| F1 | 49.13 | 45.62 | 43.80 | 0.103 |
| AUC-PR | 43.66 | 39.87 | 38.31 | 0.101 |
| R-AUC-PR | 40.50 | 42.80 | 37.73 | 0.004 |
| VUS-PR | 39.83 | 41.69 | 36.85 | 0.004 |

- **Pre-registered prediction:** removing smoothing moves all four metrics towards Table 1.
- **Result: rejected.** All four decrease, and F1 and AUC-PR move further from Table 1. Smoothing does not explain the gap.

### 6.2. Table 2 ablations (authors' code, seed 46, 64 series, VUS-PR)

We ran three ablation variants available in the authors' repository and compared them with the full model:

| Variant | VUS-PR | Diff. vs full model | Paper | Same direction as paper? |
|---|---|---|---|---|
| Mean instead of max to fuse time/frequency | 36.84 | −4.86 (p 0.061) | −6.65 | Yes, not significant |
| Hard mask instead of soft | 39.30 | −2.40 (p 0.319) | −5.07 | Indistinguishable from seed noise |
| No frequency branch | 43.60 | **+1.91** (p 0.077) | −2.91 | **No, opposite** |

- **Mean fusion:** goes in the paper's direction at a similar size, and AUC-PR drops significantly (−4.27, p = 0.006).
- **Hard mask:** goes in the paper's direction but is much smaller, and within seed noise.
- **No frequency branch:** the score **rises**, on 40/64 series. We could not reproduce the claim that the frequency branch helps on TSB-AD.

### 6.3. KDD21 column of Table 1 (extended build, 30 series, seed 46)

Table 1 also reports CoAD on KDD21. The authors' KDD21 loader reads preprocessed files that are not published. We therefore used the `UCR` family of TSB-AD-U, which contains the KDD21 series repackaged by TSB-AD.

We checked in the evaluator code that the `use_ucr` flag the authors use for KDD21 only adds a separate metric and leaves the four Table 1 metrics unchanged.

**Sample.** 30 series drawn with `default_rng(2028)`:
- length ≤ 100,000 points;
- duplicate files removed;
- no overlap with the external set of Section 9.

| Metric | Table 1 | Ours (mean) | Median | 95% CI (bootstrap) | Table 1 inside CI? |
|---|---|---|---|---|---|
| F1 | 52.82 | 47.54 | 41.16 | [35.23, 60.03] | Yes |
| AUC-PR | 48.10 | 45.69 | 33.48 | [31.42, 60.22] | Yes |
| R-AUC-PR | 46.35 | 42.42 | 24.04 | [29.06, 56.32] | Yes |
| VUS-PR | 45.67 | 41.86 | 24.91 | [28.67, 55.00] | Yes |

- **Pre-registered prediction:** Table 1's VUS-PR lies outside our 95% CI.
- **Result: rejected.** Our means are 2.4–5.3 points below Table 1 on all four metrics, but the CIs are about 26 points wide.
- **Why the CIs are wide:** the VUS-PR scores are clearly bimodal, with 16 series below 26 and 14 above 60. With 30 of 250 series, the mean cannot be estimated precisely enough.

**Conclusion:** there is no evidence of a gap on KDD21. The statement "Table 1 is not reproduced" applies only to its TSB-AD half.

**Update:** we later ran the authors' code on all 250 KDD21 series (Section 6.5). The results match Table 1 on all four metrics, confirming this conclusion.

### 6.4. Early stopping (extended build, MGAB, 9 series × 3 seeds)

The paper says the model is selected by early stopping, but the authors' training script runs a fixed number of epochs without validation-based selection. This is a 22nd difference between paper and code.

**Design.**
- **Split:** the paper does not define the validation loss, so we chose a label-free rule. The training part is split in time: the first 80% for training, the last 20% for validation.
- **Validation loss:** CoAD's own training loss on a fixed set of distorted batches from the last 20%.
- **Stopping rule:** training stops after 10 epochs without improvement, and the best epoch is restored.
- **Control:** `cut80` trains on the same 80% without early stopping. It separates the effect of early stopping from the effect of training on less data.

| Variant | Trained on | Early stopping | VUS-PR | F1 | AUC-PR | R-AUC-PR | Epochs |
|---|---|---|---|---|---|---|---|
| Reference | 100% | No | 44.73 | — | — | — | 300 |
| `cut80` | first 80% | No | 44.98 | 48.13 | 37.71 | 46.09 | 300 |
| `es` | first 80% | Yes (patience 10) | 36.40 | 44.60 | 33.44 | 37.51 | 188 (best: 179) |

- **Pre-registered prediction:** the difference between `es` and `cut80` is below the MGAB seed SD (3.57).
- **Result: rejected.**
  - **Size:** `es` − `cut80` = −8.57 VUS-PR. Only 8/27 runs improve (p = 0.025 over 27 pairs; p = 0.055 over the 9 series means), and all four metrics decrease.
  - **Not less data:** training on 80% of the data costs nothing (`cut80` − reference = +0.25, p = 0.88), so the drop is caused by early stopping itself.
  - **By seed:** the effect varies, −19.7 / −2.4 / −3.6.

**Interpretation.**
- **Why early stopping hurts.** The validation loss on synthetic anomalies stops improving at around epoch 180. Detection of real anomalies keeps improving with further training: `cut80`, trained for the full 300 epochs, does better. This loss is therefore a poor criterion for model selection. This is an interpretation, not tested separately.
- **Why it does not explain Table 1.** Early stopping lowers F1 and AUC-PR, whereas Table 1 has them *above* the code. With our rule, early stopping alone does not explain the pattern of the gap.

### 6.5. Rerunning all of Table 1 with the authors' code

The previous sections checked only the CoAD row. This section reruns the other rows of Table 1 with the code in the authors' repository, to see whether the baseline numbers reproduce.

**KDD21 data.** The authors do not publish their preprocessed KDD21 files, only `all_length.csv` (train, test and anomaly lengths and the period of each series). We rebuilt the data from the original UCR Anomaly Archive 2021:
- the training part is everything before the split point in the file name, the test part is the rest, and the label is the anomaly range in the file name;
- min–max scaling on the training part, as in the authors' TSB-AD loader (an assumption, since no preprocessing code is published);
- checked against `all_length.csv`: **250/250 series match exactly** in train, test and anomaly length. This required two special cases that the file reveals: series 239–241 (900,000 points) are downsampled by 10, and the 1-point anomalies of series 079, 108 and 187 are widened to 3 points.

**CoAD on KDD21** (authors' code, settings of `run_all_CoAD.sh`, 246/250 series; the last 4 are still running):

| Metric | Table 1 | Authors' code | Difference |
|---|---|---|---|
| F1 | 52.82 | 52.70 | −0.12 |
| AUC-PR | 48.10 | 47.96 | −0.14 |
| R-AUC-PR | 46.35 | 46.58 | +0.23 |
| VUS-PR | 45.67 | 45.94 | +0.27 |

**The KDD21 column of Table 1 reproduces**: all four metrics are within 0.3 points. This is an important control for Section 5.1. The same code matches on KDD21 but is 3–4 F1/AUC-PR points lower on TSB-AD. The gap therefore lies in the TSB-AD part of Table 1 only, not in the model or the evaluator.

**14 deep-learning baselines** (64 TSB-AD series and 250 KDD21 series, one run per series, VUS-PR):

| Method | TSB-AD: ours | Table 1 | Diff. | KDD21: ours | Table 1 | Diff. | Epochs |
|---|---|---|---|---|---|---|---|
| DADA | 14.03 | 14.53 | −0.50 | 2.08 | 2.05 | +0.03 | zero-shot |
| MOMENT | 24.55 | 24.67 | −0.12 | 9.40 | 9.13 | +0.27 | zero-shot |
| TFMAE | 6.43 | 5.77 | +0.66 | 1.86 | 1.75 | +0.11 | 0 (training disabled in the script) |
| COUTA | 10.62 | 11.01 | −0.39 | 2.62 | 3.82 | −1.20 | 1 |
| M2N2 | 9.12 | 9.25 | −0.13 | 2.77 | 3.18 | −0.41 | 1 |
| TFAD | 5.88 | 5.99 | −0.11 | 1.56 | 1.58 | −0.02 | 1 |
| TranAD | 12.28 | 12.05 | +0.23 | 2.74 | 7.89 | −5.15 | 1 |
| DCdetector | 5.34 | 5.69 | −0.35 | 1.90 | 1.85 | +0.05 | 10 (default) |
| FITS | 11.49 | 10.37 | +1.12 | 2.76 | 3.45 | −0.69 | 50 (default) |
| AnomalyTrans | 13.10 | 6.13 | +6.97 | 8.56 | 1.80 | +6.76 | 10 (default) |
| CATCH | 14.14 | 21.67 | −7.53 | 2.38 | 9.19 | −6.81 | 1 |
| FCVAE | 13.97 | 20.58 | −6.61 | 2.62 | 6.38 | −3.76 | 1 |
| MAUT | 10.34 | 14.75 | −4.41 | 2.92 | 23.65 | −20.73 | 1 |
| CutAddPaste | 11.52 | 25.08 | −13.56 | 4.14 | 18.20 | −14.06 | 1 |

How we ran them:
- The authors' public `deep_baseline_compare.py` **does not run** as published: the imports of 4 models are disabled, several calls use wrong function signatures, and TranAD and MAUT are in the runner but have no code branch. We rewrote the runner, keeping the script's data, windows, scoring and evaluator.
- The script sets **1 epoch** for every model. We ran exactly that setting (the "script" protocol). DCdetector, AnomalyTrans and FITS are not in the authors' runner, so they use default settings (DCdetector from the authors' repository; AnomalyTrans and FITS from the TSB-AD library).
- Running on the current software stack required a few compatibility fixes (NumPy 2, PyTorch 2, recent transformers). They do not change the computation. For CATCH and TFAD on series with a very short training part (TODS), the window is shrunk to fit the validation split; this is recorded per run.

Observations:
- **8 of 14 methods match Table 1 within ±1.2 points on both benchmarks**, and TranAD matches on TSB-AD. Notably, the models trained in the runner (COUTA, M2N2, TFAD, TranAD on TSB-AD) match after only **1 epoch**, and TFMAE matches with **no training**.
- This contradicts Appendix C.2, which states that every deep baseline was trained with its official repository, with early stopping, averaged over 5 seeds. Our results suggest that many baseline numbers in Table 1 were produced with the 1-epoch script. This is an inference from the agreement, not direct evidence; a "default" protocol (each model's default epochs) is prepared to test it but has not been run.
- The methods with large gaps (CutAddPaste, MAUT, CATCH, FCVAE, TranAD on KDD21) are all **below** the paper, and all have large default epoch counts (20–300). The paper most likely trained them for longer than 1 epoch, so Table 1 does not use one consistent setting for all baselines.
- AnomalyTrans is above the paper because we used TSB-AD's tuned implementation and hyperparameters.
- MMA, AnomalyBERT and TriAD have no code in the repository and the paper does not describe how they were run, so they are not reproduced.

**7 data-mining baselines.**
- On TSB-AD (64 series, run earlier): SAND 31.06 (Table 1: 31.05), Sub-PCA 23.92 (23.85), DAMP 12.99 (12.60), MP 29.48 (28.94), Series2Graph 31.44 (29.57), KMeansAD 41.01 (36.81), KShapeAD 35.86 (31.27). Five match within 2 points; KMeansAD and KShapeAD are about 4 points above the paper.
- On KDD21: about 75 of 250 series are done (the shortest series run first), too few to compare. Series2Graph failed under NumPy 2 and has been patched.

**Table 2 (ablations).** Besides the three variants in Section 6.2, the notebooks are ready to run the 9 variants of the authors' code on KDD21 and the 5 rows the code lacks. These five follow the definitions in the paper's Appendix G: OE alone (Time), OE alone (Time + Frequency), MAE alone (Random), MAE alone (Grating), Guide w/o Score. This part has not been run yet.

### 6.6. Summary: agreement with the paper

| Item | Result | Explanation |
|---|---|---|
| Table 1, R-AUC-PR and VUS-PR (TSB-AD) | ✅ Close (main build, within about 1 SD) | Partly through offsetting family differences |
| Table 1, F1 and AUC-PR (TSB-AD) | ❌ 3–4 points lower | The authors' code shows the same gap; not the pipeline, not smoothing, not early stopping |
| Table 1, CoAD on KDD21 | ✅ Match (authors' code, 246/250 series, < 0.3 on all 4 metrics) | The gap exists only in the TSB-AD part |
| Table 1, 14 deep baselines | 8/14 match on both benchmarks | Match when trained for 1 epoch as in the authors' script; gaps for models that need long training |
| Table 1, 7 data-mining baselines (TSB-AD) | 5/7 match; KMeansAD and KShapeAD about 4 points higher | Same pipeline as the authors |
| Table 2: mean fusion | ✅ Same direction, similar size | Not significant on VUS-PR (p = 0.061) |
| Table 2: hard mask | Same direction, much smaller | Indistinguishable from seed noise |
| Table 2: no frequency branch | ❌ Opposite (+1.91) | The frequency branch does not help on TSB-AD |
| Rebuildable from the paper alone | ❌ No (MGAB 18 vs 45) | Training procedure and some architecture details missing |

## 7. Error Analysis

This section asks where CoAD misses anomalies, and why. We use the extended build, which is the authors' model, so the conclusions apply to CoAD itself.

**When an event counts as detected** (diagnosis only, uses labels): an anomaly event counts as **detected** by a signal if the signal's peak inside the event is at or above that signal's 99th percentile over normal points. The qualitative conclusions hold at the 95% and 99.9% thresholds.

### 7.1. Where the model misses (8,262 events, 192 models)

We traced each event through the scoring steps:
- the raw signal of each branch;
- per-branch processing (subtracting the training mean and smoothing each branch separately; column "processing");
- fusion of the two branches;
- smoothing.

Each event is assigned to the step that loses it.

| Family | Detected | Lost at smoothing | Lost at fusion | Lost at processing | No signal detects it |
|---|---|---|---|---|---|
| TODS | 0.021 | 0.312 | 0.006 | 0.139 | **0.522** |
| IOPS | 0.421 | **0.452** | 0.009 | 0.050 | 0.067 |
| MGAB | 0.713 | 0.073 | 0.011 | 0.038 | 0.165 |
| SVDB | 0.781 | 0.158 | 0.011 | 0.043 | 0.007 |
| SED | 0.818 | 0.174 | 0.000 | 0.008 | 0.000 |

There are two types of error.

**Type 1: the signal exists, but scoring removes it.**
- **Most classifier misses are still in the raw error.** The classifier misses 5,485 of 8,262 events. For these, the raw reconstruction error still detects 53%, but after all scoring steps the final score detects only 7%.
- **IOPS loses most at smoothing:** 45% of IOPS events are lost there.
- **Order-free attribution.** The step order can affect the attribution, so we also computed order-free Shapley shares (64 series, seed 46, 3 thresholds).
  - In all 8 cells with enough data (IOPS, SVDB, SED), **smoothing accounts for 54–100% of the loss**; in TODS and MGAB, 70–98%.
  - Fusion accounts for 15–46% in IOPS and SVDB, highest in SVDB.
- **Complementary branches, lost information.** The two branches are nearly uncorrelated over events (ρ = −0.089), so they complement each other as CoAD intends. The scoring stage discards much of that complementary information.

**Type 2: no branch has a signal.**
- **Where:** for 52% of TODS events and 17% of MGAB events, no branch reaches the threshold even before any processing.
  - TODS anomalies are small: median amplitude 0.5σ, half of them below 0.5σ.
  - MGAB anomalies are subtle changes in chaotic dynamics.
- **Training contamination is not the cause.** We tested whether contaminated TODS training data made the model learn anomalies as normal. We repaired the known anomaly points in the training part (using labels, median 3.2% of points) by interpolation and retrained (15 series × 3 seeds). VUS-PR changed by only +0.07.

**Two of our initial hypotheses were also rejected:**
- *"Cooperation fails when the classifier misses":* rejected. Reconstruction error still recovers 44–59% of those segments before smoothing.
- *"Long anomalies are harder":* rejected. In a stress test, the mean percentile of the final score drops by only 0.022 when the segment grows from 1 to 4 periods.

### 7.2. Short anomalies: controlled tests

To confirm how the signal is lost, we inserted rectangular spikes of 1–16 points into real series at amplitudes 3σ and 1σ, and measured detection at each step. Results for 1-point spikes (3σ / 1σ):

| Classifier | Raw reconstruction error | Unsmoothed score | Final score |
|---|---|---|---|
| 0.34 / 0.25 | 0.95 / 0.73 | 0.88 / 0.68 | 0.71 / 0.37 |

- **The classifier is weak on short anomalies.** Detection rises by 0.32–0.34 as spikes get longer. This may be because training distortions span 10–100% of a period; it was not tested.
- **The reconstruction error catches almost all 1-point spikes**, but the final score loses most of them.
- **Two mechanisms explain the loss for isolated 1σ spikes.** Detection drops from 0.66 to 0.03, measured against the spiked series itself:
  - **dilution**, −0.36: the peak is averaged over a whole period;
  - **spill-over**, −0.29: normal points around the spike are raised too.
- **Density adds no loss:** many nearby spikes cause no extra loss in the final score.
- **The patch grid is negligible:** placing the spike mid-patch or at a patch edge changes detection by only −0.03 to +0.03.

**Conclusion:** short anomalies are lost mainly to the **one-period moving average**, not to patching.

### 7.3. Is the core mechanism faulty?

We tested five hypotheses about weaknesses in CoAD's cooperation mechanism:

| Hypothesis | Measurement | Result |
|---|---|---|
| The classifier relies on segment boundaries | Probability in the middle − at the ends | +0.065: rejected (opposite sign) |
| The classifier learns a shortcut from the jump at the boundary | Stress test inserting the same anomaly with abrupt or smooth (half-cosine) transitions; compare the OE percentile in the middle | +0.002 (480 pairs, p = 0.13): rejected |
| The soft mask leaks anomaly information into the reconstruction | Compare with hard masking | Hard masking lowers AUC by 0.097: the soft mask is selective |
| The MAE borrows anomaly information from neighbouring patches | Mask 3 consecutive patches around the centre vs the centre plus 2 distant patches; compare the error increase at anomalous vs normal centres | Mean −0.015, median +0.049 (177 models, p = 0.11): inconclusive |
| Gradient conflict between the classification and reconstruction objectives | Cosine between the two gradients; stopping gradients | Mean cosine +0.19, positive in all 9 runs (MGAB, seed 46): no conflict. Stopping gradients lowers VUS-PR by 11.5 (p = 0.074), a trend only |

### 7.4. Conclusion of the error analysis

CoAD's errors are **not in the cooperation mechanism**. The two branches complement each other, the soft mask masks the right places, and joint training shows no conflict.

The errors are **in the scoring stage**:
- the fixed one-period moving average erases most of the signal of short anomalies;
- the fusion of the two branches loses some more.

The rest are anomalies too small (TODS) or too subtle (MGAB) for either branch to see.

## 8. Comparison with Baselines

### 8.1. On the paper's 64 series (VUS-PR, Holm correction over 7 comparisons)

The baselines come from TSB-AD and run through the same pipeline. The rank and pairwise table uses the single run of the authors' code (VUS-PR 41.70); the conclusions are unchanged with the 3-seed mean of the extended build. The family table uses the extended build, 3 seeds.

| Baseline | Mean rank | CoAD wins / losses | Diff. | 95% CI | Holm p |
|---|---|---|---|---|---|
| KMeans | 2.61 | 25 / 39 | +0.69 | [−5.30, 7.19] | 0.463 |
| KShapeAD | 3.70 | 33 / 31 | +5.84 | [−1.66, 13.45] | 0.463 |
| SAND | 4.50 | 36 / 28 | +10.64 | [3.88, 17.71] | 0.060 |
| Series2Graph | 4.66 | 37 / 27 | +10.26 | [2.82, 18.02] | 0.118 |
| MP | 4.78 | 40 / 24 | +12.21 | [5.85, 18.82] | 0.003 |
| PCA | 5.50 | 48 / 16 | +17.77 | [11.30, 24.81] | < 0.001 |
| DAMP | 6.44 | 49 / 15 | +28.71 | [22.09, 35.61] | < 0.001 |

**Highest mean, third place on ranks.** CoAD has the highest mean VUS-PR, but on per-series ranks it is only 3rd (mean rank 3.81), behind KMeans and KShapeAD.

**CoAD is not significantly better than KMeans.** Its mean advantage comes almost entirely from MGAB:

| Family | CoAD | KMeans | KMeans wins |
|---|---|---|---|
| MGAB | 44.7 | 1.0 | 0/9 |
| SED | 92.0 | 92.5 | 3/3 |
| IOPS | 31.5 | 32.5 | 11/17 |
| SVDB | 54.7 | 62.5 | 14/20 |
| TODS | 24.0 | 35.7 | 14/15 |

The paper claims that CoAD outperforms other methods, based on means only and without statistical tests.

### 8.2. On 58 external series (12 other families)

On series from 12 families not used in the paper (the 60-series external set of Section 9). Only 58 of 60 series are used, because KShapeAD fails on two very short series (NEK 277, YAHOO 614); the comparison uses the series where all three methods have results.

| Method | VUS-PR |
|---|---|
| CoAD | 52.36 |
| KMeans | 58.96 |
| KShapeAD | 23.28 |

- **CoAD trails KMeans** by 6.61 points (two-sided p = 0.042).
- **Without the 4 excluded families.** The paper excludes NAB, WSD, YAHOO and Stock for data quality. Without them, CoAD trails by 2.28, which is not significant (p = 0.73).
- **Caveat:** KMeans is fit on the test series itself (transductive), which favours it.

## 9. Improvements Based on the Error Analysis

Based on Section 7, we tested improvements aimed at the places where CoAD loses signal. All are **label-free**.

**Protocol against tuning on the evaluation data.**
- **Split:** the 64 series are split into 30 *design* and 34 *confirmation* series, fixed with `default_rng(2026)`.
- **Locking and pre-registration:** all choices (variants, parameters) are locked on the design set. Predictions and pass/fail rules are written before running on the confirmation set.
- **External validation:** 60 series from 12 other TSB-AD-U families, 5 per family, drawn with `default_rng(2027)`, duplicate files removed. The families are YAHOO, UCR, WSD, SMD, Exathlon, OPPORTUNITY, NAB, Stock, SMAP, LTDB, MSL and NEK.

| Direction (error it targets) | Confirmation (34 series) | External (60 series) | Verdict |
|---|---|---|---|
| `calibrated`: divide each branch by its training SD before fusion (fusion) | +3.42 vs CoAD (Holm 0.078) | +1.44 (p 0.44); +0.20 without the 4 families | Not general |
| `calibrated` + doubled reconstruction weight (mechanism confirmed: ρ = −0.46, p = 0.003) | +0.80 vs `calibrated` (Holm 0.004); +4.22 vs CoAD (p 0.001) | +0.81 vs `calibrated` (p 0.0006, positive in 9/12 families); +2.25 vs CoAD (p 0.17) | Beats `calibrated` consistently; not general vs CoAD |
| Replace the moving average with `mix` = mean of instant and smoothed score (smoothing) | +4.79 vs CoAD (Holm 0.001); TODS +8.4 (confirmation set already seen in M11) | +1.62 (p 0.16); −1.18 without the 4 families | Helps short anomalies only (ρ = −0.44) |
| Rarity-based fusion against the normal distribution (4 variants) | All below `calibrated` | — | Negative |
| MAD instead of SD scaling | −1.37 vs `calibrated` | — | Negative |
| OE-independent residual branch (4 heads inspired by TranAD, DRAEM…) | No head beats the old residual | — | Negative |
| Context branch (checked with a splice test) | The test flagged its own control, so it is uninformative | — | No basis to build it |
| Repairing contaminated training data (TODS) | +0.07 | — | Negative |

**Observations.**
- **Gains on the paper's data.** The error analysis pointed to the right place (the scoring stage), and the scoring changes all gain on the paper's data.
- **No reliable gain on the 12 new families.** No change beats CoAD reliably there. The mean gains come from a few series: two WSD series gain +53.5 and +31.9, and without WSD the gain of `calibrated` is −0.02.
- **One consistent gain.** Up-weighting the reconstruction branch is the only change that beats `calibrated` consistently, both on the confirmation set and on the 12 new families.

**Methodological lessons.**
- Gains on the data used to find an improvement must be validated on new data.
- A diagnosis must be checked for robustness before an improvement is built on it. Example: "the residual branch is blind exactly where the classifier is blind" gave 11% detection under one measurement but 55% under another. The improvement direction based on it therefore had no basis.

**Conclusion on the improvements.** The error analysis located the signal loss correctly, and fixing it raises the scores on the paper's data. But the gains mostly exploit properties of these 5 families: on new data, no variant beats the original CoAD significantly. We therefore **do not propose replacing CoAD** with any variant; `calibrated` is kept in the demo as an analysis example, with cases where it is better and cases where it is worse.

## 10. Web Demo

We built a web demo (folder `demo/`) following the specification `CoAD_Web_Demo_Error_Analysis_Spec.md` and the design `CoAD_Web_UI_Design.md`, to present the results on real data.

**Data.**
- The demo uses **the benchmark's own GPU checkpoints** (code mode, seed 46), with no retraining. The recomputed scores match the saved scores within 4×10⁻⁶.
- Five series: 237 SVDB and 276 IOPS (`calibrated` clearly better), 234 SED (about equal), 269 IOPS and 295 TODS (`calibrated` worse).
- For each series, `precompute.py` exports every intermediate signal: the probabilities `A`, `Aʳ` and `A_c`, the reconstruction and its error, each branch score before and after smoothing, the final scores of the original CoAD, `calibrated` and KMeansAD, and the patch-level probabilities of every window.

**Features.**
- *Data:* an overview of the paper's 64 series, the 60 external series and KDD21 (lengths, training part, number of events, anomaly ratio, design/confirmation split, which series have a checkpoint).
- *Series analysis:* input, ground truth, predictions, scores with each method's own threshold, and the reconstruction. Zoom and cursor are synchronised across charts; there are tables of events and false-alarm intervals, and a patch-level Model Inspector that shows the soft-mask embedding mixing weights.
- *Error analysis:* filters for missed, false-alarm, recovered and newly missed cases. Each case has a signal trace through every stage, compared with that stage's own reference threshold. Observations are worded without claiming causes.
- *Experiment results:* the Δ of `calibrated` vs CoAD on the 34 confirmation series. The design set was only scored with a quick AUC-PR during selection, so it is shown as N/A rather than counted as 0.
- CSV upload is for validation and preview only: there is no general model, because each checkpoint is trained on one series.

**Thresholds and rules.**
- The default threshold comes from the training part (μ + 3σ of the score); no test labels are used.
- The event rules (hit by overlap, coverage, false-alarm intervals, recovered/newly missed) live in a single module shared by all screens. The module was checked against an independent count.

**What the demo shows** (train μ + 3σ threshold): `calibrated` recovers 1 of 12 events on series 237 and loses no event on any of the 5 series. But the number of false-alarm intervals rises on 237 (199 → 216) and 276 (583 → 650), and falls on 269 (15 → 10). This illustrates the conclusion of Section 9: the improvement helps in some places and costs in others.

## 11. Conclusions and Limitations

### 11.1. Conclusions

- **A verified reimplementation.** We reimplemented CoAD and verified it with 8 tests. With the authors' details switched on, our outputs match the authors' model (difference < 1e-5), and the results on 64 series do not differ from the authors' code (p ≥ 0.30).
- **The paper-based build is closest to Table 1.**
  - VUS-PR is 38.72 vs 39.83 and R-AUC-PR 39.63 vs 40.50, both within about 1 seed SD.
  - F1 and AUC-PR are 3–4 points lower, exactly as with the authors' own code.
- **The paper alone is not enough to rebuild CoAD.** The mean matches partly through offsetting families. On MGAB the paper-based build scores 18 against 45, because the training procedure is not described.
- **Extended experiments:**
  - score smoothing and early stopping are ruled out as causes of the gap to Table 1;
  - **the KDD21 column of Table 1 reproduces** with the authors' code on 246/250 series (VUS-PR 45.94 vs 45.67), so the gap lies only in the TSB-AD part;
  - the no-frequency-branch ablation goes the opposite way to Table 2.
- **Baselines in Table 1:** 8 of 14 deep-learning baselines match the paper when run exactly as the authors' script, which trains for 1 epoch, although the paper states early stopping and 5 seeds. Baselines that need long training are far below the paper. 5 of 7 data-mining baselines match on TSB-AD.
- **Error analysis:** most of the signal is lost at score smoothing, not in the cooperation mechanism. The rest are anomalies with no signal in either branch.
- **Baselines:** CoAD is not significantly better than KMeans on the 64 series, and it trails KMeans on 58 external series.
- **Improvements:** scoring changes gain on the paper's data but do not generalise to 12 new families. Up-weighting reconstruction is the only change that consistently beats `calibrated`.

### 11.2. Limitations

- **Seeds:**
  - The main results use 3 seeds, whereas Table 1 uses 5. The authors' code was run once as a reference.
  - The Table 2 ablations, KDD21, the baselines, the per-branch analysis and the Shapley attribution use one seed only.
  - The gradient test has 9 runs on MGAB.

  These conclusions need more seeds to be firm.
- **Scope:**
  - the difference groups and early stopping were tested on the 9 MGAB series only;
  - KDD21 is rebuilt from the original archive; the min–max scaling and the downsampling of series 239–241 are assumptions (lengths match 250/250);
  - the early-stopping rule is our choice, since the paper does not describe it.
- **Unfinished parts of Tables 1 and 2:** data-mining baselines on KDD21 (about 75/250 series), 4 CoAD series on KDD21, deep baselines with default epochs, and the KDD21 ablations together with the 5 Table 2 rows the code lacks. MMA, AnomalyBERT and TriAD have no code and cannot be run.
- **Error analysis:** thresholds use labels (diagnosis only); spike tests use synthetic anomalies.
- **Data reuse:** the 64 series were used across many experiments. The design/confirmation protocol applies only to the improvement experiments.
- **External baselines:** on the 60 external series, KMeans and KShapeAD only.
- **Web demo:** precomputed results for 5 series only; no inference for new CSV files.
- **Data:** univariate series only.
- **Our own design errors:**
  - splice test: SED hits the ceiling and the IOPS changes are too small;
  - residual-branch experiment: its threshold was taken from a different measurement.

## Appendix A. The 21 Differences Between the Paper and the Authors' Code (D1–D21)

Authors' code: commit `3286813`. The main build follows the "Paper" column; the extended build follows the "Authors' code" column on all 21 rows. On rows D3, D17, D19 and D21 the paper is ambiguous (e.g. the equations and Figure 2 disagree), so the main build also follows the code; the two builds differ on the remaining 17 rows.

| # | Component | Paper | Authors' code |
|---|---|---|---|
| D1 | GRU layers | 3 | 2 |
| D2 | Dropout between GRU layers | Not mentioned | 0.1 |
| D3 | Time/frequency encoder | Eqs. (1), (3) show two GRUs; Figure 2 says "Shared GRU" | One shared Bi-GRU |
| D4 | Classification head | Linear | 2-layer MLP, SELU |
| D5 | Classifier hidden size | 24 | 3 × 48 = 144 |
| D6 | BCE `pos_weight` | None | 2 |
| D7 | Classification loss | BCE on A_c | Mean of BCE(A) and BCE(Aʳ) |
| D8 | Reconstruction loss | MSE | MSE balanced 50/50 between clean and corrupted parts |
| D9 | Training batch | Not stated | Clean and corrupted copies concatenated |
| D10 | Inference | One pass (Eq. 9) | Two passes, reconstructing again with A_c |
| D11 | Final score | Sum of branches + moving average | Train-mean subtraction, moving average, mean of branches |
| D12 | Uniform replacement | Constant in [min, max] | Value of the segment's first point |
| D13 | Mirror flip | Flip along the x or y axis | Time reversal only |
| D14 | Length scale | Stretch or shrink | Stretch only, factor 2–4 |
| D15 | Jittering | N(0; variance 0.1) | σ = 5% × (max − min) of the window |
| D16 | Learning-rate schedule, precision | Not mentioned | StepLR γ = 0.99; fp16 |
| D17 | STFT bins | K = 4 (ambiguous) | `n_fft = 4`, one-sided spectrum, 3 bins |
| D18 | Activation after patch projection | None | SELU |
| D19 | Mask embedding | E_mask ∈ ℝ^{H×N} | One vector ℝ^H repeated for all patches |
| D20 | Reconstruction-loss scale | Sum within patch, mean over patches | Per-point mean MSE |
| D21 | Corrupted-segment length | "from 0 to one dominant period" | 10–100% of a period |

Impact (Section 5.3, MGAB): removing the training details (batch composition, loss balancing, class weight, learning-rate schedule, fp16, loss scale) from the extended build drops VUS-PR from 44.73 to 19.00; removing the architecture details (D1, D2, D4, D5, D18) drops it to 30.68.
