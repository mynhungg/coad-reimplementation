# CoAD Web UI Demo and Error Analysis Specification

## 1. Goals

This document proposes the features and UI flow for the CoAD course-project demo: running anomaly detection, inspecting intermediate signals, comparing methods, and analyzing errors.

This is a design specification, not a description of a completed system. Features must be limited to what the actual implementation, data and experimental results support.

### 1.1. The story to present

1. What does the input series contain?
2. Which regions does the model detect?
3. How do classification, reconstruction and residual classification respond?
4. How does the signal change through calibration, fusion and smoothing?
5. Which errors does the improved version fix, and which new errors does it introduce?
6. Does the improvement repeat across many series, rather than being a single good-looking example?

### 1.2. MVP scope

- Univariate, if that is the scope of the current implementation.
- Prepared sample data and the corresponding checkpoints.
- Original CoAD and calibrated CoAD already implemented; Kmeans if already integrated.
- Inference and benchmark results may be precomputed.
- Clearly label precomputed results versus freshly run results.
- No training on the web, streaming, accounts or permissions needed yet.

## 2. Transparency principles

- Do not present an unrun variant as a confirmed improvement.
- Do not show metrics that require labels when the data has no labels.
- Do not use test labels to tune the deployed threshold.
- Do not compare scores on different scales with one shared numeric threshold.
- Do not call an observation on a chart a definite cause.
- Do not describe a checkpoint trained separately for each series as a general model for any CSV.
- Do not display missing values as 0; use N/A and explain.
- Distinguish ground truth, prediction, manual review labels and automatic diagnostic labels.
- Every result must be traceable to its configuration, checkpoint, data and run mode.

## 3. Information architecture

| Page/tab | Purpose | Priority |
|---|---|---|
| Detection Playground | Select data, run, view predictions | P0 |
| Error Analysis | Find and inspect errors | P0 |
| Method Comparison | Compare on the same series and events | P0 |
| Model Inspector | Inspect patch probabilities, masking and feature diagnostics | P1; a compact version lives in Error Analysis |
| Experiment Results | Aggregate benchmarks by series/family | P1 |
| About / Reproducibility | Scope, methods and configuration | P0 compact version |

Detection, Inspector and Error Analysis can be merged into one workspace with tabs to avoid duplicated charts.

## 4. Detection Playground

### 4.1. Data sidebar

**Sample data**
- Family/dataset.
- Series ID.
- Number of timestamps.
- Whether labels exist.
- Normal train/calibration/test ranges, if available.
- Compatible checkpoint.
- Case tags: representative, success, failure; these are presentation metadata, not model predictions.

**Upload CSV**
- Select the timestamp column or use the index.
- Select the value column.
- Optionally select a label column.
- Data preview.
- Check numeric type, missing values, duplicate/non-increasing timestamps and minimum length.
- Do not silently impute/resample. If any processing is applied, show the rule and store it in the configuration.

**Model compatibility**
- If no suitable checkpoint exists, block inference and explain why.
- The upload flow for new data needs a separate decision: an actual general model, per-series training, or preview only.

### 4.2. Method selection

- Original CoAD.
- CoAD + calibrated.
- Kmeans if available.
- Experimental variants are labeled Experimental and appear only when valid results exist.

### 4.3. Controls

- Run / Load precomputed result.
- Select an available threshold policy.
- Drag the threshold to explore; mark the result Interactive, and never overwrite the benchmark.
- Reset view.
- Export.

### 4.4. Charts

Three charts sharing the time axis:
1. Raw input + ground-truth intervals + predicted intervals.
2. Final anomaly score + threshold.
3. Input and reconstruction.

Interactions:
- Synchronized zoom/pan.
- Hover shows timestamp/index, input, reconstruction, score, prediction, and label if available.
- Brush to select a range.
- Click an event to open its details.
- Toggle each display layer.
- A button to return to the full series.

Show ground truth and prediction with different colors/background styles. Provide a legend and options that do not rely on color alone.

### 4.5. Summary cards

- Number of flagged points.
- Number of consecutive alert segments.
- Ratio of flagged points.
- Inference time and the scope of the timing measurement.
- Model/checkpoint.
- Precomputed or live inference.
- Label-based metrics; N/A when there is no label.

## 5. Model Inspector

### 5.1. Interactive pipeline

```text
Input
  -> Time/Frequency classification
  -> Soft masking
  -> Reconstruction
  -> Residual classification
  -> Score fusion
  -> Smoothing
  -> Final score
```

The diagram must distinguish parallel branches from the actual dependency order. Do not draw it as if all scores pass through a single linear chain.

### 5.2. Patch-level heatmap

Rows, if the backend stores them:
- Time probability.
- Frequency probability.
- Max-fused probability used for masking.
- Time residual probability.
- Frequency residual probability.
- Fused residual probability.
- Classification probability after combination.

Each column is a patch within the selected window. Show the window ID, patch ID and the corresponding timestamp range.

If a timestamp belongs to several windows, the Inspector must state which window is being viewed. Do not call a patch-window value the single value of a timestamp after aggregation.

### 5.3. Soft masking

When a patch is clicked:
- Probability used for the mask.
- Mask embedding coefficient.
- Input embedding coefficient.
- Patch time range.

Mandatory note: this is an embedding mixture, not deleting a fraction of timestamps. The retained part of the input is not assumed to be normal.

### 5.4. Reconstruction and residual

- Input vs reconstruction.
- Reconstruction error according to the exact implementation formula.
- Residual-classification probability.
- If the norm of the residual feature is plotted, call it an additional diagnostic; do not treat it as an official score if the model does not use it.

## 6. Error Analysis: the core feature

### 6.1. Purpose

Distinguish:
- Output errors: missed, false alarm, partially detected.
- Intermediate observations: low classification, high reconstruction error, a peak that changes after smoothing.
- Causal hypotheses: need controlled tests to confirm; they are not inferred automatically from charts.

### 6.2. Labeled mode

Filters:
- Family/series.
- Method.
- Missed event.
- Partially covered event.
- False-alarm interval.
- Recovered by improved method.
- Lost by improved method.
- Event length.

Not every group must be implemented at P0; prioritize missed, false alarm and recovered/lost.

### 6.3. Point/event rules that must be configured explicitly

- Point TP/FP/FN: compare prediction and label at each timestamp.
- Ground-truth event: one consecutive run of anomaly labels, following the dataset's convention.
- Predicted interval: one consecutive run of predictions; gaps are not merged by default unless configured.
- Minimum event hit: at least one predicted point overlaps the event.
- Event coverage: number of correctly predicted anomaly points / event length.
- Partial coverage: coverage > 0 and < 1; it does not mean the event is missed under the any-overlap rule.
- False-alarm interval: a predicted interval that overlaps no event under the chosen rule.
- Detection delay, if used: position of the first detection inside the event minus the event start time; a missed event has delay N/A.

If tolerance, overlap ratio, gap merging or one-to-one matching rules are used, they must be displayed and exported. Do not silently change definitions between methods.

### 6.4. Case list

| Field | Content |
|---|---|
| Case ID | Stable identifier |
| Case type | GT event / false-alarm interval |
| Start/end | By index, and by timestamp if available |
| Length | Number of points |
| Method | The method |
| Status | Hit/missed/partial or false alarm |
| Coverage | If applicable |
| Peak score | According to the method's score |
| Threshold policy | Source of the threshold |
| Action | Open detail |

Do not treat false-alarm intervals and GT events as the same set of objects when aggregating counts.

### 6.5. Case detail

Layout:
1. Case and method information.
2. Raw + ground truth + prediction.
3. Input/reconstruction.
4. Classification and residual-classification probabilities.
5. Reconstruction error.
6. Score before/after calibration/fusion/smoothing, depending on the implementation.
7. Observation notes and user notes.

Extend the context on both sides of the event but mark the event boundaries clearly. The peak used to evaluate the event must be taken inside the event, not over the padding region.

### 6.6. Evidence trace

For each stage, the following can be stored:
- Stage name, score unit/scale.
- Peak within the event.
- Its own reference threshold, if any.
- Whether it exceeds the threshold.
- Peak position.
- How it is mapped to timestamps.

An example of a valid statement:
> Stage A exceeds its reference threshold, while stage B does not exceed its corresponding threshold.

Do not state:
> Stage B is definitely the sole cause of the error.

To assert an effect, a controlled test is needed that changes one factor while holding the others fixed.

A threshold obtained from the normal test data using labels, if used for diagnosis, must be labeled Oracle diagnostic and not presented as a deployment threshold.

### 6.7. Rule-based observation groups

Examples of optional groups:
- Low classification / high reconstruction error.
- Both low.
- Exceeds threshold before smoothing / does not after smoothing.
- Normal input but high reconstruction error.

The high/low thresholds, the calibration source and the unit of analysis must be published. Groups can overlap; do not sum them as a partition unless they are designed to be mutually exclusive.

### 6.8. No labels

Change the title to Review / Diagnostics:
- Flagged intervals.
- Method disagreement.
- Score trace.
- Notes: confirmed anomaly, suspected false alarm, unreviewed.

Review labels are separate metadata; they do not overwrite ground truth automatically, and must not be reused for evaluation without recording the procedure.

## 7. Method Comparison

### 7.1. Comparing on the same data

- Raw and GT are shared.
- Each method has its own score and threshold.
- Zoom and selected range are synchronized.
- State any differences in preprocessing/protocol if they are not uniform.

### 7.2. Event matrix

Use the same set of GT events and the same hit rule:

| | Improved hit | Improved missed |
|---|---|---|
| Original hit | Both hit | Lost by improvement |
| Original missed | Recovered | Both missed |

Report false-alarm intervals separately; do not infer them from this GT event table.

### 7.3. Metrics

- VUS-PR according to the confirmed implementation.
- AUC-PR if available.
- Point precision/recall at the published threshold.
- Event recall under a specific rule.
- False-alarm intervals.
- Inference time.

Do not implement a different approximation and still use the benchmark metric name. Export the library name, version and metric configuration.

## 8. Experiment Results

- Import existing CSV/JSON results.
- Per-series table.
- Filter by family, seed, split.
- Metric delta between improved and original.
- Number of series that increase/decrease/tie, with a published tolerance.
- Mean by series and by family, stating how it is aggregated.
- Design data and confirmation data are distinguished.
- Failure-case gallery that also includes cases where the improvement made results worse.

Do not merge seed rows as if they were independent series. If seeds are aggregated, publish how the average is taken and the dispersion.

## 9. Data the backend needs to return

Field names are proposals and must match the actual implementation.

### 9.1. Run metadata

```json
{
  "run_id": "run_example",
  "series_id": "series_example",
  "method": "coad_original",
  "result_mode": "precomputed",
  "checkpoint_id": "checkpoint_example",
  "config_id": "config_example",
  "has_ground_truth": true,
  "split_role": "design",
  "seed": 0
}
```

### 9.2. Timestamp-level data

| Field | Required | Notes |
|---|---|---|
| index | Yes | Alignment key |
| timestamp | No | The index may be used instead |
| input_value | Yes | State whether raw or normalized |
| ground_truth | When available | Do not treat missing as normal |
| reconstruction | When the model provides it | Same scale as input when overlaid |
| reconstruction_error | When available | State the formula |
| classification_score | When available | After timestamp mapping |
| residual_classification_score | When available | Do not confuse with feature norm |
| score_pre_smoothing | Recommended | According to the exact pipeline |
| final_score | Yes | After all processing |
| threshold | When a prediction exists | Scalar or time-varying with a description |
| prediction | When a threshold exists | Boolean |
| valid_mask | Recommended | Warm-up, padding, missing coverage |

### 9.3. Patch/window data

- Window ID, start/end.
- Patch ID within the window, start/end.
- Time/frequency probability.
- Mask probability.
- Residual probabilities.
- Mapping/aggregation version.

### 9.4. Alignment

- Use the same original index for input, label and score.
- Padding/warm-up must be marked invalid if there is no valid score.
- Publish how scores from overlapping windows are merged.
- Distinguish patch probability within a window from the timestamp score after aggregation.
- Display and export full values; do not use downsampled data to compute metrics.

## 10. Suggested backend/API

This is a proposed interface and does not constrain the framework:

| API | Purpose |
|---|---|
| GET /datasets | List of sample datasets |
| GET /series/{id}/metadata | Metadata and compatibility |
| GET /methods | Available methods |
| POST /uploads/validate | Validate a CSV |
| POST /runs | Create an inference request if live is supported |
| GET /runs/{id}/status | Processing status |
| GET /runs/{id}/result | Score, prediction, metadata |
| GET /runs/{id}/events | List of events/cases |
| GET /runs/{id}/windows/{window_id} | Patch-window diagnostics |
| GET /experiments | Precomputed benchmarks |
| GET /runs/{id}/export | Export results |

Metric computation and event matching are centralized in the backend so that all pages share the same definitions. The frontend must not reinterpret them with a different set of rules.

## 11. Export and reproducibility

### CSV
- Timestamp/index.
- Input, label if available.
- Reconstruction and the selected scores.
- Threshold and prediction.
- Validity mask.

### JSON
- Run/config/checkpoint IDs.
- Preprocessing, window, patch, stride.
- Calibration source/settings.
- Smoothing settings.
- Threshold policy.
- Metric/event rules.
- Scope of the runtime measurement.

### Figures
- The range being viewed and the legend.
- Method and series.
- Threshold policy.
- Do not export figures lacking ground-truth/prediction annotations.

## 12. UI states and performance

- Empty: instructions to select a series.
- Loading: the specific processing step.
- Ready: valid results.
- Partial diagnostics: the model has a final score but lacks intermediate signals.
- Error: an actionable message.
- Unsupported: incompatible checkpoint or data.

For long series, the overview may be downsampled for drawing. When zooming in deeply, load full-resolution data. Do not compute events/metrics on downsampled data. Ensure outliers do not disappear because of the display sampling method.

## 13. Wireframe

```text
+------------------------------------------------------------------+
| CoAD Demo | Detection | Error Analysis | Comparison | Results      |
+------------------+-----------------------------------------------+
| Dataset / Series | Metadata / model / run mode                    |
| Method           +-----------------------------------------------+
| Threshold policy | Raw + GT + prediction                          |
| Run / Load       +-----------------------------------------------+
| Export           | Input vs reconstruction                       |
|                  +-----------------------------------------------+
| Filters          | Score trace + thresholds                       |
|                  +-----------------------------------------------+
|                  | Event table / comparison / notes               |
+------------------+-----------------------------------------------+
```

Error Analysis can use the same layout: the left panel holds the case filters; clicking a row in the table controls the selected interval of all charts.

## 14. Implementation roadmap

### P0: a working research demo
- Sample data, checkpoints and precomputed results.
- Original CoAD vs calibrated.
- Raw/reconstruction/final score.
- Threshold and provenance.
- Error list: missed events, false alarms.
- Case detail with score before/after smoothing if available.
- Comparison recovered/lost.
- Export CSV + config.

### P1: improved explainability
- Patch heatmap and soft-mask panel.
- Residual-classification trace.
- Per-series/family benchmark.
- Review notes.
- Upload validation and inference only when a suitable model flow exists.

### P2: extensions if needed
- Live training with job tracking.
- More model/input types.
- Session persistence and collaboration.
- Streaming only after the online protocol is clearly defined.

P2 is not a condition for completing the MVP.

## 15. Acceptance checklist

### Data correctness
- [ ] The UI matches backend scores/predictions at the checked indices.
- [ ] Ground truth and prediction are not offset in timestamp.
- [ ] Reconstruction is drawn on the same scale as the input.
- [ ] Event counts and false-alarm interval counts match the evaluation script.
- [ ] Benchmark metrics and the UI use the same implementation/config.
- [ ] Invalid regions are not counted as normal or as negative predictions by default.

### Analysis correctness
- [ ] Without labels, FP/FN are not shown as ground truth.
- [ ] The threshold has a clear source and policy.
- [ ] Dragging the threshold does not overwrite the official benchmark.
- [ ] The evidence trace does not assert causes on its own.
- [ ] Normal-test oracle diagnostics are marked.
- [ ] Comparison includes both recovered and lost, together with false alarms.

### Demo experience
- [ ] Selecting an event synchronizes all charts.
- [ ] There is at least one case where the improvement is better and one where it is not.
- [ ] Precomputed and live are distinguished.
- [ ] Export includes a reproducible config.
- [ ] No crash when labels or intermediate scores are missing.

## 16. Presentation script

1. Select a sample series with labels.
2. Show original CoAD and its predicted regions.
3. Open a missed event in Error Analysis.
4. Inspect classification, reconstruction and the score trace.
5. Switch to calibrated, keeping the same viewing range.
6. Check whether the event is recovered or still missed, and whether false alarms increase.
7. Open a case where the improvement is worse to present the limitations.
8. Switch to the benchmark table to avoid drawing conclusions from a single case.
9. Export the results/config if needed.

This is a conditional script: do not fabricate a successful improvement case if the real data has none.

## 17. References

- CoAD, classification mechanism, masking, reconstruction and residual classification: https://arxiv.org/html/2605.26193
- Formally Exploring Time-Series Anomaly Detection Evaluation Metrics: https://arxiv.org/pdf/2510.17562.pdf
- PATE: Proximity-Aware Time series anomaly Evaluation: https://arxiv.org/html/2405.12096v1
- MathWorks, Interpret Evaluation Metrics for Time Series Anomaly Detectors: https://www.mathworks.com/help/predmaint/ug/interpret-evaluation-metrics.html

The UI rules, API, data schema and priority order in this document are design proposals for the course project, not requirements of the referenced papers.
