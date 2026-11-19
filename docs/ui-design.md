# CoAD Web Demo UI Design

## 1. Design direction

Design it as a **research workspace**, prioritizing charts and evidence analysis. The interface must support one continuous flow:

```text
Select series → View results → Select an error → Trace signals → Compare improvement
```

Do not design it as a landing page promoting AI. Avoid heavy gradients, animations, large KPI cards and purely decorative diagrams.

This document is a UI proposal for the course project, not a description of a deployed system. A feature is shown only when the actual implementation and data support it.

## 2. Navigation structure

| Screen | Main content |
|---|---|
| Series Analysis | Select data, run/load results, view reconstruction and score |
| Error Analysis | Find missed events and false alarms, inspect each case |
| Experiment Results | Compare by series/family and view failure cases |

- Model Inspector is a detail panel within the first two screens.
- Method Comparison is a view mode, not necessarily a standalone page.
- About/Reproducibility lives in a menu or an info panel.
- When switching screens, keep the selected series, method, run and time range where appropriate.

## 3. Series Analysis screen

### 3.1. Wireframe

```text
+-----------------------------------------------------------------------+
| CoAD Lab | Series Analysis | Error Analysis | Experiment Results         |
+-----------------------------------------------------------------------+
| Series: ...  Methods: CoAD, Calibrated   [Load results] [Export]         |
| Checkpoint: ...  Mode: Precomputed        Threshold: Normal calibration  |
+----------------+------------------------------------------------------+
| CONFIGURATION  | Full-series overview                                  |
|                | --------------[current range]----------------------   |
| Data           +------------------------------------------------------+
| Method         | Input + ground truth + prediction                     |
| Threshold      |                                                      |
|                +------------------------------------------------------+
| Advanced >     | Anomaly score + threshold                             |
|                +------------------------------------------------------+
| [Apply]        | Input and reconstruction                              |
|                +------------------------------------------------------+
|                | [Events] [Evidence trace] [Run configuration]         |
+----------------+------------------------------------------------------+
```

### 3.2. Initial dimensions

The following values are proposals for building the interface, not fixed constraints:

| Component | Suggested size |
|---|---|
| Header | About 56 px |
| Sidebar | 240–260 px; collapsible |
| Spacing between blocks | 16–24 px |
| Input chart | 260–300 px tall |
| Score/reconstruction chart | 180–220 px tall |
| Content | Uses the remaining horizontal space |

Prefer wide charts. Do not split the screen into many small charts that make timestamps and anomaly peaks hard to observe.

### 3.3. Configuration bar

Shown by default:
- Dataset/family and series.
- Method.
- Checkpoint or the corresponding precomputed result.
- Threshold policy.
- Run/load results button.

Hidden under Advanced:
- Calibration parameters.
- Smoothing settings.
- Read-only window/patch configuration if the checkpoint is fixed.
- Preprocessing details.

Do not allow changing parameters that break compatibility with the checkpoint without a warning or a check.

### 3.4. Result summary

Use one small line instead of many large KPI cards:

```text
True events: … · Detected: … · Missed: … · False-alarm intervals: …
```

- Shown only when labels exist.
- Tooltips explain the event hit and false-alarm interval rules.
- Without labels: replace with the number of flagged points/segments; do not call them errors.

## 4. Time interaction

### 4.1. Principles

Selecting a time range must update every related chart:
1. Brush on the overview.
2. The detail charts zoom together.
3. Hover shows a synchronized vertical cursor.
4. Clicking an event in the table jumps to that region.
5. The Full series button restores the overall view.

Always show the current range:

```text
Current range: [start]–[end]    [Full series]
```

### 4.2. Tooltip

Depending on the chart, show:
- Timestamp/index.
- Input and reconstruction.
- Score and threshold of the correct method.
- Prediction and label if available.
- Valid/invalid status if in the padding region or where there is no score.

Do not pack every signal into a long tooltip that is hard to read. Show only data relevant to the chart or mode being viewed.

## 5. Error Analysis screen

### 5.1. Wireframe

```text
+-----------------------------------------------------------------------+
| Error Analysis | Series: ... | Original CoAD ↔ Calibrated                |
+-----------------------------------------------------------------------+
| [All] [Missed] [False alarm] [Recovered] [Newly missed]                |
+---------------------+-------------------------------------------------+
| CASE LIST           | Event ...                  [Previous] [Next]    |
|                     | GT range: ... · Length: ...                     |
| Case ... Missed     | CoAD: ...     Calibrated: ...                   |
| Case ... Recovered  +-------------------------------------------------+
| Case ... False alarm| Input + GT + predictions                        |
| Case ... Newly missed+------------------------------------------------+
|                     | Reconstruction / reconstruction error           |
|                     +-------------------------------------------------+
|                     | Score before and after processing               |
|                     +-------------------------------------------------+
|                     | [Observations] [Patch detail] [Notes]           |
+---------------------+-------------------------------------------------+
```

### 5.2. Case list

Each row needs only:
- Case ID.
- Case type.
- Time range or length.
- Status across the two methods.

Full details open only after selection. Provide previous/next case buttons to ease presentation.

### 5.3. Descriptive language

Do write:
> Reconstruction error responds within the event; the final score does not exceed the corresponding threshold.

Do not write:
> The definite cause is smoothing.

Name the area **Evidence trace**. Do not call it automatic cause identification unless controlled tests exist.

### 5.4. Error display rules

- GT events and false-alarm intervals are two different kinds of objects.
- Recovered: original missed, improved detected under the same event hit rule.
- Lost: original detected, improved missed.
- Report false alarms separately; do not infer them from the GT event matrix.
- The event peak is computed within the event boundaries; padding on both sides only provides context.
- Without labels: switch the screen to Review/Diagnostics; do not assign FP/FN automatically.

## 6. Method comparison

### 6.1. Vertical stacking by default

```text
Input + ground truth

Original CoAD
Score + threshold + prediction

Calibrated
Score + threshold + prediction
```

- Share the time axis and cursor.
- Do not split the screen in half by default on a laptop.
- Score overlay is optional, not the default.
- Each method uses a threshold appropriate to its scale/policy.
- State clearly when calibration or protocol differs.

### 6.2. Change summary

Show:
- Recovered events.
- Newly missed events.
- Events missed by both.
- Number of false-alarm intervals before/after.

Do not present only the number of recovered anomalies.

## 7. Color scheme and line styles

| Component | Suggested color/style |
|---|---|
| App background | #F8FAFC |
| Panel/chart | #FFFFFF |
| Primary text | #0F172A |
| Border | #E2E8F0 |
| Raw signal | Dark gray |
| Original CoAD | #2563EB |
| Calibrated | #7C3AED |
| Reconstruction | #0F766E, dashed line |
| Ground truth | Light orange band |
| Prediction | A separate marker bar below the chart |
| Threshold | Gray, dashed line |

Principles:
- Method colors are consistent across all screens.
- Do not use red simultaneously for GT, prediction and every kind of error.
- Ground truth uses a background; prediction uses a separate marker row.
- Correct/incorrect is shown with text or symbols, not color alone.
- Check contrast before finalizing the palette.
- The legend is clearly displayed and layers can be toggled.

## 8. Model Inspector and soft masking

### 8.1. Open on demand

Selecting a window or patch opens a panel on the right or below. Do not show every heatmap, STFT, embedding and score from the start.

### 8.2. Patch panel content

```text
Patch: ...
Window: ...
Time range: ...

Time probability: ...
Frequency probability: ...
Mask probability: ...

Embedding mixing weights
Mask embedding:  ...%
Input embedding: ...%

Residual probability: ...
```

Mandatory note:
> This is the embedding mixing weight, not the fraction of timestamps removed.

Do not illustrate soft masking by deleting 9/10 of the data cells. The retained part of the input is not assumed to be normal.

### 8.3. Two viewing levels

- Window overview: patch-wise probability heatmap.
- Patch detail: values, mixing coefficients and the corresponding signals.

When windows overlap, always show the window ID. Do not confuse patch-window probability with the aggregated timestamp score.

## 9. Experiment results

Content order:
1. Filters for family, method, split.
2. Number of series that increase/decrease/tie.
3. Metric delta chart.
4. Detail table.
5. Click a series to open the analysis workspace.

Main chart:

```text
Delta VUS-PR = VUS-PR improved - VUS-PR original
```

- Horizontal bars, with a zero marker.
- Show the signed value, not just color.
- Can be sorted by delta or family.
- Distinguish the design set from the confirmation set.
- State clearly how seeds and series are aggregated.
- A win/loss pie chart is unnecessary if it adds no information.

## 10. Typography and interface components

- A sans-serif font with good Vietnamese support.
- Body text about 14–16 px.
- Numbers in tables preferably use tabular figures if the font supports them.
- Panels with thin borders and moderate rounded corners.
- Few shadows; no glassmorphism.
- A single primary color for actions.
- Buttons have clear text: Load results, Run analysis, Apply, Export.
- Tooltips explain terminology; do not replace all labels with icons.
- Advanced configuration goes in an accordion.

## 11. UI states

| State | How it is shown |
|---|---|
| No data selected | Instructions to select a series or upload |
| Loading | Report the specific step; do not fake progress if it cannot be measured |
| Results available | Show run metadata and charts |
| Missing diagnostics | Keep the final score, report that the detail is unavailable |
| Incompatible | Explain which checkpoint/input does not fit |
| No labels | Hide FP/FN evaluation, keep diagnostics |
| Error | A specific message and a retry action |

Always label results as precomputed or new inference. Changing a configuration that has not been applied needs a Pending changes state, to prevent viewers from assuming the chart has been updated.

## 12. Usability on small screens

Desktop/laptop is the priority for a research demo.
- The sidebar becomes a drawer when horizontal space is short.
- The Inspector opens at full width instead of squeezing the charts.
- Wide tables scroll horizontally.
- Do not hide the threshold policy and selected range.
- Keep the reading order: input → score → reconstruction → details.

## 13. Build order

### Step 1: A screen that already has results
- One sample series.
- Input, reconstruction, final score.
- Synchronized zoom and hover.
- Metadata and threshold policy.

### Step 2: Select a missed event
- Case list.
- Selecting a case updates the charts.
- Evidence trace and previous/next case.

### Step 3: Compare CoAD with calibrated
- Vertically stacked scores.
- Recovered/lost.
- False alarms before/after.

### Step 4: Complete the states
- Upload if a suitable protocol exists.
- Loading/error/empty.
- Export.
- Advanced configuration.

Do not start with a landing page or a pipeline animation.

## 14. UI review checklist

- [ ] Opening the page immediately shows data and results, not many decorative blocks.
- [ ] Charts are readable on a laptop.
- [ ] Zoom and selected interval are synchronized.
- [ ] Ground truth and prediction are clearly distinguished.
- [ ] Method colors are consistent.
- [ ] Errors are not represented by red/blue alone.
- [ ] The threshold has a clear policy and source.
- [ ] Charts and metrics match the backend data.
- [ ] Downsampled data is used only for drawing, not for computing metrics.
- [ ] Case detail does not conclude causes on its own.
- [ ] Without labels, FP/FN are not shown.
- [ ] Soft masking is not illustrated as deleting timestamps.
- [ ] A better case and a worse case can be demonstrated if the real data has them.
- [ ] Precomputed data is distinguished from new runs.

## 15. Design references

- Elastic EUI, Dashboard good practices: https://eui.elastic.co/v112.2.0/docs/dataviz/dashboard-good-practices/
- Microsoft Fabric, Time Series analysis in Real-Time Dashboard: https://www.learn.microsoft.com/en-us/fabric/real-time-intelligence/dashboard-time-series
- Microsoft, Customize Azure Data Explorer dashboard visuals: https://learn.microsoft.com/en-us/azure/data-explorer/dashboard-customize-visuals
- Microsoft Fabric, Customize Real-Time Dashboard Visuals: https://learn.microsoft.com/en-us/fabric/real-time-intelligence/dashboard-visuals-customize

The wireframes, sizes, palette and implementation order in this document are proposals for the CoAD demo, not mandatory requirements from the referenced sources.
