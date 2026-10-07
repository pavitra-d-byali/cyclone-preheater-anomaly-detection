# Cyclone Preheater — Abnormal Operation Detection



---

## Key Results

> Every number below is from the final reproducible run of
> `python run_analysis.py --input "Data (1).csv"` (70 s, fixed seeds).
> Nothing is hand-entered — all values are traceable to the delivered CSV/JSON outputs.

| Metric | Value |
|---|---|
| Records parsed / valid | 377,719 / 376,124 (1,595 sentinel rows → NaN, kept, never scored) |
| Operating regimes (k = 2) | HOT 301,522 samples @ 889 °C inlet · COLD 74,602 @ 33 °C · silhouette 0.849 |
| Flagged samples (raw → persisted) | 41,504 (11.0 %) → 39,441 (10.4 %) after ≥ 30 min persistence |
| **Abnormal periods** | **1,055** |
| → Rankable candidates | **879** (`process_correlated` 386 · `single_sensor_deviation` 348 · `transient_spike` 96 · `instrument_fault` 49) |
| → Expected operating context (never ranked) | `startup_context` 88 + `shutdown_transition` 87 = 175 |
| → Acquisition-level context (never ranked) | `post_outage_baseline` 1 |
| Acquisition outage runs | 357 → `acquisition_outages.csv` (never scored) |
| Invalid rows flagged as abnormal | **0** |
| Method agreement (robust z vs Isolation Forest) | Spearman **0.549** · top-1 % Jaccard 0.307 |
| Sensitivity grid (z ∈ {4,5,6} × window ∈ {24,36,54}) | Jaccard vs default **0.71 – 0.88** |
| Period durations | median 75 min · longest 5,850 min (97.5 h) |
| Multi-sensor periods (≥ 2 sensors) | 53.3 % |
| Flagged samples inside a final period | 95.1 % |

### Top-15 Severity Leaderboard

| # | Start → End | Duration | Severity | Category | Sensors |
|---|---|---|---|---|---|
| 1 | 2018-07-15 21:15 → 23:35 | 145 min | 1.000 | `instrument_fault` | 6/6 |
| 2 | 2020-05-17 10:30 → 22:20 | 715 min | 0.973 | `process_correlated` | 6/6 |
| 3 | 2017-08-06 16:30 → 2017-08-07 01:20 | 535 min | 0.968 | `process_correlated` | 6/6 |
| 4 | 2017-03-30 20:55 → 23:30 | 160 min | 0.965 | `process_correlated` | 5/6 |
| 5 | 2019-07-04 20:10 → 20:50 | 45 min | 0.963 | `instrument_fault` | 6/6 |
| 6 | 2018-09-07 00:00 → 2018-09-08 01:20 | 1,525 min | 0.936 | `process_correlated` | 6/6 |
| 7 | 2019-06-30 22:20 → 23:55 | 100 min | 0.932 | `process_correlated` | 5/6 |
| 8 | 2019-01-08 15:45 → 2019-01-09 01:20 | 580 min | 0.924 | `process_correlated` | 6/6 |
| 9 | 2017-07-31 04:30 → 17:00 | 755 min | 0.910 | `process_correlated` | 5/6 |
| 10 | 2018-08-03 00:00 → 01:20 | 85 min | 0.898 | `process_correlated` | 5/6 |
| 11 | 2017-09-13 00:00 → 2017-09-14 00:30 | 1,475 min | 0.885 | `process_correlated` | 2/6 |
| 12 | 2019-10-26 15:00 → 2019-10-27 06:55 | 960 min | 0.882 | `process_correlated` | 6/6 |
| 13 | 2019-04-14 18:40 → 2019-04-16 05:55 | 2,120 min | 0.881 | `process_correlated` | 2/6 |
| 14 | 2018-06-12 00:00 → 2018-06-13 01:20 | 1,525 min | 0.872 | `process_correlated` | 3/6 |
| 15 | 2018-05-31 19:05 → 2018-06-01 01:20 | 380 min | 0.872 | `process_correlated` | 6/6 |

Three **reference genuine events** surface naturally — no threshold was tuned to place them:

- 2020-05-17 (715 min, rank 2) ✓
- 2018-09-07 (1,525 min, rank 6) ✓
- 2019-10-26 (960 min, rank 12) ✓

The top-15 spans 11 different start hours, durations 45 – 2,120 min, and 2 categories.
No `startup_context` or `shutdown_transition` period appears in it.

---

## 1. Problem Statement

A cyclone preheater dataset (6 sensor variables, 377,719 records at 5-minute cadence)
contains instances of abnormal operation. Using Python, highlight the **time periods**
where abnormality can be observed — as actual start/end intervals with severity ranking,
without ground-truth labels.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source file | `Data (1).csv` (assignment CSV) |
| Records | **377,719** (verified against assignment spec) |
| Columns | `time` + 6 sensors = 7 |
| Sampling | 5 minutes |
| Coverage | 2017-01-01 00:00 → 2020-12-07 23:55 (1,436 days) |
| Variables | 3 temperatures (inlet gas, material, outlet gas) + 3 drafts (inlet, cone, outlet) |

### Data-Quality Findings (Phase 1 Audit)

1. **Two timestamp formats** in one file — `DD-MM-YYYY HH:MM` (150,484 rows) and
   `M/D/YYYY H:MM` (227,235 rows). Parsed per row; 0 unparseable.
2. **File is not chronologically sorted** (text-sorted export). Sorted in preprocessing.
3. **1,595 SCADA sentinel rows** (8,195 cells): `Not Connect`, `I/O Timeout`,
   `Configure`, `Unit Down`, `Scan Timeout`, `Comm Fail`. → NaN, rows kept, never dropped.
4. **14 genuine time gaps** (all ≥ 1 day, totaling ~125 missing days).
5. **Bimodal regime structure**: inlet gas < 100 °C for 16.4 % of samples (plant cycles
   between hot operation ~890 °C and cold/shutdown ~33 °C).
6. **Sensor fault modes present**: `Cyclone_Material_Temp` at exact 0.0 for 14,226
   samples while gas path is hot; stuck-value runs up to 1,276 samples; range
   saturation at 1,375 °C.
7. **Regime-dominated correlations**: temps r ≈ 0.96 – 0.99, drafts r ≈ 0.97 – 0.99,
   temp-vs-draft r ≈ −0.90. This is why regime-conditional baselines are required.

---

## 3. Pipeline

```
Data (1).csv
 → schema validation (377,719 × 7)
 → dual-format timestamp parse → chronological sort → dedupe
 → Phase 1 audit (read-only, nothing removed)
 → preprocessing (sentinels → NaN, no clipping, no interpolation, validity mask)
 → causal features (rolling median/MAD z-scores, first differences, heat & draft deltas)
 → operating-regime detection (MiniBatchKMeans, k = 2, silhouette = 0.849)
 → anomaly scoring (robust contextual z primary + Isolation Forest corroboration)
 → point flags > z = 5.0
 → temporal persistence filter (≥ 30 min runs, ≤ 15 min gaps merged)
 → contiguous periods → classification → severity ranking → CSV + plots
```

---

## 4. Preprocessing Decisions

| Decision | Why |
|---|---|
| Parse both timestamp formats per row | Locale switch is a file property; single-format parse silently fails |
| Chronological sort (stable) | File is text-sorted; every temporal feature depends on order |
| Sentinels → NaN, rows kept | A failed acquisition is a data event, not a process measurement; dropping rows corrupts the timeline |
| No clipping / winsorising | Extremes (0 °C inlet while hot, 1,375 °C saturation) are candidate anomalies — must not remove what we are trying to detect |
| No interpolation | Fabricating values inside outages would invent process behavior; validity mask marks them, never scored |
| Robust scaling (median / IQR) | 50 % breakdown point: anomalous magnitudes cannot distort the scaler |
| Causal (past-only) rolling windows | An observation is judged only against history that existed at that time — no future leakage |

---

## 5. Algorithm Selection

The data is **6 univariate channels, strongly autocorrelated, bimodal (on/off), 377k rows,
unlabeled**. On that structure:

- **Robust contextual z-score (median/MAD)** is the primary detector. Two references per
  sensor: *local* (trailing 3 h, catches sudden excursions) and *regime-conditional*
  (per-cluster envelope, catches sustained states the trailing window would absorb).
- **Isolation Forest** is corroboration, not the gate. Used threshold-free via
  `score_samples` — `predict()` is never called, so `contamination` never forces an
  arbitrary fraction. Contributes evidence columns and method-agreement validation.
- **K-Means regimes** are context, not anomalies. Clusters are validated by silhouette /
  Davies–Bouldin / Calinski–Harabasz and physically verified from medians. A small
  cluster is never equated with abnormal.
- **Rejected**: LSTM/autoencoder (no labels), DBSCAN (parameter-sensitive in 6-D mixed
  regimes), PCA reconstruction (correlates with the robust z already computed).
- **Temporal persistence** is the core: runs must last ≥ 30 min; gaps ≤ 15 min are merged.

---

## 6. Anomaly Definition

A sample is *point-anomalous* when any sensor's composite contextual score exceeds the
threshold:

$$
c_{v,t} = \max\!\bigl(\lvert z_{\text{local}}^{v,t}\rvert,\; \lvert z_{\text{regime}}^{v,t}\rvert\bigr) \qquad S_t = \max_v\; c_{v,t} \;>\; 5.0
$$

An **abnormal period** is a maximal contiguous run of point-anomalous samples that:

1. Lasts **≥ 30 minutes** (6 samples) — process upsets persist; single blips are noise.
2. Absorbs interruptions **≤ 15 minutes** (3 samples) — flicker inside one event does
   not split it.

### Classification Hierarchy (first match wins)

| Priority | Category | Evidence | Count | Rankable |
|---|---|---|---|---|
| 1 | `post_outage_baseline` | Onset at first sample after an acquisition outage where the operating regime changed *unobserved* across missing data → stale baseline | 1 | No |
| 2 | `startup_context` | ≥ 30 min COLD regime before + ≥ 2 temp channels cross 500 °C together in one valid 5-min pair + ≥ 30 min HOT regime after | 88 | No |
| 3 | `shutdown_transition` | Mirror of startup (HOT → COLD) with the same evidence chain | 87 | No |
| 4 | `instrument_fault` | Exact 0.0 while gas path hot, stuck value ≥ 80 %, or range saturation at 1,375 °C | 49 | Yes |
| 5 | `process_correlated` | ≥ 2 physically coupled sensors deviate together — the assignment's target | 386 | Yes |
| 6 | `single_sensor_deviation` | One sensor outside regime envelope, no fault signature | 348 | Yes |
| 7 | `transient_spike` | Above threshold on peak-score sensor only | 96 | Yes |

The 176 non-rankable periods (175 operating context + 1 outage boundary) are **scored,
fully reported** in `abnormal_periods.csv` with evidence sentences, but ranked **after**
all 879 rankable candidates so they never occupy the severity leaderboard.

### Severity Formula

```
magnitude = mean_composite_score / p99_of_rankable_means,  clipped to [0, 1]
severity  = 0.45 × magnitude + 0.25 × duration + 0.20 × breadth + 0.10 × fill
            rescaled so the most severe rankable period = 1.000
```

- **duration** — log1p-damped, normalized to the longest period (5,850 min)
- **breadth** — abnormal sensors / 6
- **fill** — share of interval samples flagged

---

## 7. Validation (No Ground Truth)

> **There are no ground-truth anomaly labels in the supplied dataset, so supervised
> precision/recall cannot be calculated.** Every number below is statistical or
> process-consistent evidence, not an accuracy claim.

| Check | Result |
|---|---|
| Invalid (sentinel) rows flagged | **0** |
| Method agreement (robust z vs IF) | Spearman **0.549** · top-1 % Jaccard **0.307** |
| Sensitivity grid (3 × 3) | Jaccard vs default **0.71 – 0.88** |
| Regime structure | k = 2, silhouette 0.849, Davies–Bouldin 0.213 |
| Raw runs → periods | 4,657 → 1,055 (persistence filter) |
| Flagged samples inside final intervals | **95.1 %** |
| Periods with ≥ 2 coupled sensors | **53.3 %** |
| Reference events recovered (no tuning) | 2018-09-07 (rank 6) · 2019-10-26 (rank 12) · 2020-05-17 (rank 2) ✓ |
| Leaderboard integrity | Ranks 1–879 = rankable only; 0 context periods in top-15 |

### Sensitivity Analysis

| z threshold | Window (samples) | Jaccard vs base | Flagged samples |
|---|---|---|---|
| 4.0 | 24 | 0.728 | 44,265 |
| 4.0 | 36 | 0.785 | 46,906 |
| 4.0 | 54 | 0.710 | 50,084 |
| **5.0** | 24 | 0.796 | 36,852 |
| **5.0** | **36** | **0.884** | **39,485** |
| **5.0** | 54 | 0.786 | 42,104 |
| 6.0 | 24 | 0.739 | 32,763 |
| 6.0 | 36 | 0.815 | 35,085 |
| 6.0 | 54 | 0.746 | 37,580 |

Top intervals are stable across all configurations. The default (z = 5.0, window = 36)
is highlighted; its near-neighbours (z = 5, w ∈ {24, 54}) achieve Jaccard 0.79 – 0.88.

---

## 8. How to Run

```bash
pip install -r requirements.txt

# Run on the assignment CSV:
python run_analysis.py --input "C:/Users/dell/Downloads/Data (1).csv"

# Optional: custom output directory:
python run_analysis.py --input "Data (1).csv" --outdir ./output
```

Runtime ≈ 70 seconds on a laptop CPU. Seeds fixed via `RANDOM_STATE = 42`.

---

## 9. Outputs

| File | Description |
|---|---|
| `abnormal_periods.csv` | All 1,055 periods with rank, start/end, duration, severity, magnitude, breadth, category, evidence, regime, affected sensors |
| `top_15_abnormal_periods.csv` | Top 15 by severity (rankable only) |
| `anomaly_scored_data.csv` | All 377,719 rows: raw sensors, regime, validity, composite score, IF percentile, per-sensor z, raw/persisted flags |
| `analysis_summary.csv` | Key metrics and parameters |
| `acquisition_outages.csv` | 357 sentinel runs / calendar gaps (never scored) |
| `audit_report.txt` | Full Phase 1 data audit |
| `results.json` | Machine-readable results for the PPT builder |
| `plots/*.png` | 8 diagnostic plots (timeseries, distributions, correlation, missing pattern, score distributions, regime PCA, severity timeline, top-period zoom) |
| `build_ppt.py` | Reproducible PPT builder — reads `results.json`, no hardcoded numbers |
| `*.pptx` | 4-slide submission deck |

---

## 10. Limitations

> **There are no ground-truth anomaly labels in the supplied dataset, so supervised precision/recall cannot be calculated.**

- **Anomaly evidence vs root cause**: The detected periods represent statistical and process-consistent deviation from normal operating envelopes. The system does not claim to prove physical root cause — definitive attribution requires plant historian context (kiln feed rate, ID-fan damper position, fuel inputs) and operator logs not present in the 6 SCADA channels.
- **Transductive baseline fitting**: Regime baselines and Isolation Forest are fit over the full record (transductive, unsupervised reference). For production deployment, baselines should be trained on a designated healthy operating window and refreshed periodically.
- **Acquisition outages**: Sentinel rows are excluded from scoring (`valid=False`), so any upset occurring during an acquisition outage is invisible for that duration.
- **Sensor coverage**: Only 6 process variables are available; unmeasured operational disturbances cannot be distinguished from sensor anomalies without auxiliary process tags.
- **Calendar discontinuities**: 14 multi-day acquisition dropouts limit continuous longitudinal drift modeling.
- **Operating context exclusion**: Transition periods (`startup_context`, `shutdown_transition`, `post_outage_baseline`) are separated by physical evidence chains to prevent score inflation from stale baselines. While excluded from the severity leaderboard, they remain fully recorded in `abnormal_periods.csv`.

---

## 11. Future Improvements

- Plant historian context (kiln feed rate, ID-fan damper, combustion air) → cause
  attribution instead of detection-only.
- Expert review loop: operator annotation of top-N periods → labeled validation set
  (precision/recall become measurable).
- Rolling-origin evaluation with a reference training window for online-style scoring.
- Change-point detection (e.g. PELT) to segment events without a fixed min-duration.
- Multivariate conditional models once more process tags are available.

---

*Reproducibility: fixed seeds, pinned minimum versions in `requirements.txt`,
single-command run, all thresholds centralized in `Config` and sensitivity-tested.*
