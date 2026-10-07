"""
Algo8 AI Assignment - Cyclone Preheater Abnormal Operation Detection
====================================================================

Identifies contiguous TIME PERIODS of abnormal operation in a 6-sensor cyclone
preheater dataset (5-minute cadence, ~377,719 records, no ground-truth labels).

Pipeline
--------
raw source (CSV provided with the assignment: Data (1).csv)
  -> schema validation (377,719 x 7)
  -> dual-format timestamp parsing (file switches locale mid-record)
  -> audit (nothing removed)
  -> preprocessing that PRESERVES anomalies (no clipping, no imputation)
  -> causal robust contextual features (rolling median/MAD z-scores, deltas)
  -> operating-regime detection (MiniBatchKMeans, verified physically)
  -> anomaly scoring: robust contextual z (primary) + Isolation Forest (corroboration)
  -> temporal persistence filter (min 30 min runs, 15 min gap merge)
  -> contiguous abnormal intervals, severity ranking, CSV/plot outputs
  -> validation: agreement, stability/sensitivity, process plausibility

Design notes (why, not what) are in comments where decisions were made and in
README.md. Random seeds fixed for reproducibility.

Usage
-----
    python run_analysis.py --input "C:/Users/dell/Downloads/Data (1).csv" [--outdir DIR]

Outputs (in --outdir, default = folder containing this script):
    abnormal_periods.csv, top_15_abnormal_periods.csv,
    anomaly_scored_data.csv, analysis_summary.csv, results.json,
    plots/*.png, audit_report.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

sys.stdout.reconfigure(encoding="utf-8")
RANDOM_STATE = 42  # single seed for every stochastic component

SENSORS = [
    "Cyclone_Inlet_Gas_Temp",
    "Cyclone_Material_Temp",
    "Cyclone_Outlet_Gas_draft",
    "Cyclone_cone_draft",
    "Cyclone_Gas_Outlet_Temp",
    "Cyclone_Inlet_Draft",
]
EXPECTED_COLS = ["time"] + SENSORS
SAMPLE_MINUTES = 5  # assignment: one record every 5 minutes


# --------------------------------------------------------------------------
# Configuration - every tunable threshold in one place (sensitivity-tested)
# --------------------------------------------------------------------------
@dataclass
class Config:
    # --- causal local context window for median/MAD (past-only) -----------
    # 36 samples = 3 h: long enough to absorb normal process noise,
    # short enough to adapt within a shift. Sensitivity-tested {24,36,54}.
    local_window: int = 36
    local_min_periods: int = 12          # require >=1 h of history before scoring
    # --- thresholds --------------------------------------------------------
    z_threshold: float = 5.0              # robust SDs; sensitivity {4,5,6}
    z_cap: float = 50.0                   # cap degenerate z (stuck sensors)
    # --- temporal persistence ---------------------------------------------
    min_run_samples: int = 6              # >=30 min sustained = an "period"
    merge_gap_samples: int = 3            # merge interruptions <=15 min
    # --- regime detection --------------------------------------------------
    regime_k_candidates: tuple = (2, 3, 4, 5, 6)
    regime_k: int = 3                     # overridden by selection in Phase 3
    regime_sample: int = 20_000           # rows for silhouette evaluation
    # --- Isolation Forest --------------------------------------------------
    if_estimators: int = 300
    # 'auto' (no fixed anomaly fraction). We never use .predict()/offset_,
    # only the continuous ranking score, so contamination does not drive any
    # decision - this is how we avoid blindly fixing contamination=0.1.
    if_contamination: str | float = "auto"
    # --- severity weights --------------------------------------------------
    w_magnitude: float = 0.45
    w_duration: float = 0.25
    w_breadth: float = 0.20
    w_persistence: float = 0.10
    # --- misc --------------------------------------------------------------
    score_threshold_pct: float = 99.0     # IF corroboration percentile


# ==========================================================================
# 1. LOADING - source CSV supplied with the assignment (Data (1).csv)
#    (a PDF-extraction fallback is retained for the original assignment PDF)
# ==========================================================================
_PDF_ROW_RE = re.compile(
    r"^(?:\d{2}-\d{2}-\d{4} \d{2}:\d{2}|\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2})"
    r",(?:[^,]+,){5}[^,]+$"
)
_HEADER = ("time," + ",".join(SENSORS))


def load_source(path: Path, outdir: Path) -> pd.DataFrame:
    """Load raw records from the assignment CSV (or the original PDF fallback).

    Cell values may be numeric or a sentinel token ('I/O Timeout',
    'Unit Down', ...) which is preserved and becomes NaN during numeric
    coercion - no record is dropped.
    """
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        rows, bad = [], 0
        for page in reader.pages:
            for line in (page.extract_text() or "").splitlines():
                line = line.strip()
                if not line or line == _HEADER:
                    continue
                if _PDF_ROW_RE.match(line):
                    rows.append(line)
                else:
                    bad += 1
        if bad:
            print(f"[load] WARNING: {bad} non-conforming lines skipped")
        raw = pd.DataFrame(
            [r.split(",") for r in rows], columns=EXPECTED_COLS
        )
        # persist the reconstruction so the run is auditable
        raw.to_csv(outdir / "dataset_reconstructed.csv", index=False)
        print(f"[load] reconstructed {len(raw):,} rows from PDF")
        return raw
    return pd.read_csv(path, dtype={"time": str})


def parse_timestamps(s: pd.Series) -> pd.Series:
    """Parse the file's TWO timestamp conventions (locale switched mid-export):
    'DD-MM-YYYY HH:MM' and 'M/D/YYYY H:MM'. Detected per row by separator."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    slash = s.str.contains("/", regex=False)
    out[~slash] = pd.to_datetime(s[~slash], format="%d-%m-%Y %H:%M", errors="coerce")
    out[slash] = pd.to_datetime(s[slash], format="%m/%d/%Y %H:%M", errors="coerce")
    return out


def validate_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == EXPECTED_COLS, (
        f"schema mismatch: {list(df.columns)}"
    )
    assert len(df) == 377_719, (
        f"row count {len(df)} != 377,719 (assignment spec)"
    )


# ==========================================================================
# 2. AUDIT - comprehensive, read-only (Phase 1)
# ==========================================================================
def audit(df: pd.DataFrame) -> dict:
    rep: dict = {}
    lines: list[str] = []

    def log(msg: str = ""):
        lines.append(msg)

    log("=" * 72)
    log("PHASE 1 - DATA AUDIT")
    log("=" * 72)
    log(f"shape: {df.shape}")
    log(f"columns: {list(df.columns)}")
    log(f"dtypes:\n{df.dtypes.to_string()}")
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    rep["rows"], rep["cols"] = df.shape[0], df.shape[1]
    log(f"memory: {mem_mb:.1f} MB")

    # --- timestamps ---
    ts = df["ts"]
    slash = df["time"].str.contains("/", regex=False)
    log("\n-- timestamps --")
    log(f"DD-MM-YYYY format rows: {int((~slash).sum()):,}")
    log(f"M/D/YYYY    format rows: {int(slash.sum()):,}")
    log(f"unparseable: {int(ts.isna().sum())}")
    rep["start"], rep["end"] = str(ts.min()), str(ts.max())
    rep["span_days"] = int((ts.max() - ts.min()).days)
    log(f"range: {rep['start']} -> {rep['end']}  ({rep['span_days']} days)")
    log(f"monotonic: {ts.is_monotonic_increasing}")
    dup_ts = int(ts.duplicated().sum())
    rep["dup_ts"] = dup_ts
    log(f"duplicate timestamps: {dup_ts}")
    if dup_ts:
        log(df.loc[ts.duplicated(keep=False)].head(10).to_string())

    diff = ts.diff().dt.total_seconds().div(60).dropna()
    vc = diff.value_counts()
    log(f"sampling interval value counts (minutes):\n{vc.head(8).to_string()}")
    rep["dominant_interval_min"] = float(vc.index[0])
    gaps = diff[diff > SAMPLE_MINUTES]
    rep["n_gaps"] = int(len(gaps))
    rep["max_gap_min"] = float(diff.max())
    log(f"intervals > {SAMPLE_MINUTES} min: {len(gaps)}  (max {diff.max():.0f} min)")
    for idx in gaps.sort_values(ascending=False).index[:10]:
        log(f"  gap {diff[idx]:.0f} min: {ts[idx - 1]} -> {ts[idx]}")

    # --- sentinels / missing ---
    log("\n-- missing / sentinel values --")
    sensors_raw = df[SENSORS]
    io_all = sensors_raw.eq("I/O Timeout").all(axis=1)
    any_sentinel = ~sensors_raw.apply(lambda c: pd.to_numeric(c, errors="coerce").notna()).all(axis=1)
    rep["io_timeout_rows"] = int(io_all.sum())
    rep["sentinel_rows_any"] = int(any_sentinel.sum())
    log(f"rows with 'I/O Timeout' in ALL 6 sensors: {rep['io_timeout_rows']:,}")
    log(f"rows with ANY non-numeric sensor cell:    {rep['sentinel_rows_any']:,}")
    tokens = {}
    for c in SENSORS:
        coerced = pd.to_numeric(sensors_raw[c], errors="coerce")
        mask = coerced.isna() & sensors_raw[c].notna()
        for tok, n in sensors_raw.loc[mask, c].value_counts().items():
            tokens[tok] = tokens.get(tok, 0) + int(n)
    log(f"sentinel token inventory: {tokens}")
    rep["sentinel_tokens"] = tokens
    rep["missing_per_sensor"] = {
        c: int(pd.to_numeric(sensors_raw[c], errors="coerce").isna().sum())
        for c in SENSORS
    }
    log(f"NaN per sensor after coercion: {rep['missing_per_sensor']}")

    # contiguous acquisition-outage runs
    tmask = io_all.values
    runs = []
    if tmask.any():
        bounds = np.flatnonzero(np.diff(np.r_[False, tmask, False]))
        st, en = bounds[0::2], bounds[1::2]
        for a, b in zip(st, en):
            runs.append((ts.iloc[a], ts.iloc[b - 1], int(b - a)))
        runs.sort(key=lambda r: -r[2])
        rep["n_timeout_runs"] = len(runs)
        rep["top_timeout_runs"] = [
            {"start": str(a), "end": str(b), "samples": n} for a, b, n in runs[:15]
        ]
        log(f"acquisition-outage runs: {len(runs)}, "
            f"total outage samples: {int(tmask.sum()):,}")
        for a, b, n in runs[:15]:
            log(f"  {a} -> {b}  samples={n} ({n * SAMPLE_MINUTES} min)")

    # duplicates / rows
    rep["dup_rows"] = int(df.duplicated().sum())
    log(f"\nfully duplicated rows: {rep['dup_rows']}")

    # --- numeric statistics ---
    num = df[SENSORS].apply(pd.to_numeric, errors="coerce")
    log("\n-- descriptive statistics --")
    desc = num.describe(percentiles=[0.001, 0.01, 0.05, 0.25, 0.5, 0.75,
                                      0.95, 0.99, 0.999]).T
    log(desc.to_string())
    rep["describe"] = json.loads(desc.to_json())

    log("\n-- unique values / variance --")
    log(num.nunique().to_string())
    var = num.var()
    log(f"variance:\n{var.to_string()}")
    rep["constant_cols"] = list(var[var == 0].index)

    # stuck-sensor runs (constant value repeated - data-quality artifact that
    # is ALSO a process signature; reported, not removed)
    log("\n-- longest constant-value runs --")
    stuck = {}
    for c in SENSORS:
        v = num[c].values
        change = np.flatnonzero(np.r_[True, ~(v[1:] == v[:-1]) & ~(pd.isna(v[1:]) & pd.isna(v[:-1])), True])
        # treat NaN runs separately: only count constant NON-NaN runs
        runs_len = np.diff(change)
        finite = v[change[:-1]]
        m = runs_len.max() if len(runs_len) else 0
        stuck[c] = int(m)
        log(f"  {c}: max_const_run={m}")
    rep["max_const_run"] = stuck

    # --- correlations ---
    corr = num.corr()
    log("\n-- Pearson correlation --")
    log(corr.round(3).to_string())
    rep["corr"] = json.loads(corr.to_json())

    # --- temporal coverage ---
    log("\n-- temporal coverage --")
    by_year = ts.dt.year.value_counts().sort_index()
    log(f"rows per year:\n{by_year.to_string()}")
    rep["rows_per_year"] = {int(k): int(v) for k, v in by_year.items()}
    monthly = ts.dt.to_period("M").value_counts().sort_index()
    log(f"months: {len(monthly)}, min rows/month={monthly.min()}, "
        f"max rows/month={monthly.max()}")
    expected = int((ts.max() - ts.min()).total_seconds() // (SAMPLE_MINUTES * 60)) + 1
    rep["expected_rows_5min"] = expected
    log(f"rows expected at constant 5-min cadence: {expected:,} "
        f"vs actual {len(df):,} (deficit {expected - len(df):,})")

    txt = "\n".join(lines)
    print(txt)
    rep["_report"] = txt
    return rep


# ==========================================================================
# 3. PREPROCESSING - preserve anomalies (Phase 2)
# ==========================================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological, de-duplicated, sentinel->NaN frame with an explicit
    validity mask. DELIBERATELY does NOT: clip extremes, interpolate sensor
    outages, or winsorise - all three would erase candidate anomalies."""
    out = df.copy()
    out["ts"] = parse_timestamps(out["time"])
    bad_ts = out["ts"].isna()
    if bad_ts.any():
        print(f"[preprocess] dropping {int(bad_ts.sum())} unparseable timestamps")
        out = out[~bad_ts]

    # sentinel tokens -> NaN (row kept: timeline stays intact)
    out[SENSORS] = out[SENSORS].apply(pd.to_numeric, errors="coerce")

    out = out.sort_values("ts", kind="mergesort").reset_index(drop=True)  # stable
    dup = out["ts"].duplicated(keep="first")
    if dup.any():
        print(f"[preprocess] dropping {int(dup.sum())} duplicate timestamps")
        out = out[~dup].reset_index(drop=True)

    # validity: all six sensors present. Acquisition outage is a DATA issue,
    # reported separately - it is not evidence of a process abnormality.
    out["valid"] = out[SENSORS].notna().all(axis=1)
    print(f"[preprocess] rows={len(out):,}  valid={int(out['valid'].sum()):,}  "
          f"invalid={int((~out['valid']).sum()):,}")
    return out


# ==========================================================================
# 4. FEATURE ENGINEERING - causal, justified (Phase 3)
# ==========================================================================
def robust_scale(x: pd.Series, med=None, iqr=None):
    med = x.median() if med is None else med
    iqr = (x.quantile(0.75) - x.quantile(0.25)) if iqr is None else iqr
    return (x - med) / (iqr if iqr > 1e-9 else 1.0)


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Adds ONLY justified features:

    z_local_<sensor> : causal robust z-score vs trailing median/MAD.
        WHY: contextual anomaly = unusual relative to RECENT process state;
        median/MAD have 50% breakdown so an evolving anomaly cannot mask
        itself the way mean/SD would. Uses past-only windows (no leakage).
    d_<sensor>       : first difference (rate of change, per 5 min).
        WHY: fast ramps are process upsets even when the level is normal.
    dT_inlet_outlet, dT_material_outlet : gas/material heat balance.
    dP_cone_outlet, dP_inlet_cone       : pressure drops along gas path.
        WHY: drafts are only meaningful as DIFFERENCES; a stuck or collapsing
        differential indicates blockage/fouling/flow loss.
    """
    f = pd.DataFrame({"ts": df["ts"], "valid": df["valid"]})
    w, mp = cfg.local_window, cfg.local_min_periods

    for c in SENSORS:
        x = df[c]
        med = x.rolling(w, min_periods=mp).median()
        mad = (x - med).abs().rolling(w, min_periods=mp).median()
        # global IQR floor prevents 0/0 explosions on stuck sensors while
        # keeping sensitivity (floor = 5% of the variable's overall IQR)
        iqr = x.quantile(0.75) - x.quantile(0.25)
        scale = np.maximum(1.4826 * mad, 0.05 * iqr)
        z = ((x - med) / scale).clip(-cfg.z_cap, cfg.z_cap)
        f[f"z_local_{c}"] = z
        f[f"d_{c}"] = x.diff()

    f["dT_inlet_outlet"] = (df["Cyclone_Inlet_Gas_Temp"]
                            - df["Cyclone_Gas_Outlet_Temp"])
    f["dT_material_outlet"] = (df["Cyclone_Material_Temp"]
                               - df["Cyclone_Gas_Outlet_Temp"])
    f["dP_cone_outlet"] = (df["Cyclone_cone_draft"]
                           - df["Cyclone_Outlet_Gas_draft"])
    f["dP_inlet_cone"] = (df["Cyclone_Inlet_Draft"]
                          - df["Cyclone_cone_draft"])
    return f


# ==========================================================================
# 5. OPERATING-REGIME DETECTION (Phase 3/4 - Method C)
# ==========================================================================
def detect_regimes(df: pd.DataFrame, feats: pd.DataFrame,
                   cfg: Config) -> tuple[pd.Series, dict]:
    """K-means on robust-scaled sensor levels + heat/pressure deltas.

    Clusters are NOT assumed abnormal: each cluster is characterised by its
    medians and reported as a physical process state. 'Small cluster' alone
    never triggers an anomaly flag anywhere in this pipeline.
    """
    Xcols = SENSORS + ["dT_inlet_outlet", "dT_material_outlet",
                       "dP_cone_outlet", "dP_inlet_cone"]
    derived = ["dT_inlet_outlet", "dT_material_outlet",
               "dP_cone_outlet", "dP_inlet_cone"]
    src = {c: pd.to_numeric(df[c], errors="coerce") for c in SENSORS}
    src.update({c: feats[c] for c in derived})   # physical deltas live in feats
    X = pd.DataFrame({c: robust_scale(src[c]) for c in Xcols})
    valid_mask = df["valid"].values
    Xv = X[valid_mask]

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(Xv), size=min(cfg.regime_sample, len(Xv)),
                            replace=False)
    Xs = Xv.iloc[sample_idx]

    eval_rows, best_k, best_sil = [], None, -1.0
    for k in cfg.regime_k_candidates:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             batch_size=4096, n_init=10)
        lab = km.fit_predict(Xs)
        sil = silhouette_score(Xs, lab, sample_size=10_000,
                               random_state=RANDOM_STATE)
        db = davies_bouldin_score(Xs, lab)
        ch = calinski_harabasz_score(Xs, lab)
        eval_rows.append({"k": k, "silhouette": round(sil, 4),
                          "davies_bouldin": round(db, 4),
                          "calinski_harabasz": round(ch, 1)})
        if sil > best_sil:
            best_sil, best_k = sil, k
    print("[regimes] k evaluation:")
    print(pd.DataFrame(eval_rows).to_string(index=False))

    # final fit on ALL valid rows, chosen k
    km = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE,
                         batch_size=4096, n_init=10)
    labels = pd.Series(-1, index=df.index, dtype=int)  # -1 = invalid rows
    labels[valid_mask] = km.fit_predict(Xv.values)

    # characterise clusters physically (medians in ORIGINAL units)
    prof = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce")
                         for c in SENSORS})
    prof["cluster"] = labels.values
    centers = prof[valid_mask].groupby("cluster").median()
    sizes = prof[valid_mask]["cluster"].value_counts().sort_index()
    print("[regimes] cluster medians (original units) / sizes:")
    print(centers.round(1).to_string())
    print(sizes.to_string())

    info = {
        "k": best_k,
        "metrics": eval_rows,
        "metrics_best": {"k": best_k, "silhouette": round(best_sil, 4)},
        "centers": {int(k): {c: round(float(v), 2) for c, v in row.items()}
                    for k, row in centers.iterrows()},
        "sizes": {int(k): int(v) for k, v in sizes.items()},
    }
    return labels, info


# ==========================================================================
# 6. ANOMALY SCORING (Phase 4 - Methods A/B/E combined)
# ==========================================================================
def regime_baselines(df: pd.DataFrame, labels: pd.Series,
                     cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-regime robust baselines (median, scale) for each sensor.

    z_regime answers: 'is this value unusual for the regime the plant is
    currently in?' - catches SUSTAINED abnormal states that a trailing
    window would eventually re-center onto.
    Scale floor = 25% of the global robust scale to avoid degenerate z in
    small/homogeneous clusters.
    """
    med_df = pd.DataFrame(np.nan, index=df.index, columns=SENSORS)
    z_df = pd.DataFrame(np.nan, index=df.index, columns=SENSORS)
    for k in np.unique(labels.values):
        if k < 0:
            continue
        m = labels.values == k
        for c in SENSORS:
            x = df.loc[m, c]
            med = x.median()
            mad = (x - med).abs().median()
            gscale = 1.4826 * (df[c] - df[c].median()).abs().median()
            scale = max(1.4826 * mad, 0.25 * gscale, 1e-9)
            med_df.loc[m, c] = med
            z_df.loc[m, c] = ((df.loc[m, c] - med) / scale).clip(
                -cfg.z_cap, cfg.z_cap)
    return med_df, z_df


def isolation_forest_score(df: pd.DataFrame, feats: pd.DataFrame,
                           cfg: Config) -> tuple[pd.Series, pd.Series]:
    """Threshold-FREE Isolation Forest usage.

    We deliberately do NOT use .predict() / contamination: contamination
    forces an arbitrary labeled fraction. Instead we keep the continuous
    decision score and use it (a) as corroboration evidence and (b) for
    method-agreement validation, thresholded only by percentile.
    """
    cols = [f"z_local_{c}" for c in SENSORS]  # robust-scaled already
    X = feats[cols].copy()
    # add rate-of-change and physical deltas, robust-scaled
    for c in [f"d_{c}" for c in SENSORS] + ["dT_inlet_outlet", "dT_material_outlet",
                                            "dP_cone_outlet", "dP_inlet_cone"]:
        X[c] = robust_scale(feats[c])
    Xv = X[df["valid"].values].fillna(0.0).values  # invalid rows scored as 0

    iso = IsolationForest(
        n_estimators=cfg.if_estimators,
        contamination=cfg.if_contamination,   # ranking is unaffected by this
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(Xv)
    raw = -iso.score_samples(Xv)          # higher = more anomalous
    score = pd.Series(np.nan, index=df.index)
    score[df["valid"].values] = raw
    pct = score.rank(pct=True) * 100      # percentile for thresholding
    return score, pct


def composite_score(feats: pd.DataFrame, z_regime: pd.DataFrame,
                    cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Per-sample, per-sensor contextual anomaly magnitude.

    c[v,t] = max( |z_local| , |z_regime| )
      - |z_local|  : unusual vs recent history (catches sudden excursions)
      - |z_regime| : unusual vs the regime's operating envelope
                     (catches sustained states a trailing window would
                     eventually absorb)
    S[t] = max_v c[v,t] : the plant's worst-deviating sensor at time t.
    max() keeps the score interpretable ('worst sensor') and avoids the
    dilution a mean would cause when only 1-2 sensors are affected.
    """
    c = pd.DataFrame(index=feats.index, columns=SENSORS, dtype=float)
    for v in SENSORS:
        c[v] = np.maximum(feats[f"z_local_{v}"].abs(),
                          z_regime[v].abs())
    S = c.max(axis=1)
    S[~feats["valid"]] = np.nan
    return c, S


# ==========================================================================
# 7. TEMPORAL PERSISTENCE -> CONTIGUOUS PERIODS (Phase 6)
# ==========================================================================
def runs_from_flags(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end) inclusive integer indices."""
    if not flags.any():
        return []
    b = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    return list(zip(b[0::2], b[1::2] - 1))


def persistence_filter(flags: np.ndarray, min_run: int,
                       merge_gap: int) -> list[tuple[int, int]]:
    """Keep runs >= min_run samples; merge runs separated by <= merge_gap.

    Rationale w.r.t. 5-min sampling: a genuine process abnormality persists
    (operators do not act on single 5-min blips); min_run = 6 -> 30 min.
    Brief interruptions (sensor flicker, momentary recovery) within one event
    are merged when <= 3 samples (15 min). Longer gaps stay separate events.
    """
    runs = runs_from_flags(flags)
    # merge first
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    # then filter by duration
    return [(s, e) for s, e in merged if (e - s + 1) >= min_run]


def build_periods(df: pd.DataFrame, feats: pd.DataFrame, c: pd.DataFrame,
                  S: pd.Series, if_pct: pd.Series, labels: pd.Series,
                  flag_pts: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Convert persisted runs into a ranked period table with evidence."""
    periods = []
    for s, e in persistence_filter(flag_pts, cfg.min_run_samples,
                                   cfg.merge_gap_samples):
        idx = slice(s, e + 1)
        n = e - s + 1
        span = n  # contiguous by construction; fill = 1 by definition here
        seg_c = c.iloc[idx]
        # dominant variables: ranked by median anomaly magnitude inside period
        meds = seg_c.median().sort_values(ascending=False)
        dom = [v for v in SENSORS if seg_c[v].median() >= cfg.z_threshold]
        n_vars = len(dom)
        peak = float(S.iloc[idx].max())
        mean_score = float(S.iloc[idx].mean())
        fill = 100.0 * flag_pts[idx].sum() / span
        periods.append({
            "start": df["ts"].iloc[s],
            "end": df["ts"].iloc[e],
            # wall-clock duration (accounts for any merged gap minutes,
            # which sample-count x 5 min would understate)
            "duration_min": (df["ts"].iloc[e] - df["ts"].iloc[s]
                             ).total_seconds() / 60 + SAMPLE_MINUTES,
            "n_samples": n,
            "pct_samples_flagged": round(fill, 1),
            "peak_score": round(peak, 2),
            "mean_score": round(mean_score, 2),
            "n_abnormal_vars": n_vars,
            "abnormal_variables": "; ".join(dom) if dom else
                                  "; ".join(meds.head(2).index),
            "dominant_vars_top2": "; ".join(meds.head(2).index),
            "regime": int(labels.iloc[s]),
            "if_pct_max": round(float(if_pct.iloc[idx].max()), 1),
        })
    return pd.DataFrame(periods)


# ==========================================================================
# 6b. PERIOD CLASSIFICATION - operating context vs process vs instrument
# ==========================================================================
SATURATION = 1375.0   # sensor upper range limit observed in the data
HOT_C = 500.0         # deg C: gas path considered 'hot' (operating)
COLD_C = 100.0        # deg C: gas path considered 'cold' (unit down)
CONTEXT_WINDOW = 24   # samples (2 h) after onset searched for an
                      # instantaneous state transition that explains WHY
                      # the period exists (post-transition stale-baseline
                      # flagging lasts ~85-105 min < 2 h; observed flips
                      # inside periods cluster at offsets 0-20 samples,
                      # with none between 21 and 79)
CONTEXT_RUN = 6       # samples (30 min) of sustained pre/post state
                      # required by the evidence chain
CONTEXT_SPAN = 48     # max samples walked while looking for those 6
                      # (4 h): an acquisition dropout inside the run is
                      # skipped (state unknown), a >4 h one is an outage
# categories that are scored but never compete for the severity leaderboard:
# evidence-based operating context (startup/shutdown state transitions) and
# acquisition-boundary artifacts (stale baseline across an outage)
NON_RANKABLE_CATEGORIES = ("startup_context", "shutdown_transition",
                           "post_outage_baseline")


def _flip_hour_hist(df: pd.DataFrame, inlet: np.ndarray,
                    valid_arr: np.ndarray) -> tuple:
    """Dataset-wide hourly counts of instantaneous inlet-temperature flips.

    Supporting evidence only (never a classification gate) for 'timing
    relative to recurring daily startup/shutdown behaviour': computed from
    the actual records, not from an assumed clock rule such as hour == 0.
    """
    n = len(df)
    if n < 2:
        return np.zeros(24, int), np.zeros(24, int)
    step = np.timedelta64(SAMPLE_MINUTES, "m")
    ts = df["ts"].to_numpy()
    adj = np.diff(ts) == step
    prev, cur = inlet[:-1], inlet[1:]
    pair_ok = (adj & valid_arr[:-1] & valid_arr[1:]
               & ~np.isnan(prev) & ~np.isnan(cur))
    c2h = pair_ok & (prev < COLD_C) & (cur > 400.0) & (cur - prev >= 300.0)
    h2c = pair_ok & (prev > 400.0) & (cur < COLD_C) & (prev - cur >= 300.0)
    hours = df["ts"].dt.hour.to_numpy()[1:]
    return (np.bincount(hours[c2h], minlength=24).astype(int),
            np.bincount(hours[h2c], minlength=24).astype(int))


def classify_periods(df: pd.DataFrame, c: pd.DataFrame, periods: pd.DataFrame,
                     labels: pd.Series, regime_centers: dict,
                     cfg: Config) -> pd.DataFrame:
    """Annotate every abnormal period with a physical interpretation.

    Hierarchy (first match wins):
      post_outage_baseline   - ACQUISITION level (hierarchy 1): the period
                               starts at the first sample after an
                               acquisition outage and the operating regime
                               differs from the last pre-outage sample, so
                               the state change happened UNOBSERVED during
                               missing data and the causal window still
                               holds pre-outage values. The elevated score
                               measures that stale baseline at the outage
                               boundary - not process behavior. Scored,
                               reported separately, never ranked. Periods
                               resuming after an outage with an UNCHANGED
                               regime stay rankable and carry an evidence
                               note instead.
      startup_context        - EXPECTED DAILY STARTUP operating context.
                               Evidence chain (ALL required, from data):
                               >=30 min spent in the COLD operating
                               regime (k-means cluster) before the
                               transition; >=2 temperature channels
                               crossing >=300 C over the 500 C boundary
                               TOGETHER in ONE adjacent 5-min sample pair
                               with both rows valid (sensor availability)
                               and no data gap; >=30 min in the HOT
                               operating regime after it. Acquisition
                               dropouts inside the sustained-state runs
                               are skipped (state unknown), never counted;
                               a >4 h dropout breaks the run. Timing
                               relative to the recurring daily startup
                               pattern is computed from the dataset and
                               REPORTED in the evidence string, never
                               used as the rule (never `if hour == 0`).
                               The unit cannot jump ~850 C in 5 min, so
                               the elevated score measures a stale local
                               baseline across the expected startup state
                               change - not a process failure. Scored,
                               reported separately, ranked AFTER every
                               rankable period.
      shutdown_transition     - mirror-image evidence chain (hot -> cold):
                               >=30 min in the HOT regime before,
                               instantaneous multi-channel drop across
                               500 C in one valid adjacent sample pair,
                               >=30 min in the COLD regime after: the
                               expected daily shutdown boundary with the
                               same stale-baseline score inflation.
                               Handled exactly like startup_context
                               (never silently ranked).
      instrument_fault       - explicit transmitter/PLC fault signature
                               (exact 0.0, long stuck run, range saturation)
      process_correlated     - >=2 physically coupled sensors deviate
                               together (the assignment's target: genuine
                               abnormal operation)
      single_sensor_deviation- one sensor alone, no fault signature
                               (ambiguous: early process indicator vs sensor)
      transient_spike        - above threshold on peak only (post-filter
                               rarity)

    A state transition found DEEPER than CONTEXT_WINDOW samples into the
    period cannot have caused the period's onset: the period stays
    rankable and the finding is appended to its evidence (honest record).
    Likewise, a >=300 C instantaneous temperature step near onset that
    does NOT cross into the cold boundary is neither smooth process
    dynamics nor a recorded operating-state transition: the period stays
    rankable and the step is spelled out in its evidence, so midnight-
    onset candidates carry their own explanation instead of requiring
    the reader to guess.

    Acquisition outages (sentinel rows) are valid=False, can never be
    scored, therefore can never appear as periods - a data gap is not a
    process event; likewise a period whose score originates at the
    resumption of data after an outage (regime changed across it) is
    acquisition-level, not process-level (post_outage_baseline).
    Cold-regime periods are flagged via shutdown_context so
    planned shutdowns are never silently mixed with abnormal operations.
    """
    if not len(periods):
        periods["category"] = pd.Series(dtype=str)
        periods["shutdown_context"] = pd.Series(dtype=bool)
        periods["startup_context"] = pd.Series(dtype=bool)
        periods["evidence"] = pd.Series(dtype=str)
        return periods

    cats, shut_flags, startup_flags, evidences = [], [], [], []
    centers = {int(k): v for k, v in regime_centers.items()}
    tvals = df["ts"].values
    step = np.timedelta64(SAMPLE_MINUTES, "m")
    inlet = df["Cyclone_Inlet_Gas_Temp"].to_numpy(dtype=float, copy=True)
    valid_arr = df["valid"].to_numpy(dtype=bool, copy=True)
    labels_arr = labels.to_numpy()
    cold_id = next((k for k, v in centers.items()
                    if v.get("Cyclone_Inlet_Gas_Temp", 999) < HOT_C), None)
    hot_id = next((k for k, v in centers.items()
                   if v.get("Cyclone_Inlet_Gas_Temp", 999) >= HOT_C), None)

    # temperature channels used to detect an instantaneous state transition
    temp_cols = ["Cyclone_Inlet_Gas_Temp", "Cyclone_Material_Temp",
                 "Cyclone_Gas_Outlet_Temp"]

    # dataset-wide recurring-timing evidence (data-derived hour-of-day
    # counts of instantaneous cold->hot / hot->cold inlet flips)
    c2h_hist, h2c_hist = _flip_hour_hist(df, inlet, valid_arr)
    c2h_tot, h2c_tot = int(c2h_hist.sum()), int(h2c_hist.sum())

    def _state_run(i, d, want):
        """Count valid samples whose operating-regime label equals `want`,
        walking from i in direction d (+1 forward / -1 backward).

        Sustained-state evidence for the classification chain. An
        acquisition dropout (invalid row) is state-unknown: it is skipped
        without counting, but never lets the walk claim a regime it did
        not observe; a time gap, a regime change or CONTEXT_SPAN samples
        walked ends the walk.
        """
        if want is None:
            return 0
        n = walked = 0
        cur = i
        while n < CONTEXT_RUN and walked < CONTEXT_SPAN and 0 <= cur < len(df):
            walked += 1
            if valid_arr[cur]:
                if labels_arr[cur] == want:
                    n += 1
                else:
                    break
            nxt = cur + d
            if nxt < 0 or nxt >= len(df):
                break
            if (tvals[max(cur, nxt)] - tvals[min(cur, nxt)]) != step:
                break   # data gap: state history across it is not contiguous
            cur = nxt
        return n

    def _timing(hour, hist, total, name):
        if total <= 0:
            return ""
        mode = int(hist.argmax())
        return (f"; recurring timing: {hist[hour]} of {total} dataset-wide "
                f"instantaneous {name} transitions ({100.0 * hist[hour] / total:.0f}%) "
                f"occur at {hour:02d}:00, modal {mode:02d}:00 "
                f"({100.0 * hist[mode] / total:.0f}%)")

    for _, r in periods.iterrows():
        s = int(np.searchsorted(tvals, np.datetime64(r["start"])))
        e = int(np.searchsorted(tvals, np.datetime64(r["end"])))
        seg_raw = df.iloc[s:e + 1][SENSORS]
        seg_c = c.iloc[s:e + 1]
        dom = [v for v in SENSORS if seg_c[v].median() >= cfg.z_threshold]
        med_inlet = float(seg_raw["Cyclone_Inlet_Gas_Temp"].median())
        med_outlet = float(seg_raw["Cyclone_Gas_Outlet_Temp"].median())
        k = int(labels.iloc[s])
        cold_regime = bool(centers.get(k, {}).get(
            "Cyclone_Inlet_Gas_Temp", 999) < HOT_C)

        # --- level 1: onset at an acquisition-outage boundary --------------
        # The onset IS the first sample after an acquisition outage. If the
        # operating regime differs from the last pre-outage sample, the state
        # change happened unobserved during missing data, so the causal local
        # window (positional, 36 rows) still holds pre-outage values and the
        # elevated score measures that stale baseline - acquisition context,
        # not process behavior. Regime unchanged across the outage: the
        # stale-baseline story is weak, the period stays rankable and the
        # resume is merely noted as evidence.
        outage_ev = None
        outage_note = None
        if s > 0 and (tvals[s] - tvals[s - 1]) != step:
            gap_min = int((tvals[s] - tvals[s - 1]) / np.timedelta64(1, "m"))
            if valid_arr[s - 1] and valid_arr[s]:
                if labels_arr[s - 1] != labels_arr[s]:
                    def _regime_name(idx):
                        c_in = centers.get(int(labels_arr[idx]), {}).get(
                            "Cyclone_Inlet_Gas_Temp", 0.0)
                        return "HOT" if c_in >= HOT_C else "COLD"
                    outage_ev = (
                        f"onset is the first sample after a {gap_min}-min "
                        f"acquisition outage; the operating regime changed "
                        f"across the outage ({_regime_name(s - 1)} -> "
                        f"{_regime_name(s)}, unobserved during missing data), "
                        f"so the causal local window still holds pre-outage "
                        f"values -> score measures a stale baseline at the "
                        f"outage boundary, not process behavior")
                else:
                    outage_note = (
                        f"note: onset coincides with data resumption after a "
                        f"{gap_min}-min acquisition outage; operating regime "
                        f"unchanged across the outage")

        # --- expected operating-transition evidence ------------------------
        # A preheater with massive thermal inertia cannot move ~850 C in one
        # 5-minute sample, so an instantaneous multi-channel hot<->cold jump
        # marks an operating-state transition (startup/shutdown boundary),
        # and the score built on the stale pre-transition baseline measures
        # that transition, not sustained process behavior. The transition
        # must lie within CONTEXT_WINDOW samples of onset to explain why the
        # period exists; the evidence chain (sustained state before AND
        # after, valid sensors, no data gap) keeps a genuine process step
        # change or a single-sensor glitch out of this category.
        ctx = None
        late_flip = None
        chain_fail = None
        partials = []   # >=300C single-sample steps that do NOT cross the
                        # cold boundary: large, not smooth dynamics, but
                        # also NOT a recorded operating-state transition
        w_end = min(e, s + CONTEXT_WINDOW)
        for j in range(max(s, 1), w_end + 1):
            if (pd.Timestamp(tvals[j]) - pd.Timestamp(tvals[j - 1])
                    ) != pd.Timedelta(minutes=SAMPLE_MINUTES):
                continue  # across a data gap: a legitimate restart, not a splice
            if not (valid_arr[j - 1] and valid_arr[j]):
                continue  # sensor availability required at the transition
            hits_c2h, hits_h2c = [], []
            for v in temp_cols:
                a, b = df[v].iloc[j - 1], df[v].iloc[j]
                if pd.isna(a) or pd.isna(b):
                    continue
                if a < COLD_C and b > 400.0 and b - a >= 300.0:
                    hits_c2h.append((v, a, b))
                elif a > 400.0 and b < COLD_C and a - b >= 300.0:
                    hits_h2c.append((v, a, b))
                elif abs(a - b) >= 300.0:
                    partials.append((v, a, b, j))
            lead = (j - s) * SAMPLE_MINUTES
            lead_txt = ("" if lead == 0 else
                        f"; period onset precedes the transition by {lead} min")
            if len(hits_c2h) >= 2:
                nb = _state_run(j - 1, -1, cold_id)
                nf = _state_run(j, +1, hot_id)
                if nb >= CONTEXT_RUN and nf >= CONTEXT_RUN:
                    v, a, b = max(hits_c2h, key=lambda h: abs(h[2] - h[1]))
                    ctx = (
                        "startup_context",
                        f"expected startup: the unit sits in the COLD operating "
                        f"regime for >= {CONTEXT_RUN * SAMPLE_MINUTES} min before "
                        f"the transition, then {len(hits_c2h)} temperature channels "
                        f"cross 500C TOGETHER in ONE 5-min sample with both rows "
                        f"valid and no data gap (e.g. {v} {a:.0f} -> {b:.0f} C, "
                        f"{abs(a - b):.0f}C in 5 min - far beyond preheater "
                        f"thermal-mass limits), then >= "
                        f"{CONTEXT_RUN * SAMPLE_MINUTES} min in the HOT operating "
                        f"regime{_timing(pd.Timestamp(tvals[j]).hour, c2h_hist, c2h_tot, 'cold->hot')}{lead_txt}"
                        " -> expected operating context: score reflects a stale "
                        "local baseline across the daily startup state change, "
                        "not a process failure")
                    break
                if chain_fail is None:
                    chain_fail = (
                        f"note: an instantaneous cold->hot transition at +{lead} min "
                        f"FAILED the operating-context evidence chain (cold regime "
                        f"before: {nb}/{CONTEXT_RUN} samples, hot after: "
                        f"{nf}/{CONTEXT_RUN}) -> classified by its own evidence "
                        f"instead")
            if len(hits_h2c) >= 2:
                nb = _state_run(j - 1, -1, hot_id)
                nf = _state_run(j, +1, cold_id)
                if nb >= CONTEXT_RUN and nf >= CONTEXT_RUN:
                    v, a, b = max(hits_h2c, key=lambda h: abs(h[2] - h[1]))
                    ctx = (
                        "shutdown_transition",
                        f"expected shutdown: the unit sits in the HOT operating "
                        f"regime for >= {CONTEXT_RUN * SAMPLE_MINUTES} min before "
                        f"the transition, then {len(hits_h2c)} temperature channels "
                        f"drop across 500C TOGETHER in ONE 5-min sample with both "
                        f"rows valid and no data gap (e.g. {v} {a:.0f} -> {b:.0f} C, "
                        f"{abs(a - b):.0f}C in 5 min), then >= "
                        f"{CONTEXT_RUN * SAMPLE_MINUTES} min in the COLD operating "
                        f"regime{_timing(pd.Timestamp(tvals[j]).hour, h2c_hist, h2c_tot, 'hot->cold')}{lead_txt}"
                        " -> expected operating context: score reflects a stale "
                        "local baseline across the daily shutdown boundary, "
                        "not a process failure")
                    break
                if chain_fail is None:
                    chain_fail = (
                        f"note: an instantaneous hot->cold transition at +{lead} min "
                        f"FAILED the operating-context evidence chain (hot regime "
                        f"before: {nb}/{CONTEXT_RUN} samples, cold after: "
                        f"{nf}/{CONTEXT_RUN}) -> classified by its own evidence "
                        f"instead")

        # a transition DEEP inside the period did not cause its onset: the
        # period stays rankable and the finding is recorded in its evidence
        if ctx is None:
            for j in range(w_end + 1, e + 1):
                if (pd.Timestamp(tvals[j]) - pd.Timestamp(tvals[j - 1])
                        ) != pd.Timedelta(minutes=SAMPLE_MINUTES):
                    continue
                if not (valid_arr[j - 1] and valid_arr[j]):
                    continue
                hits = 0
                for v in temp_cols:
                    a, b = df[v].iloc[j - 1], df[v].iloc[j]
                    if pd.isna(a) or pd.isna(b):
                        continue
                    if (abs(a - b) >= 300 and ((a < COLD_C and b > 400.0)
                                               or (a > 400.0 and b < COLD_C))):
                        hits += 1
                if hits >= 2:
                    late_flip = (
                        f"note: an instantaneous hot<->cold jump occurs "
                        f"{(j - s) * SAMPLE_MINUTES} min AFTER onset ({hits} "
                        f"channels); it did not create the period, so the period "
                        f"stays rankable, but scores after it carry some "
                        f"stale-baseline inflation")
                    break

        # onset carried a huge instantaneous temperature step that never
        # reached the cold boundary: physically not smooth process dynamics,
        # but also not a recorded operating-state transition, so the
        # startup/shutdown chain cannot claim it - state that explicitly
        # instead of silently leaving the reviewer to guess
        step_note = None
        if ctx is None and partials:
            v, a, b, j0 = partials[0]
            lead0 = (j0 - s) * SAMPLE_MINUTES
            where0 = "at onset" if lead0 == 0 else f"at +{lead0} min after onset"
            nch = len({pv for pv, _, _, _ in partials})
            step_note = (
                f"note: starting {where0}, {len(partials)} instantaneous "
                f"temperature step(s) >=300C across {nch} channel(s) appear "
                f"in single 5-min samples, none of them crossing into the "
                f"cold boundary (<100C) (e.g. {v} {a:.0f} -> {b:.0f} C, "
                f"{abs(a - b):.0f}C) - jumps far beyond thermal-mass limits, "
                f"but NOT a recorded hot<->cold operating-state transition "
                f"-> the startup/shutdown evidence chain does not apply, so "
                f"the period stays rankable; scores near the step(s) carry "
                f"some stale-baseline inflation")

        # explicit instrument fault signatures on dominant sensors
        zero_hit = [(v, 100.0 * (seg_raw[v] == 0).mean()) for v in dom
                    if (seg_raw[v] == 0).mean() >= 0.5]
        sat_hit = [(v, 100.0 * (seg_raw[v] >= SATURATION).mean())
                   for v in dom if (seg_raw[v] >= SATURATION).mean() > 0]
        stuck_hit = []
        for v in dom:
            vc = seg_raw[v].value_counts(normalize=True)
            if len(vc) and vc.iloc[0] >= 0.8:
                stuck_hit.append((v, 100.0 * vc.iloc[0],
                                  seg_raw[v].mode().iloc[0]))

        if outage_ev is not None:
            # hierarchy level 1: the score originates at an acquisition-
            # outage boundary (state changed unobserved across missing data)
            cat, ev = "post_outage_baseline", outage_ev
        elif ctx is not None:
            # highest-priority explanation: the period exists because the
            # operating state changed discontinuously, not because
            # operations went abnormal
            cat, ev = ctx
        elif zero_hit:
            v, pct = zero_hit[0]
            cat = "instrument_fault"
            ev = (f"{v}=0.0 in {pct:.0f}% of interval while gas path at "
                  f"{med_inlet:.0f}C -> transmitter/loop failure")
        elif sat_hit:
            v, pct = sat_hit[0]
            cat = "instrument_fault"
            ev = (f"{v} at range limit {SATURATION:.0f}C in {pct:.0f}% of "
                  f"interval -> sensor saturation")
        elif stuck_hit:
            v, pct, val = stuck_hit[0]
            cat = "instrument_fault"
            ev = (f"{v} frozen at {val} for {pct:.0f}% of interval "
                  f"({r['duration_min']:.0f} min) -> stuck value")
        elif len(dom) >= 2:
            cat = "process_correlated"
            ev = f"{len(dom)} coupled sensors deviate together: " + \
                 ", ".join(dom)
        elif len(dom) == 1:
            cat = "single_sensor_deviation"
            ev = (f"{dom[0]} alone outside regime envelope; other sensors "
                  f"consistent with current regime")
        else:
            cat = "transient_spike"
            ev = "excursion above threshold on max-score sensor only"

        if ctx is None and chain_fail is not None:
            ev += " | " + chain_fail
        if late_flip is not None:
            ev += " | " + late_flip
        if outage_note is not None:
            ev += " | " + outage_note
        if step_note is not None:
            ev += " | " + step_note
        if cold_regime:
            ev += " | inside COLD/SHUTDOWN regime"
        cats.append(cat)
        shut_flags.append(cold_regime)
        startup_flags.append(cat == "startup_context")
        evidences.append(ev)

    out = periods.copy()
    out["category"] = cats
    out["shutdown_context"] = shut_flags
    out["startup_context"] = startup_flags
    out["evidence"] = evidences
    return out


def rank_periods(periods: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Severity score and ranking, applied AFTER classification.

    SEPARATION OF QUANTITIES (assignment section 6):
      `magnitude` = ANOMALY MAGNITUDE: the period's MEAN composite score
        divided by the p99 of the rankable period means, clipped to [0,1].
        Mean, not peak: with z capped at 50 the peak sits at the cap for a
        sizeable minority of periods and stops discriminating there; the
        period mean keeps its spread (only a few rankable periods still
        reach the cap after context periods are removed).
      `severity`  = EVENT SEVERITY: magnitude 0.45 + duration 0.25 (log-
        damped, so 98 h is not 100x a 65 min event) + breadth 0.20 (share
        of the 6 sensors involved) + persistence 0.10 (share of interval
        samples flagged), rescaled so the most severe RANKABLE period is
        1.0. Weights retained after re-evaluation on the rankable set:
        severity correlates most with magnitude (intensity leads), then
        breadth and log-duration - intensity dominates while long,
        multi-sensor, fully persistent events are rewarded, and no single
        component crowds the others out. Measured correlations from the
        final run are quoted in README Key Results.

    Non-rankable context periods - `startup_context` and
    `shutdown_transition` (evidence-based expected operating-state
    transitions: the score there reflects a stale local baseline across
    the daily startup/shutdown boundary, not sustained abnormal
    operation) and `post_outage_baseline` (score originates at an
    acquisition-outage boundary whose regime change was never observed) -
    are scored with the same formula but sorted AFTER every rankable
    period, so they can never occupy the severity leaderboard. They remain
    fully reported in abnormal_periods.csv (never hidden) and carry
    startup_context / shutdown_context columns.
    """
    if not len(periods):
        return periods
    p = periods.copy()
    ctx = p["category"].isin(NON_RANKABLE_CATEGORIES).to_numpy()
    ref = p.loc[~ctx, "mean_score"]
    if not len(ref):
        ref = p["mean_score"]
    mag = (p["mean_score"] / ref.quantile(0.99)).clip(0, 1)
    dur = np.log1p(p["duration_min"]) / np.log1p(p["duration_min"].max())
    breadth = p["n_abnormal_vars"] / len(SENSORS)
    persist = p["pct_samples_flagged"] / 100.0
    sev = (cfg.w_magnitude * mag + cfg.w_duration * dur
           + cfg.w_breadth * breadth + cfg.w_persistence * persist)
    scale = sev[~ctx].max()
    if not np.isfinite(scale) or scale <= 0:
        scale = sev.max()
    p["magnitude"] = mag.round(4)
    p["breadth"] = breadth.round(4)
    p["severity"] = (sev / scale).clip(0, 1).round(4)
    p["_rankable"] = ~ctx
    p = p.sort_values(["_rankable", "severity"], ascending=[False, False],
                      kind="mergesort").drop(columns="_rankable")
    p = p.reset_index(drop=True)
    p.insert(0, "rank", np.arange(1, len(p) + 1))
    return p


# ==========================================================================
# 8. VISUALIZATION
# ==========================================================================
def make_plots(df, feats, c, S, if_pct, labels, flag_pts, periods,
               corr, cfg, outdir: Path):
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    t = df["ts"].values
    flagged = flag_pts

    def shade(ax):
        """Overlay abnormal periods as vertical bands."""
        for _, r in periods.iterrows():
            ax.axvspan(pd.Timestamp(r["start"]), pd.Timestamp(r["end"]),
                       color="red", alpha=0.35, lw=0)

    # 1) six sensor series with abnormal periods
    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)
    for ax, col in zip(axes, SENSORS):
        ax.plot(t, df[col].values, lw=0.4, color="#1f3a5f")
        shade(ax)
        ax.set_ylabel(col.replace("Cyclone_", ""), fontsize=8)
    axes[0].set_title("Six sensors with detected abnormal periods (red bands)",
                      fontsize=13)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(plots / "01_sensor_timeseries.png", dpi=110)
    plt.close(fig)

    # 2) distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), SENSORS):
        v = df[col].dropna()
        ax.hist(v, bins=120, color="#1f3a5f")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Sensor distributions (multimodality = operating regimes)")
    fig.tight_layout()
    fig.savefig(plots / "02_distributions.png", dpi=110)
    plt.close(fig)

    # 3) correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Pearson correlation (raw sensors)")
    fig.tight_layout()
    fig.savefig(plots / "03_correlation_heatmap.png", dpi=110)
    plt.close(fig)

    # 4) missing/sentinel pattern
    fig, ax = plt.subplots(figsize=(16, 2.5))
    inv = (~df["valid"]).values
    ax.fill_between(t, 0, 1, where=inv, color="orange", alpha=0.8,
                    step="mid")
    ax.set_title(f"Invalid samples (I/O Timeout / Unit Down / partial): "
                 f"{inv.sum():,} rows", fontsize=11)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(plots / "04_missing_pattern.png", dpi=110)
    plt.close(fig)

    # 5) score distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sv = S.dropna()
    axes[0].hist(sv, bins=200, color="#1f3a5f")
    axes[0].axvline(cfg.z_threshold, color="red", ls="--",
                    label=f"threshold z={cfg.z_threshold}")
    axes[0].set_yscale("log")
    axes[0].set_title("Composite anomaly score S (log count)")
    axes[0].legend()
    axes[1].hist(if_pct.dropna(), bins=200, color="#e06c2f")
    axes[1].set_title("Isolation Forest percentile score")
    fig.tight_layout()
    fig.savefig(plots / "05_score_distribution.png", dpi=110)
    plt.close(fig)

    # 6) regime view (PCA of robust features)
    Xcols = SENSORS
    X = pd.DataFrame({cc: robust_scale(pd.to_numeric(df[cc], errors="coerce"))
                      for cc in Xcols})
    m = df["valid"].values
    pcs = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        X[m].fillna(0).values)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=labels[m].values, s=1,
                    cmap="tab10", alpha=0.5)
    ax.set_title("Operating regimes (PCA of robust-scaled sensors)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(plots / "06_regimes_pca.png", dpi=110)
    plt.close(fig)

    # 7) timeline of periods by severity
    fig, ax = plt.subplots(figsize=(16, 4))
    if len(periods):
        sc = ax.scatter(periods["start"], periods["severity"],
                        c=periods["severity"], cmap="Reds", s=28)
        fig.colorbar(sc, ax=ax, label="severity")
    ax.set_title("Abnormal periods: severity over time")
    fig.tight_layout()
    fig.savefig(plots / "07_period_severity_timeline.png", dpi=110)
    plt.close(fig)

    # 8) zoom on the top period
    if len(periods):
        r = periods.iloc[0]
        window = (r["start"] - pd.Timedelta(hours=6),
                  r["end"] + pd.Timedelta(hours=6))
        msk = (df["ts"] >= window[0]) & (df["ts"] <= window[1])
        sub = df[msk]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for col in SENSORS:
            axes[0].plot(sub["ts"], sub[col], lw=1.2, label=col)
        axes[0].legend(fontsize=7, ncol=3)
        axes[0].set_title(f"Top period zoom: {r['start']} -> {r['end']} "
                          f"(severity {r['severity']:.3f})")
        axes[1].plot(sub["ts"], S[msk], color="red", lw=1.2,
                     label="composite score S")
        axes[1].axhline(cfg.z_threshold, color="k", ls="--")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("S")
        axes[1].legend()
        for ax in axes:
            ax.axvspan(r["start"], r["end"], color="red", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots / "08_top_period_zoom.png", dpi=110)
        plt.close(fig)

    return plots


# ==========================================================================
# 9. VALIDATION WITHOUT GROUND TRUTH (Phase 8)
# ==========================================================================
def validate(df, feats, c, S, if_score, if_pct, labels, flag_pts,
             periods, regimes, cfg) -> dict:
    v = {}
    sv = S.dropna()
    v["p50"], v["p99"], v["max"] = (float(sv.quantile(0.5)),
                                    float(sv.quantile(0.99)),
                                    float(sv.max()))
    # method agreement: rank correlation + top-1% overlap.
    # S is capped at z_cap, so >q99 can be EMPTY when q99 equals the cap
    # (that produced a spurious Jaccard of 0). Use >= to include the tied
    # cap-block, and report set sizes so the tie effect is visible.
    both = pd.DataFrame({"S": S, "IF": if_score}).dropna()
    v["method_corr"] = float(both["S"].rank().corr(both["IF"].rank()))
    q_s, q_i = both["S"].quantile(0.99), both["IF"].quantile(0.99)
    top_s = set(both.index[both["S"] >= q_s])
    top_i = set(both.index[both["IF"] >= q_i])
    v["n_top_s"], v["n_top_i"] = len(top_s), len(top_i)
    v["top1pct_jaccard"] = round(len(top_s & top_i) / max(len(top_s | top_i), 1), 3)
    v["pct_if_top1pct_also_robust_flagged"] = round(
        100.0 * len(top_i & set(df.index[flag_pts])) / max(len(top_i), 1), 1)
    # regime separation - recomputed on an INDEPENDENT subsample with the
    # SAME 10 features used for clustering, as a replication check
    m = df["valid"].values
    derived = ["dT_inlet_outlet", "dT_material_outlet",
               "dP_cone_outlet", "dP_inlet_cone"]
    src = {c: pd.to_numeric(df[c], errors="coerce") for c in SENSORS}
    src.update({c: feats[c] for c in derived})
    Xv = pd.DataFrame({c: robust_scale(src[c]) for c in
                       SENSORS + derived})[m].fillna(0)
    rng = np.random.RandomState(RANDOM_STATE)
    sub = rng.choice(len(Xv), size=min(10_000, len(Xv)), replace=False)
    lab = labels[m].values[sub]
    v["silhouette"] = float(silhouette_score(Xv.values[sub], lab,
                                             random_state=RANDOM_STATE))
    v["db"] = float(davies_bouldin_score(Xv.values[sub], lab))
    v["regime_metrics"] = regimes["metrics"]
    # temporal persistence
    raw_runs = runs_from_flags(flag_pts)
    v["n_raw_runs"] = len(raw_runs)
    v["n_periods"] = len(periods)
    if len(periods):
        v["pct_samples_in_intervals"] = round(
            100.0 * periods["n_samples"].sum() / max(flag_pts.sum(), 1), 1)
        # process validation: multi-sensor simultaneity
        v["pct_multisensor"] = round(
            100.0 * (periods["n_abnormal_vars"] >= 2).mean(), 1)
        v["median_dur_min"] = float(periods["duration_min"].median())
        v["max_dur_min"] = float(periods["duration_min"].max())
    # sensitivity: thresholds x windows (recompute cheaply)
    sens = []
    base_flags = flag_pts.copy()
    for tau in (4.0, 5.0, 6.0):
        for w in (24, 36, 54):
            cfg2 = Config(local_window=w, z_threshold=tau,
                          min_run_samples=cfg.min_run_samples,
                          merge_gap_samples=cfg.merge_gap_samples)
            # recompute only z_local (the threshold-dependent part)
            fl = _flags_for(df, feats, labels, cfg2)
            jac = len(np.flatnonzero(fl & base_flags)) / max(
                len(np.flatnonzero(fl | base_flags)), 1)
            sens.append({"z": tau, "window": w,
                         "jaccard_vs_base": round(jac, 3),
                         "flagged": int(fl.sum())})
    v["sensitivity"] = sens
    v["jaccard_range"] = (f"{min(s['jaccard_vs_base'] for s in sens):.2f}"
                          f"-{max(s['jaccard_vs_base'] for s in sens):.2f}")
    return v


def _flags_for(df, feats, labels, cfg2) -> np.ndarray:
    """Recompute point flags under an alternative (threshold, window) config.
    z_regime and IF are config-invariant here; only the causal window and
    threshold change - exactly what sensitivity analysis should probe."""
    _, z_regime = regime_baselines(df, labels, cfg2)
    c2 = pd.DataFrame(index=feats.index, columns=SENSORS, dtype=float)
    w, mp = cfg2.local_window, cfg2.local_min_periods
    for v in SENSORS:
        x = df[v]
        med = x.rolling(w, min_periods=mp).median()
        mad = (x - med).abs().rolling(w, min_periods=mp).median()
        iqr = x.quantile(0.75) - x.quantile(0.25)
        zl = ((x - med) / np.maximum(1.4826 * mad, 0.05 * iqr)).clip(
            -cfg2.z_cap, cfg2.z_cap)
        c2[v] = np.maximum(zl.abs(), z_regime[v].abs())
    S2 = c2.max(axis=1)
    S2[~df["valid"].to_numpy()] = np.nan
    fl = (S2 > cfg2.z_threshold).to_numpy(copy=True)
    fl[~df["valid"].to_numpy()] = False
    # persistence applied for comparability
    out = np.zeros(len(fl), bool)
    for s, e in persistence_filter(fl, cfg2.min_run_samples,
                                   cfg2.merge_gap_samples):
        out[s:e + 1] = True
    return out


# ==========================================================================
# 10. MAIN
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help='assignment CSV, e.g. "C:/Users/dell/Downloads/Data (1).csv" '
                         "(the original assignment PDF is also accepted)")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    t0 = time.time()

    # -- load & validate -------------------------------------------------
    raw = load_source(Path(args.input), outdir)
    validate_schema(raw)
    df = preprocess(raw)

    # -- Phase 1 audit ---------------------------------------------------
    rep = audit(df)
    (outdir / "audit_report.txt").write_text(rep.pop("_report"),
                                             encoding="utf-8")

    # -- Phase 3 features & regimes --------------------------------------
    feats = build_features(df, cfg)
    labels, regimes = detect_regimes(df, feats, cfg)
    df["regime"] = labels

    # -- Phase 4/5 scoring -----------------------------------------------
    _, z_regime = regime_baselines(df, labels, cfg)
    c, S = composite_score(feats, z_regime, cfg)
    if_score, if_pct = isolation_forest_score(df, feats, cfg)

    # point flags: composite contextual score above threshold.
    # (IF is corroboration/evidence, not a gate - see README.)
    # .copy(): pandas CoW returns read-only views from .values
    flag_pts = (S > cfg.z_threshold).to_numpy(copy=True)
    flag_pts[~df["valid"].to_numpy()] = False   # belt-and-braces

    # -- Phase 6/7 periods ------------------------------------------------
    periods = build_periods(df, feats, c, S, if_pct, labels, flag_pts, cfg)
    periods = classify_periods(df, c, periods, labels, regimes["centers"], cfg)
    periods = rank_periods(periods, cfg)

    # -- Phase 8 validation ----------------------------------------------
    val = validate(df, feats, c, S, if_score, if_pct, labels, flag_pts,
                   periods, regimes, cfg)

    # -- outputs ----------------------------------------------------------
    periods.to_csv(outdir / "abnormal_periods.csv", index=False)
    periods.head(15).to_csv(outdir / "top_15_abnormal_periods.csv", index=False)

    # acquisition outages (sentinel rows) reported SEPARATELY: they are
    # data-availability events, never scored as process abnormalities
    invalid = (~df["valid"]).values
    outages = []
    for s, e in runs_from_flags(invalid):
        outages.append({
            "start": df["ts"].iloc[s],
            "end": df["ts"].iloc[e],
            "duration_min": (e - s + 1) * SAMPLE_MINUTES,
            "n_samples": e - s + 1,
            "n_sensors_missing": int(df.iloc[s:e + 1][SENSORS].isna()
                                     .any(axis=1).sum()),
        })
    pd.DataFrame(outages).to_csv(outdir / "acquisition_outages.csv",
                                 index=False)

    scored = df[["ts"] + SENSORS + ["regime", "valid"]].copy()
    scored["composite_score"] = S.round(3)
    scored["if_percentile"] = if_pct.round(2)
    for v in SENSORS:
        scored[f"z_{v}"] = c[v].round(2)
    scored["flag_raw"] = flag_pts.astype(int)
    persisted = np.zeros(len(df), bool)
    for s, e in persistence_filter(flag_pts, cfg.min_run_samples,
                                   cfg.merge_gap_samples):
        persisted[s:e + 1] = True
    # The merge step can bridge a <=15-min sentinel outage inside one event,
    # so interval spans may include invalid rows. Sample-level flags must not:
    # a sentinel row is never evidence of a process abnormality.
    persisted[~df["valid"].to_numpy()] = False
    scored["flag_period"] = persisted.astype(int)
    scored.to_csv(outdir / "anomaly_scored_data.csv", index=False)

    # summary
    summary_rows = [
        ("records", len(df)),
        ("valid_records", int(df["valid"].sum())),
        ("sentinel_records", int((~df["valid"]).sum())),
        ("date_start", str(df["ts"].min())),
        ("date_end", str(df["ts"].max())),
        ("sampling_interval_min", SAMPLE_MINUTES),
        ("local_window_samples", cfg.local_window),
        ("z_threshold", cfg.z_threshold),
        ("min_period_samples", cfg.min_run_samples),
        ("merge_gap_samples", cfg.merge_gap_samples),
        ("regime_k", regimes["k"]),
        ("regime_silhouette", regimes["metrics_best"]["silhouette"]),
        ("flagged_samples_raw", int(flag_pts.sum())),
        ("flagged_samples_persisted", int(persisted.sum())),
        ("n_abnormal_periods", len(periods)),
        ("periods_process_correlated",
         int((periods["category"] == "process_correlated").sum()) if len(periods) else 0),
        ("periods_instrument_fault",
         int((periods["category"] == "instrument_fault").sum()) if len(periods) else 0),
        ("periods_single_sensor",
         int((periods["category"] == "single_sensor_deviation").sum()) if len(periods) else 0),
        ("periods_startup_context",
         int((periods["category"] == "startup_context").sum()) if len(periods) else 0),
        ("periods_shutdown_transition",
         int((periods["category"] == "shutdown_transition").sum()) if len(periods) else 0),
        ("periods_post_outage_baseline",
         int((periods["category"] == "post_outage_baseline").sum()) if len(periods) else 0),
        ("periods_rankable",
         int((~periods["category"].isin(NON_RANKABLE_CATEGORIES)).sum()) if len(periods) else 0),
        ("periods_in_shutdown_context",
         int(periods["shutdown_context"].sum()) if len(periods) else 0),
        ("acquisition_outage_runs", len(outages)),
        ("median_duration_min", float(periods["duration_min"].median()) if len(periods) else 0),
        ("max_duration_min", float(periods["duration_min"].max()) if len(periods) else 0),
        ("sensitivity_jaccard_range", val["jaccard_range"]),
        ("method_agreement_spearman", round(val["method_corr"], 3)),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(
        outdir / "analysis_summary.csv", index=False)

    # results.json for the PPT builder
    top = periods.head(15) if len(periods) else pd.DataFrame()
    res = {
        "data": {
            "rows": len(df), "cols": 7,  # raw schema: time + 6 sensors
            "start": rep["start"], "end": rep["end"],
            "span_days": rep["span_days"],
            "years": f"{df['ts'].dt.year.min()}-{df['ts'].dt.year.max()}",
            "missing": int((~df["valid"]).sum()),
            "dup_ts": rep["dup_ts"], "dup_rows": rep["dup_rows"],
            "max_gap_min": rep["max_gap_min"], "n_gaps": rep["n_gaps"],
            "quality_findings": [
                "two timestamp locales (DD-MM-YYYY and M/D/YYYY) in one file",
                f"file is NOT time-sorted (text-sorted export) - sorted in preprocessing",
                f"{rep['sentinel_rows_any']:,} rows contain SCADA sentinel tokens "
                f"({rep['sentinel_tokens']})",
            ],
        },
        "results": {
            "n_flagged_samples": int(persisted.sum()),
            "pct_flagged": 100.0 * persisted.sum() / len(df),
            "n_periods": len(periods),
            "n_process_correlated": int((periods["category"] == "process_correlated").sum()) if len(periods) else 0,
            "n_instrument_fault": int((periods["category"] == "instrument_fault").sum()) if len(periods) else 0,
            "n_single_sensor": int((periods["category"] == "single_sensor_deviation").sum()) if len(periods) else 0,
            "n_startup_context": int((periods["category"] == "startup_context").sum()) if len(periods) else 0,
            "n_shutdown_transition": int((periods["category"] == "shutdown_transition").sum()) if len(periods) else 0,
            "n_post_outage_baseline": int((periods["category"] == "post_outage_baseline").sum()) if len(periods) else 0,
            "n_rankable": int((~periods["category"].isin(NON_RANKABLE_CATEGORIES)).sum()) if len(periods) else 0,
            "n_shutdown_context": int(periods["shutdown_context"].sum()) if len(periods) else 0,
            "n_outage_runs": len(outages),
            "median_dur_min": val.get("median_dur_min", 0),
            "max_dur_min": val.get("max_dur_min", 0),
            "max_dur_h": val.get("max_dur_min", 0) / 60.0,
            "dominant_vars": _dominant_vars(periods),
        },
        "validation": val,
        "top15_table": _top_table(top),
    }
    (outdir / "results.json").write_text(json.dumps(res, indent=2,
                                                    default=str),
                                         encoding="utf-8")

    # -- plots -------------------------------------------------------------
    make_plots(df, feats, c, S, if_pct, labels, flag_pts, periods,
               pd.DataFrame(rep["corr"]), cfg, outdir)

    # -- final report -------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL REPORT")
    print("=" * 72)
    print(f"records                 : {len(df):,} "
          f"({int(df['valid'].sum()):,} valid)")
    print(f"regimes (k)             : {regimes['k']}  "
          f"silhouette={regimes['metrics_best']['silhouette']}")
    print(f"flagged samples (raw)   : {int(flag_pts.sum()):,} "
          f"({100 * flag_pts.mean():.2f}%)")
    print(f"flagged (persisted)     : {int(persisted.sum()):,} "
          f"({100 * persisted.mean():.2f}%)")
    print(f"abnormal periods        : {len(periods)}")
    if len(periods):
        print("  by category           :")
        for cat, n in periods["category"].value_counts().items():
            print(f"    {cat:26s}: {n}")
        n_ctx = int(periods["category"].isin(NON_RANKABLE_CATEGORIES).sum())
        n_post = int((periods["category"] == "post_outage_baseline").sum())
        print(f"  expected operating context (startup_context + "
              f"shutdown_transition): {n_ctx - n_post}")
        print(f"  acquisition-level context (post_outage_baseline): {n_post}")
        print(f"  rankable candidates   : {len(periods) - n_ctx} "
              f"(severity leaderboard built from these only)")
        print(f"  in shutdown context   : "
              f"{int(periods['shutdown_context'].sum())}")
        print(f"median duration         : {periods['duration_min'].median():.0f} min")
        print(f"longest period          : {periods['duration_min'].max():.0f} min "
              f"({periods['duration_min'].max() / 60:.1f} h)")
        print(f"periods with >=2 sensors: "
              f"{int((periods['n_abnormal_vars'] >= 2).sum())}/{len(periods)}")
        print("\nTOP 10 BY SEVERITY (rankable periods only):")
        cols = ["rank", "start", "end", "duration_min", "severity",
                "magnitude", "category", "n_abnormal_vars", "abnormal_variables"]
        print(periods[cols].head(10).to_string(index=False))
    print(f"\nacquisition outage runs : {len(outages)} "
          f"(reported separately, never scored)")
    print(f"\nmethod agreement (S vs IF, Spearman): {val['method_corr']:.3f}")
    print(f"sensitivity Jaccard range            : {val['jaccard_range']}")
    print(f"elapsed: {time.time() - t0:.0f}s")
    print(f"outputs written to: {outdir}")


def _dominant_vars(periods: pd.DataFrame) -> str:
    if not len(periods):
        return "n/a"
    from collections import Counter
    cnt = Counter()
    for s in periods["abnormal_variables"]:
        for v in str(s).split("; "):
            if v:
                cnt[v] += 1
    return ", ".join(f"{k} ({v})" for k, v in cnt.most_common(3))


def _top_table(top: pd.DataFrame):
    rows = []
    for _, r in top.iterrows():
        rows.append([
            int(r["rank"]), str(r["start"]), str(r["end"]),
            int(r["duration_min"]), f"{r['severity']:.3f}",
            str(r.get("category", "")),
            str(r["abnormal_variables"])[:52],
        ])
    return rows


if __name__ == "__main__":
    main()
