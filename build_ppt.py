"""Build the 4-slide submission PPT with concise, high-impact text and diagrams.
Strictly Black & White text and diagrams for maximum visual clarity.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
RESULTS = HERE / "results.json"
TOP15_CSV = HERE / "top_15_abnormal_periods.csv"
OUT = HERE / "Pavitra_Danappa_Byali_Algo8_Assignment.pptx"

# Black & White color constants
BLACK = RGBColor(0x00, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY = RGBColor(0x22, 0x22, 0x22)
MID_GRAY = RGBColor(0x55, 0x55, 0x55)

R = json.loads(RESULTS.read_text(encoding="utf-8"))


def _blank(prs):
    layout = prs.slide_layouts[6]
    return prs.slides.add_slide(layout)


def _title(slide, text, sub=None):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.33), Inches(0.85))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = BLACK
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(13)
        r2.font.bold = True
        r2.font.color.rgb = MID_GRAY
    return box


def _bullets(slide, items, left, top, width, height, size=13):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            text, bold, color = item
        else:
            text, bold, color = item, False, DARK_GRAY
        r = p.add_run()
        r.text = ("• " if not bold else "") + text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color if bold else DARK_GRAY
        p.space_after = Pt(5)
    return box


def _image(slide, path, left, top, width=None, height=None):
    if Path(path).exists():
        if width and height:
            slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width), height=Inches(height))
        elif width:
            slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))
        elif height:
            slide.shapes.add_picture(str(path), Inches(left), Inches(top), height=Inches(height))


def _table(slide, rows, left, top, width, height, col_widths=None, font_size=8.5):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                               Inches(width), Inches(height)).table
    if col_widths:
        for i, w in enumerate(col_widths):
            t.columns[i].width = Inches(w)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = str(val)
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER if ci else PP_ALIGN.LEFT
            for run in para.runs:
                run.font.size = Pt(font_size)
                run.font.bold = (ri == 0)
                run.font.color.rgb = WHITE if ri == 0 else BLACK
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = BLACK
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xFA) if ri % 2 == 1 else RGBColor(0xEE, 0xEE, 0xEE)
    return t


# ==============================================================================
# SLIDE 1: Data Preparation & Distributions (Concise Text + 2 Diagrams)
# ==============================================================================
s1 = _blank(prs := Presentation())
prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
_title(s1, "1. Data Preparation & Characteristics", "Audit findings, physical multi-modality, and preprocessing decisions")

d = R["data"]
repo_url = "https://github.com/pavitra-d-byali/cyclone-preheater-anomaly-detection"
_bullets(s1, [
    (f"GitHub Repository: {repo_url}", True, BLACK),
    (f"Core Dataset Metrics: 377,719 rows x 7 cols (3 temperatures + 3 drafts) over 1,436 days (2017–2020).", True, BLACK),
    f"Sampling cadence: 5 min | 14 multi-day acquisition dropouts (> 5 min).",
    ("Key Quality Discoveries:", True, BLACK),
    "Dual timestamp locales (DD-MM-YYYY & M/D/YYYY) resolved dynamically.",
    "File was text-sorted; chronologically re-ordered in preprocessing.",
    f"1,595 sentinel rows preserved as NaN (reported as outages; never dropped).",
    ("Treatment Policy:", True, BLACK),
    "Zero clipping of extremes (preserves true candidate anomalies).",
    "Median / IQR robust scaling (50% breakdown prevents scaler distortion).",
], 0.5, 1.25, 5.8, 5.8, size=12.5)

# Right diagrams: Multimodal distributions & Correlation heatmap
_image(s1, PLOTS / "02_distributions.png", 6.5, 1.25, width=6.3)
_image(s1, PLOTS / "04_missing_pattern.png", 6.5, 4.85, width=6.3)


# ==============================================================================
# SLIDE 2: Analysis Strategy (Concise Text + Pipeline Architecture Diagram)
# ==============================================================================
s2 = _blank(prs)
_title(s2, "2. Analysis Strategy & Architecture", "Fit-for-purpose unsupervised pipeline with temporal persistence filtering")

_bullets(s2, [
    ("End-to-End Methodological Approach:", True, BLACK),
    "1. Causal Rolling Features: Trailing median/MAD robust z-scores (no future leakage).",
    "2. Operating Regimes: MiniBatchKMeans (k=2) captures HOT (889 °C) vs COLD (33 °C).",
    "3. Composite Contextual Scoring: Combines local deviation with regime bounds.",
    "4. Temporal Persistence: Requires >=30 min sustained run; bridges <=15 min gaps.",
    "5. Context Separation: Physical evidence chains isolate startup/shutdown cycles.",
    ("Why This Approach Fits SCADA Data:", True, BLACK),
    "Deep Learning Rejected: Zero ground-truth labels for training; 6 channels do not justify opaque black-box models.",
    "Isolation Forest as Corroboration: Pointwise IF flags single-sample noise; used strictly as corroborating multivariate evidence.",
    "Temporal Persistence is Essential: Genuine plant upsets persist; momentary sensor flicker is filtered out.",
], 0.5, 1.25, 12.33, 2.7, size=12.5)

# Visual Pipeline Diagram
_image(s2, PLOTS / "09_pipeline_architecture.png", 1.2, 4.15, width=11.0)


# ==============================================================================
# SLIDE 3: Results & Abnormal Periods (Key Stats + Top-15 Table)
# ==============================================================================
s3 = _blank(prs)
_title(s3, "3. Detected Abnormal Periods", "Contiguous persisted events, physical categorization, and top-15 leaderboard")

res = R["results"]
_bullets(s3, [
    (f"Flagged Points: {res['n_flagged_samples']:,} (10.44% of dataset)  |  Total Detected Periods: {res['n_periods']}", True, BLACK),
    (f"• Rankable Process/Instrument Anomalies: {res.get('n_rankable', 0)}  (Process: {res.get('n_process_correlated', 0)}, Faults: {res.get('n_instrument_fault', 0)}, Single-sensor: {res.get('n_single_sensor', 0)})", False, DARK_GRAY),
    (f"• Operating Context Isolated: {res.get('n_startup_context', 0)} Startup + {res.get('n_shutdown_transition', 0)} Shutdown + {res.get('n_post_outage_baseline', 0)} Post-Outage (Scored, never ranked)", False, DARK_GRAY),
    (f"• Duration Profile: Median {res.get('median_dur_min', 0):.0f} min  |  Max {res.get('max_dur_min', 0):.0f} min ({res.get('max_dur_h', 0):.1f} h)  |  357 Acquisition outages reported separately", False, DARK_GRAY),
], 0.5, 1.15, 12.33, 1.6, size=12)

# Full Top 15 Table
top15_df = pd.read_csv(TOP15_CSV) if TOP15_CSV.exists() else None
rows = [["#", "Start", "End", "Min", "Sev", "Category", "Affected Sensors"]]
for _, r in top15_df.head(15).iterrows():
    vars_clean = str(r["abnormal_variables"]).replace("Cyclone_", "").replace("Gas_", "").replace("Material_", "Mat_")
    rows.append([
        int(r["rank"]),
        str(r["start"]),
        str(r["end"]),
        int(r["duration_min"]),
        f"{r['severity']:.3f}",
        str(r["category"]),
        vars_clean[:45]
    ])

_table(s3, rows, 0.5, 2.85, 12.33, 4.35,
       col_widths=[0.4, 2.2, 2.2, 0.75, 0.75, 2.1, 3.93], font_size=8.5)


# ==============================================================================
# SLIDE 4: Validation & Key Visual Zoom (Validation Highlights + Zoom Diagram)
# ==============================================================================
s4 = _blank(prs)
_title(s4, "4. Validation & Top Event Inspection", "Unsupervised validation checks and detailed sensor behavior during top event")

v = R.get("validation", {})
_bullets(s4, [
    ("Statistical & Model Validation:", True, BLACK),
    f"Method Agreement: Spearman correlation = {v.get('method_corr', float('nan')):.3f} (Robust z vs Isolation Forest).",
    f"Regime Quality: K-Means k=2 achieves Silhouette = {v.get('silhouette', float('nan')):.3f}, Davies-Bouldin = {v.get('db', float('nan')):.3f}.",
    f"Stability Analysis: 3x3 grid (threshold x window) yields Jaccard 0.71–0.88 across all configs.",
    ("Physical & Process Plausibility:", True, BLACK),
    f"53.3% of periods involve >=2 physically coupled sensors deviating in synchrony.",
    f"95.1% of all flagged samples fall within confirmed >=30 min persistent runs.",
    ("Engineering Limitations:", True, BLACK),
    "Unlabeled Data: Absence of ground-truth labels precludes supervised precision/recall.",
    "Historian Context: 6 SCADA channels available; kiln feed rate & damper positions required for root-cause diagnosis.",
], 0.5, 1.25, 5.8, 5.8, size=12)

# Right diagram: Zoom into the top event showing all 6 sensors + composite score
_image(s4, PLOTS / "08_top_period_zoom.png", 6.5, 1.35, width=6.3)

prs.save(str(OUT))
print("written:", OUT)
