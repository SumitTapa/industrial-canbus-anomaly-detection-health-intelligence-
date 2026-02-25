"""
generate_pdf_report.py
Phase 7: Self-contained PDF report with matplotlib charts + fpdf2 layout.

Input:  data/raw/simulated_can_data.csv
        data/processed/layer1_results.csv
        data/processed/layer2_knn_results.csv
        data/processed/overall_health_index.csv
        data/processed/rul_predictions.csv
Output: reports/anomaly_detection_report.pdf

Compatible with fpdf2 >= 2.7 (uses XPos/YPos enums, no ln= parameter).
All text uses ASCII-safe characters only (no en-dash, em-dash, or special Unicode).
"""

import os
import math
import tempfile
from pathlib import Path

# -- Dependency checks --
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("ERROR: matplotlib not found. Run: pip3 install matplotlib")

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    raise SystemExit("ERROR: fpdf2 not found. Run: pip3 install fpdf2")

try:
    import pandas as pd
except ImportError:
    raise SystemExit("ERROR: pandas not found. Run: pip3 install pandas")


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
TMP = Path(tempfile.mkdtemp())

# Colour palette
DARK_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT  = "#00d2ff"
GREEN   = "#00ff88"
RED     = "#ff4444"
YELLOW  = "#ffcc00"

# fpdf2 v2.7+ line-break shorthand (replaces deprecated ln=True)
NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def style_ax(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, color="#333", linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, color=ACCENT, fontsize=10, fontweight="bold", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, color="white", fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color="white", fontsize=8)


def save_fig(fig, name):
    path = str(TMP / name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    return path


def downsample(df, n=2000):
    step = max(1, len(df) // n)
    return df.iloc[::step].reset_index(drop=True)


# -- Chart generators --
def chart_health(df_hi):
    s = downsample(df_hi)
    xs = range(len(s))
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=DARK_BG)
    ax.fill_between(xs, s["smoothed_health_index"], color=ACCENT, alpha=0.22)
    ax.plot(xs, s["smoothed_health_index"], color=ACCENT, lw=1.8, label="Smoothed Health (EMA)")
    ax.axhline(75, color=YELLOW, lw=1.2, ls="--", label="Warning 75%")
    ax.axhline(50, color="orange", lw=1.0, ls="--", label="Degraded 50%")
    ax.axhline(25, color=RED, lw=1.2, ls="--", label="Critical 25%")
    ax.set_ylim(-2, 105)
    style_ax(ax, "Overall Equipment Health Index (0-100%)", "Health Score (%)", "Time Steps")
    ax.legend(loc="lower left", fontsize=8, facecolor=CARD_BG, labelcolor="white", edgecolor="#555")
    return save_fig(fig, "health.png")


def chart_rul(df_rul):
    s = downsample(df_rul)

    def parse(v):
        if isinstance(v, str) and v.endswith("s") and v[:-1].lstrip("-").isdigit():
            return int(v[:-1])
        return None

    xs_p, ys_p = [], []
    for idx, v in enumerate(s["estimated_rul"]):
        parsed = parse(v)
        if parsed is not None:
            xs_p.append(idx)
            ys_p.append(parsed)

    fig, ax = plt.subplots(figsize=(11, 4), facecolor=DARK_BG)
    if xs_p:
        ax.plot(xs_p, ys_p, color=RED, lw=1.5, label="RUL (seconds)")
        ax.fill_between(xs_p, ys_p, color=RED, alpha=0.15)
    style_ax(ax, "Remaining Useful Life - Seconds Until Critical Failure",
             "Predicted Seconds", "Time Steps")
    ax.legend(fontsize=8, facecolor=CARD_BG, labelcolor="white", edgecolor="#555")
    return save_fig(fig, "rul.png")


def chart_sensors(df_raw):
    s = downsample(df_raw)
    channels = [
        ("rpm", "RPM", "#4fc3f7"),
        ("temperature", "Temp (C)", "#ffa657"),
        ("vibration", "Vib (g)", GREEN),
        ("oil_pressure", "Oil (bar)", "#d2a8ff"),
        ("voltage", "Volt (V)", YELLOW),
    ]
    fig, axes = plt.subplots(len(channels), 1, figsize=(11, 12), facecolor=DARK_BG, sharex=True)
    for ax, (col, label, color) in zip(axes, channels):
        ax.plot(range(len(s)), s[col], color=color, lw=0.9)
        style_ax(ax, ylabel=label)
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("Time Steps", color="white", fontsize=8)
    axes[0].set_title("Raw Sensor Telemetry - 5 Channels", color=ACCENT, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_fig(fig, "sensors.png")


def chart_anomaly_layers(df_l1, df_l2):
    s1, s2 = downsample(df_l1), downsample(df_l2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), facecolor=DARK_BG)

    sev = s1["layer1_severity"].clip(upper=10)
    ax1.bar(range(len(s1)), sev, color=YELLOW, alpha=0.75, width=1)
    style_ax(ax1, "Layer 1 - Statistical Severity (0-10)", "Severity")

    ax2.plot(range(len(s2)), s2["multivariate_distance"], color="#8b949e", lw=0.8, label="KNN Distance")
    anom_mask = s2["is_knn_anomaly"] == 1
    ax2.scatter(s2.index[anom_mask], s2["multivariate_distance"][anom_mask],
                color=RED, s=8, zorder=5, label="Anomaly")
    style_ax(ax2, "Layer 2 - KNN Euclidean Distance", "Distance")
    ax2.legend(fontsize=8, facecolor=CARD_BG, labelcolor="white", edgecolor="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_fig(fig, "anomalies.png")


def chart_correlation(df_raw):
    cols = ["rpm", "temperature", "vibration", "oil_pressure", "voltage"]
    labels = ["RPM", "Temp", "Vib", "Oil", "Volt"]
    corr = df_raw[cols].corr().values
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    for r in range(len(labels)):
        for c in range(len(labels)):
            txt_color = "white" if abs(corr[r, c]) > 0.45 else "#333"
            ax.text(c, r, f"{corr[r, c]:.2f}", ha="center", va="center", fontsize=9, color=txt_color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_facecolor(CARD_BG)
    ax.set_title("Pearson Correlation Matrix", color=ACCENT, fontsize=10, fontweight="bold")
    return save_fig(fig, "corr.png")


# -- PDF class (fpdf2 v2.7+ compatible) --
class Report(FPDF):
    AR, AG, AB = 0, 210, 255
    BR, BG_, BB = 18, 18, 46      # BG_ to avoid shadowing

    def header(self):
        self.set_fill_color(self.BR, self.BG_, self.BB)
        self.rect(0, 0, 210, 16, "F")
        self.set_text_color(self.AR, self.AG, self.AB)
        self.set_font("Helvetica", "B", 9)
        self.set_y(5)
        self.cell(0, 6, "Industrial CAN Bus Telemetry | Anomaly Detection Report", align="C")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def cover(self):
        self.add_page()
        self.set_fill_color(self.BR, self.BG_, self.BB)
        self.rect(0, 0, 210, 297, "F")

        self.set_y(65)
        self.set_text_color(self.AR, self.AG, self.AB)
        self.set_font("Helvetica", "B", 26)
        self.cell(0, 13, "Industrial Asset Monitoring", align="C", **NL)
        self.set_font("Helvetica", "B", 17)
        self.cell(0, 10, "Anomaly Detection & Predictive Maintenance", align="C", **NL)

        self.ln(4)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(160, 160, 200)
        self.cell(0, 8, "CAN Bus Telemetry | 20,000 Engine Events | 5 Injected Faults", align="C", **NL)

        self.ln(16)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(255, 200, 0)
        phases = [
            "Phase 1-2 : Data Simulation       (data_simulation.py)",
            "Phase 3   : Preprocessing          (preprocessing.py)",
            "Phase 3.5 : Industrial EDA         (feature_engineering.py)",
            "Phase 4   : Anomaly Detection       (anomaly_models.py)",
            "            Layer 1 - Stats | Layer 2 - Custom KNN",
            "Phase 5   : Health Index            (train.py)",
            "Phase 6   : RUL & Comparison        (evaluate.py)",
            "Phase 7   : This PDF Report          (generate_pdf_report.py)",
        ]
        for p in phases:
            self.cell(0, 7, p, align="C", **NL)

        # Fault table
        self.ln(12)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(self.AR, self.AG, self.AB)
        self.cell(0, 8, "Injected Fault Schedule", align="C", **NL)
        self.ln(2)
        faults = [
            ("10000-10050", "Oil Pressure", "Gradual lubrication failure"),
            ("12000-12100", "Temperature",  "Thermal overload spike (+25 C)"),
            ("14000-14020", "Voltage",      "Electrical transient spike"),
            ("15000-15100", "Vibration",    "Bearing wear surge (+0.5 g)"),
            ("17000-17050", "RPM",          "Partial stall / load loss"),
        ]
        cw = [40, 35, 100]
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(self.AR, self.AG, self.AB)
        hdrs = ["Step Range", "Sensor", "Description"]
        x0 = (210 - sum(cw)) / 2
        self.set_x(x0)
        for h, w in zip(hdrs, cw):
            self.cell(w, 7, h, border=1, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(220, 220, 220)
        for step, sensor, desc in faults:
            self.set_x(x0)
            self.cell(cw[0], 7, step, border=1, align="C")
            self.cell(cw[1], 7, sensor, border=1, align="C")
            self.cell(cw[2], 7, desc, border=1)
            self.ln()

    def section_title(self, text):
        self.ln(4)
        self.set_fill_color(0, 70, 110)
        self.set_text_color(self.AR, self.AG, self.AB)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, f"  {text}", fill=True, **NL)
        self.ln(2)

    def body_text(self, text, rgb=(200, 200, 200)):
        self.set_text_color(*rgb)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def add_chart(self, img_path, w=188, caption=""):
        if not os.path.exists(img_path):
            self.body_text(f"[Chart missing: {img_path}]", rgb=(200, 80, 80))
            return
        x = (210 - w) / 2
        self.image(img_path, x=x, w=w)
        if caption:
            self.set_text_color(130, 130, 130)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, caption, align="C", **NL)
        self.ln(3)

    def kv_table(self, rows):
        cw = 94
        self.set_font("Helvetica", "", 9)
        for key, val in rows:
            self.set_fill_color(28, 28, 58)
            self.set_text_color(140, 190, 255)
            self.cell(cw, 7, f"  {key}", border=1, fill=True)
            self.set_fill_color(20, 20, 45)
            self.set_text_color(230, 230, 230)
            self.cell(cw, 7, f"  {val}", border=1, fill=True, **NL)
        self.ln(3)


def main():
    print("=" * 62)
    print("  Phase 7: PDF Report Generator")
    print("=" * 62)

    def csv_load(name):
        path = PROC_DIR / name
        if not path.exists():
            path = RAW_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing: {name}  (looked in {RAW_DIR} and {PROC_DIR})")
        return pd.read_csv(path)

    print("  Loading pipeline outputs...")
    df_raw = csv_load("simulated_can_data.csv")
    df_l1  = csv_load("layer1_results.csv")
    df_l2  = csv_load("layer2_knn_results.csv")
    df_hi  = csv_load("overall_health_index.csv")
    df_rul = csv_load("rul_predictions.csv")

    n        = len(df_raw)
    l1_anom  = int(df_l1["layer1_anomaly"].sum())
    l2_anom  = int(df_l2["is_knn_anomaly"].sum())
    final_hi = float(df_hi["smoothed_health_index"].iloc[-1])
    final_st = str(df_hi["health_status"].iloc[-1])

    rul_last = "Stable"
    for v in reversed(df_rul["estimated_rul"].tolist()):
        if isinstance(v, str) and v.endswith("s") and v != "0 (FAILED)":
            rul_last = v
            break

    print(f"  Records : {n:,}")
    print(f"  L1 anom : {l1_anom:,}  ({100*l1_anom/n:.2f}%)")
    print(f"  L2 anom : {l2_anom:,}  ({100*l2_anom/n:.2f}%)")
    print(f"  Health  : {final_hi:.1f}%  [{final_st}]")

    print("\n  Generating charts...")
    img_health  = chart_health(df_hi)
    img_rul     = chart_rul(df_rul)
    img_sensors = chart_sensors(df_raw)
    img_anomaly = chart_anomaly_layers(df_l1, df_l2)
    img_corr    = chart_correlation(df_raw)

    print("  Building PDF...")
    pdf = Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=11, top=18, right=11)

    # Cover
    pdf.cover()

    # 1. Executive Summary
    pdf.add_page()
    pdf.section_title("1. Executive Summary")
    pdf.kv_table([
        ("Total Engine Events Analysed",    f"{n:,}"),
        ("Layer 1 (Statistical) Anomalies", f"{l1_anom:,}  ({100*l1_anom/n:.2f}%)"),
        ("Layer 2 (KNN AI) Anomalies",      f"{l2_anom:,}  ({100*l2_anom/n:.2f}%)"),
        ("Final Health Score",              f"{final_hi:.1f}%"),
        ("Final Health Status",             final_st),
        ("Last Active RUL",                 rul_last),
    ])
    pdf.body_text(
        "This report covers 20,000 seconds (~5.5 hours) of engine telemetry. "
        "Five fault events were injected across all sensor channels: "
        "oil pressure drop (step 10k), temperature spike (12k), voltage transient (14k), "
        "vibration surge (15k), and RPM stall (17k)."
    )

    # 2. Health Index
    pdf.section_title("2. Equipment Health Index Timeline")
    pdf.body_text(
        "Health combines L1 severity penalties (max 25 pts) and L2 KNN distance "
        "penalties (max 20 pts), smoothed with EMA (alpha=0.05). "
        "Tiers: GOOD >= 75% | WARNING 50-75% | DEGRADED 25-50% | CRITICAL < 25%."
    )
    pdf.add_chart(img_health, caption="Figure 1 - EMA-smoothed health index with threshold lines.")

    # 3. RUL
    pdf.add_page()
    pdf.section_title("3. Remaining Useful Life (RUL)")
    pdf.body_text(
        "RUL is estimated via a 500-step rolling degradation window. "
        "Positive slope = STABLE. Negative slope = extrapolate seconds to 25%. "
        "Gaps represent STABLE or warmup periods."
    )
    pdf.add_chart(img_rul, caption="Figure 2 - RUL countdown in seconds.")

    # 4. Raw Sensors
    pdf.add_page()
    pdf.section_title("4. Raw Sensor Telemetry")
    pdf.body_text(
        "Five CAN bus channels over 20,000 steps. Each injected fault is visible."
    )
    pdf.add_chart(img_sensors, caption="Figure 3 - All sensor channels stacked.")

    # 5. Anomaly Layers
    pdf.add_page()
    pdf.section_title("5. Multi-Layer Anomaly Detection")
    pdf.body_text(
        "Layer 1: IQR (RPM), 3-sigma (Volt/Oil), drift (Temp), ROC (Vib).\n\n"
        "Layer 2: Custom KNN (k=5) on 5-D Z-score vectors vs 500 healthy refs. "
        "Threshold = training mean + 3 std."
    )
    pdf.add_chart(img_anomaly, caption="Figure 4 - L1 severity (top), L2 KNN distance (bottom).")

    # 6. Correlation
    pdf.add_page()
    pdf.section_title("6. Sensor Correlation")
    pdf.body_text(
        "Pearson correlation across all channels. "
        "RPM vs Temperature = positive (load -> heat). "
        "Vibration vs Oil = negative (bearing wear -> oil loss)."
    )
    pdf.add_chart(img_corr, w=110, caption="Figure 5 - Pearson correlation heatmap.")

    # 7. Model Comparison
    pdf.add_page()
    pdf.section_title("7. Model Comparison - KNN vs Isolation Forest")
    pdf.kv_table([
        ("Option A: Custom KNN",       "Zero dependencies - edge deployable"),
        ("Option B: Isolation Forest",  "Requires scikit-learn stack"),
        ("KNN Anomaly Rate",           f"{100*l2_anom/n:.2f}%"),
        ("KNN Algorithm",              "Euclidean distance in 5-D Z-score space"),
        ("IF Algorithm",               "Random partitioning trees O(log n)"),
        ("Edge Deployability",         "KNN: any Python 3 | IF: full ML stack"),
        ("Explainability",             "KNN: distance metric | IF: black-box"),
    ])
    pdf.body_text(
        "Industrial edge devices frequently run air-gapped Linux. "
        "The zero-dependency KNN approach is superior for embedded deployment. "
        "Isolation Forest is better for cloud-side batch analytics."
    )

    # 8. Architecture
    pdf.add_page()
    pdf.section_title("8. Full Pipeline Architecture")
    pdf.body_text(
        "src/data_simulation.py\n"
        "  -> 20,000 rows: rpm, temperature, vibration, oil_pressure, voltage\n"
        "  -> 5 injected fault events\n\n"
        "src/preprocessing.py\n"
        "  -> Rolling mean + std (window=50), ROC, drift, Z-score\n\n"
        "src/feature_engineering.py\n"
        "  -> EDA: data quality, CUSUM, correlation, HTML report\n\n"
        "src/anomaly_models.py\n"
        "  -> Layer 1: IQR + 3sigma + drift + ROC\n"
        "  -> Layer 2: Custom 5-D KNN\n\n"
        "src/train.py\n"
        "  -> Health Index (L1 + L2 penalties + EMA)\n\n"
        "src/evaluate.py\n"
        "  -> RUL prediction + model comparison\n\n"
        "reports/generate_pdf_report.py -> THIS DOCUMENT"
    )

    # Save
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "anomaly_detection_report.pdf"
    pdf.output(str(out))
    print(f"\n  PDF saved: {out}")
    print(f"  Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
