"""
feature_engineering.py
Phase 3.5: Industrial time-series EDA - data quality, CUSUM, correlation, 3D scatter.

Input:  data/raw/simulated_can_data.csv
        data/processed/preprocessed_can_data.csv
Output: reports/industrial_eda_report.html

Note: This is the pure-Python EDA engine. A Jupyter notebook version exists at notebooks/01_eda.ipynb.
"""

import csv
import json
import math
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"


# -- Utilities --
def calc_mean(data):
    return sum(data) / len(data) if data else 0


def calc_std(data, mean):
    return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data)) if len(data) > 1 else 0


def calculate_pearson(x, y):
    n = len(x)
    if n == 0:
        return 0
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum(v ** 2 for v in x)
    sum_y_sq = sum(v ** 2 for v in y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x_sq - sum_x ** 2) * (n * sum_y_sq - sum_y ** 2))
    if denominator == 0:
        return 0
    return numerator / denominator


def cusum(data, target=None, shift=None):
    """Cumulative Sum control chart for change-point detection."""
    if not data:
        return [], []
    if target is None:
        target = calc_mean(data)
    if shift is None:
        shift = calc_std(data, target) * 0.5
    pos_cusum = [0]
    neg_cusum = [0]
    for i in range(1, len(data)):
        pos_cusum.append(max(0, pos_cusum[-1] + (data[i] - target) - shift))
        neg_cusum.append(min(0, neg_cusum[-1] + (data[i] - target) + shift))
    return pos_cusum, neg_cusum


def load_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, data


def run_eda():
    """Execute full EDA pipeline and produce interactive HTML report."""
    print("=" * 62)
    print("  Phase 3.5: Industrial Time-Series EDA")
    print("=" * 62)

    raw_file = RAW_DIR / "simulated_can_data.csv"
    prep_file = PROC_DIR / "preprocessed_can_data.csv"

    print(f"  Loading: {raw_file}")
    raw_header, raw_data = load_csv(raw_file)
    print(f"  Loading: {prep_file}")
    prep_header, prep_data = load_csv(prep_file)

    n = len(raw_data)
    sensors = ["rpm", "temperature", "vibration", "oil_pressure", "voltage"]
    sensor_idx = {s: raw_header.index(s) for s in sensors}

    # Extract sensor columns
    sensor_data = {}
    for s in sensors:
        idx = sensor_idx[s]
        sensor_data[s] = [float(row[idx]) for row in raw_data]

    # -- Data Quality Summary --
    print("  Running data quality checks...")
    timestamps = [row[0] for row in raw_data]
    gaps = 0
    for i in range(1, len(timestamps)):
        t0 = datetime.strptime(timestamps[i - 1], "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(timestamps[i], "%Y-%m-%d %H:%M:%S")
        diff = (t1 - t0).total_seconds()
        if diff != 1.0:
            gaps += 1

    # Sensor freeze detection
    freeze_counts = {}
    freeze_window = 50
    for s in sensors:
        vals = sensor_data[s]
        fz = 0
        for i in range(freeze_window, len(vals)):
            window = vals[i - freeze_window:i]
            if max(window) - min(window) < 1e-6:
                fz += 1
                break
        freeze_counts[s] = fz

    # -- CUSUM on Vibration --
    print("  Computing CUSUM on vibration channel...")
    vib_data = sensor_data["vibration"]
    vib_cusum_pos, vib_cusum_neg = cusum(vib_data)

    # -- Pearson Correlation --
    print("  Computing Pearson correlation matrix...")
    corr_matrix = {}
    for s1 in sensors:
        corr_matrix[s1] = {}
        for s2 in sensors:
            corr_matrix[s1][s2] = round(calculate_pearson(sensor_data[s1], sensor_data[s2]), 4)

    # -- Print summary --
    print(f"\n  Data Quality:")
    print(f"    Records       : {n:,}")
    print(f"    Timestamp gaps: {gaps}")
    for s in sensors:
        mean = calc_mean(sensor_data[s])
        std = calc_std(sensor_data[s], mean)
        print(f"    {s:16s}  mean={mean:.4f}  std={std:.4f}  freeze={freeze_counts[s]}")

    print(f"\n  Correlation (RPM vs Temp): {corr_matrix['rpm']['temperature']}")
    print(f"  Correlation (Vib vs Oil):  {corr_matrix['vibration']['oil_pressure']}")

    # -- Generate HTML Report --
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORT_DIR / "industrial_eda_report.html"

    # Downsample for plotting
    step = max(1, n // 2000)
    xs = list(range(0, n, step))

    html_data = {
        "sensors": {},
        "cusum_pos": [vib_cusum_pos[i] for i in xs],
        "cusum_neg": [vib_cusum_neg[i] for i in xs],
        "corr_matrix": corr_matrix,
        "sensor_names": sensors,
        "xs": xs,
    }
    for s in sensors:
        html_data["sensors"][s] = [sensor_data[s][i] for i in xs]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Industrial EDA Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ background:#0d1117; color:#c9d1d9; font-family:monospace; padding:20px; }}
h1 {{ color:#58a6ff; text-align:center; }}
h2 {{ color:#58a6ff; border-bottom:1px solid #30363d; padding-bottom:4px; }}
.chart {{ width:100%; height:350px; margin-bottom:20px; }}
.kpi {{ display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }}
.kpi-card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; flex:1; min-width:150px; text-align:center; }}
.kpi-val {{ font-size:24px; font-weight:bold; color:#58a6ff; }}
.kpi-label {{ font-size:12px; color:#8b949e; margin-top:4px; }}
</style>
</head>
<body>
<h1>Industrial CAN Bus - Exploratory Data Analysis</h1>
<div class="kpi">
  <div class="kpi-card"><div class="kpi-val">{n:,}</div><div class="kpi-label">Total Records</div></div>
  <div class="kpi-card"><div class="kpi-val">{gaps}</div><div class="kpi-label">Timestamp Gaps</div></div>
  <div class="kpi-card"><div class="kpi-val">{len(sensors)}</div><div class="kpi-label">Sensor Channels</div></div>
</div>

<h2>Sensor Time Series</h2>
<div id="sensor_chart" class="chart" style="height:600px;"></div>

<h2>CUSUM - Vibration Change Point Detection</h2>
<div id="cusum_chart" class="chart"></div>

<h2>Pearson Correlation Matrix</h2>
<div id="corr_chart" class="chart" style="height:400px;"></div>

<script>
var D = {json.dumps(html_data)};
var config = {{responsive:true, displayModeBar:false}};
var darkLayout = function(t,ya,xa) {{
  return {{
    paper_bgcolor:'#0d1117', plot_bgcolor:'#161b22',
    title:{{text:t||'', font:{{color:'#58a6ff'}}}},
    font:{{color:'#c9d1d9', size:10}},
    xaxis: Object.assign({{gridcolor:'#21262d', zerolinecolor:'#30363d'}}, xa||{{}}),
    yaxis: Object.assign({{gridcolor:'#21262d', zerolinecolor:'#30363d'}}, ya||{{}}),
    margin:{{l:60,r:20,t:40,b:40}}
  }};
}};

// Sensor subplots
var sTraces = [];
var colors = {{'rpm':'#4fc3f7','temperature':'#ffa657','vibration':'#3fb950','oil_pressure':'#d2a8ff','voltage':'#f0e68c'}};
D.sensor_names.forEach(function(s,i) {{
  sTraces.push({{x:D.xs, y:D.sensors[s], name:s, type:'scatter', xaxis:'x', yaxis:'y'+(i+1),
    line:{{color:colors[s], width:1}} }});
}});
var sLayout = Object.assign(darkLayout('All Sensor Channels'), {{
  grid:{{rows:5, columns:1, pattern:'independent', roworder:'top to bottom'}},
  height:600, showlegend:true, legend:{{font:{{color:'#c9d1d9'}}}},
}});
for(var i=1;i<=5;i++) {{
  sLayout['yaxis'+(i>1?i:'')] = {{gridcolor:'#21262d',tickfont:{{color:'#c9d1d9'}}}};
  sLayout['xaxis'+(i>1?i:'')] = {{gridcolor:'#21262d',tickfont:{{color:'#c9d1d9'}}}};
}}
Plotly.newPlot('sensor_chart', sTraces, sLayout, config);

// CUSUM
Plotly.newPlot('cusum_chart', [
  {{x:D.xs, y:D.cusum_pos, name:'CUSUM+', line:{{color:'#f85149'}}}},
  {{x:D.xs, y:D.cusum_neg, name:'CUSUM-', line:{{color:'#58a6ff'}}}}
], darkLayout('Vibration CUSUM Change-Point Detection'), config);

// Correlation heatmap
var corrZ = D.sensor_names.map(function(s1) {{
  return D.sensor_names.map(function(s2) {{ return D.corr_matrix[s1][s2]; }});
}});
var labels = ['RPM','Temp','Vib','Oil','Volt'];
Plotly.newPlot('corr_chart', [{{
  z:corrZ, x:labels, y:labels, type:'heatmap', colorscale:'RdBu',
  zmin:-1, zmax:1, text:corrZ.map(r=>r.map(v=>v.toFixed(2))), texttemplate:'%{{text}}'
}}], Object.assign(darkLayout('Pearson Correlation'), {{height:400}}), config);
</script>
</body>
</html>"""

    with open(report_file, "w") as f:
        f.write(html)

    print(f"\n  Saved: {report_file}")
    return report_file


if __name__ == "__main__":
    run_eda()
