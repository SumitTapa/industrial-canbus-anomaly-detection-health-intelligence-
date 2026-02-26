"""
Generate a GitHub-friendly static markdown dashboard.

Unlike the interactive HTML dashboards, this markdown report renders directly
in GitHub's blob viewer without JavaScript.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "dashboard_static.md"


def read_csv(path: Path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> Path:
    health = read_csv(PROC / "overall_health_index.csv")
    l1 = read_csv(PROC / "layer1_results.csv")
    l2 = read_csv(PROC / "layer2_knn_results.csv")
    rul = read_csv(PROC / "rul_predictions.csv")

    total = len(health)
    final = health[-1]
    health_values = [float(r["smoothed_health_index"]) for r in health]
    status_counts = Counter(r["health_status"] for r in health)

    l1_count = sum(int(r["layer1_anomaly"]) for r in l1)
    l2_count = sum(int(r["is_knn_anomaly"]) for r in l2)

    last_rul = rul[-1]["estimated_rul"]
    active_rul = sum(
        1
        for r in rul
        if r["estimated_rul"].endswith("s") and r["estimated_rul"] != "0 (FAILED)"
    )

    md = f"""# Static Dashboard (GitHub-friendly)\n\nThis report is a **non-interactive fallback** for GitHub viewing.\nFor interactive charts, use `reports/dashboard.html` via a local server.\n\n## Snapshot\n\n| Metric | Value |\n|---|---:|\n| Total records | {total:,} |\n| Final smoothed health | {float(final['smoothed_health_index']):.2f}% |\n| Final status | {final['health_status']} |\n| Layer 1 anomalies | {l1_count:,} ({(100*l1_count/total):.2f}%) |\n| Layer 2 anomalies | {l2_count:,} ({(100*l2_count/total):.2f}%) |\n| Latest RUL estimate | {last_rul} |\n| Active RUL countdown points | {active_rul:,} |\n| Min smoothed health | {min(health_values):.2f}% |\n| Max smoothed health | {max(health_values):.2f}% |\n\n## Health status distribution\n\n| Status | Count |\n|---|---:|\n"""

    for key in ["GOOD", "WARNING", "DEGRADED", "CRITICAL"]:
        md += f"| {key} | {status_counts.get(key, 0):,} |\n"

    md += """\n## Recent rows (tail)\n\n### Health index (last 10)\n\n| Timestamp | Raw health | Smoothed health | Status |\n|---|---:|---:|---|\n"""

    for row in health[-10:]:
        md += (
            f"| {row['timestamp']} | {row['instant_health_raw']} | "
            f"{row['smoothed_health_index']} | {row['health_status']} |\n"
        )

    md += """\n\n### RUL (last 10)\n\n| Timestamp | Smoothed health | Status | Estimated RUL |\n|---|---:|---|---|\n"""

    for row in rul[-10:]:
        md += (
            f"| {row['timestamp']} | {row['smoothed_health_index']} | "
            f"{row['health_status']} | {row['estimated_rul']} |\n"
        )

    md += """\n\n## Open interactive dashboard locally\n\n```bash
python3 -m http.server 8000
# then open http://localhost:8000/reports/dashboard.html
```
"""

    OUT.write_text(md)
    return OUT


if __name__ == "__main__":
    out = main()
    print(f"Saved: {out}")
