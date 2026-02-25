"""
preprocessing.py
Phase 3: Rolling statistics, rate-of-change features, cumulative drift, and Z-score normalization.

Input:  data/raw/simulated_can_data.csv
Output: data/processed/preprocessed_can_data.csv
"""

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


def calc_mean_std(values):
    """Return (mean, std) for a list of floats."""
    if not values:
        return 0, 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)


def preprocess_data(window_size=50):
    """Compute rolling stats, ROC, drift, and Z-score normalize all features."""
    print("=" * 62)
    print("  Phase 3: Preprocessing & Feature Engineering")
    print("=" * 62)

    input_file = RAW_DIR / "simulated_can_data.csv"
    output_file = PROC_DIR / "preprocessed_can_data.csv"

    print(f"  Reading: {input_file}")
    with open(input_file, mode="r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)

    # Column indices
    t_idx = header.index("timestamp")
    rpm_idx = header.index("rpm")
    temp_idx = header.index("temperature")
    vib_idx = header.index("vibration")
    oil_idx = header.index("oil_pressure")
    volt_idx = header.index("voltage")

    data.sort(key=lambda x: x[t_idx])
    print(f"  Records loaded: {len(data):,}")
    print(f"  Computing rolling stats (window={window_size})...")

    # Baseline temperature (first 1000 records)
    base_temp_sum = sum(float(row[temp_idx]) for row in data[:min(1000, len(data))])
    base_temp = base_temp_sum / min(1000, len(data))
    cumulative_drift = 0.0

    enriched_data = []

    for i in range(len(data)):
        row = data[i]
        rpm = float(row[rpm_idx])
        temp = float(row[temp_idx])
        vib = float(row[vib_idx])
        oil = float(row[oil_idx])
        volt = float(row[volt_idx])

        # Rolling window
        start_idx = max(0, i - window_size + 1)
        window_rows = data[start_idx:i + 1]

        rpm_mean, rpm_std = calc_mean_std([float(r[rpm_idx]) for r in window_rows])
        temp_mean, temp_std = calc_mean_std([float(r[temp_idx]) for r in window_rows])
        vib_mean, vib_std = calc_mean_std([float(r[vib_idx]) for r in window_rows])
        oil_mean, oil_std = calc_mean_std([float(r[oil_idx]) for r in window_rows])
        volt_mean, volt_std = calc_mean_std([float(r[volt_idx]) for r in window_rows])

        # Rate of change
        shift_idx = max(0, i - window_size)
        temp_roc = (temp - float(data[shift_idx][temp_idx])) / window_size if i >= window_size else 0
        vib_roc = (vib - float(data[shift_idx][vib_idx])) / window_size if i >= window_size else 0

        # Cumulative thermal drift
        drift = temp - base_temp
        if drift > 0:
            cumulative_drift += drift

        enriched_data.append([
            row[t_idx], rpm, temp, vib, oil, volt,
            rpm_mean, rpm_std, temp_mean, temp_std,
            vib_mean, vib_std, oil_mean, oil_std,
            volt_mean, volt_std,
            temp_roc, vib_roc, cumulative_drift,
        ])

    # Z-score normalization (all columns except timestamp)
    print("  Normalizing features (Z-score)...")
    num_cols = len(enriched_data[0])
    col_means = []
    col_stds = []
    for col_idx in range(1, num_cols):
        col_values = [row[col_idx] for row in enriched_data]
        mean, std = calc_mean_std(col_values)
        col_means.append(mean)
        col_stds.append(std if std != 0 else 1)

    for i in range(len(enriched_data)):
        for col_idx in range(1, num_cols):
            val = enriched_data[i][col_idx]
            mean = col_means[col_idx - 1]
            std = col_stds[col_idx - 1]
            enriched_data[i][col_idx] = round((val - mean) / std, 6)

    # Write output
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_headers = [
        "timestamp", "rpm", "temperature", "vibration", "oil_pressure", "voltage",
        "rpm_rolling_mean", "rpm_rolling_std",
        "temperature_rolling_mean", "temperature_rolling_std",
        "vibration_rolling_mean", "vibration_rolling_std",
        "oil_pressure_rolling_mean", "oil_pressure_rolling_std",
        "voltage_rolling_mean", "voltage_rolling_std",
        "temperature_roc", "vibration_roc", "temperature_cumulative_drift",
    ]

    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_headers)
        writer.writerows(enriched_data)

    print(f"\n  Saved: {output_file}")
    print(f"  Features: {len(out_headers)}")
    print("  Preprocessing complete!")
    return output_file


if __name__ == "__main__":
    preprocess_data()
