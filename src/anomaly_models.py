"""
anomaly_models.py
Phase 4: Multi-layer anomaly detection.

Layer 1 - Statistical rules: IQR, 3-sigma, drift, rate-of-change
Layer 2 - Custom KNN AI: 5-D Euclidean distance in Z-score space

Input:  data/raw/simulated_can_data.csv
        data/processed/preprocessed_can_data.csv
Output: data/processed/layer1_results.csv
        data/processed/layer2_knn_results.csv
"""

import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


# -- Shared Utilities --
def calc_mean(data):
    return sum(data) / len(data) if data else 0


def calc_std(data, mean):
    return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data)) if len(data) > 1 else 0


def calc_iqr_bounds(data, k=1.5):
    sdata = sorted(data)
    n = len(sdata)
    q1 = sdata[n // 4]
    q3 = sdata[(n * 3) // 4]
    iqr = q3 - q1
    return q1 - (k * iqr), q3 + (k * iqr)


def euclidean_dist(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def load_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, data


# ======================================================================
# LAYER 1: Statistical Rules Engine
# ======================================================================
def run_layer1():
    """Apply IQR, 3-sigma, drift, and ROC rules across all telemetry."""
    print("  --- Layer 1: Statistical Rules Engine ---")
    raw_header, raw_data = load_csv(RAW_DIR / "simulated_can_data.csv")
    prep_header, prep_data = load_csv(PROC_DIR / "preprocessed_can_data.csv")

    n_samples = len(raw_data)
    train_end = 5000

    # RPM - IQR bounds
    rpm_raw = [float(row[raw_header.index("rpm")]) for row in raw_data]
    rpm_lower, rpm_upper = calc_iqr_bounds(rpm_raw[:train_end], k=1.5)

    # Voltage - 3-sigma
    volt_raw = [float(row[raw_header.index("voltage")]) for row in raw_data]
    volt_train = volt_raw[:train_end]
    volt_mean = calc_mean(volt_train)
    volt_std = calc_std(volt_train, volt_mean) or 1.0
    volt_lower = volt_mean - (3 * volt_std)
    volt_upper = volt_mean + (3 * volt_std)

    # Oil Pressure - 3-sigma
    oil_raw = [float(row[raw_header.index("oil_pressure")]) for row in raw_data]
    oil_train = oil_raw[:train_end]
    oil_mean = calc_mean(oil_train)
    oil_std = calc_std(oil_train, oil_mean) or 1.0
    oil_lower = oil_mean - (3 * oil_std)
    oil_upper = oil_mean + (3 * oil_std)

    # Temperature drift
    drift_feat = [float(row[prep_header.index("temperature_cumulative_drift")]) for row in prep_data]
    drift_train = drift_feat[:train_end]
    drift_mean = calc_mean(drift_train)
    drift_std = calc_std(drift_train, drift_mean) or 1.0
    drift_threshold = drift_mean + (3 * drift_std)

    # Vibration ROC
    vib_roc_feat = [float(row[prep_header.index("vibration_roc")]) for row in prep_data]
    vib_roc_train = vib_roc_feat[:train_end]
    vib_roc_mean = calc_mean(vib_roc_train)
    vib_roc_std = calc_std(vib_roc_train, vib_roc_mean) or 1.0
    vib_roc_threshold = vib_roc_mean + (4 * vib_roc_std)

    # Execute rules
    results = []
    anomaly_count = 0

    for i in range(n_samples):
        timestamp = raw_data[i][0]
        v_rpm = rpm_raw[i]
        v_volt = volt_raw[i]
        v_oil = oil_raw[i]
        v_drift = drift_feat[i]
        v_vib_roc = vib_roc_feat[i]

        flags = []
        severity_score = 0

        if v_rpm < rpm_lower or v_rpm > rpm_upper:
            flags.append("RPM_IQR_VIOLATION")
            deviation = min(10, abs(v_rpm - (rpm_upper if v_rpm > rpm_upper else rpm_lower)) / 50.0)
            severity_score += deviation

        if v_volt < volt_lower or v_volt > volt_upper:
            flags.append("VOLTAGE_3SIGMA")
            severity_score += min(10, abs(v_volt - volt_mean) / volt_std)

        if v_oil < oil_lower or v_oil > oil_upper:
            flags.append("OIL_PRESSURE_ANOMALY")
            severity_score += min(10, abs(v_oil - oil_mean) / oil_std)

        if v_drift > drift_threshold:
            flags.append("TEMP_DRIFT_CRITICAL")
            severity_score += min(10, (v_drift - drift_threshold) / drift_std)

        if v_vib_roc > vib_roc_threshold:
            flags.append("VIBRATION_ROC_SPIKE")
            severity_score += min(10, (v_vib_roc - vib_roc_threshold) / vib_roc_std)

        final_severity = round(min(10.0, severity_score), 2)
        is_anomaly = 1 if flags else 0
        if is_anomaly:
            anomaly_count += 1

        results.append([
            timestamp, is_anomaly, final_severity,
            " | ".join(flags) if is_anomaly else "NORMAL",
        ])

    output_file = PROC_DIR / "layer1_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "layer1_anomaly", "layer1_severity", "layer1_reasons"])
        writer.writerows(results)

    print(f"    Saved: {output_file}")
    print(f"    Anomalies: {anomaly_count} / {n_samples} ({100 * anomaly_count / n_samples:.2f}%)")
    print(f"    RPM IQR: [{rpm_lower:.1f}, {rpm_upper:.1f}]")
    print(f"    Volt 3s: [{volt_lower:.1f}, {volt_upper:.1f}]")
    print(f"    Oil  3s: [{oil_lower:.2f}, {oil_upper:.2f}]")
    print(f"    Drift threshold: {drift_threshold:.2f}")
    print(f"    Vib ROC threshold: {vib_roc_threshold:.6f}")
    return output_file


# ======================================================================
# LAYER 2: Custom K-Nearest Neighbors AI
# ======================================================================
def run_layer2():
    """5-D KNN distance anomaly detection on Z-scored sensor data."""
    print("  --- Layer 2: Custom KNN Anomaly Detector ---")
    prep_header, prep_data = load_csv(PROC_DIR / "preprocessed_can_data.csv")

    features = ["rpm", "temperature", "vibration", "oil_pressure", "voltage"]
    feat_indices = [prep_header.index(f) for f in features]

    vectors = [[float(row[i]) for i in feat_indices] for row in prep_data]
    n = len(vectors)
    print(f"    Loaded {n:,} 5-D vectors")

    # Train: sample 500 healthy refs from first 5000 records
    random.seed(42)
    healthy_refs = random.sample(vectors[:5000], 500)

    k = 5
    anomaly_scores = []

    print("    Computing KNN distances...")
    for i, v in enumerate(vectors):
        distances = sorted(euclidean_dist(v, ref) for ref in healthy_refs)
        score = sum(distances[:k]) / k
        anomaly_scores.append(score)
        if (i + 1) % 5000 == 0:
            print(f"      Processed {i + 1:,} / {n:,}")

    # Threshold from training distribution
    train_scores = anomaly_scores[:5000]
    train_mean = calc_mean(train_scores)
    train_std = calc_std(train_scores, train_mean)
    dist_threshold = train_mean + (3 * train_std)

    output_file = PROC_DIR / "layer2_knn_results.csv"
    anomaly_count = 0
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "multivariate_distance", "is_knn_anomaly"])
        for i in range(n):
            ts = prep_data[i][0]
            dist = anomaly_scores[i]
            flag = 1 if dist > dist_threshold else 0
            if flag:
                anomaly_count += 1
            writer.writerow([ts, round(dist, 4), flag])

    print(f"    Saved: {output_file}")
    print(f"    Anomalies: {anomaly_count} / {n} ({100 * anomaly_count / n:.2f}%)")
    print(f"    Threshold (mean+3std): {dist_threshold:.4f}")
    return output_file


# ======================================================================
# Main entry point
# ======================================================================
def main():
    print("=" * 62)
    print("  Phase 4: Multi-Layer Anomaly Detection")
    print("=" * 62)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    run_layer1()
    print()
    run_layer2()
    print("\n  Phase 4 complete!")


if __name__ == "__main__":
    main()
