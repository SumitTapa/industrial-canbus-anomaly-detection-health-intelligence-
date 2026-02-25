"""
train.py
Phase 5: Equipment Health Index calculation.

Combines Layer 1 severity + Layer 2 KNN distance into a unified 0-100% health score,
then smooths with EMA (alpha=0.05) to model physical inertia.

Input:  data/processed/layer1_results.csv
        data/processed/layer2_knn_results.csv
Output: data/processed/overall_health_index.csv
"""

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"


def load_csv(filepath):
    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)
        return header, data
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None, None


def calculate_ema(data, alpha):
    """Exponential Moving Average."""
    ema = [data[0]]
    for point in data[1:]:
        ema.append(alpha * point + (1 - alpha) * ema[-1])
    return ema


def main():
    print("=" * 62)
    print("  Phase 5: Equipment Health Index")
    print("=" * 62)

    h1, l1_data = load_csv(PROC_DIR / "layer1_results.csv")
    h2, l2_data = load_csv(PROC_DIR / "layer2_knn_results.csv")

    if not l1_data or not l2_data:
        print("  Required anomaly files missing. Run anomaly_models.py first.")
        return

    n_samples = len(l1_data)

    # L2 distance normalization bounds from training data
    l2_distances = [float(row[1]) for row in l2_data]
    train_dist = l2_distances[:5000]
    mean_dist = sum(train_dist) / len(train_dist)
    std_dist = math.sqrt(sum((x - mean_dist) ** 2 for x in train_dist) / len(train_dist))
    std_dist = std_dist if std_dist > 0 else 0.001
    max_expected_dist = mean_dist + (6 * std_dist)

    # Compute raw health scores
    raw_health_scores = []
    for i in range(n_samples):
        health = 100.0

        # L1 penalty: severity * 2.5 (max 25 pts)
        l1_severity = float(l1_data[i][2])
        health -= l1_severity * 2.5

        # L2 penalty: proportional distance (max 20 pts)
        l2_dist = float(l2_data[i][1])
        l2_ratio = min(1.0, max(0.0, (l2_dist - mean_dist) / (max_expected_dist - mean_dist)))
        health -= l2_ratio * 20.0

        raw_health_scores.append(max(0.0, health))

    # Smooth with EMA
    print("  Applying EMA smoothing (alpha=0.05)...")
    smoothed_health = calculate_ema(raw_health_scores, alpha=0.05)

    # Write output
    output_file = PROC_DIR / "overall_health_index.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "instant_health_raw", "smoothed_health_index", "health_status"])
        for i in range(n_samples):
            ts = l1_data[i][0]
            raw_h = round(raw_health_scores[i], 2)
            smooth_h = round(smoothed_health[i], 2)

            status = "GOOD"
            if smooth_h < 75:
                status = "WARNING"
            if smooth_h < 50:
                status = "DEGRADED"
            if smooth_h < 25:
                status = "CRITICAL"

            writer.writerow([ts, raw_h, smooth_h, status])

    min_h = round(min(smoothed_health), 2)
    max_h = round(max(smoothed_health), 2)
    final_h = round(smoothed_health[-1], 2)
    print(f"\n  Saved: {output_file}")
    print(f"  Health range: min={min_h}%  max={max_h}%  final={final_h}%")
    print(f"  L2 distance: mean={mean_dist:.4f}  cutoff={max_expected_dist:.4f}")


if __name__ == "__main__":
    main()
