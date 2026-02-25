"""
evaluate.py
Phase 6: Remaining Useful Life (RUL) prediction + Model comparison.

RUL: rolling 500-step health decline extrapolation to 25% critical threshold.
Comparison: Custom KNN (Option A) vs Isolation Forest (Option B).

Input:  data/processed/overall_health_index.csv
        data/processed/layer2_knn_results.csv
        data/processed/layer2_sk_results.csv  (optional, requires scikit-learn)
Output: data/processed/rul_predictions.csv
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"


def load_csv(filepath):
    path = Path(filepath)
    if not path.exists():
        return None, None

    with open(path, "r") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None, None
        data = list(reader)
    return header, data


# ======================================================================
# RUL Prediction
# ======================================================================
def run_rul():
    """Estimate Remaining Useful Life from rolling health decline rate."""
    print("  --- Phase 6a: RUL Prediction ---")
    header, data = load_csv(PROC_DIR / "overall_health_index.csv")
    if not data:
        print("  Missing overall_health_index.csv. Run train.py first.")
        return

    n_samples = len(data)
    window = 500
    critical_threshold = 25.0

    rul_estimates = []
    for i in range(n_samples):
        timestamp = data[i][0]
        current_health = float(data[i][2])
        status = data[i][3]

        if i < window:
            rul_estimates.append([timestamp, current_health, status, "CALCULATING..."])
            continue

        past_health = float(data[i - window][2])
        health_drop = past_health - current_health

        if health_drop <= 0.05 or current_health <= critical_threshold:
            if current_health <= critical_threshold:
                rul_estimates.append([timestamp, current_health, status, "0 (FAILED)"])
            else:
                rul_estimates.append([timestamp, current_health, status, "STABLE"])
        else:
            drop_per_second = health_drop / window
            health_remaining = max(0, current_health - critical_threshold)
            seconds_remaining = health_remaining / drop_per_second
            rul_estimates.append([timestamp, current_health, status, f"{int(seconds_remaining)}s"])

    output_file = PROC_DIR / "rul_predictions.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "smoothed_health_index", "health_status", "estimated_rul"])
        for row in rul_estimates:
            writer.writerow(row)

    # Summary
    stable_count = sum(1 for r in rul_estimates if r[3] == "STABLE")
    calc_count = sum(1 for r in rul_estimates if r[3] == "CALCULATING...")
    failed_count = sum(1 for r in rul_estimates if r[3] == "0 (FAILED)")
    active_rul = sum(1 for r in rul_estimates
                     if isinstance(r[3], str) and r[3].endswith("s")
                     and r[3] != "0 (FAILED)")

    print(f"    Saved: {output_file}")
    print(f"    CALCULATING (warmup) : {calc_count}")
    print(f"    STABLE               : {stable_count}")
    print(f"    Active RUL countdown : {active_rul}")
    print(f"    FAILED               : {failed_count}")


# ======================================================================
# Model Comparison: KNN vs Isolation Forest
# ======================================================================
def run_comparison():
    """Compare Option A (KNN) vs Option B (Isolation Forest) results."""
    print("\n  --- Phase 6b: Model Comparison ---")
    _, data_knn = load_csv(PROC_DIR / "layer2_knn_results.csv")
    _, data_sk = load_csv(PROC_DIR / "layer2_sk_results.csv")

    if data_knn:
        knn_anomalies = sum(1 for row in data_knn if int(row[2]) == 1)
        knn_total = len(data_knn)
        print(f"    Option A (Custom KNN): {knn_anomalies} / {knn_total} ({100 * knn_anomalies / knn_total:.2f}%)")
        print(f"    Zero external dependencies - edge-deployable")
    else:
        print("    Option A results not found.")

    if data_sk:
        sk_anomalies = sum(1 for row in data_sk if int(row[2]) == 1)
        sk_total = len(data_sk)
        print(f"    Option B (Isolation Forest): {sk_anomalies} / {sk_total} ({100 * sk_anomalies / sk_total:.2f}%)")
        overlap = sum(1 for i in range(len(data_knn))
                      if int(data_knn[i][2]) == 1 and int(data_sk[i][2]) == 1)
        print(f"    Model overlap: {overlap}")
    else:
        print("    Option B results not found (requires scikit-learn).")


def main():
    print("=" * 62)
    print("  Phase 6: Evaluation & Prediction")
    print("=" * 62)
    run_rul()
    run_comparison()
    print("\n  Phase 6 complete!")


if __name__ == "__main__":
    main()
