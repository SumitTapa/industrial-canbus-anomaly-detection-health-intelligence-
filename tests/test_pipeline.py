"""
tests/test_pipeline.py
End-to-end and output-quality tests for the CAN bus anomaly detection pipeline.
Run: python3 -m pytest tests/ -v   (from project root)
"""

import csv
import sys
from pathlib import Path

import pytest

# Add src to path so we can import modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_models import run_layer1, run_layer2
from data_simulation import N, generate_telemetry
from evaluate import run_rul
from preprocessing import preprocess_data
from train import main as train_main

PROC_DIR = ROOT / "data" / "processed"


def _csv_rows(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


@pytest.fixture(scope="module")
def pipeline_outputs():
    """Run full pipeline once and return output paths for downstream assertions."""
    raw_path = Path(generate_telemetry())
    prep_path = Path(preprocess_data())
    l1_path = Path(run_layer1())
    l2_path = Path(run_layer2())
    train_main()
    run_rul()

    return {
        "raw": raw_path,
        "prep": prep_path,
        "l1": l1_path,
        "l2": l2_path,
        "health": PROC_DIR / "overall_health_index.csv",
        "rul": PROC_DIR / "rul_predictions.csv",
    }


def test_pipeline_end_to_end_outputs_exist_and_shape(pipeline_outputs):
    """Validate all expected artifacts exist and row counts match telemetry length."""
    for path in pipeline_outputs.values():
        assert path.exists(), f"Expected artifact missing: {path}"

    # Header + data rows in raw csv
    with open(pipeline_outputs["raw"]) as f:
        lines = f.readlines()
    assert len(lines) == N + 1, f"Expected {N + 1} rows including header, got {len(lines)}"

    # Processed files should match telemetry length
    for key in ("prep", "l1", "l2", "health", "rul"):
        rows = _csv_rows(pipeline_outputs[key])
        assert len(rows) == N, f"Expected {N} rows in {pipeline_outputs[key].name}, got {len(rows)}"


def test_pipeline_semantic_quality_checks(pipeline_outputs):
    """Validate key semantic properties of model outputs."""
    l1_rows = _csv_rows(pipeline_outputs["l1"])
    l2_rows = _csv_rows(pipeline_outputs["l2"])
    health_rows = _csv_rows(pipeline_outputs["health"])
    rul_rows = _csv_rows(pipeline_outputs["rul"])

    l1_anomalies = sum(int(r["layer1_anomaly"]) for r in l1_rows)
    l2_anomalies = sum(int(r["is_knn_anomaly"]) for r in l2_rows)

    # There should be meaningful but not universal anomaly detection
    assert 0 < l1_anomalies < N
    assert 0 < l2_anomalies < N

    # Health index should stay in valid bounds and include at least one warning tier
    health_values = [float(r["smoothed_health_index"]) for r in health_rows]
    statuses = {r["health_status"] for r in health_rows}
    assert all(0.0 <= h <= 100.0 for h in health_values)
    assert "WARNING" in statuses or "DEGRADED" in statuses or "CRITICAL" in statuses

    # RUL output should include warmup phase and at least one non-warmup state
    rul_states = [r["estimated_rul"] for r in rul_rows]
    assert "CALCULATING..." in rul_states
    assert any(state != "CALCULATING..." for state in rul_states)

    # Fault-injection window should include anomalies from at least one layer
    start, end = 10000, 15100
    l1_fault_hits = sum(int(l1_rows[i]["layer1_anomaly"]) for i in range(start, end))
    l2_fault_hits = sum(int(l2_rows[i]["is_knn_anomaly"]) for i in range(start, end))
    assert l1_fault_hits > 0 or l2_fault_hits > 0
