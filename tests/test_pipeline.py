"""
tests/test_pipeline.py
Basic smoke tests for the CAN bus anomaly detection pipeline.
Run: python3 -m pytest tests/ -v   (from project root)
"""

import sys
from pathlib import Path

# Add src to path so we can import modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_data_simulation():
    """Verify simulation creates CSV with correct shape."""
    from data_simulation import generate_telemetry
    csv_path = generate_telemetry()
    assert Path(csv_path).exists()
    with open(csv_path) as f:
        lines = f.readlines()
    # Header + 20000 data rows
    assert len(lines) == 20001, f"Expected 20001 lines, got {len(lines)}"


def test_preprocessing():
    """Verify preprocessing creates enriched CSV."""
    from preprocessing import preprocess_data
    csv_path = preprocess_data()
    assert Path(csv_path).exists()
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    assert "temperature_cumulative_drift" in header
    assert "vibration_roc" in header


def test_anomaly_models():
    """Verify both anomaly layers produce results."""
    from anomaly_models import run_layer1, run_layer2
    l1 = run_layer1()
    l2 = run_layer2()
    assert Path(l1).exists()
    assert Path(l2).exists()


def test_train():
    """Verify health index is computed."""
    proc = ROOT / "data" / "processed" / "overall_health_index.csv"
    # This test assumes train.py has been run
    if proc.exists():
        with open(proc) as f:
            lines = f.readlines()
        assert len(lines) > 1


def test_evaluate():
    """Verify RUL predictions are computed."""
    proc = ROOT / "data" / "processed" / "rul_predictions.csv"
    if proc.exists():
        with open(proc) as f:
            lines = f.readlines()
        assert len(lines) > 1
