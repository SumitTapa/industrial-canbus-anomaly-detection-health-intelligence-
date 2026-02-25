# CAN Bus Anomaly Detection & Predictive Maintenance

Industrial asset monitoring system for CAN bus telemetry data. Implements a multi-layer anomaly detection pipeline with statistical rules (Layer 1) and custom KNN AI (Layer 2), followed by health index calculation and remaining useful life (RUL) prediction.

## Project Structure

```
can-bus-anomaly-detection/
|
|-- data/
|   |-- raw/                  # Simulated CAN bus CSV (20,000 records)
|   +-- processed/            # Preprocessed features, anomaly results, health index, RUL
|
|-- notebooks/
|   +-- 01_eda.ipynb          # Interactive EDA notebook
|
|-- src/
|   |-- __init__.py
|   |-- data_simulation.py    # Phase 1-2: Generate synthetic telemetry with fault injection
|   |-- preprocessing.py      # Phase 3:   Rolling stats, ROC, drift, Z-score normalization
|   |-- feature_engineering.py# Phase 3.5: EDA, CUSUM, correlation, HTML report
|   |-- anomaly_models.py     # Phase 4:   Layer 1 (stats) + Layer 2 (KNN) anomaly detection
|   |-- train.py              # Phase 5:   Health index (L1+L2 penalties + EMA smoothing)
|   +-- evaluate.py           # Phase 6:   RUL prediction + model comparison
|
|-- reports/
|   +-- generate_pdf_report.py# Phase 7:   PDF report (matplotlib + fpdf2)
|
|-- tests/
|   +-- test_pipeline.py      # Smoke tests
|
|-- requirements.txt
|-- README.md
+-- .gitignore
```

## Quickstart

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (only needed for PDF report + EDA notebook)
pip install -r requirements.txt

# 3. Run the full pipeline
cd can-bus-anomaly-detection

python3 src/data_simulation.py       # Phase 1-2: Generate data
python3 src/preprocessing.py         # Phase 3:   Preprocess & normalize
python3 src/feature_engineering.py   # Phase 3.5: EDA report
python3 src/anomaly_models.py        # Phase 4:   Anomaly detection
python3 src/train.py                 # Phase 5:   Health index
python3 src/evaluate.py              # Phase 6:   RUL prediction
python3 reports/generate_pdf_report.py  # Phase 7: PDF report
```

## Fault Injection Schedule

| Step Range   | Sensor       | Fault Type                        |
|-------------|-------------|-----------------------------------|
| 10000-10050 | Oil Pressure | Gradual lubrication failure       |
| 12000-12100 | Temperature  | Thermal overload (+25 C)          |
| 14000-14020 | Voltage      | Electrical transient spike        |
| 15000-15100 | Vibration    | Bearing wear surge (+0.5 g)       |
| 17000-17050 | RPM          | Partial stall / load loss (-400)  |

## Pipeline Architecture

```
data_simulation.py  -->  data/raw/simulated_can_data.csv
        |
preprocessing.py    -->  data/processed/preprocessed_can_data.csv
        |
anomaly_models.py   -->  data/processed/layer1_results.csv
                         data/processed/layer2_knn_results.csv
        |
train.py            -->  data/processed/overall_health_index.csv
        |
evaluate.py         -->  data/processed/rul_predictions.csv
        |
generate_pdf_report.py -> reports/anomaly_detection_report.pdf
```

## Anomaly Detection Layers

### Layer 1: Statistical Rules Engine
- **RPM**: IQR bounds (k=1.5) from first 5000 healthy records
- **Voltage**: 3-sigma Z-score deviation
- **Oil Pressure**: 3-sigma Z-score deviation (both drop and spike)
- **Temperature**: Cumulative thermal drift threshold
- **Vibration**: Rate-of-change spike detection (4-sigma)

### Layer 2: Custom K-Nearest Neighbors
- **Algorithm**: Pure Python, zero external dependencies
- **Features**: 5-D Z-score vectors (rpm, temp, vib, oil, volt)
- **Training**: 500 randomly sampled healthy reference points from first 5000 records
- **Scoring**: Mean Euclidean distance to k=5 nearest healthy neighbors
- **Threshold**: Training mean + 3 standard deviations

## Health Index & RUL

- **Health**: 0-100% score combining L1 severity (max 25pt penalty) and L2 distance (max 20pt penalty), smoothed with EMA (alpha=0.05)
- **Status tiers**: GOOD (>=75%) | WARNING (50-75%) | DEGRADED (25-50%) | CRITICAL (<25%)
- **RUL**: Rolling 500-step degradation rate extrapolated to 25% critical threshold

## Model Comparison

| Feature          | Option A (Custom KNN) | Option B (Isolation Forest) |
|-----------------|----------------------|----------------------------|
| Dependencies    | None (pure Python)   | scikit-learn, numpy        |
| Edge deployment | Yes                  | Requires ML stack          |
| Explainability  | Transparent distance | Black-box                  |
| Training        | Reference sampling   | Ensemble trees             |

## Dependencies

**Core pipeline**: Zero dependencies (pure Python 3)
**Reports & notebooks**: matplotlib, pandas, fpdf2 (see requirements.txt)
