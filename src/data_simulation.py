"""
data_simulation.py
Generates 20,000 seconds of synthetic CAN-bus telemetry with 5 injected fault events.

Fault schedule
--------------
Step 10000-10050 : Oil Pressure Drop      (lubrication failure)
Step 12000-12100 : Temperature Spike      (thermal overload)
Step 14000-14020 : Voltage Spike          (electrical instability)
Step 15000-15100 : Vibration Surge        (bearing wear)
Step 17000-17050 : RPM Drop              (load loss / stall)

Output: data/raw/simulated_can_data.csv
"""

import math
import random
import csv
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent        # can-bus-anomaly-detection/
OUT_DIR = ROOT / "data" / "raw"

random.seed(42)

N = 20000
START_TIME = datetime(2025, 1, 1)


def generate_telemetry():
    """Generate N seconds of telemetry with 5 fault events."""
    print("=" * 62)
    print("  Phase 1-2: Data Simulation")
    print("=" * 62)
    print(f"  Generating {N:,} seconds of CAN-bus telemetry (pure Python)...")

    data = []
    for i in range(N):
        current_time = START_TIME + timedelta(seconds=i)

        # -- Base signals --
        rpm = 1500 + 30 * math.sin((i / N) * 100 * math.pi / 50) + random.gauss(0, 8)
        temperature = 70 + 0.0008 * i + random.gauss(0, 1.0)
        vibration = 0.20 + 0.00005 * i + random.gauss(0, 0.020)
        oil_pressure = 5.0 + random.gauss(0, 0.10)
        voltage = 220.0 + random.gauss(0, 2.0)

        # -- Fault Injection --
        # Fault 1 - Oil pressure drop (lubrication failure)
        if 10000 <= i < 10050:
            oil_pressure -= 2.5 * ((i - 10000) / 50)

        # Fault 2 - Sudden temperature spike (thermal overload)
        if 12000 <= i < 12100:
            temperature += 25.0

        # Fault 3 - Voltage spike (electrical transient)
        if 14000 <= i < 14020:
            voltage += 18.0 * math.sin((i - 14000) / 20.0 * math.pi)

        # Fault 4 - Vibration surge (bearing wear / imbalance)
        if 15000 <= i < 15100:
            vibration += 0.5

        # Fault 5 - RPM drop (partial stall / load loss)
        if 17000 <= i < 17050:
            rpm -= 400.0

        data.append([
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            round(rpm,          2),
            round(temperature,  2),
            round(vibration,    4),
            round(oil_pressure, 2),
            round(voltage,      2),
        ])

    # -- Write CSV --
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = OUT_DIR / "simulated_can_data.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rpm", "temperature", "vibration",
                         "oil_pressure", "voltage"])
        writer.writerows(data)

    print(f"\n  Saved: {csv_file}")
    print(f"  Records: {N:,}")
    print(f"\n  Fault event summary:")
    print(f"    Step 10000-10050 : Oil Pressure Drop")
    print(f"    Step 12000-12100 : Temperature Spike (+25 C)")
    print(f"    Step 14000-14020 : Voltage Spike (+18 V transient)")
    print(f"    Step 15000-15100 : Vibration Surge (+0.5 g)")
    print(f"    Step 17000-17050 : RPM Drop (-400 RPM)")
    return csv_file


if __name__ == "__main__":
    generate_telemetry()
