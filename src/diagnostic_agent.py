from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CRITICAL_SENSORS = ["sensor_3", "sensor_4", "sensor_9", "sensor_14", "sensor_17"]


@dataclass
class DiagnosticResult:
    status: str
    severity: str
    probable_causes: list[str]
    recommendation: str


def compute_engine_baseline(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    baseline: dict[str, dict[str, float]] = {}
    normal_df = train_df[train_df["anomaly_label"] == 0]

    for sensor in CRITICAL_SENSORS:
        baseline[sensor] = {
            "mean": float(normal_df[sensor].mean()),
            "std": float(normal_df[sensor].std()),
        }

    return baseline


def diagnose_engine(engine_df: pd.DataFrame, baseline: dict[str, dict[str, float]]) -> DiagnosticResult:
    latest = engine_df.sort_values("time_in_cycles").iloc[-1]
    anomaly_label = int(latest["anomaly_label"])
    rul = float(latest["RUL"])
    cycle_ratio = float(latest["cycle_ratio"])

    probable_causes: list[str] = []

    for sensor in CRITICAL_SENSORS:
        mean = baseline[sensor]["mean"]
        std = baseline[sensor]["std"] or 1e-6
        value = float(latest[sensor])
        z_score = abs((value - mean) / std)

        if z_score >= 2.0:
            probable_causes.append(f"{sensor} deviates strongly from normal behavior")

    if cycle_ratio > 0.85:
        probable_causes.append("engine is operating in a late life-cycle phase")

    if rul <= 15:
        severity = "Critical"
    elif rul <= 30 or anomaly_label == 1:
        severity = "Warning"
    else:
        severity = "Normal"

    if not probable_causes:
        probable_causes.append("no major sensor drift detected at the latest cycle")

    if severity == "Critical":
        recommendation = "Schedule immediate inspection and prioritize preventive maintenance."
    elif severity == "Warning":
        recommendation = "Inspect the highlighted sensors and monitor the engine closely over the next cycles."
    else:
        recommendation = "Continue standard monitoring and periodic maintenance."

    status = "Anomaly detected" if anomaly_label == 1 else "Normal behavior"

    return DiagnosticResult(
        status=status,
        severity=severity,
        probable_causes=probable_causes,
        recommendation=recommendation,
    )
