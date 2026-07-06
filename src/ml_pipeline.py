from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(r"C:\Users\yassi\OneDrive\Documents\projet pfa")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TRAIN_PATH = PROCESSED_DIR / "train_anomaly.csv"
MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"

TOP_SENSORS = ["sensor_3", "sensor_4", "sensor_9", "sensor_14", "sensor_17"]
FEATURES = [
    "time_in_cycles",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_14",
    "sensor_17",
    "sensor_20",
    "sensor_3_rolling_mean",
    "sensor_4_rolling_mean",
    "sensor_9_rolling_mean",
    "sensor_14_rolling_mean",
    "sensor_17_rolling_mean",
    "sensor_3_delta",
    "sensor_4_delta",
    "sensor_9_delta",
    "sensor_14_delta",
    "sensor_17_delta",
    "cycle_ratio",
]
RAW_COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


@dataclass
class PredictionBundle:
    dataframe: pd.DataFrame
    summary: dict


def ensure_model() -> RandomForestClassifier:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    train_df = pd.read_csv(TRAIN_PATH)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(train_df[FEATURES], train_df["anomaly_label"])
    joblib.dump(model, MODEL_PATH)
    return model


def _load_uploaded_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None)

    if len(df.columns) == 26:
        df.columns = RAW_COLUMNS

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Upload a file matching the NASA C-MAPSS style schema."
        )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[RAW_COLUMNS]
    df = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

    for sensor in TOP_SENSORS:
        df[f"{sensor}_rolling_mean"] = (
            df.groupby("unit_number")[sensor]
            .transform(lambda series: series.rolling(window=5, min_periods=1).mean())
        )
        df[f"{sensor}_delta"] = df.groupby("unit_number")[sensor].diff().fillna(0)

    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["cycle_ratio"] = df["time_in_cycles"] / max_cycle.clip(lower=1)
    return df


def add_predictions(df: pd.DataFrame, model: RandomForestClassifier) -> PredictionBundle:
    feature_df = build_features(df)
    probabilities = model.predict_proba(feature_df[FEATURES])[:, 1]
    predictions = model.predict(feature_df[FEATURES])

    feature_df["anomaly_probability"] = probabilities.round(4)
    feature_df["anomaly_prediction"] = predictions
    feature_df["severity"] = feature_df["anomaly_probability"].apply(_severity_from_probability)
    feature_df["maintenance_window"] = feature_df["anomaly_probability"].apply(_maintenance_window)

    summary = {
        "rows": int(len(feature_df)),
        "engines": int(feature_df["unit_number"].nunique()),
        "predicted_anomalies": int(feature_df["anomaly_prediction"].sum()),
        "critical_rows": int((feature_df["severity"] == "Critical").sum()),
        "warning_rows": int((feature_df["severity"] == "Warning").sum()),
        "normal_rows": int((feature_df["severity"] == "Normal").sum()),
    }
    return PredictionBundle(dataframe=feature_df, summary=summary)


def _severity_from_probability(probability: float) -> str:
    if probability >= 0.85:
        return "Critical"
    if probability >= 0.55:
        return "Warning"
    return "Normal"


def _maintenance_window(probability: float) -> str:
    if probability >= 0.85:
        return "Failure risk very soon"
    if probability >= 0.55:
        return "Inspect in the next operating cycles"
    return "Continue routine monitoring"


def predict_from_file(path: Path) -> PredictionBundle:
    model = ensure_model()
    df = _load_uploaded_dataframe(path)
    _validate_columns(df)
    return add_predictions(df, model)
