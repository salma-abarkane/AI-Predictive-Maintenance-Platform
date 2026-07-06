import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(r"C:\Users\yassi\OneDrive\Documents\projet pfa")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

TRAIN_PATH = PROCESSED_DIR / "train_anomaly.csv"
TEST_PATH = PROCESSED_DIR / "test_anomaly.csv"
METRICS_PATH = REPORTS_DIR / "isolation_forest_metrics.json"

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


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    x_train = train_df[FEATURES]
    x_test = test_df[FEATURES]
    y_test = test_df["anomaly_label"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        random_state=42,
    )
    model.fit(x_train_scaled)

    raw_predictions = model.predict(x_test_scaled)
    y_pred = pd.Series(raw_predictions).map({1: 0, -1: 1})

    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        "features": FEATURES,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Isolation Forest trained successfully")
    print("Features used:", len(FEATURES))
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print(f"\nMetrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
