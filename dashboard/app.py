import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

import sys
sys.path.append(str(BASE_DIR))

from src.diagnostic_agent import compute_engine_baseline, diagnose_engine
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

TRAIN_DATA_PATH = PROCESSED_DIR / "train_anomaly.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test_anomaly.csv"
IF_METRICS_PATH = REPORTS_DIR / "isolation_forest_metrics.json"
RF_METRICS_PATH = REPORTS_DIR / "random_forest_metrics.json"
XGB_METRICS_PATH = REPORTS_DIR / "xgboost_metrics.json"


st.set_page_config(page_title="Industrial Predictive Maintenance Dashboard", layout="wide")


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics: dict, label: str, score: str) -> float:
    value = metrics["classification_report"][label]
    if isinstance(value, dict):
        return value[score]
    return value


def add_progress_panel() -> None:
    st.markdown(
        """
        <style>
        .progress-box {
            position: fixed;
            top: 1rem;
            right: 1rem;
            width: 320px;
            background: #f6f8fb;
            border: 1px solid #d6dbe4;
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
            z-index: 9999;
        }
        .progress-box h4 {
            margin: 0 0 10px 0;
            font-size: 16px;
        }
        .progress-box ul {
            margin: 0;
            padding-left: 18px;
            font-size: 13px;
        }
        .progress-box p {
            margin: 0 0 8px 0;
            font-size: 13px;
        }
        </style>
        <div class="progress-box">
            <h4>Avancement du projet</h4>
            <p><strong>Progression globale :</strong> 65%</p>
            <ul>
                <li>Bloc 1 : Dataset integre</li>
                <li>Bloc 2 : Pretraitement + RUL termine</li>
                <li>Bloc 3 : Dataset anomalie + features termine</li>
                <li>Bloc 4 : Isolation Forest et Random Forest termines</li>
                <li>Bloc 5 : Agent diagnostic a faire</li>
                <li>Bloc 6 : Dashboard en cours</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    add_progress_panel()

    train_df = load_csv(TRAIN_DATA_PATH)
    test_df = load_csv(TEST_DATA_PATH)
    if_metrics = load_json(IF_METRICS_PATH)
    rf_metrics = load_json(RF_METRICS_PATH)
    xgb_metrics = load_json(XGB_METRICS_PATH) if XGB_METRICS_PATH.exists() else None
    baseline = compute_engine_baseline(train_df)

    st.title("Industrial Predictive Maintenance Dashboard")
    st.caption("NASA C-MAPSS-based predictive maintenance prototype")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train Rows", f"{len(train_df):,}")
    col2.metric("Test Rows", f"{len(test_df):,}")
    col3.metric("Train Anomalies", int(train_df["anomaly_label"].sum()))
    col4.metric("Test Anomalies", int(test_df["anomaly_label"].sum()))

    st.subheader("Model Comparison")
    comparison_rows = [
        {
            "Model": "Isolation Forest",
            "Accuracy": metric_value(if_metrics, "accuracy", "precision"),
            "Anomaly Precision": metric_value(if_metrics, "1", "precision"),
            "Anomaly Recall": metric_value(if_metrics, "1", "recall"),
            "Anomaly F1": metric_value(if_metrics, "1", "f1-score"),
        },
        {
            "Model": "Random Forest",
            "Accuracy": metric_value(rf_metrics, "accuracy", "precision"),
            "Anomaly Precision": metric_value(rf_metrics, "1", "precision"),
            "Anomaly Recall": metric_value(rf_metrics, "1", "recall"),
            "Anomaly F1": metric_value(rf_metrics, "1", "f1-score"),
        },
    ]
    if xgb_metrics is not None:
        comparison_rows.append(
            {
                "Model": "XGBoost",
                "Accuracy": metric_value(xgb_metrics, "accuracy", "precision"),
                "Anomaly Precision": metric_value(xgb_metrics, "1", "precision"),
                "Anomaly Recall": metric_value(xgb_metrics, "1", "recall"),
                "Anomaly F1": metric_value(xgb_metrics, "1", "f1-score"),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Top Random Forest Features")
        st.caption("Left axis = feature importance score. Higher values mean the feature influences the model more.")
        feature_df = pd.DataFrame(
            rf_metrics["top_feature_importance"].items(),
            columns=["Feature", "Importance"],
        ).set_index("Feature")
        st.bar_chart(feature_df)

    with right:
        st.subheader("Anomaly Distribution")
        st.caption("Left axis = number of rows in the selected dataset. It counts how many samples are normal or anomalous.")
        anomaly_counts = test_df["anomaly_label"].value_counts().sort_index()
        anomaly_chart = pd.DataFrame(
            {
                "Label": ["Normal", "Anomaly"],
                "Count": [int(anomaly_counts.get(0, 0)), int(anomaly_counts.get(1, 0))],
            }
        ).set_index("Label")
        st.bar_chart(anomaly_chart)

    st.subheader("Sensor Trends by Engine")
    engine_id = st.selectbox("Select Engine", sorted(test_df["unit_number"].unique()))
    sensor_name = st.selectbox(
        "Select Sensor",
        ["sensor_3", "sensor_4", "sensor_9", "sensor_14", "sensor_17"],
    )

    engine_df = test_df[test_df["unit_number"] == engine_id].copy()
    engine_df = engine_df.sort_values("time_in_cycles")
    diagnostic = diagnose_engine(engine_df, baseline)

    st.subheader("Diagnostic Assistant")
    diag_col1, diag_col2 = st.columns(2)
    diag_col1.metric("Status", diagnostic.status)
    diag_col2.metric("Severity", diagnostic.severity)
    st.write("Probable Causes")
    for cause in diagnostic.probable_causes:
        st.write(f"- {cause}")
    st.info(diagnostic.recommendation)

    st.subheader("Sensor Trends by Engine")
    st.caption(
        "Left axis = sensor value and RUL value across cycles. "
        "For sensors, it is the measured signal level. For RUL, it is the estimated remaining useful life."
    )
    chart_df = engine_df.set_index("time_in_cycles")[[sensor_name, "RUL"]]
    st.line_chart(chart_df)

    st.subheader("Critical Samples")
    critical_df = test_df[test_df["anomaly_label"] == 1].copy()
    critical_df = critical_df[
        ["unit_number", "time_in_cycles", "RUL", "sensor_3", "sensor_4", "sensor_9", "sensor_14", "sensor_17"]
    ].head(50)
    st.dataframe(critical_df, use_container_width=True)


if __name__ == "__main__":
    main()
