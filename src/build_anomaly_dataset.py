import pandas as pd
from pathlib import Path


BASE_DIR = Path(r"C:\Users\yassi\OneDrive\Documents\projet pfa")
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train_with_rul.csv"
TEST_PATH = PROCESSED_DIR / "test_with_rul.csv"

TRAIN_OUTPUT = PROCESSED_DIR / "train_anomaly.csv"
TEST_OUTPUT = PROCESSED_DIR / "test_anomaly.csv"

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
NEAR_FAILURE_THRESHOLD = 30


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def select_informative_sensors(df: pd.DataFrame) -> list[str]:
    std_series = df[SENSOR_COLUMNS].std().sort_values(ascending=False)
    informative = std_series[std_series > 0.0].index.tolist()
    return informative


def build_anomaly_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anomaly_label"] = (df["RUL"] <= NEAR_FAILURE_THRESHOLD).astype(int)
    return df


def add_health_features(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    df = df.copy()

    for sensor in sensors[:5]:
        df[f"{sensor}_rolling_mean"] = (
            df.groupby("unit_number")[sensor]
            .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        )
        df[f"{sensor}_delta"] = df.groupby("unit_number")[sensor].diff().fillna(0)

    df["cycle_ratio"] = df["time_in_cycles"] / df.groupby("unit_number")["time_in_cycles"].transform("max")
    return df


def print_dataset_summary(name: str, df: pd.DataFrame, sensors: list[str]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print("Shape:", df.shape)
    print("Engines:", df["unit_number"].nunique())
    print("Anomaly distribution:")
    print(df["anomaly_label"].value_counts().sort_index())
    print("\nTop variable sensors:")
    print(df[sensors].std().sort_values(ascending=False).head(10))


def main() -> None:
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    sensors = select_informative_sensors(train_df)
    train_df = build_anomaly_labels(train_df)
    test_df = build_anomaly_labels(test_df)

    train_df = add_health_features(train_df, sensors)
    test_df = add_health_features(test_df, sensors)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    print_dataset_summary("Train anomaly dataset", train_df, sensors)
    print_dataset_summary("Test anomaly dataset", test_df, sensors)


if __name__ == "__main__":
    main()
