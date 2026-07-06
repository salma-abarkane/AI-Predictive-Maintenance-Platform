import pandas as pd
from pathlib import Path


BASE_DIR = Path(r"C:\Users\yassi\OneDrive\Documents\projet pfa")
RAW_DIR = BASE_DIR / "data" / "raw" / "PM"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = RAW_DIR / "PM_train.txt"
TEST_PATH = RAW_DIR / "PM_test.txt"
TRUTH_PATH = RAW_DIR / "PM_truth.txt"

COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def load_pm_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLUMNS
    return df


def load_truth_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :1]
    df.columns = ["RUL"]
    return df


def add_rul_to_train(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df = df.copy()
    df["RUL"] = max_cycles - df["time_in_cycles"]
    return df


def build_test_truth(test_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = test_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]

    truth_df = truth_df.copy()
    truth_df["unit_number"] = range(1, len(truth_df) + 1)

    merged = max_cycles.merge(truth_df, on="unit_number", how="left")
    merged["failure_cycle"] = merged["max_cycle"] + merged["RUL"]

    test_df = test_df.merge(merged[["unit_number", "failure_cycle"]], on="unit_number", how="left")
    test_df["RUL"] = test_df["failure_cycle"] - test_df["time_in_cycles"]
    test_df = test_df.drop(columns=["failure_cycle"])
    return test_df


def print_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    if "unit_number" in df.columns:
        print("Engines:", df["unit_number"].nunique())
    print(df.head())


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_pm_file(TRAIN_PATH)
    test_df = load_pm_file(TEST_PATH)
    truth_df = load_truth_file(TRUTH_PATH)

    train_processed = add_rul_to_train(train_df)
    test_processed = build_test_truth(test_df, truth_df)

    train_processed.to_csv(PROCESSED_DIR / "train_with_rul.csv", index=False)
    test_processed.to_csv(PROCESSED_DIR / "test_with_rul.csv", index=False)
    truth_df.to_csv(PROCESSED_DIR / "truth_rul.csv", index=False)

    print_summary("Train processed", train_processed)
    print_summary("Test processed", test_processed)
    print_summary("Truth", truth_df)


if __name__ == "__main__":
    main()
