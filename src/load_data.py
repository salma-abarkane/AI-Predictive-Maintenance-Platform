import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "PM"

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


train_df = load_pm_file(TRAIN_PATH)
test_df = load_pm_file(TEST_PATH)
truth_df = load_truth_file(TRUTH_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Truth shape:", truth_df.shape)

print(train_df.head())
print(truth_df.head())