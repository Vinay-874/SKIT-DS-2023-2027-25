from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nsrdb_cleaned_base.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_splitting"
)

TIME_COLUMN = "Timestamp"
TARGET_COLUMN = "Target_GHI_NextHour"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Please check data/processed/nsrdb_cleaned_base.csv"
        )

    print("=" * 60)
    print("LOADING SOLAR FORECASTING DATASET")
    print("=" * 60)

    print(f"Dataset path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    print("\nOriginal dataset shape:", df.shape)
    print("\nAvailable columns:")
    print(df.columns.tolist())

    # Check required columns without depending on capitalization
    column_lookup = {
        column.casefold(): column
        for column in df.columns
    }

    actual_time_column = column_lookup.get(
        TIME_COLUMN.casefold()
    )

    actual_target_column = column_lookup.get(
        TARGET_COLUMN.casefold()
    )

    if actual_time_column is None:
        raise ValueError(
            f"Missing timestamp column: {TIME_COLUMN}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if actual_target_column is None:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Standardize required column names
    df.rename(
        columns={
            actual_time_column: TIME_COLUMN,
            actual_target_column: TARGET_COLUMN,
        },
        inplace=True,
    )

    # Convert timestamp and target to correct types
    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        errors="coerce",
        utc=True,
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    original_rows = len(df)

    # Remove rows with missing timestamp or target
    df = df.dropna(
        subset=[TIME_COLUMN, TARGET_COLUMN]
    ).copy()

    # Remove infinite target values
    df = df.loc[
        np.isfinite(df[TARGET_COLUMN])
    ].copy()

    removed_rows = original_rows - len(df)

    # Sort chronologically
    df = (
        df.sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )

    if len(df) < 10:
        raise ValueError(
            "Insufficient valid data for splitting."
        )

    print("\nValid rows:", len(df))
    print("Invalid rows removed:", removed_rows)
    print("Data validation completed successfully.")

    return df


# ==========================================
# 3. SPLIT DATA CHRONOLOGICALLY
# ==========================================

def split_data(df):

    total_rows = len(df)

    if not np.isclose(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO,
        1.0,
    ):
        raise ValueError(
            "Train, validation and test ratios must sum to 1."
        )

    # Preserve chronological order; do not shuffle
    train_end = int(total_rows * TRAIN_RATIO)

    validation_end = int(
        total_rows * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise ValueError(
            "One of the data splits is empty."
        )

    # Verify chronological ordering between splits
    if not (
        train_df[TIME_COLUMN].max()
        <= validation_df[TIME_COLUMN].min()
        and
        validation_df[TIME_COLUMN].max()
        <= test_df[TIME_COLUMN].min()
    ):
        raise ValueError(
            "Chronological split validation failed."
        )

    # Verify that no rows have been lost
    if (
        len(train_df)
        + len(validation_df)
        + len(test_df)
        != total_rows
    ):
        raise ValueError(
            "The split row counts do not match the input dataset."
        )

    print("\n" + "=" * 60)
    print("DATA SPLITTING COMPLETED")
    print("=" * 60)

    print(f"Total valid rows: {total_rows}")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Testing rows: {len(test_df)}")

    return train_df, validation_df, test_df


# ==========================================
# 4. CREATE EVALUATION REPORT
# ==========================================

def build_evaluation_report(
    train_df,
    validation_df,
    test_df,
):

    def summarize_split(name, frame):

        target = frame[TARGET_COLUMN]

        return {
            "split": name,
            "rows": int(len(frame)),
            "start_timestamp": str(
                frame[TIME_COLUMN].min()
            ),
            "end_timestamp": str(
                frame[TIME_COLUMN].max()
            ),
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
            "target_min": float(target.min()),
            "target_max": float(target.max()),
        }

    report = {
        "project": "Solar Power Forecasting",
        "task": "Data Splitting and Evaluation Framework",
        "dataset_path": str(DATA_PATH),
        "timestamp_column": TIME_COLUMN,
        "target_column": TARGET_COLUMN,
        "split_method": "Chronological splitting without shuffling",

        "split_ratios": {
            "training": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "testing": TEST_RATIO,
        },

        "data_leakage_prevention": [
            "Data is sorted chronologically.",
            "Training observations precede validation observations.",
            "Validation observations precede test observations.",
            "Data is not randomly shuffled.",
        ],

        "evaluation_metrics": {
            "MAE": "Mean Absolute Error",
            "RMSE": "Root Mean Squared Error",
            "R2": "Coefficient of Determination",
        },

        "model_metrics_status": (
            "Not calculated yet. Model predictions are required."
        ),

        "splits": [
            summarize_split("training", train_df),
            summarize_split("validation", validation_df),
            summarize_split("testing", test_df),
        ],
    }

    return report


# ==========================================
# 5. SAVE OUTPUT FILES
# ==========================================

def save_outputs(
    train_df,
    validation_df,
    test_df,
    report,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"
    report_path = OUTPUT_DIR / "evaluation_report.json"

    train_df.to_csv(
        train_path,
        index=False,
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("OUTPUT FILES SAVED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTraining file: {train_path}")
    print(f"Validation file: {validation_path}")
    print(f"Testing file: {test_path}")
    print(f"Evaluation report: {report_path}")


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def main():

    print("\n")
    print("=" * 60)
    print("TASK 1: DATA SPLITTING AND EVALUATION")
    print("=" * 60)

    df = load_data()

    train_df, validation_df, test_df = split_data(df)

    report = build_evaluation_report(
        train_df,
        validation_df,
        test_df,
    )

    save_outputs(
        train_df,
        validation_df,
        test_df,
        report,
    )

    print("\n" + "=" * 60)
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()