

from pathlib import Path
import json

import numpy as np
import pandas as pd

#1.Project configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nsrdb_raw.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_splitting"
)

# Column names from your actual dataset
TIME_COLUMN = "Timestamp"
TARGET_COLUMN = "Target_GHI_NextHour (W/m²)"

# Chronological split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# 2. LOAD AND VALIDATE DATA

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Please check that nsrd_raw.csv exists inside "
            "the data/raw folder."
        )

    print("Loading dataset...")
    print(f"Dataset path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Remove accidental spaces around column names
    df.columns = df.columns.astype(str).str.strip()

    print(f"Original dataset shape: {df.shape}")
    print("Available columns:")
    print(df.columns.tolist())

    # Identify columns without depending on exact capitalization
    column_lookup = {
        col.casefold(): col
        for col in df.columns
    }

    actual_time_column = column_lookup.get(
        TIME_COLUMN.casefold()
    )

    actual_target_column = column_lookup.get(
        TARGET_COLUMN.casefold()
    )

    if actual_time_column is None:
        raise ValueError(
            f"Time column '{TIME_COLUMN}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if actual_target_column is None:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Use the exact column names found in the CSV
    df.rename(
        columns={
            actual_time_column: TIME_COLUMN,
            actual_target_column: TARGET_COLUMN,
        },
        inplace=True,
    )

    # Convert timestamp to datetime
    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        errors="coerce",
        utc=True,
    )

    # Convert target values to numeric
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    # Remove rows without a valid timestamp or target
    original_rows = len(df)

    df = df.dropna(
        subset=[TIME_COLUMN, TARGET_COLUMN]
    ).copy()

    removed_rows = original_rows - len(df)

    # Infinite target values are invalid
    valid_target = np.isfinite(df[TARGET_COLUMN])

    df = df.loc[valid_target].copy()

    # Sort chronologically; do not shuffle
    df = df.sort_values(
        TIME_COLUMN
    ).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "Not enough valid observations to create "
            "train, validation and test datasets."
        )

    print(f"Valid rows: {len(df)}")
    print(f"Rows removed for missing time/target: {removed_rows}")
    print("Data validation completed.")

    return df

# 3. CHRONOLOGICAL DATA SPLITTING
def split_data(df):

    total_rows = len(df)

    if not np.isclose(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO,
        1.0,
    ):
        raise ValueError(
            "Train, validation and test ratios must sum to 1."
        )

    # Calculate chronological split boundaries
    train_end = int(total_rows * TRAIN_RATIO)

    validation_end = int(
        total_rows * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    # Keep all observations in chronological order
    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[validation_end:].copy()

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise ValueError(
            "One of the data splits is empty. "
            "Check the dataset size and split ratios."
        )

    # Confirm chronological ordering across boundaries
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

    return train_df, validation_df, test_df

# 4. EVALUATION FRAMEWORK

def build_evaluation_report(
    train_df,
    validation_df,
    test_df,
):

    def split_summary(name, frame):

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
        "task": "Data Splitting & Evaluation Framework Setup",
        "timestamp_column": TIME_COLUMN,
        "target_column": TARGET_COLUMN,
        "split_method": "Chronological; no random shuffling",
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "data_leakage_prevention": [
            "Observations remain in chronological order.",
            "Training data occurs before validation data.",
            "Validation data occurs before test data.",
            "No random train_test_split is used.",
        ],
        "evaluation_metrics": {
            "MAE": (
                "Mean Absolute Error: average absolute "
                "difference between actual and predicted values."
            ),
            "RMSE": (
                "Root Mean Squared Error: square root of "
                "the average squared prediction error."
            ),
            "R2": (
                "R-squared: compares model predictions with "
                "a mean-based reference."
            ),
        },
        "model_metrics_status": (
            "Not calculated yet; models must be trained first."
        ),
        "splits": [
            split_summary("train", train_df),
            split_summary("validation", validation_df),
            split_summary("test", test_df),
        ],
    }

    return report

# 5. SAVE OUTPUTS
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

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    report_path = (
        OUTPUT_DIR / "evaluation_report.json"
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

    print("\n" + "=" * 50)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Testing rows: {len(test_df)}")

    print("\nChronological ranges:")

    for name, frame in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: "
            f"{frame[TIME_COLUMN].min()} -> "
            f"{frame[TIME_COLUMN].max()}"
        )

    print("\nFiles saved:")

    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "validation.csv")
    print(OUTPUT_DIR / "test.csv")
    print(report_path)

# 6. MAIN PROGRAM
def main():

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


if __name__ == "__main__":
    main()