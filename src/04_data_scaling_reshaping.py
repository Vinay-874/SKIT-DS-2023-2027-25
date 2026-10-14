from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "reports" / "data_splitting"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "scaled_data"

TARGET_COLUMN = "Target_GHI_NextHour (W/m²)"
TIME_COLUMN = "Timestamp"

# LOAD TRAIN, VALIDATION AND TEST DATA
def load_splits():

    train_path = SPLIT_DIR / "train.csv"
    validation_path = SPLIT_DIR / "validation.csv"
    test_path = SPLIT_DIR / "test.csv"

    for path in [train_path, validation_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required split file not found: {path}\n"
                "Please run Task 1 first."
            )

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print("Train dataset shape:", train_df.shape)
    print("Validation dataset shape:", validation_df.shape)
    print("Test dataset shape:", test_df.shape)

    return train_df, validation_df, test_df

# PREPARE NUMERIC FEATURES
def prepare_features(train_df, validation_df, test_df):

    for name, df in [
        ("training", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' is missing "
                f"from {name} data.\n"
                f"Available columns: {df.columns.tolist()}"
            )

    # Exclude timestamp and target from input features.
    excluded_columns = {TARGET_COLUMN, TIME_COLUMN}

    feature_columns = [
        column
        for column in train_df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(train_df[column])
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric input features were found."
        )

    # Check that validation and test contain the same features.
    for column in feature_columns:
        if column not in validation_df.columns:
            raise ValueError(
                f"Feature '{column}' is missing from validation data."
            )

        if column not in test_df.columns:
            raise ValueError(
                f"Feature '{column}' is missing from test data."
            )

    # Convert features to numeric values.
    train_x = train_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    validation_x = validation_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    test_x = test_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Replace infinite values with missing values.
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    validation_x = validation_x.replace(
        [np.inf, -np.inf], np.nan
    )
    test_x = test_x.replace([np.inf, -np.inf], np.nan)

    # Calculate imputation values from training data only.
    medians = train_x.median().fillna(0)

    train_x = train_x.fillna(medians)
    validation_x = validation_x.fillna(medians)
    test_x = test_x.fillna(medians)

    # Final safety check.
    for name, frame in [
        ("training", train_x),
        ("validation", validation_x),
        ("test", test_x),
    ]:
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Non-finite values remain in {name} features."
            )

    print("\nSelected input features:")
    for column in feature_columns:
        print("-", column)

    return (
        feature_columns,
        train_x,
        validation_x,
        test_x,
        medians,
    )
# SCALE FEATURES

def scale_features(train_x, validation_x, test_x):

    scaler = StandardScaler()

    # Fit the scaler only on training features.
    train_scaled = scaler.fit_transform(train_x)

    # Use the same fitted scaler for validation and test.
    validation_scaled = scaler.transform(validation_x)
    test_scaled = scaler.transform(test_x)

    return (
        scaler,
        train_scaled,
        validation_scaled,
        test_scaled,
    )

# SAVE RESULTS

def save_results(
    train_df,
    validation_df,
    test_df,
    feature_columns,
    medians,
    scaler,
    train_scaled,
    validation_scaled,
    test_scaled,
):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the fitted scaler.
    joblib.dump(
        scaler,
        OUTPUT_DIR / "feature_scaler.joblib"
    )

    # Save configuration for reproducibility.
    config = {
        "target_column": TARGET_COLUMN,
        "time_column": TIME_COLUMN,
        "feature_columns": feature_columns,
        "scaling_method": "StandardScaler",
        "scaler_fit_on": "training_data_only",
        "imputation_medians": {
            column: float(value)
            for column, value in medians.items()
        },
    }

    with open(
        OUTPUT_DIR / "scaling_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)

    # Save scaled feature datasets.
    splits = [
        ("train", train_df, train_scaled),
        ("validation", validation_df, validation_scaled),
        ("test", test_df, test_scaled),
    ]

    for name, original_df, scaled_array in splits:

        scaled_df = pd.DataFrame(
            scaled_array,
            columns=feature_columns,
        )

        # Keep the original target unscaled.
        scaled_df[TARGET_COLUMN] = (
            pd.to_numeric(
                original_df[TARGET_COLUMN],
                errors="coerce",
            ).to_numpy()
        )

        # Preserve timestamps where available.
        if TIME_COLUMN in original_df.columns:
            scaled_df.insert(
                0,
                TIME_COLUMN,
                original_df[TIME_COLUMN].to_numpy(),
            )

        scaled_df.to_csv(
            OUTPUT_DIR / f"{name}_scaled.csv",
            index=False,
        )

    # Save model-ready numeric arrays.
    np.savez_compressed(
        OUTPUT_DIR / "scaled_arrays.npz",
        X_train=train_scaled,
        X_validation=validation_scaled,
        X_test=test_scaled,
        y_train=pd.to_numeric(
            train_df[TARGET_COLUMN], errors="coerce"
        ).to_numpy(),
        y_validation=pd.to_numeric(
            validation_df[TARGET_COLUMN], errors="coerce"
        ).to_numpy(),
        y_test=pd.to_numeric(
            test_df[TARGET_COLUMN], errors="coerce"
        ).to_numpy(),
        feature_columns=np.array(feature_columns),
    )

    print("\nAll output files saved successfully.")

# MAIN EXECUTION
def main():

    print("=" * 50)
    print("TASK 2: DATA SCALING & RESHAPING")
    print("=" * 50)

    train_df, validation_df, test_df = load_splits()

    (
        feature_columns,
        train_x,
        validation_x,
        test_x,
        medians,
    ) = prepare_features(
        train_df,
        validation_df,
        test_df,
    )

    (
        scaler,
        train_scaled,
        validation_scaled,
        test_scaled,
    ) = scale_features(
        train_x,
        validation_x,
        test_x,
    )

    save_results(
        train_df,
        validation_df,
        test_df,
        feature_columns,
        medians,
        scaler,
        train_scaled,
        validation_scaled,
        test_scaled,
    )

    print("\n" + "=" * 50)
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print("Training array shape:", train_scaled.shape)
    print("Validation array shape:", validation_scaled.shape)
    print("Testing array shape:", test_scaled.shape)

    print("\nTarget column remains unscaled.")
    print("Scaler was fitted only on training data.")

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()