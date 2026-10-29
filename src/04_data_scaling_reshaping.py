from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "reports" / "data_splitting"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "scaled_data"

TARGET_COLUMN = "Target_GHI_NextHour"
TIME_COLUMN = "Timestamp"

FEATURE_MAP = {
    "GHI": "GHI",
    "DNI": "DNI",
    "DHI": "DHI",
    "Temperature": "Temperature",
    "Humidity": "Relative Humidity",
    "Pressure": "Pressure",
    "Wind Speed": "Wind Speed",
    "Solar Zenith Angle": "Solar Zenith Angle",
}

FEATURE_COLUMNS = list(FEATURE_MAP.keys())


def load_splits():
    paths = {
        "train": SPLIT_DIR / "train.csv",
        "validation": SPLIT_DIR / "validation.csv",
        "test": SPLIT_DIR / "test.csv",
    }

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. Run Task 1 first."
            )

    train_df = pd.read_csv(paths["train"])
    validation_df = pd.read_csv(paths["validation"])
    test_df = pd.read_csv(paths["test"])

    print("Train dataset shape:", train_df.shape)
    print("Validation dataset shape:", validation_df.shape)
    print("Test dataset shape:", test_df.shape)

    return train_df, validation_df, test_df


def prepare_features(train_df, validation_df, test_df):
    required = list(FEATURE_MAP.values()) + [TARGET_COLUMN]

    for name, df in [
        ("training", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in {name} data: {missing}\n"
                f"Available columns: {df.columns.tolist()}"
            )

    source_columns = list(FEATURE_MAP.values())

    train_x = train_df[source_columns].copy()
    validation_x = validation_df[source_columns].copy()
    test_x = test_df[source_columns].copy()

    rename_map = {
        source: model_name
        for model_name, source in FEATURE_MAP.items()
    }

    train_x.rename(columns=rename_map, inplace=True)
    validation_x.rename(columns=rename_map, inplace=True)
    test_x.rename(columns=rename_map, inplace=True)

    train_x = train_x[FEATURE_COLUMNS]
    validation_x = validation_x[FEATURE_COLUMNS]
    test_x = test_x[FEATURE_COLUMNS]

    frames = [train_x, validation_x, test_x]

    for frame in frames:
        frame[:] = frame.apply(pd.to_numeric, errors="coerce")
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    medians = train_x.median().fillna(0)

    train_x = train_x.fillna(medians)
    validation_x = validation_x.fillna(medians)
    test_x = test_x.fillna(medians)

    for name, frame in [
        ("training", train_x),
        ("validation", validation_x),
        ("test", test_x),
    ]:
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values remain in {name} features.")

    print("\nSelected features:")
    for column in FEATURE_COLUMNS:
        print("-", column)

    return train_x, validation_x, test_x, medians


def scale_features(train_x, validation_x, test_x):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    test_scaled = scaler.transform(test_x)

    return scaler, train_scaled, validation_scaled, test_scaled




def save_results(
    train_df,
    validation_df,
    test_df,
    medians,
    scaler,
    train_scaled,
    validation_scaled,
    test_scaled,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, OUTPUT_DIR / "feature_scaler.joblib")

    config = {
        "target_column": TARGET_COLUMN,
        "time_column": TIME_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "feature_source_columns": list(FEATURE_MAP.values()),
        "feature_mapping": FEATURE_MAP,
        "scaling_method": "StandardScaler",
        "scaler_fit_on": "training_data_only",
        "imputation_medians": {
            column: float(value) for column, value in medians.items()
        },
    }

    with open(OUTPUT_DIR / "scaling_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    splits = [
        ("train", train_df, train_scaled),
        ("validation", validation_df, validation_scaled),
        ("test", test_df, test_scaled),
    ]

    for name, original_df, scaled_array in splits:
        scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS)

        # Keep the target in its original units (W/mÂ²).
        scaled_df[TARGET_COLUMN] = pd.to_numeric(
            original_df[TARGET_COLUMN], errors="coerce"
        ).to_numpy()

        if TIME_COLUMN in original_df.columns:
            scaled_df.insert(0, TIME_COLUMN, original_df[TIME_COLUMN].to_numpy())

        scaled_df.to_csv(OUTPUT_DIR / f"{name}_scaled.csv", index=False)

    y_train = pd.to_numeric(train_df[TARGET_COLUMN], errors="coerce").to_numpy()
    y_validation = pd.to_numeric(
        validation_df[TARGET_COLUMN], errors="coerce"
    ).to_numpy()
    y_test = pd.to_numeric(test_df[TARGET_COLUMN], errors="coerce").to_numpy()

    if not (
        np.isfinite(y_train).all()
        and np.isfinite(y_validation).all()
        and np.isfinite(y_test).all()
    ):
        raise ValueError("Target column contains missing or non-finite values.")

    np.savez_compressed(
        OUTPUT_DIR / "scaled_arrays.npz",
        X_train=train_scaled,
        X_validation=validation_scaled,
        X_test=test_scaled,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        feature_columns=np.array(FEATURE_COLUMNS),
    )

    print("\nAll output files saved successfully.")


def main():
    print("=" * 50)
    print("TASK 2: DATA SCALING & RESHAPING")
    print("=" * 50)

    train_df, validation_df, test_df = load_splits()

    train_x, validation_x, test_x, medians = prepare_features(
        train_df, validation_df, test_df
    )

    scaler, train_scaled, validation_scaled, test_scaled = scale_features(
        train_x, validation_x, test_x
    )

    save_results(
        train_df,
        validation_df,
        test_df,
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
    print("Number of input features:", len(FEATURE_COLUMNS))
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
