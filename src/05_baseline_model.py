from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "reports" / "scaled_data"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "baseline_model"

TARGET_COLUMN = "Target_GHI_NextHour (W/m²)"
TIME_COLUMN = "Timestamp"
 
# LOAD DATA

def load_data():
    files = {
        "train": DATA_DIR / "train_scaled.csv",
        "validation": DATA_DIR / "validation_scaled.csv",
        "test": DATA_DIR / "test_scaled.csv",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {path}\n"
                "Run Task 1 and Task 2 successfully first."
            )

    train_df = pd.read_csv(files["train"])
    val_df = pd.read_csv(files["validation"])
    test_df = pd.read_csv(files["test"])

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column missing in {name}: {TARGET_COLUMN}\n"
                f"Available columns: {df.columns.tolist()}"
            )

    return train_df, val_df, test_df

# PREPARE FEATURES AND TARGET 
def prepare_xy(train_df, val_df, test_df):
    excluded = {TARGET_COLUMN, TIME_COLUMN}

    features = [
        col for col in train_df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    if not features:
        raise ValueError("No numeric input features found.")

    for name, df in [
        ("validation", val_df),
        ("test", test_df),
    ]:
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing features in {name} data: {missing}"
            )

    X_train = train_df[features].replace(
        [np.inf, -np.inf], np.nan
    )
    X_val = val_df[features].replace(
        [np.inf, -np.inf], np.nan
    )
    X_test = test_df[features].replace(
        [np.inf, -np.inf], np.nan
    )

    # Use training-set medians only.
    medians = X_train.median().fillna(0)

    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    y_train = pd.to_numeric(
        train_df[TARGET_COLUMN], errors="coerce"
    )
    y_val = pd.to_numeric(
        val_df[TARGET_COLUMN], errors="coerce"
    )
    y_test = pd.to_numeric(
        test_df[TARGET_COLUMN], errors="coerce"
    )

    # Remove rows whose target is missing/non-finite.
    def clean(X, y):
        valid = y.notna() & np.isfinite(y)
        return X.loc[valid], y.loc[valid]

    X_train, y_train = clean(X_train, y_train)
    X_val, y_val = clean(X_val, y_val)
    X_test, y_test = clean(X_test, y_test)

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError("One of the datasets has no valid target rows.")

    return (
        features,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

# TRAIN AND EVALUATE BASELINE
def evaluate(model, X, y, split_name):
    predictions = model.predict(X)

    return {
        "dataset": split_name,
        "rows": int(len(y)),
        "MAE": float(mean_absolute_error(y, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y, predictions))),
        "R2": float(r2_score(y, predictions)) if len(y) >= 2 else None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_data()

    (
        features,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = prepare_xy(train_df, val_df, test_df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    results = [
        evaluate(model, X_train, y_train, "train"),
        evaluate(model, X_val, y_val, "validation"),
        evaluate(model, X_test, y_test, "test"),
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "baseline_metrics.csv",
        index=False,
    )

    # Save test predictions for inspection.
    predictions_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": model.predict(X_test),
    })
    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # Save model and feature configuration.
    import joblib
    joblib.dump(model, OUTPUT_DIR / "baseline_model.joblib")

    with open(
        OUTPUT_DIR / "baseline_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model": "LinearRegression",
                "target_column": TARGET_COLUMN,
                "feature_columns": features,
                "training_rows": len(X_train),
                "validation_rows": len(X_val),
                "test_rows": len(X_test),
            },
            f,
            indent=4,
        )

    print("\nBASELINE MODEL SETUP COMPLETED")
    print("-" * 45)
    print("Features:", features)
    print("\nEvaluation metrics:")
    print(results_df.to_string(index=False))
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()