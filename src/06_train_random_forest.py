from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "reports" / "sequence_data"
OUTPUT_DIR = ROOT / "reports" / "model_training" / "random_forest"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sequences(filename):
    path = DATA_DIR / filename

    with np.load(path, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"].astype(np.float64)
        timestamps = data["timestamps"]

    # Convert (samples, 24, 8) into (samples, 192)
    X = X.reshape(X.shape[0], -1)

    if len(X) != len(y) or len(y) != len(timestamps):
        raise ValueError(
            f"Length mismatch in {filename}: "
            f"X={len(X)}, y={len(y)}, timestamps={len(timestamps)}"
        )

    return X, y, timestamps


def calculate_metrics(y_true, predictions):
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R2": float(r2_score(y_true, predictions)),
    }


def main():
    print("Loading sequence datasets...")

    X_train, y_train, _ = load_sequences("train_sequences.npz")
    X_val, y_val, val_timestamps = load_sequences(
        "validation_sequences.npz"
    )
    X_test, y_test, test_timestamps = load_sequences(
        "test_sequences.npz"
    )

    print(f"Train:      X={X_train.shape}, y={y_train.shape}")
    print(f"Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"Test:       X={X_test.shape}, y={y_test.shape}")

    # Check for invalid values
    for name, X, y in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(
                f"{name} data contains NaN or infinite values."
            )

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start_time

    # Validation evaluation
    val_predictions = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_predictions)

    print("\nValidation Results")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Final test evaluation
    test_predictions = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    print("\nTest Results")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save model
    model_path = OUTPUT_DIR / "random_forest_model.joblib"
    dump(model, model_path)

    # Save metrics
    results = {
        "model": "RandomForestRegressor",
        "parameters": model.get_params(),
        "input_features": int(X_train.shape[1]),
        "sequence_length": 24,
        "features_per_timestep": 8,
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "training_time_seconds": round(training_seconds, 2),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Save actual and predicted test values
    predictions_df = pd.DataFrame({
        "Timestamp": test_timestamps,
        "Actual_GHI": y_test,
        "Predicted_GHI": test_predictions,
    })

    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print("\nTraining completed successfully!")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()