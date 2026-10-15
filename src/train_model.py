
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "Member-4" / "app" / "solar_model.joblib"

def train_model():
    X_path = DATA_DIR / "X_train_base.npy"
    y_path = DATA_DIR / "y_train_base.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Processed dataset missing. Run prepare_dataset.py first."
        )

    X = np.load(X_path)
    y = np.load(y_path)

    if X.ndim != 2 or X.shape[1] != 8:
        raise ValueError(f"Expected 8 features; got shape {X.shape}")

    # Training aur testing data separate karo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel training complete!")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"MAE: {mae:.2f} W/m²")
    print(f"RMSE: {rmse:.2f} W/m²")
    print(f"R² score: {r2:.4f}")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
