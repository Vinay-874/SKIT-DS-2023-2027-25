
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "raw" / "nsrdb_raw.csv"

FEATURES = [
    "GHI (W/m²)",
    "DNI (W/m²)",
    "DHI (W/m²)",
    "Temp (°C)",
    "RH (%)",
    "Pressure (hPa)",
    "Wind Speed (m/s)",
    "Solar Zenith Angle (°)",
]
TARGET = "Target_GHI_NextHour (W/m²)"


def evaluate(name, X_train, X_test, y_train, y_test):
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }

    print(f"\n--- {name} ---")

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        print(
            f"{model_name}: "
            f"MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}"
        )


def main():
    df = pd.read_csv(CSV_PATH)

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print("Required columns not found:", missing)
        print("Actual CSV columns:", df.columns.tolist())
        return

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    print("Valid rows:", len(df))
    print("Feature shape:", X.shape)
    print("Negative DHI:", int((df["DHI (W/m²)"] < 0).sum()))

    ghi = df["GHI (W/m²)"]
    print("\nTarget alignment check:")
    print("Same-row GHI correlation:", round(ghi.corr(df[TARGET]), 4))
    print("GHI 1 row earlier correlation:",
          round(ghi.shift(1).corr(df[TARGET]), 4))
    print("GHI 2 rows earlier correlation:",
          round(ghi.shift(2).corr(df[TARGET]), 4))

    # Test A: reproduce the existing random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    evaluate("Random split", X_train, X_test, y_train, y_test)

    # Test B: train on earlier rows, test on later rows
    split = int(len(X) * 0.8)
    evaluate(
        "Chronological split",
        X[:split], X[split:],
        y[:split], y[split:],
    )

    print("\nDiagnostic complete. No model or dataset files were changed.")


if __name__ == "__main__":
    main()
