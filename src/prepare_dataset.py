
import pandas as pd
import numpy as np
from pathlib import Path

# Project ke root folder ka path automatically find karega
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def prepare_ml_dataset():
    data_file = PROJECT_ROOT / "data" / "raw" / "nsrdb_raw.csv"
    output_dir = PROJECT_ROOT / "data" / "processed"

    if not data_file.exists():
        print(f"Error: Dataset not found at {data_file}")
        return

    print(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)

    feature_columns = [
        "GHI (W/m²)",
        "DNI (W/m²)",
        "DHI (W/m²)",
        "Temp (°C)",
        "RH (%)",
        "Pressure (hPa)",
        "Wind Speed (m/s)",
        "Solar Zenith Angle (°)"
    ]

    target_column = "Target_GHI_NextHour (W/m²)"
    all_needed = feature_columns + [target_column]

    # Required columns verify karo
    missing_columns = [col for col in all_needed if col not in df.columns]
    if missing_columns:
        print("Error: Required columns missing:", missing_columns)
        return

    # Required columns ko numeric format mein convert karo
    for col in all_needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing values wali rows remove karo
    original_rows = len(df)
    df = df.dropna(subset=all_needed).copy()
    print(f"Rows removed: {original_rows - len(df)}")

    if df.empty:
        print("Error: No valid rows available.")
        return

    X = df[feature_columns].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train_base.npy", X)
    np.save(output_dir / "y_train_base.npy", y)

    # Feature order ko save karo, taaki API same order use kare
    with open(output_dir / "feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_columns))

    print("\nDataset successfully prepared!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Saved files in: {output_dir}")

if __name__ == "__main__":
    prepare_ml_dataset()
