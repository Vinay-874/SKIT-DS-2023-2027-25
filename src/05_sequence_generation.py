from pathlib import Path

import numpy as np
import pandas as pd

from importlib.util import spec_from_file_location, module_from_spec


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "reports" / "scaled_data"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "sequence_data"

WINDOW_SIZE = 24
TARGET_COLUMN = "Target_GHI_NextHour"
TIME_COLUMN = "Timestamp"

FEATURE_COLUMNS = [
    "GHI",
    "DNI",
    "DHI",
    "Temperature",
    "Humidity",
    "Pressure",
    "Wind Speed",
    "Solar Zenith Angle",
]


def load_sequence_function():
    script_path = PROJECT_ROOT / "src" / "02_sequence_windows.py"

    spec = spec_from_file_location("sequence_windows", script_path)

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load sequence function from: {script_path}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.create_sequences


def process_split(name, create_sequences):
    path = DATA_DIR / f"{name}_scaled.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = FEATURE_COLUMNS + [TARGET_COLUMN, TIME_COLUMN]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {name} data: {missing}")

    # Ensure timestamps are valid and chronological.
    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        errors="raise",
    )

    if not df[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError(
            f"Timestamps in {name} data are not chronological."
        )

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(
            f"Non-finite values found in {name} data."
        )

    # Generate sequences.
    X_seq, y_seq = create_sequences(
        X,
        y,
        window_size=WINDOW_SIZE,
    )

    # Timestamp of the last observation in each input window.
    timestamps = (
        df[TIME_COLUMN]
        .iloc[WINDOW_SIZE - 1:]
        .astype(str)
        .to_numpy()
    )

    # Verify all generated arrays have the same sample count.
    if not (len(X_seq) == len(y_seq) == len(timestamps)):
        raise ValueError(
            f"Sequence length mismatch in {name}: "
            f"X={len(X_seq)}, y={len(y_seq)}, "
            f"timestamps={len(timestamps)}"
        )

    result = {
        "X": X_seq,
        "y": y_seq,
        "timestamps": timestamps,
    }

    return result


def main():
    print("=" * 55)
    print("TASK 3: SEQUENCE GENERATION")
    print("=" * 55)

    create_sequences = load_sequence_function()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["train", "validation", "test"]:
        result = process_split(name, create_sequences)

        print(f"\n{name.capitalize()} split:")
        print("Input sequence shape:", result["X"].shape)
        print("Target shape:", result["y"].shape)
        print("Timestamp count:", len(result["timestamps"]))

        output_path = OUTPUT_DIR / f"{name}_sequences.npz"

        np.savez_compressed(
            output_path,
            X=result["X"],
            y=result["y"],
            timestamps=result["timestamps"],
        )

        print("Saved:", output_path)

    print("\nAll sequence files saved successfully.")
    print("Window size:", WINDOW_SIZE)
    print("Feature count:", len(FEATURE_COLUMNS))
    print("Output directory:", OUTPUT_DIR)
    print("\nTASK 3 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()