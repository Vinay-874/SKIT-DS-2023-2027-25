
import pandas as pd
from pathlib import Path

files = [
    "data/raw/nsrdb_raw.csv",
    "data/nsrdb_mock_dataset.csv",
    "data/processed/nsrdb_cleaned_base.csv"
]

cols = [
    "GHI (W/m²)",
    "Target_GHI_NextHour (W/m²)"
]

for file in files:
    path = Path(file)

    print(f"\n--- FILE: {file} ---")

    if not path.exists():
        print("File not found")
        continue

    df = pd.read_csv(path)
    print("Total rows:", len(df))

    if all(col in df.columns for col in cols):
        print(df[cols].head(4).to_string(index=False))
    else:
        print("Required columns missing")
