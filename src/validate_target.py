
import pandas as pd
from pathlib import Path

# Dataset path
ROOT = Path(__file__).resolve().parent.parent
file_path = ROOT / "data" / "raw" / "nsrdb_raw.csv"

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

df = pd.read_csv(file_path)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="coerce"
)

df = (
    df.dropna(subset=["Timestamp"])
    .sort_values("Timestamp")
    .reset_index(drop=True)
)

ghi_col = "GHI (W/m²)"
dhi_col = "DHI (W/m²)"
target_col = "Target_GHI_NextHour (W/m²)"

required = [ghi_col, dhi_col, target_col]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 1. Dataset quality
print("=" * 50)
print("DATASET QUALITY CHECK")
print("=" * 50)

print("Total rows:", len(df))
print("Duplicate timestamps:", df["Timestamp"].duplicated().sum())
print("Negative GHI values:", (df[ghi_col] < 0).sum())
print("Negative DHI values:", (df[dhi_col] < 0).sum())
print("Negative target values:", (df[target_col] < 0).sum())

# 2. Timestamp intervals
print("\nMOST COMMON TIMESTAMP INTERVALS")
print(df["Timestamp"].diff().dropna().value_counts().head())

# 3. Target correlations
print("\nCORRELATION WITH TARGET")
print("Current GHI:", df[ghi_col].corr(df[target_col]))
print("GHI one row later:", df[ghi_col].shift(-1).corr(df[target_col]))
print("GHI two rows later:", df[ghi_col].shift(-2).corr(df[target_col]))

# 4. Exact target alignment
future_ghi = df[ghi_col].shift(-2)

comparison = pd.DataFrame({
    "target": df[target_col],
    "future_ghi": future_ghi
}).dropna()

difference = (comparison["target"] - comparison["future_ghi"]).abs()
exact_matches = (difference < 1e-8).sum()

print("\nEXACT TARGET ALIGNMENT CHECK")
print("Rows compared:", len(comparison))
print("Exact matches:", exact_matches)
print("Non-matching rows:", len(comparison) - exact_matches)

if len(comparison) > 0:
    print(f"Match percentage: {exact_matches / len(comparison) * 100:.2f}%")
    print("Maximum absolute difference:", difference.max())

# 5. GHI relationship across different time lags
print("\nGHI RELATIONSHIP WITH FUTURE VALUES")

for lag in [1, 2, 3, 4, 6, 12]:
    correlation = df[ghi_col].corr(df[ghi_col].shift(-lag))
    print(f"GHI correlation at lag {lag}: {correlation:.4f}")

# 6. Inspect negative DHI examples
print("\nNEGATIVE DHI VALUE EXAMPLES")

negative_dhi = df.loc[
    df[dhi_col] < 0,
    ["Timestamp", ghi_col, dhi_col]
]

print("Total negative DHI rows:", len(negative_dhi))
print(negative_dhi.head(10).to_string(index=False))

# 7. First 10 records
print("\nFIRST 10 RECORDS")
print(
    df[["Timestamp", ghi_col, target_col]]
    .head(10)
    .to_string(index=False)
)
