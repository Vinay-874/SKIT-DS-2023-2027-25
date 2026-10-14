
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
file_path = ROOT / "data" / "raw" / "nsrdb_raw.csv"

df = pd.read_csv(file_path)

features = [
    "GHI (W/m²)",
    "DNI (W/m²)",
    "DHI (W/m²)",
    "Temp (°C)",
    "RH (%)",
    "Pressure (hPa)",
    "Wind Speed (m/s)",
    "Solar Zenith Angle (°)"
]

target = "Target_GHI_NextHour (W/m²)"

print("\n--- First 5 rows ---")
print(df[features + [target]].head().to_string(index=False))

print("\n--- Target statistics ---")
print(df[target].describe())

print("\n--- Feature correlations with target ---")
print(
    df[features + [target]]
    .corr(numeric_only=True)[target]
    .sort_values(ascending=False)
)

print("\n--- Timestamp sample ---")
if "Timestamp" in df.columns:
    print(df["Timestamp"].head(10).to_string(index=False))
