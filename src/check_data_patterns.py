
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "raw" / "nsrdb_raw.csv")

columns = [
    "GHI (W/m²)",
    "DNI (W/m²)",
    "DHI (W/m²)",
    "Temp (°C)",
    "RH (%)",
    "Pressure (hPa)",
    "Wind Speed (m/s)",
    "Solar Zenith Angle (°)",
    "Target_GHI_NextHour (W/m²)"
]

print("--- Unique values ---")
for col in columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n--- Irradiance ranges ---")
print(df[["GHI (W/m²)", "DNI (W/m²)", "DHI (W/m²)"]].describe())

print("\n--- First 20 GHI and target values ---")
print(
    df[["GHI (W/m²)", "Target_GHI_NextHour (W/m²)"]]
    .head(20)
    .to_string(index=False)
)
