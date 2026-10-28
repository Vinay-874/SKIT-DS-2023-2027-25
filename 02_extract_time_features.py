import os
import numpy as np
import pandas as pd

# 1. Define Paths Relative to Project Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "nsrdb_cleaned_base.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "nsrdb_time_features.csv")

print("=== Sprint 1: Member 2 - User Story 4: Time Feature Extraction ===")

# 2. Load Processed Base Dataset
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"[!] Input file not found at: {INPUT_PATH}")

print(f"[+] Loading base cleaned dataset from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# Ensure Timestamp is datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# 3. Extract Discrete Temporal Components
print("[+] Extracting discrete time features (Hour, Day, Month, DayOfWeek, Is_Weekend)...")
df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day
df["Month"] = df["Timestamp"].dt.month
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["Is_Weekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)

# 4. Extract Cyclical Sine/Cosine Transformations
print("[+] Computing cyclical sine and cosine encodings for Hour and Month...")
# Hour encoding (24-hour cycle)
df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)

# Month encoding (12-month cycle)
df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

# 5. Save Enriched Dataset
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"[+] Successfully extracted time features!")
print(f"[+] Output shape: {df.shape} (Rows, Columns)")
print(f"[+] Enriched dataset saved to: {OUTPUT_PATH}")
print("[SUCCESS] Time Feature Extraction completed successfully!")