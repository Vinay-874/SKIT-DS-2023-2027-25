import os
import pandas as pd

# 1. Define Paths Relative to Project Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "nsrdb_time_features.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "nsrdb_engineered_features.csv")

print("=== Sprint 2: Member 2 - User Story: Lag & Rolling Features ===")

# 2. Load Time-Enriched Dataset
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"[!] Input file not found at: {INPUT_PATH}. Run 02_extract_time_features.py first!")

print(f"[+] Loading time-enriched dataset from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# Ensure chronological ordering prior to creating lag/rolling windows
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# 3. Create Lagged Features (t-1, t-2, t-3)
print("[+] Generating lagged meteorological features (t-1, t-2, t-3 hours)...")
target_vars = ["GHI", "Temperature", "Relative Humidity"]
existing_vars = [var for var in target_vars if var in df.columns]

# Assuming 30-min interval: 2 steps = 1 hour lag
lag_steps = [2, 4, 6]  # 1h, 2h, 3h lags

for var in existing_vars:
    for step in lag_steps:
        hours = step // 2
        df[f"{var}_lag_{hours}h"] = df[var].shift(step)

# 4. Create Rolling Window Statistics (Moving Averages & Std Dev)
print("[+] Generating rolling window statistics (3-hour & 6-hour moving averages)...")
for var in existing_vars:
    # 3-hour window (6 steps of 30 mins)
    df[f"{var}_rolling_mean_3h"] = df[var].shift(1).rolling(window=6).mean()
    df[f"{var}_rolling_std_3h"] = df[var].shift(1).rolling(window=6).std()
    
    # 6-hour window (12 steps of 30 mins)
    df[f"{var}_rolling_mean_6h"] = df[var].shift(1).rolling(window=12).mean()

# 5. Clean Initial NaN Rows Caused by Lags
initial_shape = df.shape
df_cleaned = df.dropna().reset_index(drop=True)

# 6. Save Final Engineered Feature Matrix
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_cleaned.to_csv(OUTPUT_PATH, index=False)

print(f"[+] Successfully generated lag and rolling features!")
print(f"[+] Initial Shape : {initial_shape}")
print(f"[+] Cleaned Shape : {df_cleaned.shape} (Rows, Columns)")
print(f"[+] Engineered dataset saved to: {OUTPUT_PATH}")
print("[SUCCESS] Lag & Rolling Features completed successfully!")