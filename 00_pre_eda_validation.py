import os
import pandas as pd

# Define paths
RAW_FILE_PATH = "data/raw/nsrdb_raw.csv"  # Ensure your downloaded file is named nsrdb_raw.csv
PROCESSED_FILE_PATH = "data/processed/nsrdb_cleaned_base.csv"

# Step 1: Load Raw Dataset
print("=== Step 1: Loading Dataset ===")
df = pd.read_csv(RAW_FILE_PATH)
print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Check Data Types & Schema
print("\n=== Step 2: Schema & Data Types ===")
print(df.dtypes)

# Step 3: Missing Values & Duplicate Inspection
print("\n=== Step 3: Missing Values & Duplicates ===")
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate rows count:", df.duplicated().sum())

# Step 4: Timestamp Processing & Continuity Check
print("\n=== Step 4: Timestamp Validation ===")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

time_diffs = df["Timestamp"].diff().value_counts()
print("Detected Sampling Intervals:\n", time_diffs)

# Step 5: Physical Bounds Audit & Cleaning
print("\n=== Step 5: Physical Bounds Audit ===")
negative_dhi = (df["DHI (W/m²)"] < 0).sum()
print(f"Negative DHI records detected: {negative_dhi}")

# Clip invalid negative solar irradiance values to zero
df["DHI (W/m²)"] = df["DHI (W/m²)"].clip(lower=0)
df["GHI (W/m²)"] = df["GHI (W/m²)"].clip(lower=0)
df["DNI (W/m²)"] = df["DNI (W/m²)"].clip(lower=0)

# Save Staged Preprocessed Dataset
os.makedirs("data/processed", exist_ok=True)
df.to_csv(PROCESSED_FILE_PATH, index=False)
print(f"\nSanitized base dataset saved to: {PROCESSED_FILE_PATH}")