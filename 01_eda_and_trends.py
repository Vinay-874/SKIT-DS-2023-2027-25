import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "nsrdb_cleaned_base.csv"
)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# 2. Load preprocessed dataset
print("Loading cleaned dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Convert Timestamp and extract Hour
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour

# Helper function to detect matching column names flexibly
def get_col(possible_names, df_cols):
    for name in possible_names:
        if name in df_cols:
            return name
    return None

ghi_col = get_col(["GHI (W/m²)", "GHI", "Target_GHI_NextHour (W/m²)", "Target_GHI"], df.columns)
temp_col = get_col(["Temp (°C)", "Temperature", "Temp"], df.columns)
rh_col = get_col(["RH (%)", "Relative Humidity", "RH"], df.columns)
cloud_col = get_col(["Cloud Type", "Cloud_Type", "Cloud Coverage"], df.columns)
zenith_col = get_col(["Solar Zenith Angle (°)", "Solar Zenith Angle", "Zenith"], df.columns)

print(f"\n[+] Detected Active Columns:")
print(f"    - GHI Column   : {ghi_col}")
print(f"    - Temp Column  : {temp_col}")
print(f"    - RH Column    : {rh_col}")
print(f"    - Cloud Column : {cloud_col}")

# 3. Create Multi-Panel Visualization
sns.set_theme(style="whitegrid")  
plt.figure(figsize=(14, 10))

# Subplot 1: Hourly Solar Irradiance (Diurnal Pattern)
if ghi_col and "Hour" in df.columns:
    plt.subplot(2, 2, 1)
    sns.boxplot(
        data=df,
        x="Hour",
        y=ghi_col,
        hue="Hour",
        palette="YlOrRd",
        legend=False
    )
    plt.title(f"Diurnal {ghi_col} Distribution by Hour", fontsize=11, fontweight="bold")
    plt.xlabel("Hour of Day")
    plt.ylabel(ghi_col)

# Subplot 2: Temperature vs. Relative Humidity by Cloud Type
if temp_col and rh_col:
    plt.subplot(2, 2, 2)
    sns.scatterplot(
        data=df,
        x=temp_col,
        y=rh_col,
        hue=cloud_col if cloud_col else None,
        palette="tab10" if cloud_col else None,
        alpha=0.7,
    )
    plt.title("Temperature vs Relative Humidity", fontsize=11, fontweight="bold")

# Subplot 3: Solar Zenith Angle Distribution
if zenith_col:
    plt.subplot(2, 2, 3)
    sns.histplot(df[zenith_col], kde=True, color="teal")
    plt.title(f"{zenith_col} Distribution", fontsize=11, fontweight="bold")

# Subplot 4: Average Solar Output by Cloud Cover
if cloud_col and ghi_col:
    df[cloud_col] = df[cloud_col].astype(str)
    plt.subplot(2, 2, 4)
    sns.barplot(
        data=df,
        x=cloud_col,
        y=ghi_col,
        hue=cloud_col,
        estimator="mean",
        palette="Blues_d",
        legend=False
    )
    plt.title(f"Average {ghi_col} Across Cloud Coverage Types", fontsize=11, fontweight="bold")

plt.tight_layout()

# 4. Save figure
output_chart_path = os.path.join(REPORTS_DIR, "eda_plots.png")
plt.savefig(output_chart_path, dpi=300)
print(f"\nEDA chart successfully saved to: {output_chart_path}")
plt.show()