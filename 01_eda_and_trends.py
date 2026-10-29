import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Define paths relative to your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "nsrdb_cleaned_base.csv"
)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# 2. Load the preprocessed dataset
print("Loading cleaned dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour

# 3. Create Multi-Panel Visualization
sns.set_theme(style="whitegrid")  
plt.figure(figsize=(14, 10))

# Subplot 1: Hourly Solar Irradiance (Diurnal Pattern)
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x="Hour", y="GHI (W/m²)", palette="YlOrRd")
plt.title("Diurnal GHI Distribution by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("GHI (W/m²)")

# Subplot 2: Temperature vs. Relative Humidity by Cloud Type
plt.subplot(2, 2, 2)
sns.scatterplot(
    data=df,
    x="Temp (°C)",
    y="RH (%)",
    hue="Cloud Type",
    palette="tab10",
    alpha=0.7,
)
plt.title("Temperature vs Relative Humidity by Cloud Type")

# Subplot 3: Solar Zenith Angle Distribution
plt.subplot(2, 2, 3)
sns.histplot(df["Solar Zenith Angle (°)"], kde=True, color="teal")
plt.title("Solar Zenith Angle Distribution")

# Subplot 4: Average Solar Output by Cloud Cover
plt.subplot(2, 2, 4)
sns.barplot(
    data=df, x="Cloud Type", y="GHI (W/m²)", estimator="mean", palette="Blues_d"
)
plt.title("Average GHI Across Cloud Coverage Types")

plt.tight_layout()

# 4. Save figure for project report
output_chart_path = os.path.join(REPORTS_DIR, "eda_plots.png")
plt.savefig(output_chart_path, dpi=300)
print("EDA chart successfully saved to:", output_chart_path)
plt.show()