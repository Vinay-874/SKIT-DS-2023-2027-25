import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Define Paths Relative to Project Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "nsrdb_cleaned_base.csv"
)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# 2. Load the Cleaned Dataset
print("=== Sprint 1: User Story 3 - Analyze Feature Correlations ===")
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Cleaned dataset not found at '{INPUT_PATH}'. Make sure you've run the cleaning script first!"
    )

print(f"[+] Loading processed dataset from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# Helper function to detect target/GHI column name flexibly
def get_ghi_column(df_cols):
    possible_names = ["GHI (W/m²)", "GHI", "Target_GHI_NextHour (W/m²)", "Target_GHI"]
    for name in possible_names:
        if name in df_cols:
            return name
    return None

target_ghi_col = get_ghi_column(df.columns)

# Select numerical meteorological columns for correlation analysis
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
numeric_df = df[numeric_cols]

# 3. Compute Pearson and Spearman Correlation Matrices
pearson_corr = numeric_df.corr(method="pearson")
spearman_corr = numeric_df.corr(method="spearman")

# 4. Generate & Save Pearson Correlation Heatmap
sns.set_theme(style="white")
plt.figure(figsize=(10, 8))
sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation Coefficient"},
)
plt.title("Pearson Correlation Heatmap (Feature Dependencies)", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()

heatmap_path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"[+] Saved Pearson correlation heatmap to: {heatmap_path}")

# 5. Display Strongest Predictors for GHI / Target Variable
if target_ghi_col and target_ghi_col in pearson_corr.columns:
    ghi_corr = pearson_corr[target_ghi_col].drop(target_ghi_col).sort_values(ascending=False)
    print(f"\n[+] Feature Correlation relative to '{target_ghi_col}':")
    for feature, val in ghi_corr.items():
        print(f"    - {feature:<30}: {val:.4f}")
else:
    print("\n[!] Primary GHI column not found in correlation matrix.")

# 6. Check for Multicollinearity between Predictors
print("\n[!] Checking for Multicollinearity (|r| >= 0.85 between features):")
multicollinear_pairs = []
features = [c for c in numeric_df.columns if c != target_ghi_col]

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        f1, f2 = features[i], features[j]
        r = pearson_corr.loc[f1, f2]
        if abs(r) >= 0.85:
            multicollinear_pairs.append((f1, f2, r))

if multicollinear_pairs:
    for f1, f2, r in multicollinear_pairs:
        print(f"    - High correlation: {f1} <--> {f2} (r = {r:.4f})")
else:
    print("    - No severe multicollinearity detected between independent predictors.")

print("\n[SUCCESS] User Story 3 execution completed!")