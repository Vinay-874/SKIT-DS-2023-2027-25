import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_model_review():
    print("=" * 60)
    print("--- MODEL REVIEW & ERROR AUDIT ---")
    print("=" * 60)
    
    # 1. Load cleaned dataset
    data_path = 'data/processed/nsrdb_cleaned_base.csv'
    df = pd.read_csv(data_path)
    print(f"Dataset successfully loaded from {data_path}. Shape: {df.shape}")
    
    # 2. Define features and target
    features = [
        'Latitude', 'Longitude', 'GHI', 'DNI', 'DHI', 
        'Temperature', 'Relative Humidity', 'Pressure', 
        'Dew Point', 'Wind Speed', 'Solar Zenith Angle'
    ]
    target = 'Target_GHI_NextHour'
    
    X = df[features]
    y = df[target]
    
    # 3. Chronological Time-Series Split (No shuffle to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"Training set records: {len(X_train)} | Testing set records: {len(X_test)}")
    
    # 4. Train Random Forest Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 5. Predictions & Core Performance Metrics
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- 1. OVERALL PERFORMANCE METRICS ---")
    print(f"MAE  (Mean Absolute Error): {mae:.4f} W/m²")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f} W/m²")
    print(f"R²   (Coefficient of Determination): {r2:.4f}")
    
    # 6. Advanced Residual & Error Analysis
    residuals = y_test.values - y_pred
    print("\n--- 2. RESIDUAL & BIAS AUDIT ---")
    print(f"Mean Residual (Model Bias): {np.mean(residuals):.4f} W/m²")
    print(f"Standard Deviation of Residuals: {np.std(residuals):.4f} W/m²")
    print(f"Max Over-estimation Error: {np.min(residuals):.4f} W/m²")
    print(f"Max Under-estimation Error: {np.max(residuals):.4f} W/m²")
    
    # 7. Error Breakdown by Solar Intensity Tiers
    eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residual': residuals})
    eval_df['GHI_Tier'], bins = pd.qcut(eval_df['Actual'], q=4, retbins=True, duplicates='drop')
    
    tier_errors = eval_df.groupby('GHI_Tier', observed=False).apply(
        lambda g: pd.Series({
            'MAE': mean_absolute_error(g['Actual'], g['Predicted']),
            'RMSE': np.sqrt(mean_squared_error(g['Actual'], g['Predicted']))
        })
    )
    
    print("\n--- 3. ERROR BREAKDOWN BY SOLAR INTENSITY ---")
    print(tier_errors)
    
    # 8. Feature Importances Review
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- 4. TOP 5 FEATURE IMPORTANCES ---")
    print(importances.head(5))
    
    print("\n" + "=" * 60)
    print("[AUDIT COMPLETE]: Model review successfully verified.")
    print("=" * 60)

if __name__ == "__main__":
    run_model_review()