import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "processed", "nsrdb_cleaned_base.csv")

if os.path.exists(file_path):
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Standardize timestamp and sort chronologically
    time_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    
    target_col = "Clearsky_DHI" if "Clearsky_DHI" in df.columns else df.columns[-1]
    
    # Select numeric feature columns only
    feature_cols = [
        c for c in df.columns 
        if c not in [time_col, target_col] and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"Target variable: {target_col}")
    print(f"Selected numeric features ({len(feature_cols)}): {feature_cols}")

    # Chronological 80/20 train-test split to prevent leakage
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col].fillna(0)

    print(f"\nTraining set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    # Train Random Forest Regressor
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Evaluation
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Random Forest Model Performance ---")
    print(f"MAE  (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R²   (Coefficient of Determination): {r2:.4f}")

    # Feature Importance Analysis
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    print("\n--- Top 5 Most Important Features ---")
    print(importances.nlargest(5))
else:
    print(f"❌ Dataset not found at {file_path}")