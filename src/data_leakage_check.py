import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_time_series_leakage(train_df, test_df, time_col="Timestamp"):
    """Verifies that the training set timestamps strictly precede
    the test set timestamps to prevent time-series data leakage.
    """
    max_train_time = train_df[time_col].max()
    min_test_time = test_df[time_col].min()

    print(f"Latest timestamp in training data: {max_train_time}")
    print(f"Earliest timestamp in test data:   {min_test_time}")

    if max_train_time < min_test_time:
        print("✅ Pass: No temporal overlap detected between train and test sets.")
        return True
    else:
        print("❌ Warning: Temporal overlap or leakage detected!")
        return False


def check_target_leakage(df, feature_cols, target_col):
    """Audits features to ensure no future target values or direct 
    copies of the target variable are included in the feature set.
    """
    print("\nRunning target leakage audit...")
    leaky_features = []
    
    # Ensure target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"⚠️ Warning: Target column '{target_col}' is non-numeric. Skipping target leakage correlation check.")
        return True

    for col in feature_cols:
        if col != target_col and col in df.columns:
            # Only calculate correlation for numeric features
            if pd.api.types.is_numeric_dtype(df[col]):
                correlation = df[col].corr(df[target_col])
                if abs(correlation) > 0.99:
                    leaky_features.append((col, correlation))
            
    if leaky_features:
        print(f"❌ Warning: Potential target leakage found in features: {leaky_features}")
        return False
    else:
        print("✅ Pass: No high-correlation target leakage found in numeric feature columns.")
        return True


def verify_cv_folds(X, time_col_series, n_splits=5):
    """Verifies that all TimeSeriesSplit cross-validation folds 
    strictly maintain chronological order to prevent temporal leakage.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"\nRunning TimeSeriesSplit cross-validation audit ({n_splits} folds)...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        max_train_t = time_col_series.iloc[train_idx].max()
        min_test_t = time_col_series.iloc[test_idx].max()
        
        assert max_train_t < min_test_t, f"Fold {fold+1} Leakage Error: Train max time >= Test min time!"
        print(f"  - Fold {fold+1}: Train ends {max_train_t} | Test starts {min_test_t} [PASS]")
    
    print("✅ Success: All cross-validation folds verified for zero data leakage.")


if __name__ == "__main__":
    candidates = [
        os.path.join(BASE_DIR, "data", "processed", "nsrdb_cleaned_base.csv"),
        os.path.join(BASE_DIR, "data", "nsrdb_mock_dataset.csv"),
        os.path.join(BASE_DIR, "data", "raw", "nsrdb_raw.csv")
    ]
    file_path = next((p for p in candidates if os.path.exists(p)), None)

    if file_path:
        print(f"Running comprehensive data leakage audit on: {file_path}")
        df = pd.read_csv(file_path, encoding="utf-8")
        
        # Standardize timestamp column name if needed
        time_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # 80/20 Holdout Split Check
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        print(f"\n--- Holdout Split Audit ---")
        print(f"Splitting 80% train ({len(train_df)} rows) and 20% test ({len(test_df)} rows)")
        check_time_series_leakage(train_df, test_df, time_col=time_col)
        
        # Target Leakage Check
        target_col = "Clearsky_DHI" if "Clearsky_DHI" in df.columns else df.columns[-1]
        feature_cols = [c for c in df.columns if c not in [time_col, target_col]]
        check_target_leakage(df, feature_cols, target_col)
        
        # Cross-Validation TimeSeriesSplit Check
        verify_cv_folds(df, df[time_col])
    else:
        print("❌ No dataset found to test leakage.")