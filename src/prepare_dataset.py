import pandas as pd
import numpy as np
import os

def prepare_ml_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Select features and target columns based on your CSV
    feature_columns = [
        'GHI (W/m²)', 'DNI (W/m²)', 'DHI (W/m²)', 
        'Temp (°C)', 'RH (%)', 'Pressure (hPa)', 
        'Wind Speed (m/s)', 'Solar Zenith Angle (°)'
    ]
    target_column = 'Target_GHI_NextHour (W/m²)'
    
    # Drop rows with missing values in these columns
    all_needed = feature_columns + [target_column]
    df = df.dropna(subset=all_needed)
    
    # Extract matrices X and y
    X = df[feature_columns].values
    y = df[target_column].values
    
    print("Dataset successfully processed and formatted!")
    print(f"Feature Matrix Shape (X): {X.shape}")
    print(f"Target Array Shape (y): {y.shape}")
    
    # Save processed arrays for baseline ML modeling
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train_base.npy', X)
    np.save('data/processed/y_train_base.npy', y)
    print("Saved processed matrices to data/processed/ successfully!")

if __name__ == "__main__":
    # Updated with your exact file name
    data_file = "data/nsrdb_mock_dataset.csv"
    
    if os.path.exists(data_file):
        prepare_ml_dataset(data_file)
    else:
        print(f"Error: Could not find '{data_file}'. Make sure the file is inside the data/ folder!")