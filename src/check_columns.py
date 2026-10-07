import pandas as pd
import os

print("--- STARTING SCRIPT ---")

# Replace 'your_file_name.csv' with the EXACT name of your file in the data folder
file_path = 'data/nsrdb_mock_dataset.csv'


if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("SUCCESS: File loaded!")
    print("--- ALL COLUMN NAMES IN YOUR CSV ---")
    print(df.columns.tolist())
else:
    print(f"ERROR: Could not find the file at {file_path}. Check the file name!")