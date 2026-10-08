import pandas as pd


def check_time_series_leakage(train_df, test_df, time_col):
  """Verifies that the training set timestamps strictly precede

  the test set timestamps to prevent time-series data leakage.
  """
  max_train_time = train_df[time_col].max()
  min_test_time = test_df[time_col].min()

  print(f"Latest timestamp in training data: {max_train_time}")
  print(f"Earliest timestamp in test data: {min_test_time}")

  if max_train_time < min_test_time:
    print("✅ Pass: No temporal overlap detected between train and test sets.")
    return True
  else:
    print("❌ Warning: Temporal overlap or leakage detected!")
    return False