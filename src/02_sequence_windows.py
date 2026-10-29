import numpy as np


def create_sequences(X, y, window_size=24):
    """
    Create sequences for one-hour-ahead GHI forecasting.

    Each input sequence contains the previous `window_size`
    observations. The target is aligned with the final input
    timestamp's one-hour-ahead target value.

    X: Feature matrix, shape (n_samples, n_features)
    y: Target array, where y[t] = GHI at t + 1 hour
    window_size: Number of historical observations

    Returns:
        X_seq: (n_sequences, window_size, n_features)
        y_seq: (n_sequences,)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")

    if window_size >= len(X):
        raise ValueError(
            "window_size must be smaller than the number of rows."
        )

    X_seq = []
    y_seq = []

    for i in range(window_size, len(X) + 1):
        X_seq.append(X[i - window_size:i])
        y_seq.append(y[i - 1])

    return np.asarray(X_seq), np.asarray(y_seq)