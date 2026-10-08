import numpy as np


def create_sequences(X, y, window_size=24):
    """
    Create sliding time-series windows.

    Parameters
    ----------
    X : array-like
        Input features ordered by time.
    y : array-like
        Target values ordered by time.
    window_size : int
        Number of previous time steps used for prediction.

    Returns
    -------
    X_seq : numpy.ndarray
        Sequence input data.
    y_seq : numpy.ndarray
        Corresponding target values.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")

    if window_size >= len(X):
        raise ValueError("window_size must be smaller than the number of rows.")

    X_seq = []
    y_seq = []

    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size:i])
        y_seq.append(y[i])

    return np.asarray(X_seq), np.asarray(y_seq)


if __name__ == "__main__":

    # Small example for testing
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7]
    ])

    y = np.array([
        10,
        20,
        30,
        40,
        50,
        60,
        70
    ])

    X_seq, y_seq = create_sequences(X, y, window_size=3)

    print("Input sequence shape:", X_seq.shape)
    print("Target shape:", y_seq.shape)

    print("\nGenerated sequences:")
    for i in range(len(X_seq)):
        print(X_seq[i].flatten(), "->", y_seq[i])