"""Obtaining features for timeseries prediction

    Returns
    -------
    np array
        returns features
    """
import numpy as np


def create_features(data, lag=1):
    """This function generates features to predict PM10 level in the next hour

    Parameters
    ----------
    data : float
        PM10 value per hour
    lag : int, optional
        number of measurements to predict the next one, by default 1

    Returns
    -------
    numpy array
        returns a numpy array with the features
    """
    X, y = [], []
    # Adjust the range to stop before the last `lag` elements
    for i in range(len(data) - lag):
        # Ensure indexing works by converting to a list or directly accessing values
        X.append(data.iloc[i:(i + lag)].values.tolist())  # Using .iloc for DataFrame or Series
        y.append(data.iloc[i + lag])
    return np.array(X), np.array(y)