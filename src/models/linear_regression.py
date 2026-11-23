import numpy as np


def pad_one(x):
    return np.hstack((x, np.ones((len(x), 1))))


def fit(X, y):
    W, *_ = np.linalg.lstsq(pad_one(X), y, rcond=None)  # least squares, last weight is the bias
    return W


def predict(X, W):
    return pad_one(X) @ W
