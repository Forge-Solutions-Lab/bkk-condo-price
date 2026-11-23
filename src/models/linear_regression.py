import numpy as np


def pad_one(x):
    return np.hstack((x, np.ones((len(x), 1))))


def fit(X, y, l2=0.0):
    A = pad_one(X)
    if l2 > 0:
        # ridge penalty on feature weights only; the last weight (bias) is unpenalised
        reg = np.eye(A.shape[1]) * l2
        reg[-1, -1] = 0.0
        return np.linalg.solve(A.T @ A + reg, A.T @ y)
    W, *_ = np.linalg.lstsq(A, y, rcond=None)  # least squares, last weight is the bias
    return W


def predict(X, W):
    return pad_one(X) @ W
