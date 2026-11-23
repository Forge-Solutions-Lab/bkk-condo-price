import numpy as np


def pad_one(x):
    return np.hstack((x, np.ones((len(x), 1))))


def fit(X, y):
    W, *_ = np.linalg.lstsq(pad_one(X), y, rcond=None)  # least squares, last weight is the bias
    return W


def fit_ridge(X, y, alpha=1.0):
    Xp = pad_one(X)
    reg = np.eye(Xp.shape[1]) * float(alpha)
    reg[-1, -1] = 0.0  # do not penalize the intercept
    return np.linalg.solve(Xp.T @ Xp + reg, Xp.T @ y)


def predict(X, W):
    return pad_one(X) @ W
