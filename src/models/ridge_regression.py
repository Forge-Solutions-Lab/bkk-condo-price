import numpy as np


def pad_one(x):
    return np.hstack((x, np.ones((len(x), 1))))


def fit(X, y, lam=1.0):
    Xp = pad_one(X)
    if lam == 0:
        W, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        return W
    I = np.eye(Xp.shape[1])
    I[-1, -1] = 0.0  # do not penalise the bias term
    W = np.linalg.solve(Xp.T @ Xp + lam * I, Xp.T @ y)
    return W


def predict(X, W):
    return pad_one(X) @ W
