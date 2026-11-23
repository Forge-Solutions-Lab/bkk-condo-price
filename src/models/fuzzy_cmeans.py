import numpy as np


def fuzzy_cmeans(X, K, m=2, max_iter=300, tol=1e-6, seed=42):
    rng = np.random.default_rng(seed)
    C = np.zeros((K, X.shape[1]))
    mu = rng.random((len(X), K))  # membership of each point in each cluster
    for _ in range(max_iter):
        mu /= np.sum(mu, axis=1)[:, None]  # memberships of a point add up to 1
        Cold = C.copy()
        for k in range(K):
            C[k] = np.dot(mu[:, k] ** m, X) / np.sum(mu[:, k] ** m)
            D = np.sqrt(np.sum((X - C[k]) ** 2, axis=1))  # Euclidean distance
            D = np.maximum(D, 1e-12) ** (-2 / (m - 1))  # a point on a centre would divide by zero
            mu[:, k] = D
        if np.mean(np.abs(Cold - C)) < tol:
            break
    mu /= np.sum(mu, axis=1)[:, None]
    return C, mu  # centres, membership n x K
