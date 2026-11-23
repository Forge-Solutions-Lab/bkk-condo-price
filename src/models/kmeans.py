import numpy as np


def distance(X, c, metric='euclidean'):
    if metric == 'manhattan':
        return np.sum(np.abs(X - c), axis=1)
    return np.sum((X - c) ** 2, axis=1)  # squared Euclidean distance


def assign(X, C, metric='euclidean'):
    D = np.zeros((len(X), len(C)))  # n x K
    for k in range(len(C)):
        D[:, k] = distance(X, C[k], metric)
    return np.argmin(D, axis=1), np.min(D, axis=1)


def init_plus_plus(X, K, metric, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(1, K):
        _, d = assign(X, np.array(C), metric)
        p = d / np.sum(d) if np.sum(d) > 0 else None  # far points are more likely to be picked
        C.append(X[rng.choice(len(X), p=p)])
    return np.array(C, dtype=float)


def kmeans(X, K, metric='euclidean', n_init=10, max_iter=300, seed=42):
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        C = init_plus_plus(X, K, metric, rng)  # k-means++ start instead of a plain random pick
        for _ in range(max_iter):
            idx, d = assign(X, C, metric)
            Cold = C.copy()
            for k in range(K):
                if np.any(idx == k):
                    C[k] = np.mean(X[idx == k], axis=0)
            if np.sum(np.abs(Cold - C)) == 0:
                break
        idx, d = assign(X, C, metric)
        if np.sum(d) < best[2]:
            best = (C, idx, np.sum(d))
    return best  # centers, cluster of each point, total distance (inertia)
