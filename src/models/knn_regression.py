import numpy as np


def distance(X, z, metric='euclidean'):
    if metric == 'manhattan':
        return np.sum(np.abs(X - z), axis=1)
    return np.sqrt(np.sum((X - z) ** 2, axis=1))  # Euclidean distance


def neighbors(X, z, k=5, metric='euclidean'):
    d = distance(X, z, metric)
    idx = np.argsort(d, kind='stable')[:k]
    return idx, d[idx]


def knn(X, y, z, k=5, metric='euclidean', weight='uniform'):
    idx, d = neighbors(X, z, k, metric)
    if weight == 'distance':
        w = 1 / np.maximum(d, 1e-12)  # closer neighbour, bigger vote
        return np.sum(w * y[idx]) / np.sum(w)
    return np.mean(y[idx])  # every neighbour votes equally


def knn_predict(X, y, Z, k=5, metric='euclidean', weight='uniform'):
    return np.array([knn(X, y, z, k, metric, weight) for z in Z])
