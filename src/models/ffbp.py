import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30.0, 30.0)))


def pad_one(x):
    return np.hstack((x, np.ones((len(x), 1))))


def forward(X, W0, W1):
    input0 = pad_one(X)  # n x (d+1)
    input1 = pad_one(sigmoid(input0 @ W0))  # n x (H+1)
    output = input1 @ W1  # n x 1, linear output because the target is a price, not a class
    return input0, input1, output


def ffbp(X, Y, H=8, lr=0.5, epochs=3000, seed=42):
    # d -> H -> 1, trained with feed-forward back-propagation
    rng = np.random.default_rng(seed)
    W0 = rng.normal(scale=0.5, size=(X.shape[1] + 1, H))
    W1 = rng.normal(scale=0.5, size=(H + 1, 1))
    for epoch in range(epochs):
        # Forward
        input0, input1, output = forward(X, W0, W1)
        # Backward
        # output
        e = Y[:, None] - output
        delta = e  # derivative of a linear output is 1
        W1_old = W1.copy()
        W1 = W1 + lr * (input1.T @ delta) / len(X)  # mean over rows so lr does not depend on data size
        # hidden
        delta = input1 * (1 - input1) * (delta @ W1_old.T)
        delta = delta[:, :-1]  # drop the bias column
        W0 = W0 + lr * (input0.T @ delta) / len(X)
    return W0, W1


def ffbp_stable(X, Y, H=12, lr=0.35, epochs=2500, weight_decay=4e-3, grad_clip=1.5, lr_decay=5e-4, seed=42):
    """Regularized FFBP with scaled init, decaying step size, L2 decay, and clipped residuals."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W0 = rng.normal(scale=1.0 / np.sqrt(d + 1), size=(d + 1, H))
    W1 = rng.normal(scale=1.0 / np.sqrt(H + 1), size=(H + 1, 1))
    ones = np.ones((n, 1))
    input0 = np.hstack((X, ones))
    for epoch in range(epochs):
        step = lr / (1.0 + lr_decay * epoch)
        hidden = sigmoid(input0 @ W0)
        input1 = np.hstack((hidden, ones))
        output = input1 @ W1
        delta = np.clip(Y[:, None] - output, -grad_clip, grad_clip)
        W1_old = W1.copy()
        pen1 = weight_decay * W1
        pen1[-1, :] = 0.0
        W1 = W1 + step * ((input1.T @ delta) / n - pen1)
        dh = hidden * (1.0 - hidden) * (delta @ W1_old[:-1].T)
        pen0 = weight_decay * W0
        pen0[-1, :] = 0.0
        W0 = W0 + step * ((input0.T @ dh) / n - pen0)
    return W0, W1


def predict(X, W0, W1):
    return forward(X, W0, W1)[2][:, 0]
