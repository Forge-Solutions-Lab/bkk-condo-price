import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def predict(X, W0, W1):
    return forward(X, W0, W1)[2][:, 0]
