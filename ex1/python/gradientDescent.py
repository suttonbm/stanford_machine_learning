import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    history = np.zeros(iterations)

    for n in range(iterations):
        val = np.dot(np.transpose(np.dot(X, theta)) - y, X)
        theta = np.add(theta, -1 * alpha * (1./m) * np.transpose(val))

        history[n] = computeCost(X, y, theta)
    # END for
    return (theta, history)
