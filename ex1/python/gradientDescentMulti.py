import numpy as np
from computeCostMulti import computeCost

def gradientDescent(X, y, theta, alpha, nIters):
    m = len(y)
    history = np.zeros(nIters)

    for k in range(nIters):
        t1 = np.add(np.transpose(np.dot(X, theta)), -1*y)
        t2 = np.dot(t1, X)
        theta = np.add(theta, -1 * alpha * np.transpose(t2) / m)

        history[k] = computeCost(X, y, theta)
    # END for

    return (theta, history)
