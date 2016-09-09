import numpy as np

def computeCost(X, y, theta):
    m = len(y)

    J = np.sum((np.transpose(np.dot(X, theta)) - y)**2) / (2*m)
    return J
