import numpy as np

def computeCost(X, y, theta):
    err = np.add(np.transpose(np.dot(X, theta)), -1*y)
    J = np.dot(err, np.transpose(err))[0]/(2*len(y))
    return J
