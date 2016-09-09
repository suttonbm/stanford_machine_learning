import numpy as np
from numpy.linalg import inv

def normalEqn(X, y):
    t1 = inv(np.dot(np.transpose(X), X))
    t2 = np.dot(t1, np.transpose(X))
    theta = np.dot(t2, y)
    return theta
