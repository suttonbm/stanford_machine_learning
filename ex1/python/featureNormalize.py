import numpy as np

def normalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)

    X_norm = np.zeros_like(X)
    for n in range(np.shape(X)[0]):
        X_norm[n,:] = np.divide(np.add(X[n,:], -1*mu), sigma)
    # END for

    return (X_norm, mu, sigma)
