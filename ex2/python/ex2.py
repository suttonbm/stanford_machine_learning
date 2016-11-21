## Machine Learning Online Class - Exercise 2: Logistic Regression
#
# Python port of octave/matlab code

#> ------------ Initialization ------------------------------------------------

from ex2_functions import *
import numpy as np
import matplotlib.pyplot as plt

# Import the data
data = np.matrix(np.genfromtxt('../ex2data1.txt', delimiter=','))
X = data[:, 0:2]
y = data[:, 2]

#>------------- Part 1: Plotting ----------------------------------------------

print("""
Plotting data with + indicating 'positive' examples and o indicating 'negative'
examples.""")

plotData(X, y)

#>------------- Part 2: Compute Cost and Gradient -----------------------------

m, n = X.shape

# Add intercept term to X
Xnew = np.matrix((m, n+1))
Xnew[:, 1:] = X
Xnew[:, 0] = np.ones(m)
X = Xnew

# Initialize theta
initial_theta = np.matrix(np.zeros((n+1, 1)))

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

#>------------- Part 3: Optimize Function -------------------------------------
