###############################################################################
#
#   Port of Stanford Machine Learning, Assignment #1 from Octave to Python
#
#   X refers to population size in 10,000s
#   y refers to profit in $10,000s
#

## ============ Initialization ===============================================
#   Supporting Python Modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#   User scripts
from computeCostMulti import computeCost
from gradientDescentMulti import gradientDescent
from featureNormalize import normalize
from normalEqn import normalEqn

## ============ Part 1: Feature Normalization =================================
print("Loading data...")

data = np.genfromtxt('../ex1data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

print("First 10 examples from the dataset:")
for k in range(10):
    print("  x = [{0:.1f}, {1:.1f}], y = {2:.1f}".format(X[k,0], X[k,1], y[k]))
# END for

print("Normalizing features...")

(X, mu, sigma) = normalize(X)

X = np.transpose(np.array([np.ones(m), X[:,0], X[:,1]]))

## ============ Part 2: Gradient Descent ======================================

print("Running gradient descent...")
nIters = 400

plt.figure()
for alpha in np.logspace(-2, 0, 5):
    theta = np.zeros((3,1))
    (theta, history) = gradientDescent(X, y, theta, alpha, nIters)

    plt.plot(history)
# END for
plt.show()

print("Theta from gradient descent: ")
print(" {0:.2f}, {1:.2f}, {2:.2f}".format(theta[0,0], theta[1,0], theta[2,0]))

# Estimate the price of a 1650 sq-ft, 3 br house
house = np.array([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]])
price = np.dot(house, theta)[0]

print("The predicted price of a 1650 sq-ft, 3 br house")
print("(using gradient descent): ${0:.2f}".format(price))

## ============ Part 3: Normal Equation =======================================

X = np.transpose(np.array([np.ones(m), data[:,0], data[:,1]]))

theta = normalEqn(X, y)

print("Theta from normal equation: ")
print(" {0:.2f}, {1:.2f}, {2:.2f}".format(theta[0], theta[1], theta[2]))

# Estimate the price of a 1650 sq-ft, 3 br house
house = np.array([1, 1650., 3.])
price = np.dot(house, theta)

print("The predicted price of a 1650 sq-ft, 3 br house")
print("(using normal equation): ${0:.2f}".format(price))
