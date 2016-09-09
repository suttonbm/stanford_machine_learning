###############################################################################
#
#   Port of Stanford Machine Learning, Assignment #1 from Octave to Python
#
#   X refers to population size in 10,000s
#   y refers to profit in $10,000s
#

## ============ Initialization ===============================================
#   Supporting Python Module
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#   User scripts
from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

## ============ Part 1: Basic Function =======================================
print("Running warmUpExercise...")
print("5x5 Identity Matrix:")

warmUpExercise()

## ============ Part 2: Plotting =============================================
print("Plotting data...")
data = np.genfromtxt('../ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

plotData(X, y)

## ============ Part 3: Gradient Descent ======================================
print("Running Gradient Descent...")

X = np.transpose(np.array([np.ones_like(X), X]))
theta = np.zeros((2,1))

# Set gradient descent parameters
iterations = 1500
alpha = 0.01

# Compute and display initial cost
print computeCost(X, y, theta)

# Run gradient descent
theta, cost_History = gradientDescent(X, y, theta, alpha, iterations)

print("Theta found by gradient descent:")
print theta
print("theta_0 = {0:.2f}; theta_1 = {1:.2f}".format(theta[0,0], theta[1,0]))

plt.figure()
plt.plot(X[:,1], y, 'x', X[:,1], np.dot(X, theta), '-')
plt.show()

# Predict values
predict1 = np.dot([1., 3.5], theta)[0]
print("For population=35,000, predicted profit is: ${0:.2f}".format(predict1 * 10000))
predict2 = np.dot([1., 7.], theta)[0]
print("For population=70,000, predicted profit is: %{0:.2f}".format(predict2 * 10000))

## ============ Part 4: Visualizing J(theta_0, theta_1) =======================
print("Visualizing J(theta)...")

# Grid to calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# Initialize J
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Calculate J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i,j] = computeCost(X, y, t)
    # END for
# END for

# Generate a surface plot


# Fix rotation
J_vals = np.transpose(J_vals)
tX, tY = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(tX, tY, J_vals, cmap=cm.coolwarm)
plt.show()

# Generate a contour plot
fig = plt.figure()
plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
plt.plot(theta[0,0], theta[1,0], 'x')
plt.show()
