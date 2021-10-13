import numpy as np
import matplotlib.pyplot as plt
import pdb
import matplotlib
from mpl_toolkits.mplot3d import axes3d

# Setting up the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Variables
theta = 0  # BECVF Theta
theta1 = 0  # OCVF Theta
i = 0  # While loop iteration number
n = 100  # number of times 2 pi is split up for BECVF. Theta
n_1 = 500  # number of times 2 pi is split up for OCVF. Theta1

# Array Variables
X = np.linspace(0, 2 * np.pi, n_1)  # BECVF Phi Variable
X1 = np.linspace(-np.pi, np.pi, n_1)  # OCVF Phi Variable
Zeros = np.linspace(0, 0, n_1)  # Array of Zeros matching the size of Phi Variables

# Rotation of the BECVF and OCVF about their angular momentum vectors
while i < n:
    # Increment's of rotation for BECVF and OCVF
    theta = (2 * i * np.pi) / n
    theta1 = (i * np.pi) / n

    # ---------------------------------------- BECVF -----------------------------------------------
    # BECVF Great Circle Basis Elements *** Page 70 ***
    # ax.plot(Zeros, np.cos(X), -np.sin(X), color='red')
    # ax.plot(np.cos(X), Zeros, -np.sin(X), color='red')

    # Elements of the Rotated BECVF *** Page 71 *** BECVF Matrices
    X1_BECVF = np.cos(X) * (-0.5 + 0.5 * np.cos(theta)) + np.sin(X) * (np.sin(theta) / np.sqrt(2))
    Y1_BECVF = np.cos(X) * (0.5 + 0.5 * np.cos(theta)) + np.sin(X) * (np.sin(theta) / np.sqrt(2))
    Z1_BECVF = np.cos(X) * (np.sin(theta) / np.sqrt(2)) - np.sin(X) * np.cos(theta)

    X2_BECVF = np.cos(X) * (0.5 + 0.5 * np.cos(theta)) + np.sin(X) * (np.sin(theta) / np.sqrt(2))
    Y2_BECVF = np.cos(X) * (-0.5 + 0.5 * np.cos(theta)) + np.sin(X) * (np.sin(theta) / np.sqrt(2))
    Z2_BECVF = np.cos(X) * (np.sin(theta) / np.sqrt(2)) - np.sin(X) * np.cos(theta)

    # Plot of the Elements of the Rotated BECVF *** Page 72 ***
    ax.plot(X1_BECVF, Y1_BECVF, Z1_BECVF, color='blue')
    ax.plot(X2_BECVF, Y2_BECVF, Z2_BECVF, color='red')

    # ---------------------------------------- OCVF -----------------------------------------------
    # OCVF Great Circle Basis Elements *** Page 73 ***
    # ax.plot(np.sin(np.pi/4)*np.cos(X), np.cos(np.pi/4)*np.cos(X), -np.sin(X), color='red')
    # ax.plot(np.cos(X), np.sin(X), Zeros, color='red')

    # Elements of the Rotated OCVF *** Page 74 *** OCVF Matrices
    X1_OCVF = (1 / (4 * np.sqrt(2))) * np.cos(X) * (1 + 3 * np.cos(theta1)) + (1 / (4 * np.sqrt(2))) * np.cos(X) * (
                -1 + np.cos(theta1) + 2 * np.sqrt(2) * np.sin(theta1)) - 0.25 * np.sin(X) * (
                          -np.sqrt(2) + np.sqrt(2) * np.cos(theta1) - 2 * np.sin(theta1))
    Y1_OCVF = (1 / (4 * np.sqrt(2))) * np.cos(X) * (-1 + np.cos(theta1) - 2 * np.sqrt(2) * np.sin(theta1)) + (
                1 / (4 * np.sqrt(2))) * np.cos(X) * (1 + 3 * np.cos(theta1)) - 0.25 * np.sin(X) * (
                          np.sqrt(2) - np.sqrt(2) * np.cos(theta1) - 2 * np.sin(theta1))
    Z1_OCVF = (1 / (2 * np.sqrt(2))) * np.cos(X) * (-1 + np.cos(theta1) + np.sqrt(2) * np.sin(theta1)) + (
                1 / (4 * np.sqrt(2))) * np.cos(X) * (
                          np.sqrt(2) - np.sqrt(2) * np.cos(theta1) + 2 * np.sin(theta1)) - np.sin(X) * (
                          (np.cos(theta1 / 2)) * (np.cos(theta1 / 2)))

    X2_OCVF = 0.25 * np.cos(X) * (1 + 3 * np.cos(theta1)) + 0.25 * np.sin(X) * (
                -1 + np.cos(theta1) + 2 * np.sqrt(2) * np.sin(theta1))
    Y2_OCVF = 0.25 * np.cos(X) * (-1 + np.cos(theta1) - 2 * np.sqrt(2) * np.sin(theta1)) + 0.25 * np.sin(X) * (
                1 + 3 * np.cos(theta1))
    Z2_OCVF = (1 / (2 * np.sqrt(2))) * np.cos(X) * (-1 + np.cos(theta1) + np.sqrt(2) * np.sin(theta1)) + 0.25 * np.sin(
        X) * (np.sqrt(2) - np.sqrt(2) * np.cos(theta1) + 2 * np.sin(theta1))

    # Plot of the Elements of the Rotated OCVF *** Page 75 ***
    ax.plot(X1_OCVF, Y1_OCVF, Z1_OCVF, color='red')
    ax.plot(X2_OCVF, Y2_OCVF, Z2_OCVF, color='blue')

    i += 1

# Figure Features

# Limits of the Figure
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

# Set Tick Marks
ax.set_xticks(np.arange(-1, 1, step=0.5))
ax.set_yticks(np.arange(-1, 1, step=0.5))
ax.set_zticks(np.arange(-1, 1, step=0.5))

# Axes in the Figure
ax.plot(Zeros, np.linspace(-1, 1, n_1), Zeros, color='black')
ax.plot(np.linspace(-1, 1, n_1), Zeros, Zeros, color='black')
ax.plot(Zeros, Zeros, np.linspace(-1, 1, n_1), color='black')

# BECVF Rotational Axis *** Page 70 ***
# ax.plot(-np.linspace(0, 1, n_1), np.linspace(0, 1, n_1), Zeros, color='green')

# OCVF Rotational Axis *** Page 73 ***
# ax.plot(-(1/np.sqrt(2))*X1, (1/np.sqrt(2))*X1, X1, color = 'green')

# Labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
