import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pdb


class BECVF:
    def __init__(self):
        self.X1 = []
        self.Y1 = []
        self.Z1 = []
        self.X2 = []
        self.Y2 = []
        self.Z2 = []

    def BECVF_M1(self, theta, phi):
        BECVF_X1 = np.cos(phi) * (-0.5 + 0.5 * np.cos(theta)) + np.sin(phi) * (np.sin(theta) / np.sqrt(2))
        BECVF_Y1 = np.cos(phi) * (0.5 + 0.5 * np.cos(theta)) + np.sin(phi) * (np.sin(theta) / np.sqrt(2))
        BECVF_Z1 = np.cos(phi) * (np.sin(theta) / np.sqrt(2)) - np.sin(phi) * np.cos(theta)

        self.X1.append(BECVF_X1)
        self.Y1.append(BECVF_Y1)
        self.Z1.append(BECVF_Z1)

    def BECVF_M2(self, theta, phi):
        BECVF_X2 = np.cos(phi) * (0.5 + 0.5 * np.cos(theta)) + np.sin(phi) * (np.sin(theta) / np.sqrt(2))
        BECVF_Y2 = np.cos(phi) * (-0.5 + 0.5 * np.cos(theta)) + np.sin(phi) * (np.sin(theta) / np.sqrt(2))
        BECVF_Z2 = np.cos(phi) * (np.sin(theta) / np.sqrt(2)) - np.sin(phi) * np.cos(theta)
        self.X2.append(BECVF_X2)
        self.Y2.append(BECVF_Y2)
        self.Z2.append(BECVF_Z2)


class OCVF:
    def __init__(self):
        self.X1 = []
        self.Y1 = []
        self.Z1 = []
        self.X2 = []
        self.Y2 = []
        self.Z2 = []

    def OCVF_M1(self, theta, phi):
        OCVF_X1 = (1 / (4 * np.sqrt(2))) * np.cos(phi) * (1 + 3 * np.cos(theta)) + \
                  (1 / (4 * np.sqrt(2))) * np.cos(phi) * (-1 + np.cos(theta) + 2 * np.sqrt(2) * np.sin(theta)) - \
                  0.25 * np.sin(phi) * (-np.sqrt(2) + np.sqrt(2) * np.cos(theta) - 2 * np.sin(theta))
        OCVF_Y1 = (1 / (4 * np.sqrt(2))) * np.cos(phi) * (-1 + np.cos(theta) - 2 * np.sqrt(2) * np.sin(theta)) + \
                  (1 / (4 * np.sqrt(2))) * np.cos(phi) * (1 + 3 * np.cos(theta)) - \
                  0.25 * np.sin(phi) * (np.sqrt(2) - np.sqrt(2) * np.cos(theta) - 2 * np.sin(theta))
        OCVF_Z1 = (1 / (2 * np.sqrt(2))) * np.cos(phi) * (-1 + np.cos(theta) + np.sqrt(2) * np.sin(theta)) + \
                  (1 / (4 * np.sqrt(2))) * np.cos(phi) * \
                  (np.sqrt(2) - np.sqrt(2) * np.cos(theta) + 2 * np.sin(theta)) - \
                  np.sin(phi) * ((np.cos(theta / 2)) * (np.cos(theta / 2)))
        self.X1.append(OCVF_X1)
        self.Y1.append(OCVF_Y1)
        self.Z1.append(OCVF_Z1)

    def OCVF_M2(self, theta, phi):
        OCVF_X2 = 0.25 * np.cos(phi) * (1 + 3 * np.cos(theta)) + \
                  0.25 * np.sin(phi) * (-1 + np.cos(theta) + 2 * np.sqrt(2) * np.sin(theta))
        OCVF_Y2 = 0.25 * np.cos(phi) * (-1 + np.cos(theta) - 2 * np.sqrt(2) * np.sin(theta)) + \
                  0.25 * np.sin(phi) * (1 + 3 * np.cos(theta))
        OCVF_Z2 = (1 / (2 * np.sqrt(2))) * np.cos(phi) * (-1 + np.cos(theta) + np.sqrt(2) * np.sin(theta)) + \
                  0.25 * np.sin(phi) * (np.sqrt(2) - np.sqrt(2) * np.cos(theta) + 2 * np.sin(theta))
        self.X2.append(OCVF_X2)
        self.Y2.append(OCVF_Y2)
        self.Z2.append(OCVF_Z2)


def setup_plot(title_str, azim=30, elev=30):
    # Setting up the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Limits of the Figure
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # Set Tick Marks
    ax.set_xticks(np.arange(-1, 1, step=0.5))
    ax.set_yticks(np.arange(-1, 1, step=0.5))
    ax.set_zticks(np.arange(-1, 1, step=0.5))

    # Axes in the Figure
    ax.plot(Zeros, np.linspace(-1, 1, n_OCVF), Zeros, color='black')
    ax.plot(np.linspace(-1, 1, n_OCVF), Zeros, Zeros, color='black')
    ax.plot(Zeros, Zeros, np.linspace(-1, 1, n_OCVF), color='black')

    # View Direction
    ax.view_init(azim=azim, elev=elev)

    # Labels and title
    ax.set_title(title_str)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    return ax


if __name__ == '__main__':
    # ---------------------------------------- Define Variables -----------------------------------------------
    theta_BECVF = 0  # BECVF Theta
    theta_OCVF = 0  # OCVF Theta
    n_BECVF = 50  # number of times 2 pi is split up for BECVF.
    n_OCVF = 50  # number of times 2 pi is split up for OCVF.
    r = 1  # Radius for both

    # Array Variables
    phi_BECVF = np.linspace(0, 2 * np.pi, n_OCVF)  # BECVF Phi Variable (see bottom of page 71)
    phi_OCVF = np.linspace(-np.pi, np.pi, n_OCVF)  # OCVF Phi Variable
    Zeros = np.linspace(0, 0, n_OCVF)  # Array of Zeros matching the size of Phi Variables

    # ---------------------------------------- Run Calculations -----------------------------------------------
    BECVF_matrix = BECVF()
    OCVF_matrix = OCVF()

    for indx in range(n_BECVF):
        # Increment's of rotation for BECVF and OCVF
        theta_BECVF = (2 * indx * np.pi) / n_BECVF
        theta_OCVF = (indx * np.pi) / n_BECVF

        BECVF_matrix.BECVF_M1(theta_BECVF, phi_BECVF)
        OCVF_matrix.OCVF_M1(theta_BECVF, phi_BECVF)
        BECVF_matrix.BECVF_M2(theta_BECVF, phi_BECVF)
        OCVF_matrix.OCVF_M2(theta_BECVF, phi_BECVF)

    # ---------------------------------------- Plot Figures -----------------------------------------------
    # BECVF Great Circle Basis Elements and rotated axis ***Reproduction: See Page 70, Figure 1.4***
    ax1 = setup_plot('BECVF Great Circles (pg 70)')
    ax1.plot(Zeros, r*np.cos(phi_BECVF), r*-np.sin(phi_BECVF), color='red')
    ax1.plot(r*np.cos(phi_BECVF), Zeros, r*-np.sin(phi_BECVF), color='red')
    ax1.plot(-np.linspace(0, 1, n_OCVF), np.linspace(0, 1, n_OCVF), Zeros, color='green')

    # OCVF Great Circle Basis Elements and rotated axis ***Reproduction: See Page 73, Figure 1.8***
    # (note the abs is the magnitude of the XY components)
    ax2 = setup_plot('OCVF Great Circles (pg 73)')
    ax2.plot(r*np.sin(np.pi/4)*np.cos(phi_BECVF), r*np.cos(np.pi/4)*np.cos(phi_BECVF), r*-np.sin(phi_BECVF), color='red')
    ax2.plot(r*np.cos(phi_BECVF), r*np.sin(phi_BECVF), Zeros, color='red')
    ax2.plot(abs(-(1/np.sqrt(2))*phi_OCVF), abs((1/np.sqrt(2))*phi_OCVF), abs(phi_OCVF), color='green')

    # Plot of the Elements of the Rotated OCVF *** Page 75 ***
    ax3 = setup_plot('OCVF Current Pattern (pg 75)', azim=0, elev=90)
    for indx in range(n_BECVF):
        ax3.plot(OCVF_matrix.X1[indx], OCVF_matrix.Y1[indx], OCVF_matrix.Z1[indx], color='blue')
        ax3.plot(OCVF_matrix.X2[indx], OCVF_matrix.Y2[indx], OCVF_matrix.Z2[indx], color='blue')

    # Plot of the Elements of the Rotated BECVF *** Page 72 ***
    ax4 = setup_plot('BECVF Current Pattern (pg 72)', azim=0, elev=90)
    for indx in range(n_BECVF):
        ax4.plot(BECVF_matrix.X1[indx], BECVF_matrix.Y1[indx], BECVF_matrix.Z1[indx], color='blue')
        ax4.plot(BECVF_matrix.X2[indx], BECVF_matrix.Y2[indx], BECVF_matrix.Z2[indx], color='blue')

    # Plot of the Elements of the Rotated OCVF *** Page 75 ***
    ax5 = setup_plot('BECVF and OCVF Current Pattern (pg 72, 75)', azim=0, elev=90)
    for indx in range(n_BECVF):
        ax5.plot(OCVF_matrix.X1[indx], OCVF_matrix.Y1[indx], OCVF_matrix.Z1[indx], color='green')
        ax5.plot(OCVF_matrix.X2[indx], OCVF_matrix.Y2[indx], OCVF_matrix.Z2[indx], color='green')
        ax5.plot(BECVF_matrix.X1[indx], BECVF_matrix.Y1[indx], BECVF_matrix.Z1[indx], color='blue')
        ax5.plot(BECVF_matrix.X2[indx], BECVF_matrix.Y2[indx], BECVF_matrix.Z2[indx], color='blue')

    plt.show()
