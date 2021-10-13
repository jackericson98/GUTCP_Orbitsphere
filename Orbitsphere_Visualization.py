from setup_plot import setup_plot
import matplotlib.pyplot as plt
import numpy as np
from Orbitsphere_Matrix import Orbitsphere
import pdb


# M and N values, as seen in Fig. 1.16 in text, "N=M=12"
iterate_value = 12
M = iterate_value
N = iterate_value

# Radius
r = 1

# Phi Variable
phi = np.linspace(0, 2*np.pi, 100)

orb = Orbitsphere()
orb.Orbitsphere_78_exact(M, N, r, phi)
orb.Orbitsphere_80_exact(M, N, r, phi)

ax1 = setup_plot('Orbitsphere (Fig. 1.12 pg 78)', iterate_value, azim=0, elev=90)
for indx in range(orb.len_78):
    ax1.plot(orb.X_78[indx], orb.Y_78[indx], orb.Z_78[indx], color='blue')

ax2 = setup_plot('Orbitsphere (Fig. 1.16 pg 80)', iterate_value, azim=0, elev=90)
for indx in range(orb.len_80):
    ax2.plot(orb.X_80[indx], orb.Y_80[indx], orb.Z_80[indx], color='blue')
plt.show()
