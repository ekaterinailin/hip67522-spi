"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Harish K. Vedantham, 2025


Calculate the pointing flux intercepting a planet in the presence of a stellar wind.
"""

import numpy as np
import matplotlib.pyplot as plt
#
Nmc = 10000 # NUMBER OF MONTE CARLO RUNS
mag_flux = np.zeros((Nmc)) # INTERCEPTED POYNTING FLUX
ram_flux = np.zeros((Nmc)) # INTERCEPTED RAM PRESSURE FLUX
#
vwind = 462e5 # STELLAR WIND SPEED CM/S
Rp = 4.7e9 # RADIUS OF OBSTACLE
#
for i in range(Nmc): # FOR EACH MC RUN
   Bstar = (np.random.rand()*2.5+1)*1e3 # CHOOSE A STELLAR SURFACE FIELD VALUE (GAUSS)
   rhostar = 10**(np.random.rand()*2+8) # CHOOSE A BASE DENSITY VALUE (CM^-3)
   # B = Bstar*(12**-2) # FOR AN OPEN FIELD
   B = Bstar*(12**-3) # FOR A DIPOLE FIELD
   rho = rhostar*(12**-2) # NUMBER DENSITY OF WIND AT PLANET
   mag_pressure = B**2/(8*np.pi) # MAGNETIC ENERGY DENSITY AT PLANET
   ram_pressure = 1.67e-24*rho*vwind**2 # RAM PRESSURE AT PLANET
   mag_flux[i] = mag_pressure*np.pi*Rp**2*vwind # INTERCEPTED MAGNETIC FLUX
   ram_flux[i] = ram_pressure*np.pi*Rp**2*vwind # INTERCEPTED KINETIC FLUX

plt.hist(np.log10(mag_flux),alpha=0.5,label="MAGNETIC")
plt.hist(np.log10(ram_flux),alpha=0.5,label="KINETIC")
plt.legend()
plt.xlabel(r"log$_{10}$ (energy flux in erg/s)")
plt.ylabel("Incidence [arb. units]")
plt.show()
plt.close() 