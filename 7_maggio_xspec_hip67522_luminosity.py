"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


This script calculates the radio luminosity of HIP 67522 based on
the X-ray observations in Maggio+2024, using their 3T model
and XSPEC to convert to luminosity in 0.2-2 keV band.
"""

import xspec
import numpy as np
import astropy.units as u

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    xspec.Model("vapec+vapec+vapec")

    # fill in parameters from Maggio et al. 2024
    kT1 = xspec.AllModels(1)(1)
    kT1.values = ".558,,,,,"
    norm1 = xspec.AllModels(1)(16)
    norm1.values = "0.000427"
    kT2 = xspec.AllModels(1)(17)
    kT2.values = ".98,,,,,"
    norm2 = xspec.AllModels(1)(32)
    norm2.values = "0.000427"
    kT3 = xspec.AllModels(1)(33)
    kT3.values = "1.98,,,,,"
    norm3 = xspec.AllModels(1)(48)
    norm3.values = "0.000427"

    # get flux in 0.2-2 keV band
    xspec.AllModels.calcFlux("0.2,2.0,0")

    # calculate luminosity in that band using the flux result 3.117e-12
    lum = ((3.1117e-12 * 4 * np.pi * 124.7**2) * u.erg / u.cm**2 / u.s * u.pc**2).to("erg/s")

    print(lum)

    print(np.log10(lum.value))
    