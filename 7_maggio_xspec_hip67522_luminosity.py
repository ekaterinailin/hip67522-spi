import xspec
import numpy as np
import astropy.units as u

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

    # calculate luminosity in that band
    lum = ((3.1117e-12 * 4 * np.pi * 124.7**2) * u.erg / u.cm**2 / u.s * u.pc**2).to("erg/s")

    print(lum)

    print(np.log10(lum.value))
    