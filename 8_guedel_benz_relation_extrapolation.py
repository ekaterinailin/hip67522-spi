
"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl


This script calculates the radio luminosity of HIP 67522 based on 
the X-ray luminosity and the radio spectrum.
"""

import numpy as np
import astropy.units as u



def lr_gbr(loglx):
    """GBR relation from Williams 2014.
    
    Parameters
    ----------
    loglx : float
        log10 X-ray luminosity in erg/s.
    
    Returns
    -------
    loglir : float
        log10 radio luminosity in erg/s/Hz.
    """

    return 1.36 * (loglx - 18.97) 

def spectrum(nu, alpha, beta, err=False, errvals=None):
    """Radio spectrum.
    
    Parameters
    ----------
    nu : float
        Frequency in Hz.
    alpha : float
        Low-frequency spectral index.
    beta : float
        offset
    
    Returns
    -------
    flux : float
        Flux density in Jy.
    """
    val = alpha * np.log10(nu) + beta

    if err:
        alphaerr, betaerr = errvals 
        valerr = np.sqrt(np.log10(nu)**2 * alphaerr**2 + betaerr**2)
        return val, valerr
    else:
        return val
    
if __name__ == "__main__":

    loglx = 30.76 # 0.2-2 keV from Xpsec conversion
    logr = lr_gbr(loglx)

    print(f"log10 L_R @ 5-8 GHz based on X-ray luminosity = {logr:.2f}")

    # spectral properties from results/spectral_index.txt and results/spectral_offset.txt
    alpha, alphaerr = np.loadtxt("results/atca/spectral_index.txt", dtype=float, delimiter=",")
    beta, betaerr = np.loadtxt("results/atca/spectral_offset.txt", dtype=float, delimiter=",")

    # distance of the source
    d = 124.7 * u.pc
    derr = 0.3 * u.pc # distance error

    # flux at 6.75 GHz extrapolated from ATCA spectra
    val = spectrum(6.75e9, alpha, beta)

    # convert to luminosity
    l65ghz = ((10**val * u.Jy * 4 * np.pi * d**2)).to(u.erg / u.s / u.Hz)
    log10l65ghz = np.log10(l65ghz.value)

    # convert to surface brightness
    ofs = np.log10(u.Jy.to("erg * s^-1 * pc^-2 * Hz^-1") )

    val, valerr = spectrum(6.75e9, alpha, beta, err=True, errvals = [alphaerr, betaerr])
    log10l65ghz = val + ofs + np.log10(4 * np.pi) + 2 * np.log10(d.value)
    err = np.sqrt(valerr**2 + (2 / np.log(10) / d.value * derr.value)**2)
    
    print(f"log10 L_R @ 6.75 GHz based on extrapolation = {log10l65ghz:.2f} +/- {err:.2f} erg/s/Hz")
