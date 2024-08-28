"""
UTF-8, Python 3

------------
HIP 67522 CHEOPS SPI
------------

Ekaterina Ilin, 2024, MIT License


IO functions for CHEOPS data reduced with PIPE.
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def extract(data, stri):
    """Quick function to extract light curve columns from a fits file"""
    return data[stri].byteswap().newbyteorder()


def read_lc_from_pipe(file, outdata="00000", path="../data/hip67522"):
    """Read the light curve data from PIPE reduced LC.
    
    Parameters:
    -----------
    file : str
        The file name of the PIPE reduced LC.
    outdata : str
        The outdata folder number, i.e. version of the PIPE reduction.
    path : str
        The path to the data.

    Returns:
    --------
    lc : dict
        Dictionary with the light curve data. Contains the following
        keys: "t", "f", "ferr", "roll", "dT", "flag", "bg", "xc", "yc"
        
    """

    # file name
    IMG = f'{path}/CHEOPS-products-{file}/Outdata/{outdata}/hip67522_CHEOPS-products-{file}_im.fits'

    # open the fits file
    hdulist = fits.open(IMG)
    print(f"Imagette file found for {file}:\n {IMG}\n")

    # get the image data
    image_data = hdulist[1].data

    # get LC data
    t = extract(image_data, "BJD_TIME")
    f = extract(image_data, "FLUX")
    ferr = extract(image_data, "FLUXERR")
    roll = extract(image_data, "ROLL")
    dT = extract(image_data, "thermFront_2")
    flag = extract(image_data, "FLAG")
    bg = extract(image_data, "BG")
    xc = extract(image_data, "XC")
    yc = extract(image_data, "YC")

    # make sure the data is in fact 10s cadence
    assert np.diff(t).min() * 24 * 60 * 60 < 10.05, "Time series is not 10s cadence"

    # make a dictionary of the data
    lc = {"t": t, "f": f, "ferr": ferr, "roll": roll, "dT": dT, "flag": flag, "bg": bg, "xc": xc, "yc": yc}

    return lc

def get_residual_image(file, index=664, vmin=-600, vmax=6000):

    IMG = f'../data/hip67522/CHEOPS-products-{file}/Outdata/00000/residuals_sa.fits'
    hdulist = fits.open(IMG)
    print(f"Residuals image file found for {file}:\n {IMG}\n")

    # get the image data
    image_data = hdulist[0].data
    print(hdulist[0].header)

    print(f"Image shape: {image_data.shape}")
    # sum over the first axis
    image_data = image_data[index:index+20].sum(axis=0)

    # show the image
    plt.imshow(image_data, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(label=r"Flux [e$^{-}$/s]")

    plt.xlabel("x pixel number")
    plt.ylabel("y pixel number")

    plt.tight_layout()


def extract(data, stri):
    """Quick function to extract light curve columns from a fits file"""
    return data[stri].byteswap().newbyteorder()