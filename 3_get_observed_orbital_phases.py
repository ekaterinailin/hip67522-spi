"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script compiles the observed orbital phases of the TESS and CHEOPS missions for HIP 67522.

"""

import numpy as np 
import pandas as pd
from astropy.io import fits

def get_tess_orbital_phases(period, sectors,  midpoints ):
    """Download the TESS light curves for HIP 67522 and calculate the observing 
    time in the first 10% and last 90% of the orbit, or any other split.  
    
    Parameters
    ----------
    period : float
        The orbital period of the planet in days.
    sectors : list
        The TESS sectors to download the light curves from.
    midpoints : list
        The midpoints of the TESS sectors in BJD. 
        Predictions from Rizzuto et al. 2020 using the NASA prediction tool for simplicity for each Sector

    Returns
    -------
    tessphases : array
        The phases of the TESS light curves
    """

    # get LC data
    lcs = []
    for sector in sectors:
        hdu = fits.open(f"../data/tess/tess_hip67522_{sector}.fits")

        t = hdu[1].data["TIME"]
        f = hdu[1].data["PDCSAP_FLUX"]
        ferr = hdu[1].data["PDCSAP_FLUX_ERR"]
        flag = hdu[1].data["QUALITY"]

        mask =  ((flag==0) &
                 np.isfinite(t) &
                 np.isfinite(f) &
                 np.isfinite(ferr) &
                 (~np.isnan(t)) &
                 (~np.isnan(f)) &
                 (~np.isnan(ferr)))

        lc = {"time": t[mask] + 2457000., "flux": f[mask], "flux_err": ferr[mask], "quality": flag[mask]}
        lcs.append(pd.DataFrame(lc))


    # get all phases for the TESS light curves
    tessphases = []
    for lc, midpoint in zip(lcs, midpoints):
        tessphases.append(((lc.time.values - midpoint) % period) / period)


    return np.concatenate(tessphases)
        


def get_cheops_orbital_phases(period, midpoint):
    """Grab the CHEOPS light curves for HIP 67522 and calculate the orbital phases.

    Parameters
    ----------
    period : float
        The orbital period of the planet in days.
    midpoint : float
        The midpoint of the CHEOPS observations in BJD.

    Returns
    -------
    cheopsphases : array
        The phases of the CHEOPS light curves

    """

    time = np.array([])
    cheopsphases = np.array([])

     # load the file names
    files = np.loadtxt("data/cheops_files.txt", dtype=str)

    # read in all the de-trended light curves
    dlcs = []
    for pi, file in files:
        location = f"data/cheops/HIP67522_{file}{pi}_detrended_lc.csv"
        dlcs.append(pd.read_csv(location))

    for dlc in dlcs:

        t = dlc["time"].values

        # make sure the data is in fact 10s cadence
        assert np.diff(t).min() * 24 * 60 * 60 < 10.05, "Time series is not 10s cadence"

        time = np.concatenate([time, t])
        cheopsphases = np.concatenate([cheopsphases, ((t - midpoint) % period) / period])

    return cheopsphases



if __name__ == "__main__":

    hip67522params = pd.read_csv("../data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    


    # GET OBSERVED CHEOPS PHASES ---------------------------------------------------------------

    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]
    cheops_phases, _, _, _ = get_cheops_orbital_phases(period, midpoint)


    # GET OBSERVED TESS PHASES ---------------------------------------------------------------

    # predictions from Rizzuto et al. 2020 using the NASA prediction tool for simplicity for each Sector
    sector_midpoints = [2458694.49725,2459425.24506,2460155.99288] 
    tess_phases, ttess01, ttess09, ttess = get_tess_orbital_phases(period, [11, 38, 64], sector_midpoints) 

    # write phases to file
    np.savetxt("data/tess_phases.txt", tess_phases)
    np.savetxt("data/cheops_phases.txt", cheops_phases)
