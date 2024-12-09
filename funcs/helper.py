"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl


This module contains helpful IO functions.
"""

import numpy as np
import pandas as pd

import lightkurve as lk

COLORS = [
    "cyan",  # Blue
    "#D95F0E",  # Vermilion
    "#009E73",  # Teal
    "maroon",  # maroon
    "#CC79A7",  # Pink
    "#56B4E9",  # Sky Blue
    "#FF7F00",  # Orange
    "olive",   # Dark Red
    "#FF4500",   # Orange Red
    "#1F78B4" # Blue
        ]

def get_tess_orbital_phases(period, split=0.1, by_sector = False, usemask=False, mask=None):
    """Download the TESS light curves for HIP 67522 and calculate the observing 
    time in the first 10% and last 90% of the orbit, or any other split.  
    
    Parameters
    ----------
    period : float
        The orbital period of the planet in days.
    split : float
        The phase split of the light curve to calculate the observing time for.
        Default is 0.1.
    by_sector : bool
        If True, return the phases for each sector separately. Default is False.

    Returns
    -------
    tessphases : array
        The phases of the TESS light curves
    ttess01 : float
        The observing time in days in the first 10% of the light curve.
    ttess09 : float
        The observing time in days in the last 90% of the light curve.
    ttess : float
        The total observing time in days of the TESS light curves.
    """
    lcs = lk.search_lightcurve("HIP 67522", mission="TESS",author="SPOC", exptime=120)

    # predictions from Rizzuto et al. 2020 using the NASA prediction tool for simplicity for each Sector
    midpoints = [2458694.49725,2459425.24506,2460155.99288] 

    # get the light curves
    lcs = [lc.download() for lc in lcs]

    # get all phases for the TESS light curves
    tessphases = []
    for lc, midpoint in zip(lcs, midpoints):
        lc = lc[lc.quality < 1]    
        tessphases.append(((lc.time.value - midpoint) % period) / period)


    if by_sector:
    
        return tessphases
    
    else:

        tessphases = np.concatenate(tessphases)
        # get the observing time for first 10% and last 90% of the light curve
        if usemask:
            m = (tessphases > mask[0]) & (tessphases < mask[1])
            ttess01 = len(tessphases[m]) * 2. / 60. / 24.
            ttess09 = len(tessphases[~m]) * 2. / 60. / 24.
        else:
            ttess01 = len(tessphases[tessphases < split]) * 2. / 60. / 24.
            ttess09 = len(tessphases[tessphases > split]) * 2. / 60. / 24.

        ttess = ttess01 + ttess09
        return tessphases, ttess01, ttess09, ttess

def get_cheops_orbital_phases(period, midpoint, split=0.1, usemask=False, mask=None):

    time = np.array([])
    cheopsphases = np.array([])

     # load the file names
    files = np.loadtxt("files.txt", dtype=str)

    # read in all the de-trended light curves
    dlcs = []
    for pi, file in files:
        location = f"../data/hip67522/pipe_HIP67522/HIP67522_{file}{pi}_detrended_lc.csv"
        dlcs.append(pd.read_csv(location))



    for dlc in dlcs:

        t = dlc["time"].values

        # make sure the data is in fact 10s cadence
        assert np.diff(t).min() * 24 * 60 * 60 < 10.05, "Time series is not 10s cadence"

        time = np.concatenate([time, t])
        cheopsphases = np.concatenate([cheopsphases, ((t - midpoint) % period) / period])

    tot_obs_time_d_cheops = len(time) * 10. / 60. / 60. / 24.
    if usemask:
        m = (cheopsphases > mask[0]) & (cheopsphases < mask[1])
        tcheops01 = len(time[m]) * 10. / 60. / 60. / 24.
        tcheops09 = len(time[~m]) * 10. / 60. / 60. / 24.
    else:
        tcheops01 = len(time[cheopsphases < split]) * 10. / 60. / 60. / 24.
        tcheops09 = len(time[cheopsphases > split]) * 10. / 60. / 60. / 24.

    return cheopsphases, tcheops01, tcheops09, tot_obs_time_d_cheops

