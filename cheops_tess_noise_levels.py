"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

Determine the mean noise level in the CHEOPS and TESS light curves.
"""

import numpy as np
import pandas as pd
from lightkurve import search_lightcurve

from altaipony.flarelc import FlareLightCurve
from altaipony.customdetrend import custom_detrending

if __name__ == "__main__":

    # TESS ------------------------------------------------
    lcs = search_lightcurve('HIP 67522', mission='TESS', author='SPOC')
    lcs = lcs[[0,1,3]].download_all()

    stds, meds = [], []
    for lc in lcs:
        flc = FlareLightCurve(time=lc.time.value, flux=lc.flux.value, flux_err=lc.flux_err.value)
        flcd = flc.detrend(mode="custom", func=custom_detrending, **{"savgol1":6., "savgol2":3., "spline_coarseness":6.})

        # remove outliers more than 7 sigma
        flcd = flcd.remove_outliers(sigma=5) # mask the flaring regions

        # plt.figure(figsize=(15,5))
        # plt.plot(flc.time.value.astype(float), flc.flux.value.astype(float), c="k")
        # plt.plot(flcd.time.value.astype(float), flcd.detrended_flux, c="r", label="detrended")

        stds.append(np.nanstd(flcd.detrended_flux))
        meds.append(np.nanmedian(flcd.detrended_flux))

    # get the mean noise level from meds and stds
    tessmed = np.mean(np.array(stds) / np.array(meds))
    tessstd = np.std(np.array(stds) / np.array(meds))

    # CHEOPS ------------------------------------------------
    # now load cheops lightcurves
    # load the file names
    files = np.loadtxt("files.txt", dtype=str)

    # read the flare table
    flares = pd.read_csv("../results/cheops_flares.csv")

    # read in all the de-trended light curves
    dlcs = []
    for pi, file in files:
        location = f"../data/hip67522/pipe_HIP67522/HIP67522_{file}{pi}_detrended_lc.csv"
        dlcs.append(pd.read_csv(location))


    # now loop over the light curves and calculate the mean and std of each detrended light curve
    stds, meds = [], []
    for dlc in dlcs:
        # check if there are flares in the light curve
        if len(flares[(flares["tmin"] > dlc["time"].min()) & (flares["tmax"] < dlc["time"].max())]) == 0:
            print("No flares in this light curve")
            
        else:
            for i, flare in flares[(flares["tmin"] > dlc["time"].min()) & (flares["tmax"] < dlc["time"].max())].iterrows():
                # mask flare
                dlc = dlc[((dlc["time"] < flare["tmin"]) | (dlc["time"] > flare["tmax"]))]
                print("Flare masked")

        meds.append(np.nanmedian(dlc["flux"]))
        stds.append(np.nanstd(dlc["flux"]))

    # get the mean noise level from meds and stds
    cheopsmed = np.mean(np.array(stds) / np.array(meds))
    cheopsstd = np.std(np.array(stds) / np.array(meds))

    # SYNTHESIS ------------------------------------------------
    print(f"CHEOPS noise level: {cheopsmed:.5f}, {cheopsstd:.5f}")
    print(f"TESS noise level: {tessmed:.5f}, {tessstd:.5f}")


