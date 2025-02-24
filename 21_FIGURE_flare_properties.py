"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

This script shows the properties of the flares in the CHEOPS and TESS light curves of HIP 67522.
It plots the relative amplitude vs. duration, relative amplitude vs. flare energy, and duration vs. flare energy.
The flares are color-coded by whether they are in or out of the planet-induced cluster.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# increase font size to 12
plt.rcParams.update({'font.size': 12})

if __name__ == "__main__":

    # read in CHEOPS and TESS flare catalogs
    cflares = pd.read_csv("results/cheops_flares.csv")
    tflares = pd.read_csv("results/tess_flares.csv")

    # remove the smallest CHEOPS flare
    cflares = cflares.sort_values("mean_bol_energy", ascending=True).iloc[1:]

    # combine the two catalogs
    flares = pd.concat([cflares, tflares])

    # get HIP 67522 orbital period and transit midpoint 
    hip67522params = pd.read_csv("data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]

    # GET FLARE ORBITAL PHASES ---------------------------------------------------------------

    flares["phase"] = (flares["t_peak_BJD"] - midpoint) % period / period

    # COLOR BY WHETHER IN or OUT OF CLUSTER --------------------------------------------------

    flares["color"] = "peru"
    flares.loc[flares.phase < 0.2, "color"] = "navy"

    flares["marker"] = "o"
    flares.loc[flares.phase < 0.2, "marker"] = "*"

    # calculate relative amplitude and duration in hours
    flares["rel_amp"] = flares["amplitude"] / flares["med_flux"]
    flares["dur_h"] = flares["dur_d"] * 24


    # plot diagnostics
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))


    ax[0].scatter(flares["rel_amp"], flares["dur_h"], color=flares["color"], alpha=1)  
    ax[0].set_xlabel("Relative Amplitude")
    ax[0].set_ylabel("Duration [h]")

    ax[1].scatter(flares["rel_amp"], flares["mean_bol_energy"], color=flares["color"], alpha=1)
    ax[1].set_xlabel("Relative Amplitude")
    ax[1].set_ylabel("Flare energy [erg]")

    ax[2].scatter(flares["dur_h"], flares["mean_bol_energy"], color=flares["color"], alpha=1)
    ax[2].set_xlabel("Duration [h]")
    ax[2].set_ylabel("Flare energy [erg]")

    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")

    # in ax[2] set xticks to 0.2, 0.4, 0.6
    ax[2].set_xticks([0.2, 0.3, 0.4, 0.6])
    ax[0].set_yticks([0.2, 0.3, 0.4, 0.6])
    # and label accordingly
    ax[2].set_xticklabels(["0.2", "0.3", "0.4", "0.6"])
    ax[0].set_yticklabels(["0.2", "0.3", "0.4", "0.6"])


    # make legend handles for peru=out of cluster, navy=in cluster
    outofcluster = mpatches.Patch(color='peru', label='Out of cluster')
    incluster = mpatches.Patch(color='navy', label='In cluster')
    ax[0].legend(handles=[outofcluster, incluster], loc=4, frameon=False)   

    plt.tight_layout()
    plt.savefig("plots/flare_properties.png", dpi=300)