"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


Estimate the upper limit on the flare rate of HIP 67522 using the data from Tu et al. 2020.
"""

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from altaipony.ffd import FFD

if __name__ == "__main__":

    # Load the data from Tu et al
    print("Loading data from Tu et al. 2020")
    stars = pd.read_csv("../data/tu/tu2020_stars.tsv", sep="\t", skiprows=45, header=None,
                        names=["TIC", "Teff_K", "logg", "radius_rsun","Prot_d","Nflares",
                            "Nset","freq_per_year", "flag", "simbad",
                            "RA_deg","DEC_deg"])

    flares = pd.read_csv("../data/tu/tu2020_flares.tsv", sep="\t", skiprows=38, header=None, 
                        names=["TIC","peak_time","peak_flux_erg_s","ed_rec","duration_s"])


    # get HIP 67522 flare energies
    hip = pd.read_csv("../results/hip67522_flares.csv")

    # convert for FFD in altaipony
    hip["ed_rec"] = hip["mean_bol_energy"]

    # minimum energy is the second smallest flare in the full sample
    emin = hip["ed_rec"].sort_values().values[1]

    # get all the stars that have more than 2 flares
    stars = stars[(stars.Nflares > 2) &  (stars.flag != 'GM   ')].set_index("TIC")

    # what are the parameters of the stars in the sample?
    teffmin, teffmax, rotmin, rotmax = stars["Teff_K"].min(), stars["Teff_K"].max(), stars["Prot_d"].min(), stars["Prot_d"].max()

    print(f"Sample contains {len(stars)} stars with more than 2 flares.")
    print(f"Sample covers Teff from {teffmin} to {teffmax} K and Prot from {rotmin} to {rotmax} days.\n")

    # Load the data from Ziegler et al
    print("Loading data from Ziegler et al. 2020")
    ebs = pd.read_csv("../results/ziegler/EBs.tsv", sep="\t", skiprows=44, header=None,
                        names=["TIC", "yr","Obs","Sep","PA","Con","MinSep","Con0.15","Con1","Simbad","RA","DEC"])
    ebs.TIC = ebs.TIC.astype(int)
    etic = ebs.TIC.values
    stic = stars.index.astype(int)

    # find the overlap between the two sets
    overlap = np.intersect1d(stic, etic)

    print(f"EBs in Tu et al sample: {len(overlap)} stars.")

    # remove the overlap
    sss = stars.drop(overlap)

    # get the FFDs
    eds, freqs = [], [], 
    for tic, s in sss.iterrows():
        f = flares[flares.TIC == tic]
        ffd = FFD(f)

        obstime = len(f) / (s.freq_per_year / 365.25) # days 
        print(f"TIC {tic} {obstime} days")
        ffd.tot_obs_time = obstime
        ed, freq, counts = ffd.ed_and_freq()
        eds.append(ed)
        freqs.append(freq)


    # PLOT THE FIGURE -------------------------------------------------------------

    plt.figure(figsize=(6,5))

    # make an FFD for hip and plot it
    ffdhip = FFD(hip[hip.ed_rec >= emin])
    ffdhip.tot_obs_time = 74.5
    ed, freq, counts = ffdhip.ed_and_freq()
    plt.errorbar(ed, freq, c="navy",zorder=10, fmt="o-")

    # plot FFDs
    for ed, freq in zip(eds, freqs):
        plt.errorbar(ed, freq, alpha=0.8, color="steelblue", fmt="o-")

    # plot the limits
    plt.axvline(emin, c="navy", linestyle="--", alpha=0.8)
    plt.axhline(2, c="navy", linestyle="--", alpha=0.8)
    plt.text(emin*1.1, 2*1.05, f"flare rate upper limit", fontsize=11, va="bottom", ha="left")


    # make legend handles ------
    handle = plt.Line2D([0], [0], marker="o", color='w', markerfacecolor='steelblue', markersize=10, 
                        label=f"Tu et al. 2020 (N={len(eds)})")

    handlehip = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=10,
                        label="HIP 67522, 5650 K, 1.42 d")


    plt.legend(handles=[handle, handlehip], loc=(0.48, 0.85), frameon=False)
    # ----------------------------

    plt.xlabel(r"$E_{\rm flare}$ [erg]", fontsize=12)
    plt.ylabel("cumulative number of flares per day", fontsize=12)
    plt.xlim(3e33, 1e37)
    plt.ylim(1e-3, 1e1)
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("../plots/paper/tu.png", dpi=300, bbox_inches="tight")

