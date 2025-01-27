"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl


Calculate the flare energies for the TESS and CHEOPS flares of HIP 67522 and compare the FFDs.
Fit power laws to the FFDs in different orbital phase ranges.
Determine the detection thresholds for TESS and CHEOPS flares.
The TESS flares are from Ilin+2024, the CHEOPS flares are from this work.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from altaipony.ffd import FFD



# set default matplotlib fontsize to 13
plt.rcParams.update({'font.size': 13})



if __name__ == "__main__":


    # GET STELLAR AND PLANET PARAMETERS -----------------------------------------------------

    hip67522params = pd.read_csv("../data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]
    teff = hip67522params[hip67522params.param=="teff_K"].val.values[0]
    tefferr = hip67522params[hip67522params.param=="teff_K"].err.values[0]
    radius = hip67522params[hip67522params.param=="radius_rsun"].val.values[0]
    radiuserr = hip67522params[hip67522params.param=="radius_rsun"].err.values[0]

    # ----------------------------------------------------------------------------------------

    # GET ALL FLARES

    flares = pd.read_csv("../results/hip67522_flares.csv")
    flares = flares.sort_values("mean_bol_energy", ascending=True).iloc[1:] # exclude the smallest flare
    flares["ed_rec"] = flares["mean_bol_energy"]


    # GET ALL OBSERVING PHASES

    tess_phases = np.loadtxt("../data/tess_phases.txt")
    cheops_phases = np.loadtxt("../data/cheops_phases.txt")
    
    # weigh by observing cadence
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. / 60. / 60. / 24., np.ones_like(tess_phases) * 2. / 60. / 24.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])
    
    # FULL SAMPLE FFD

    ffd = FFD(flares)
    ffd.tot_obs_time = np.sum(weights)
    ed, freq, counts = ffd.ed_and_freq()
    bfa = ffd.fit_powerlaw("mcmc")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ffd.plot_mcmc_powerlaw(ax, bfa, c="steelblue", subset=400, alpha=0.01, custom_xlim=(1e33,1e36))
    ax.scatter(ed, freq, c="k", s=45, zorder=1000)
    ax.scatter(ed, freq, c="peru", s=25, zorder=1001)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(3e33, 1e36)
    plt.ylim(4e-3, 1)
    plt.xlabel("Bolometric Flare Energy [erg]")
    plt.ylabel("Cumulative number of flares per day")
    plt.savefig("../plots/paper/ffd_full_sample.png", dpi=300)

    # alpha values histogram
    fig, ax = plt.subplots()
    ax.hist(bfa.samples[:,1], bins=20, color="k", alpha=0.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig("../plots/ffd/alpha_hist.png", dpi=300)

    print("Full sample alpha mean:", np.mean(bfa.samples[:,1]))
    print("Full sample alpha median:", np.median(bfa.samples[:,1]))
    print("Full sample alpha std:", np.std(bfa.samples[:,1]))

    # write median alpha to file
    with open("../results/ffd_full_sample_alpha.txt", "w") as f:
        f.write(str(np.median(bfa.samples[:,1])))

