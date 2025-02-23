"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


Calculate the flare energies for the TESS and CHEOPS 
flares of HIP 67522 and compare the FFDs
for the clustered and baseline flares.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from altaipony.ffd import FFD



# set default matplotlib fontsize to 13
plt.rcParams.update({'font.size': 13})



if __name__ == "__main__":

    # if plots/diagnostic/ffd/ does not exist, create it
    if not os.path.exists("plots/diagnostic/ffd"):
        os.makedirs("plots/diagnostic/ffd")

    # GET STELLAR AND PLANET PARAMETERS -----------------------------------------------------

    hip67522params = pd.read_csv("data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]
    teff = hip67522params[hip67522params.param=="teff_K"].val.values[0]
    tefferr = hip67522params[hip67522params.param=="teff_K"].err.values[0]
    radius = hip67522params[hip67522params.param=="radius_rsun"].val.values[0]
    radiuserr = hip67522params[hip67522params.param=="radius_rsun"].err.values[0]

    # ----------------------------------------------------------------------------------------

    # GET ALL FLARES

    flares = pd.read_csv("results/hip67522_flares.csv")
    flares = flares.sort_values("mean_bol_energy", ascending=True).iloc[1:] # exclude the smallest flare
    flares["ed_rec"] = flares["mean_bol_energy"]
    flares = flares.reset_index()


    # GET ALL OBSERVING PHASES

    tess_phases = np.loadtxt("results/tess_phases.txt")
    cheops_phases = np.loadtxt("results/cheops_phases.txt")
    
    # weigh by observing cadence
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. / 60. / 60. / 24., np.ones_like(tess_phases) * 2. / 60. / 24.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])

    phase_cuts = [(0.0, 0.2, "navy", "off"),(0.2,1.0, "peru", "none")]

    fig, ax = plt.subplots(figsize=(6.5, 5))

    alphas_clustered = {}
    alphas_baseline = {}
    
    for i in range(10):

        for phase_cut_min, phase_cut_max, color, flip in phase_cuts:

            obs_phase_mask = (obs_phases < phase_cut_max) & (obs_phases > phase_cut_min)
            flare_phase_mask = (flares["orb_phase"] < phase_cut_max) & (flares["orb_phase"] > phase_cut_min)

            if flip == "off":
                # of those in the observing phase range, remove one randomly from the flare phase range
                choose = np.where(flare_phase_mask)[0][i]
                print(choose)
                flare_phase_mask[choose] = False
                print("Random flare removed from observing phase range.")
            elif flip == "none":
                flare_phase_mask[choose] = True
                obs_phase_mask = np.ones_like(obs_phases, dtype=bool)
        
            flares_cut = flares[flare_phase_mask]
            weights_cut = weights[obs_phase_mask]

            # FULL SAMPLE FFD

            ffd = FFD(flares_cut)
            ffd.tot_obs_time = np.sum(weights_cut)

            print(ffd.tot_obs_time)
            print(len(flares_cut))

            ed, freq, counts = ffd.ed_and_freq()
            bfa = ffd.fit_powerlaw("mcmc")

            if flip == "off":
                alphas_clustered[i] = bfa.samples[:,1]
            elif flip == "none":
                alphas_baseline[i] = bfa.samples[:,1]

            ffd.plot_mcmc_powerlaw(ax, bfa, c=color, subset=100, alpha=0.005, custom_xlim=(1e33,1e36))
            ax.scatter(ed, freq, c="k", s=45, zorder=1000)
            ax.scatter(ed, freq, c="w", s=25, zorder=1001)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(5e33, 1.5e36)
    plt.ylim(4e-3, 2)
    plt.xlabel("Bolometric Flare Energy [erg]")
    plt.ylabel("cumulative number of flares per day")
    plt.savefig("plots/diagnostic/ffd/ffd_cut.png", dpi=300)


    # save alpha values to file

    # take the longest array
    max_len = max([len(alphas_clustered[i]) for i in range(10)])
    # pad the remaining arrays with NaN
    alphas_clustered = [np.concatenate([alphas_clustered[i], np.ones(max_len - len(alphas_clustered[i])) * np.nan]) for i in range(10)]

    # take the longest array
    max_len = max([len(alphas_baseline[i]) for i in range(10)])
    # pad the remaining arrays with NaN
    alphas_baseline = [np.concatenate([alphas_baseline[i], np.ones(max_len - len(alphas_baseline[i])) * np.nan]) for i in range(10)]

    df_clustered = pd.DataFrame(alphas_clustered)
    df_baseline = pd.DataFrame(alphas_baseline)
    df_clustered.to_csv("results/ffd_alpha_clustered_cut.csv", index=False)
    df_baseline.to_csv("results/ffd_alpha_baseline_cut.csv", index=False)

    # alpha values histogram
    fig, ax = plt.subplots()
    for i in range(10):
        ax.hist(alphas_clustered[i], bins=20, color="navy", alpha=0.2)
    for i in range(10):
        ax.hist(alphas_baseline[i], bins=20, color="peru", alpha=0.2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("count")
    plt.xlim(1, 3)
    plt.tight_layout()
    plt.savefig("plots/diagnostic/ffd/alpha_hist_cut.png", dpi=300)

    print("Full sample alpha mean:", np.mean(bfa.samples[:,1]))
    print("Full sample alpha median:", np.median(bfa.samples[:,1]))
    print("Full sample alpha std:", np.std(bfa.samples[:,1]))

    # write median alpha to file
    with open("results/ffd_alpha_cut.txt", "w") as f:
        f.write(str(np.median(bfa.samples[:,1])))

