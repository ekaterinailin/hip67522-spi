"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

Simple script to calculate the flare rate in the cluster and 
compare it to the expected rate based on the out-of-cluster rate, as
per manuscript.
"""

import numpy as np 
import pandas as pd

if __name__ == "__main__":

    # read phases from file
    tess_phases = np.loadtxt("results/tess_phases.txt")
    cheops_phases = np.loadtxt("results/cheops_phases.txt")

    # weigh by observing cadence
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. / 60. / 60. / 24., 
                                np.ones_like(tess_phases) * 2. / 60. / 24.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])

    # flare phases

    flares = pd.read_csv("results/hip67522_flares.csv")
    flares = flares.sort_values("mean_bol_energy", ascending=True).iloc[1:] # exclude the smallest flare
    phases = flares.orb_phase.values

    phasemask = obs_phases < 0.201 # let's be generous and include the flare at 0.2005, choosing smaller does not make it worse

    # flare rates within and outside the phase mask
    rate_in_cluster = len(phases[phases<0.201]) / np.sum(weights[phasemask])
    rate_out_cluster = len(phases[phases>0.201]) / np.sum(weights[~phasemask])

    print(f"Flare rate in cluster: {rate_in_cluster:.2f} flares/day")
    print(f"Flare rate outside cluster: {rate_out_cluster:.2f} flares/day")

    print(f"Rate ratio: {rate_in_cluster / rate_out_cluster:.2f}")

    # expected flare number in cluster based on out-of-cluster rate
    expected_in_cluster = rate_out_cluster * np.sum(weights[phasemask])

    print(f"Expected number of flares in cluster: {expected_in_cluster:.2f}")
    print(f"Observed number of flares in cluster: {len(phases[phases<0.201])}") 