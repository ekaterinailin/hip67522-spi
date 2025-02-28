"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script computes the probability of flares aligning 5 rotations apart 
four times or more in a random distribution of flares along rotations covered
by TESS observations.

"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Get total observing times
    tess_phases = len(np.loadtxt("data/tess_phases.txt"))

    tot_obs = np.sum(np.ones(tess_phases) * 2 / 60 / 24)

    nrots = tot_obs / 1.418

    print(f"Total observing time: {tot_obs:.2f} days")
    print(f"Total observing time: {nrots:.2f} rotations")

    # 2 times waiting for 0 rotation period -- cluster into one
    # 4 times waiting for 5 rotation periods
    # 3 times waiting for not 5 or 0 rotation periods

    rotations = np.arange(nrots)
    n_5_waits = []
    N = 100000
    for i in range(N):
        rotwaits = np.diff(np.sort(np.random.choice(rotations, 8))) 
        # count how many rotwaits are multiples of 5
        n_5_waits.append(rotwaits[(rotwaits%5 > -0.01) & (rotwaits%5 < .01)  & (rotwaits > 0.01)].shape[0])

    n_5_waits = np.array(n_5_waits)

    plt.figure()

    # histogram of trials
    bins = np.linspace(0.5, 8.5, 9)
    plt.hist(n_5_waits, bins=bins, alpha=0.5, edgecolor="k", facecolor="k")

    # highlight the 4 or more times waiting for 5 rotations
    plt.axvspan(3.5, 8.5, color="peru", alpha=0.2)

    plt.xlim(0.5, 8.5)

    plt.xlabel(f"Number of times one waits for a multiple of five rotations "
            f"between flares\n(Total number of rotations covered by TESS): {nrots:.0f}")

    plt.ylabel("Number of trials")

    frac = n_5_waits[(n_5_waits > 3.99) ].shape[0] / N

    plt.title(f"Fraction of trials where one waits for a multiple of five "
            F"rotations\n between flares 4 or more times: {frac*100:.2f}%.")

    plt.tight_layout()
    plt.savefig("plots/harmonic.png", dpi=300)