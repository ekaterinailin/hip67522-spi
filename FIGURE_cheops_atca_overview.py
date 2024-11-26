
"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl


Plot the ATCA and CHEOPS light curves of HIP 67522.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# set font size to 13
plt.rcParams.update({'font.size': 13})

if __name__ == "__main__":

    # read the light curves -----------------------------------------------------

    atca = pd.read_csv("../data/atca_all_timeseries.csv")
    cheops = pd.read_csv("../data/cheops_all_timeseries.csv")

    # plot the light curves -----------------------------------------------------
    
    # set up the figure
    fig, ax1 = plt.subplots(figsize=(16,4))
    ax2 = ax1.twinx()

    # plot both data sets
    ax1.scatter(atca.jd, atca.source_J*1e3, s=10, c="blue")
    ax2.scatter(cheops.time, cheops.masked_raw_flux, s=1, c="olive")

    # legend
    custom_lines = [Line2D([0], [0], color="blue", lw=2),
                    Line2D([0], [0], color="olive", marker="o", lw=2),
                    ]
    ax1.legend(custom_lines, ['2 GHz radio - ATCA', 'broadband optical - CHEOPS'], loc=(0.3, 0.75),
            frameon = False)

    # layout
    ax2.set_ylabel(r"CHEOPS flux [e$^{-}$/s]")
    ax1.set_xlabel("Time [JD]")
    ax1.set_ylabel("ATCA L band Stokes I [mJy]")
    plt.tight_layout()

    # save the figure
    plt.savefig("../plots/paper/atca_cheops_timeseries.png", dpi=300)