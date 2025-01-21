# -*- coding: utf-8 -*-
"""
@author: Ekaterina Ilin, 2024, ilin@astron.nl

This script is used to plot the raw and detrended CHEOPS 
light curves of HIP67522, highlighting the flares.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


flux_label = r"Flux [e$^{-}$/s]"
time_label = "Time [BJD]"

# make font size larger
plt.rcParams.update({'font.size': 12})


if __name__ == "__main__":

    # load the file names
    files = np.loadtxt("data/cheops_files.txt", dtype=str)

    # read the flare table
    flares = pd.read_csv("results/cheops_flares.csv")

    # read in all the de-trended light curves
    dlcs = []
    for pi, file in files:
        location = f"results/cheops/HIP67522_{file}{pi}_detrended_lc.csv"
        # df = pd.read_csv(location)
        # df = df[df["flag"] == 0]
        dlcs.append(pd.read_csv(location))  
    
    # make a list of the first time stamp in each lc in lcs and sort the lcs by this time
    t0s = [lc["time"].min() for lc in dlcs]
    dlcs = [dlcs[i] for i in np.argsort(t0s)]

    print("Number of light curves: ", len(dlcs))


    # MERGE ALL THE LIGHT CURVES INTO ONE DATA FRAME for later use --------------------------------------------

    # combine all the dlcs into one data frame, adding the file+pi as identifier column
    for i, dlc in enumerate(dlcs):
        dlcs[i]["file"] = files[i][0] + files[i][1]

    # concatenate the dlcs
    dlcpd = pd.concat(dlcs)

    # save the combined dlcs to a csv file
    dlcpd.to_csv("../data/cheops_all_timeseries.csv", index=False)

    # ----------------------------------------------------------------------------------------------------------

    fig = plt.figure(figsize=(17, 23))
    flataxes = []   

    gs = fig.add_gridspec(7, 1, hspace=0.25, wspace=0.3, 
                          top=0.98, bottom=0.03, left=0.06, right=0.99)
    gss = []
    for i in range(7):
        # add subgridspec with two rows and three columns
        gss.append(gs[i].subgridspec(2, 3, hspace=0.0, wspace=0.2))

    for i in range(7):
        for j in range(2):
            for k in range(3):
                flataxes.append(fig.add_subplot(gss[i][j, k]))

    # Loop over the flattened axes array (14 rows, 3 columns)
    for i in [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38]:

        # Determine the corresponding original index (0-based index of the original grid)
        original_idx = (i//2) + i%2   # Every two axes correspond to one original subplot 36 + x = 18,  37 + x = 19, 38 + x = 20
        if i%3 ==2:
            original_idx = original_idx + 1


        idup, iddown = i, i+3
        ax1, ax2 = flataxes[idup], flataxes[iddown]

        print(original_idx, idup, iddown)
        ax2.set_xlim(dlcs[original_idx]["time"].min(), dlcs[original_idx]["time"].max())
        ax1.set_xticklabels([])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False)
        ax1.set_xlim(ax2.get_xlim())

        
        flataxes[idup].scatter(dlcs[original_idx]["time"], dlcs[original_idx]["masked_raw_flux"] , s=1, color="steelblue", label="CHEOPS")
        flataxes[idup].plot(dlcs[original_idx]["time"], dlcs[original_idx]["model"], color="orange", label="Model")

        # mark transits with a grey axvspan if in the transit_mask range
        mask = np.where(dlcs[original_idx]["transit_mask"].values==True)[0]
        
        if len(mask) > 0:
            print(mask[0], mask[-1])
            mintrans, maxtrans = dlcs[original_idx].iloc[mask[0]], dlcs[original_idx].iloc[mask[-1]]
            print(mintrans["time"], maxtrans["time"])
            flataxes[idup].axvspan(mintrans["time"], maxtrans["time"], color="steelblue", alpha=0.3, 
                                   zorder=-10)
            # Plot the second "bottom" subplot (even rows in the new grid)
    
        flataxes[iddown].scatter(dlcs[original_idx]["time"], dlcs[original_idx]["flux"], s=1, color="navy", label="CHEOPS")
        
    
        # Mark flares if they are within the time range of the current light curve
        for flare in flares[(flares["tmin"] > dlcs[original_idx]["time"].min()) & 
                            (flares["tmax"] < dlcs[original_idx]["time"].max())].iterrows():
            flataxes[iddown].axvspan(flare[1]["tmin"], flare[1]["tmax"], color="peru", alpha=0.3, zorder=-10)


    # # only set y label for the first column
    for i in range(0, 42, 3):
        flataxes[i].set_ylabel(flux_label)

    # # only set x label for the last row
    for ax in flataxes[-3:]:
        ax.set_xlabel(time_label)


    plt.savefig("../plots/paper/cheops_lc.png", dpi=300)
