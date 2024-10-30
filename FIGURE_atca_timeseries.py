
"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

Read atca time series and plot the flux density vs orbital and rotational phase,
as well as into a grid of subplots for each observing run.

The file names of the resulting plots are:
- phase_flux_density.png
- orbital_phase_flux_density_grid.png

"""

import pandas as pd
import matplotlib.pyplot as plt


# set default font size in matplotlib
plt.rcParams.update({'font.size': 12})

colors = [
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
        ]*2

if __name__ == "__main__":

    # read in both data sets
    df = pd.read_csv('../data/atca_all_timeseries.csv')
    full_integration_fluxes = pd.read_csv('../data/atca_full_integration_time_series.csv')

    # PLOT THE FLUX DENSITY VS ORBITAL AND ROTATIONAL PHASE --------------------------------------------

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    versions = [("Orbital phase of HIP 67522 b", "phase", axs[0]), 
                ("Rotational phase of HIP 67522", "rot_phase", axs[1])] 


    for version, phase, ax in versions:
        # plot the flux density vs time
        

        for obsname, g in df.groupby('obsname'):
            
            # insert slashes after YYYY/MM/DD
            obsname = str(obsname)
            obsname = obsname[:4] + "/" + obsname[4:6] + "/" + obsname[6:]

            c = colors.pop()

            g1 = g[g['source_J_val']]
            ax.errorbar(g1[phase], g1['source_J'], yerr=g1['bkg_rms_J'], fmt='o', c=c, label=obsname)
            g2 = g[~g['source_J_val']]
            ax.errorbar(g2[phase], g2['source_J'], yerr=g2['bkg_rms_J'], fmt='.', color=c, uplims=True, alpha=0.3)

        # plot the full integration fluxes
        f2 = full_integration_fluxes[~full_integration_fluxes['source_J_val']]
        ax.errorbar(f2[phase], f2['source_J'], yerr=f2['bkg_rms_J'], fmt='v', color="grey", markersize=10)
            
        ax.set_xlabel(version)
        ax.set_ylabel('Flux Density [Jy]')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1.6e-3)
        if version == "Rotational phase of HIP 67522":
            ax.legend(ncol=2, frameon=False, loc=2, fontsize=10)


    plt.tight_layout()
    plt.xlim(0.,1)

    plt.savefig(f'../plots/paper/phase_flux_density.png', dpi=300)

    # ---------------------------------------------------------------------------------------------------
    

    # PLOT ALL THE OBSERVATIONS IN A GRID ---------------------------------------------------------------

    # set up a vertical 3 x 5 grid of subplots
    fig, axs = plt.subplots(5, 2, figsize=(21/2, 29.7/2), sharey=True)
    axs = axs.flatten()

    # loop over each observation and plot the flux density vs orbital phase
    for i, (obsname, group) in enumerate(df.groupby('obsname')):

        # separate detections from non-detections
        ax = axs[i]
        
        g1 = group[group["source_J_val"]]
        ax.errorbar(g1['phase'], g1['source_J'], yerr=g1['bkg_rms_J'], fmt='o', c="blue")
        
        g2 = group[~group["source_J_val"]]
        ax.errorbar(g2['phase'], 3 * g2['bkg_rms_J'], yerr=g2['bkg_rms_J'], fmt='o', color="grey", uplims=True)
        
        ax.set_title(group["date"].iloc[0], fontsize=12)

        
    # set x-labels for the bottom row
    for ax in axs[-2:]:
        ax.set_xlabel('Orbital Phase of HIP 67522 b')

    # set y-labels for the left column
    for ax in axs[::2]:
        ax.set_ylabel('Flux Density [Jy]')

    plt.tight_layout()

    # reduce vertical spacing between subplots
    plt.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(wspace=0.06)

    plt.savefig(f'../plots/paper/orbital_phase_flux_density_grid.png', dpi=300)

    # ---------------------------------------------------------------------------------------------------