# -*- coding: utf-8 -*-
"""
@author: Ekaterina Ilin, 2025, ilin@astron.nl


Plots TESS light curves in full, marking the flares, and then
a set of zoom-ins on the flares.
"""

from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import batman

if __name__ == "__main__":

    # get flare data
    location = "results/hip67522_flares.csv"
    flares = pd.read_csv(location)

    # get stellar and planet parameters
    hip67522params = pd.read_csv("data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]

    # use batman to create a transit model
    params = batman.TransitParams()

    params.t0 = midpoint - 2457000.            #time of inferior conjunction in BTJD
    params.per = period               #orbital period
    params.rp = 0.0668                      #planet radius (in units of stellar radii)
    params.a = 11.74                       #semi-major axis (in units of stellar radii)
    params.inc = 89.46                     #orbital inclination (in degrees)
    params.ecc = 0.053                      #eccentricity
    params.w = 199.1                       #longitude of periastron (in degrees)
    params.u = [0.22, 0.27]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model


    # FULL LIGHT CURVES --------------------------------------------------------

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))

    for sector, ax in zip([11, 38, 64], axes):

        # get LC data
        hdu = fits.open(f"data/tess/tess_hip67522_{sector}.fits")

        t = hdu[1].data["TIME"]
        f = hdu[1].data["PDCSAP_FLUX"]
        ferr = hdu[1].data["PDCSAP_FLUX_ERR"]
        flag = hdu[1].data["QUALITY"]

        # mask out bad quality data
        m = ((flag == 0) &
             np.isfinite(t) &
             np.isfinite(f) &
             np.isfinite(ferr) &
             (~ np.isnan(t)) &
             (~ np.isnan(f)) & 
             (~ np.isnan(ferr)))
                                                                        
        t, f, ferr = t[m], f[m], ferr[m]

        t = t.astype(float)
        f = f.astype(float)
        ferr = ferr.astype(float)

        # plot the light curve
        ax.scatter(t, f, c="navy", s=0.1)

        # set xlims to the light curve
        ax.set_xlim(t.min(), t.max())

        # plot the flares
        for i, flare in flares.iterrows():
            if flare.t_peak_BJD - 2457000 > t.min() and flare.t_peak_BJD - 2457000 < t.max():
                ax.axvline(flare.t_peak_BJD - 2457000, color="peru", lw=1.5, zorder=-1)


        # highlight transits
        tinterpolate = np.linspace(t.min(), t.max(), 1000)
        m = batman.TransitModel(params, tinterpolate)    #initializes model
        transit = m.light_curve(params)          #calculates light curve
        # make a mask of the transit defined a everything below 1
        transit_mask = transit < 1

        # find all edges of the transit mask
        edges = np.diff(transit_mask.astype(int))
        # find the indices of the edges
        indices = np.where(edges != 0)[0]

        # loop over each two subsequent indices
        for s, f in zip(indices[::2], indices[1::2]):
            ax.axvspan(tinterpolate[s], tinterpolate[f], color="steelblue", alpha=0.3)

    
    # last subplot xlabel
    ax.set_xlabel("Time [BJD - 2457000]")

    # all subplots ylabel
    for ax in axes:
        ax.set_ylabel(r"Flux [e$^{-}$/s]")

    plt.tight_layout()

    plt.savefig("plots/paper/tess_lc.png", dpi=300)


    # PLOT ALL THE FLARES ---------------------------------------------------------------

    fig = plt.figure(figsize=(9, 12))
    flataxes = []   
    w = 2
    h= 5
    tess_flares = flares[flares.mission == "TESS"].reset_index()

    gs = fig.add_gridspec(h, 1, hspace=0.25, wspace=0.15, 
                            top=0.98, bottom=0.03, left=0.06, right=0.99)
    gss = []
    for i in range(h):
        # add subgridspec with two rows and three columns
        gss.append(gs[i].subgridspec(2, w, hspace=0.0, wspace=0.2))

    for i in range(h):
        for j in range(2):
            for k in range(w):
                flataxes.append(fig.add_subplot(gss[i][j, k]))


    flareids = [0,1,2,5,6,7,8,9,10,11]
    flareids = flareids[::-1]


    for i in [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]:
        # plot the mask_raw flux to subplot 0
        
        ax = flataxes[i]
        ax2 = flataxes[i+2]
        ind = flareids.pop()
        flare = tess_flares.loc[ind]
        tpeak = flare.t_peak_BJD - 2457000
        if tpeak < 2000.:
            sector = 11
        elif tpeak < 3000.:
            sector = 38
        elif tpeak < 4000.:
            sector = 64

        lc = pd.read_csv(f"../data/tess/HIP67522_detrended_lc_{ind}_{sector}.csv")
        ax2.scatter(lc.time, lc.flux, c="navy", s=0.2)
        ax.scatter(lc.time, lc.masked_raw_flux, c="steelblue", s=0.2) 
        ax.plot(lc.time, lc.model, c="orange", lw=0.5) 


        # highlight the transit mask
        if (lc.transit_mask == 1).any():
            transit_edge = np.where(lc.transit_mask == 1)[0][0]
            transit_end = np.where(lc.transit_mask == 1)[0][-1]

            # highlight the transit region from edge to end
            ax.axvspan(lc.time[transit_edge], lc.time[transit_end], color="steelblue", alpha=0.3)

        for a in [ax,ax2]:
            a.set_xlim(lc.time.min(), lc.time.max())

        # turn off the x-axis labels in ax
        plt.setp(ax.get_xticklabels(), visible=False)

        # set x and y-ticks
        ax2.set_xticks(np.linspace(lc.time.min(), lc.time.max(), 5))
        ax2.set_yticks(np.linspace(lc.flux.min()-150, lc.flux.max()+150, 3).round(decimals=-1))
        ax2.set_ylim(lc.flux.min()-200, lc.flux.max()+200)
        ax.set_yticks(np.linspace(lc.masked_raw_flux.min()-200, lc.masked_raw_flux.max()+200, 3).round(decimals=-2))
        ax.set_ylim(lc.masked_raw_flux.min()-400, lc.masked_raw_flux.max()+400)



        # enforce the +offset notation for the x-axis tick labels
        ax2.get_xaxis().get_major_formatter().set_useOffset(lc.time.min().round())

        if i == 16 or i == 17:
            ax2.set_xlabel("Time [BJD - 2457000]")

        if i in [4*n for n in range(5)]:
            ax.set_ylabel("Flux [e$^{-}$/s]")
            ax2.set_ylabel("Flux [e$^{-}$/s]")

    plt.savefig("plots/paper/tess_flares.png", dpi=300, bbox_inches="tight")

    # --------------------------------------------------------------------------------------------
    