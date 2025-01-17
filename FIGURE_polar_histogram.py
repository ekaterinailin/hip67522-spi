# -*- coding: utf-8 -*-
"""
@author: Ekaterina Ilin, 2025, ilin@astron.nl

Produces the polar histogram of the flare rate on HIP 67522.
Includes star and planet, and the expected flare rate, which is:

flare_rate(longitude) = flare_rate(longitude = 0) * cos(longitude)^(alpha - 1)

where alpha is the power law index of the flare rate distribution. We choose
alpha=1.8 from the fit, which is consistent with solar and stellar observations.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

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

    # read phases from file
    tess_phases = np.loadtxt("../data/tess_phases.txt")
    cheops_phases = np.loadtxt("../data/cheops_phases.txt")
    flares = pd.read_csv("../results/hip67522_flares.csv")

    # weigh by observing cadence
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. / 60. / 60. / 24., 
                                np.ones_like(tess_phases) * 2. / 60. / 24.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])

    flares = flares.sort_values(by="mean_bol_energy", ascending=True).iloc[1:]
    phases = flares.orb_phase.values


    # BIN THE DATA ---------------------------------------------------------------------------
    nbins = 31
    bins = np.linspace(0, 1, nbins) 
    binmids= (bins[1:] + bins[:-1]) / 2

    # bin the phases
    arr = np.digitize(obs_phases, bins)

    # sum the observing times in each bin to binned weights
    # unit of entries in binned is [days]
    binned = np.array([np.sum(weights[arr==i]) for i in range(1, len(bins))]) 

    hist, bins = np.histogram(phases, bins=bins)

    # ----------------------------------------------------------------------------------------

    # MAKE A POLAR HISTORGRAM ----------------------------------------------------------------

    # set up the figure ----------------------------------------------------------
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # make the axes grey
    ax.spines['polar'].set_color('grey')

    # set the zero to the top
    ax.set_theta_zero_location('E')

    # set the direction to clockwise
    ax.set_theta_direction(1)
    ax.set_rlabel_position(118)

    # add the grid
    ax.grid(zorder=1, linestyle='--', color='gray')

    # Change the color of the radial (r-axis) labels only
    for label in ax.get_yaxis().get_ticklabels():
        label.set_color('navy')  # Set radial labels color to red
        label.set_fontsize(11)  # Optionally adjust the font size
        label.set_fontweight('bold')  # Optionally adjust the font weight

    # set the radius limits
    ax.set_ylim(0, 1.2)
    # ------------------------------------------------------------------------------

    # plot the histogram of flare rates --------------------------------------------
    ax.bar(binmids*2*np.pi , width=np.pi*2/nbins, height=hist/binned,  
        facecolor='steelblue', alpha=0.8, edgecolor='navy', linewidth=2)

    # annotate the histogram
    ax.text(0.2, 1.03, r'Flare rate [d$^{-1}$]', fontsize=12, rotation=0, 
            ha='center', va='center', transform=ax.transAxes, color="navy", fontweight='bold')

    # plot the star
    circle = Circle((0, 0), radius=0.12, transform=ax.transData._b, 
                    facecolor='orange', alpha=0.9, zorder=-1, edgecolor='black')
    ax.add_artist(circle)

    # plot the expected flare rate
    x = np.linspace(-np.pi/2, np.pi/2, 100)
    # the exponent accounts for the flare rate being changed to the power of alpha - 1
    y = np.cos(x)**0.8 * 1.2 
    ax.plot(x, y, color='orange', linewidth=2, linestyle='--', alpha=0.8)
    ax.fill_between(x, y, 0, color='orange', alpha=0.2)

    # ------------------------------------------------------------------------------

    # annotate the figure ----------------------------------------------------------

    # add a new axis for the annotations
    # The new axes will slightly overlap and be 20% larger
    ax_larger = fig.add_axes([0.25, 0.0, 1, 1], frameon=False) 
    # remove axes
    ax_larger.yaxis.set_visible(False)
    ax_larger.xaxis.set_visible(False)
    ax_larger.set_xlim(0,3)
    ax_larger.set_ylim(0,1)


    # arrow to the observer
    ax_larger.arrow(2.2, 0.5, 0.5, 0, head_width=0.02, head_length=0.05, 
                    fc='grey', ec='steelblue', linewidth=1)
    ax_larger.annotate('Observer', xy=(2.5, 0.6), xytext=(2.4, 0.515), 
                    fontsize=12, ha='center', color="navy",zorder=10)


    # Ccurved arrow around the polar plot
    arrow = FancyArrowPatch(
        (2.1, 0.6),             # Arrow start (on the edge of the circle)
        (1.9, 0.75),             # Arrow end (90 degrees from the start point)
        mutation_scale=2,  # Scale of the arrow
        color='steelblue',        # Color of the arrowhead
        alpha=0.8,          # Transparency of the arrow
        linewidth=1,        # Line thickness
        arrowstyle='Simple,head_width=4,head_length=4',    # Arrow style
        connectionstyle="arc3,rad=.1"  # Curved connection for the arrow
    )

    # Add the arrow to the plot
    ax_larger.add_patch(arrow)

    # annotate the arrow
    ax_larger.annotate(r'HIP 67522 b', xy=(2.3, 0.57), xytext=(2.08, 0.67), fontsize=12, ha='left', color="navy")

    # add planet as a small circle -- to scale!
    circle = Circle((1.357 ,0.352), radius=0.12* 0.0668, transform=ax.transData._b, facecolor='black', alpha=1, zorder=1, edgecolor='black')
    ax_larger.add_artist(circle)

    # annotate expected flare rate
    ax_larger.annotate(r'Expected flare rate', xy=(2.1, .2), xytext=(2.4, 0.4), fontsize=12, ha='center', color="peru")



    plt.savefig("../plots/paper/polar.png", dpi=300, bbox_inches='tight')
