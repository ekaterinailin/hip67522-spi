"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

Take the Bayes Factor and AIC output tables and plot the results.
Get some diagnostics.
"""

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # read ../results/bayes_factor.txt
    df = pd.read_csv('results/bayes_factor.txt', sep=',', header=None, #skiprows=32, 
                    names = ["nflares","nbinedges","maxlambda0","maxlambda1","l0gridsize",
                            "l1gridsize","phi0gridsize","dphigridsize","K", "marglnLmod", 
                            "marglnLunmod", "minlambda0","minlambda1"] )
    # read AIC values
    aics = pd.read_csv('results/bestfit_parameters_aic.txt', header=None,
                       names=["nbinedges", "AICmod", "AICunmod","deltaAIC","logmod","logunmod",
                              "l0mod","l1","phi0","dphi","l0unmod"])

    df = df[df["nbinedges"] > 50]
    aics = aics[aics["nbinedges"] > 50]

    print("\nBayes Factor statistics:")
    print(df["K"].describe())

    print("\nAkaike Information Criterion statistics:")
    print(aics["deltaAIC"].describe())


    # MAKE A PLOT OF BOTH
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,5.5), sharex=True)

    ax1.scatter(df['nbinedges'] - 1, df['K'], s=10, marker='x', color="navy")
    ax1.set_ylabel(r'$K$')
    ax1.set_ylim(0,19)

    ax2.scatter(aics["nbinedges"] - 1, aics["deltaAIC"], s=10, marker='x', color="navy")
    ax2.set_ylabel(r'$\Delta$ AIC')
    ax2.set_ylim(-12,0)

    plt.xlabel('Number of bins')

    # remove space between subplots
    plt.tight_layout()

    # reduce space between subplot to 0
    plt.subplots_adjust(hspace=0)

    # remove the 0 tick from ax1 y-axis
    ax1.yaxis.get_major_ticks()[0].label1.set_visible(False)

    plt.savefig('plots/paper/aic_and_bayes_factor.png', dpi=300)

    