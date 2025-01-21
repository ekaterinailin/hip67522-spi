"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

Double check that the solution in various bins are consistent
with the quoted number (99 bins)
"""

import pandas as pd 
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # read AIC values
    aics = pd.read_csv('results/bestfit_parameters_aic.txt', header=None,
                        names=["nbinedges", "AICmod", "AICunmod","deltaAIC","logmod","logunmod",
                                "l0mod","l1","phi0","dphi","l0unmod"])

    # take the value where AIC and K have converged
    aics = aics[aics.nbinedges > 75]

    # calculate the center of the phase bin, too
    aics["center_phase"] = aics["phi0"] + aics["dphi"]/2

    # make histograms

    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(18, 4))

    axes = list(axes.flatten())

    colnames = {"l0mod" : "$\lambda_0$ (mod) [d$^{-1}$]", 
                "l1" : "$\lambda_1$ [d$^{-1}$]", 
                "phi0":"$\phi_0$", 
                "center_phase":"mid phase",
                "dphi":"$\Delta\phi$", 
                "l0unmod":"$\lambda_0$ (unmod) [d$^{-1}$]"}

    for col in ["l0mod","l1","center_phase", "phi0", "dphi","l0unmod"]:
        ax = axes.pop(0)
        ax.hist(aics[col], bins=20, color="steelblue", alpha=0.7)
        ax.set_xlabel(colnames[col])
        print(f"{colnames[col]}: {aics[col].mean():.3f} +/- {aics[col].std():.3f}")

    plt.savefig("plots/diagnostic/fit_parameter_histograms.png", dpi=300)
    