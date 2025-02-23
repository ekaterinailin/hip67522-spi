"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

This script generates corner plots and flare rate posteriors 
for the modulated and unmodulated flare rate models. Run script
10_ to generate the mcmc samples for the same bin edge number first.
"""

import numpy as np 
import matplotlib.pyplot as plt
import corner
import pandas as pd

# set font size to 12
plt.rcParams.update({'font.size': 14})


if __name__ == "__main___":

    # number of bin edges
    n = 101

    # mcmc samples 
    samples = pd.read_csv(f"results/modulated_samples_{n}.csv")
    samples_unmod = pd.read_csv(f"results/unmodulated_samples_{n}.csv")

    # parameter labels
    labels = [r"$\lambda_0$", r"$\lambda_1$", r"$\phi_0$",r"$\Delta\phi$",]

    # CORNER plots
    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], 
                    show_titles=False, title_kwargs={"fontsize": 12}, figsize=(6,6))  
    plt.tight_layout()
    plt.savefig(f"plots/corner_plot_{n}.png", dpi=300)

    # flare rate posteriors
    plt.figure()
    bins = np.linspace(0, 3, 100)

    # unmodulated base rate
    plt.hist(samples_unmod,bins=bins, histtype='step', label='$\lambda_{0,unmod}$ - unmodulated flare rate',
            color='navy', hatch='//', density=True, linewidth=1.5, alpha=0.7)

    # modulated base and elevated flare rates
    plt.hist(samples["l0"],bins=bins, histtype='step', label='$\lambda_{0,mod}$ - modulated base flare rate',
            color='peru', hatch='//', density=True, linewidth=1.5, alpha=0.7)
    plt.hist(samples["l1"], bins=bins, histtype='step', label='$\lambda_{1,mod}$ - modulated elevated flare rate',
            color='peru', hatch=r'||', density=True, linewidth=1.5, alpha=0.7)
    
    # layout
    plt.xlim(0,2)
    plt.xlabel(r"Flare rate [1/day]")
    plt.ylabel("Density")
    plt.legend(loc=1, fontsize=12, frameon=False)
    plt.savefig(f"plots/flare_rate_comparison_{n}.png", dpi=300)