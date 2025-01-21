"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

This script plots the best-fit solutions and samples from the posterior
to visualize the range of elevated flaring.

Pass number of bin edges as an argument to the script.
"""
import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# set font size to 12
plt.rcParams.update({'font.size': 11.5})

if __name__ == "__main__":

    # if plots/paper does not exist, create it
    if not os.path.exists("plots/paper"):
        os.makedirs("plots/paper")

    # read phases from file
    tess_phases = np.loadtxt("../data/tess_phases.txt")
    cheops_phases = np.loadtxt("../data/cheops_phases.txt")
    flares = pd.read_csv("../results/flare_phases_and_energies.csv")
    flares = flares.sort_values("ed_rec", ascending=True).iloc[1:] # exclude the smallest flare

    # weigh by observing cadence
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. / 60. / 60. / 24., np.ones_like(tess_phases) * 2. / 60. / 24.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])

    # flare phases
    phases = flares.phase.values

    # shift by 0.5
    obs_phases = (obs_phases + 0.5) % 1
    phases = (phases + 0.5) % 1

    # define binning
    nbins = int(os.sys.argv[1])
    bins = np.linspace(0, 1, nbins - 1)
    binmids= (bins[1:] + bins[:-1]) / 2

    # assign bin number to each observed phase
    arr = np.digitize(obs_phases, bins)

    # sum the observing times in each bin to binned weights
    # unit of entries in binned is [days]
    binned = np.array([np.sum(weights[arr==i]) for i in range(1, len(bins))]) 

    # observed numbers of flares per time bin
    hist, bins = np.histogram(phases, bins=bins)

    chainmod = pd.read_csv(f'results/modulated_samples_{nbins}.csv')
    chainunmod = pd.read_csv(f'results/unmodulated_samples_{nbins}.csv')

    bestfit = pd.read_csv(f"results/bestfit_parameters_{nbins}.csv")

    print(f"\nBest-fit parameters for modulated model with {nbins-1} bins:")
    print(bestfit)


    # modulated model
    def modulated_model(binmids, lambda0, lambda1, phase0, dphase):

        # make a mask for the elevated flare rate
        mask = (binmids > phase0) & (binmids < (phase0 + dphase)%1)

        # initialize the rate values with zeros
        result = np.zeros_like(binmids)

        result[~mask] = lambda0
        result[mask] = lambda1 

        return result # rate of observed flares per bin



    # PLOT THE BEST-FIT

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # plot the observed flares as vertical lines at their respective phases
    for phase in phases:
        ax.axvline(phase - 0.5, color='navy', alpha=0.7, lw=1, zorder=-500, linestyle=':')


    # sample from the modulated chain and plot a 100 models
    for i in range(150):
        sample = chainmod.sample(n=1)

        # fill between vertical lines  between phi0 and phi0+dphi
        vals = [sample.phi0.values[0]-0.5, (sample.phi0.values[0] + sample.dphi.values[0])-0.5]
        
        ax.fill_between(x=vals, y1=4,
                        color='steelblue', alpha=0.008, zorder=-1000)
        
        if vals[1] > 0.5:
            vals = [-.5, vals[1] - 1]
            ax.fill_between(x=vals, y1=4,
                        color='steelblue', alpha=0.008, zorder=-1000)
        elif vals[0] < -0.5:
            vals = [vals[0] + 1, 0.5]
            ax.fill_between(x=vals, y1=4,
                        color='steelblue', alpha=0.008, zorder=-1000)
        
            
    # plot the best fit model for the modulated model
    modmodel = modulated_model(binmids, bestfit.lambda0.values[1],
                            bestfit.lambda1.values[1], bestfit.phi0.values[1],
                            bestfit.dphi.values[1])
    ax.step(binmids -0.5, y=modmodel, color='grey', lw=3.5, alpha=0.7)
    ax.step(binmids -0.5, y=modmodel, color='navy', lw=2.5, alpha=0.7)


    # plot the best fit model for the unmodulated model
    unmodmodel = bestfit.lambda0_unmod.values[1] * np.ones_like(binmids)
    ax.step(binmids -0.5, y=unmodmodel , color='grey', lw=3.5, alpha=0.7)
    ax.step(binmids -0.5, y=unmodmodel , color='peru', lw=2.5, alpha=0.7)

    # add a cosine curve to show the phase of the planet
    x = np.linspace(-0.25, 0.25, 100)
    y = np.cos(2 * np.pi * x)**0.8 * 0.6

    ax.plot(x, y, c="k", zorder=-1, linestyle='-', lw=1.4, alpha=0.8)   
    ax.plot(x, y, c="orange", zorder=0, linestyle='-', lw=1.1, alpha=0.8)   

    # ADD PHASE COVERAGE ----------------------------------------------
    # add right axis with the visibility of the planet
    ax2 = plt.gca().twinx()
    ax2.set_ylabel("Phase coverage [d]")

    # plot histogram of observed phases
    ax2.step(binmids -0.5, binned, color='k', lw=.8, label='phase coverage', 
            linestyle='-', alpha=1)

    # LAYOUT ----------------------------------------------------------
    for a in [ax, ax2]:
        a.set_xlim(-.5, .5)
    ax.set_ylim(0, 1.6)
    ax2.set_ylim(0,1.4)
    ax.set_xlabel('Obital Phase of HIP 67522 b')
    ax.set_ylabel('Flare rate [d$^{-1}$]', color="navy")
    # color the ax y tick labels in navy too
    ax.tick_params(axis='y', colors='navy')


    # put a text next to peru line saying unmodulated flare rate
    ax.text(-0.48, 0.235, 'flare rate without interaction', color='k', fontsize=10.5)
    ax.text(-0.48, 0.12, 'flare rate with interaction', color='k', fontsize=10.5)

    # replace x-tick labels with -.5, -0.25, 0, 0.25, 0.5
    plt.xticks([-0.5, -0.25, 0, 0.25, 0.5], ['-0.5', '-0.25', '0', '0.25', '0.5'])

    # plt.legend(handles=handles, loc='upper right', frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/paper/mcmc.png', dpi=300)