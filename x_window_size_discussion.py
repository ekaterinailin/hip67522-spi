"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

Generate a random power-law distribution of flares and illustrate the effect 
of different detection thresholds on the flare rate per orbital phase bin.

Also shows the effect of small numbers on the apparent window size, and how
limited observing duration can affect the window size.

"""

import numpy as np 
import matplotlib.pyplot as plt

# set fontsize to 12
plt.rcParams.update({'font.size': 12})

def generate_random_power_law_distribution(a, b, g, size=1, seed=None):
    """Power-law generator for pdf(x)\propto x^{g-1}
    for a<=x<=b
    """
    if seed is not None:
        np.random.seed(seed)
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r)**(1. / g)


def get_flares(pl, emin, emax=None):
    
    if emax is not None:
        pl = pl[pl < emax] 
    phases = np.random.uniform(-.5, 0.5, size=pl.size)
    attenuation = np.cos(2* np.pi * phases)
    attenuation[(phases > 0.25) | (phases < -0.25)] = np.nan
    attenuation[(pl * attenuation) < emin] = np.nan
    
    return attenuation, pl, phases

if __name__ == "__main__":

    g = -0.6 # gives -1.6 slope of power law
    N = 40000 # number of generated flares

    # generate random power law distribution from a wide range of energies
    pl = generate_random_power_law_distribution(1e30, 1e45, g, size=N, seed=437)

    # set the color scheme
    colors = ["navy", "steelblue","peru"]

    # plot the histogram of the generated flares
    plt.figure()

    # select a random threshold for observing duration
    plt.axhline(10, color="black", lw=0.5, linestyle="--", 
                label="Example observing time per phase bin")

    # plot the histogram of the generated flares for different detection thresholds
    for i, emin in enumerate([1e32, 1e33, 1e34]):
        attenuation, npl, phases = get_flares(pl, emin)
        phases_sel = phases[~np.isnan(attenuation)]
        plt.hist(phases_sel, bins=np.linspace(-0.5, 0.5, 40), histtype="step", 
                edgecolor=colors[i], label=fr"$\log_{{10}}E_{{\rm min}}$={np.log10(emin):.0f}")
        
    # layout
    plt.legend(frameon=False, loc=2)   
    plt.xlabel("Orbital phase")
    plt.ylabel("Flare rate per orbital phase bin")
    plt.ylim(0,100)
    plt.tight_layout()
    plt.savefig("plots/window_size_illustration.png", dpi=300)


