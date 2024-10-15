"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script calculates the expected and observed cumulative distribution of flares
from the TESS and CHEOPS light curves, and uses an Anderson-Darling test to check
if the flares are uniformly distributed in the orbital phase of the innermost planet, 
or clustered.
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy.stats import kstest

from scipy import interpolate

import pandas as pd


from funcs.ad import sample_AD_for_custom_distribution, get_pvalue_from_AD_statistic
from funcs.helper import get_tess_orbital_phases, get_cheops_orbital_phases

# set default matplotlib fontsize to 13
plt.rcParams.update({'font.size': 13})

def get_observed_phases(p, lcs, cadences, qcs, phaseshift=0.):
    """Takes 
    
    ps, bins = get_observed_phases(np.sort(pss), lcs, [2,2,2,10./60.], [11, 38, 64,1], phaseshift=phaseshift)#

    
    Parameters:
    -----------
    mode : str
        either "Rotation" or "Orbit"
    p : array
        array of phases, for the KS-test it should be the 
        measured flare phases, for the AD test, any well 
        sampled grid of phases works
    phaseshift : float
        default = 0
        
    Return:
    -------
    pandas.DataFrame - one column per Sector or Quarter,
                       and observing time per phase in each row
    list - mids of bins for each phase row in the DataFrame
    """
    
    # bin array is two elements longer than the number of bins
    # to include 0 and 1
    bins = p 
    
    # add zero and one
    bins = np.concatenate([bins, [1]])
    bins = np.concatenate([[0], bins])
    phases = pd.DataFrame()

    for q, cadence, ph in list(zip(qcs, cadences, lcs)):
        
        counts, bins = np.histogram((ph + phaseshift) % 1, bins=bins)
            
        # get observing times for each Sector
        phases[q] = counts * cadence

    return phases, (bins[1:] + bins[:-1])/2



def get_cumulative_distributions(df, dfphases, sectors, F_k_cheops):
    """Calculate the cumulative distribution function of observed
    flares, while accounting for the number of observed flares in
    a light curve with a given detection threshold.

    Parameters:
    -----------
    df : DataFrame
        detected flares table, with a flag qcs that denotes the LC
    dfphases : DataFrame
        table that lists the observed phases, with a column for each LC
    get_secs_cadences : list
        format: [(Sector, cadence in minutes), (.., ..), (.., ..)]
        Sector is an int
        cadence is a float
    """

    # Measured number of flares in each bin
    F = dfphases.shape[0]
    n_i = np.full(F, 1)

    # Calculate cumulative frequency distribution
    cum_n_i = np.cumsum(n_i) / F

    # Expected number of flares in each bin

    # setup array for each bin
    n_exp = np.zeros_like(n_i).astype(float)

    # sum over different sectors to account for different detection thresholds
    for sec in sectors:

        obstimes_k = dfphases[sec]
        
        tot_obstimes_k = obstimes_k.sum() # total observation time in that Sector in minutes
        
        if sec == 1:
            F_k = F_k_cheops
        else:   
            F_k = df[df.qcs==sec].shape[0] # number of flares in that Sector

        n_exp_k = (obstimes_k * F_k / tot_obstimes_k).values 

        n_exp += n_exp_k

    # for debugging: 
    # calculate the old cumulative distribution that 
    # ignores the different detection thresholds
    # cum_n_exp_alt = dfphases.sum(axis=1).values.cumsum() / dfphases.sum(axis=1).values.sum()
    
    # Calculate cumulative frequency distribution
    cum_n_exp = n_exp.cumsum() / n_exp.sum()
    
    return n_i, n_exp, cum_n_exp, cum_n_i#, cum_n_exp_alt return alternative only if necessary


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

    # GET OBSERVED CHEOPS PHASES --------------------------------------------------------------

    cheops_phases, _, _, tot_obs_time_d_cheops = get_cheops_orbital_phases(period, midpoint, split=0.1)

    # ----------------------------------------------------------------------------------------

    # GET CHEOPS FLARES ----------------------------------------------------------------------

    cheopsflares = pd.read_csv("../results/cheops_flares.csv")
    cheopsflares["phase"] = ((cheopsflares.t_peak_BJD - midpoint) % period) / period

    # ----------------------------------------------------------------------------------------

    # GET OBSERVED TESS PHASES ---------------------------------------------------------------

    tess_phases = get_tess_orbital_phases(period, split=0.1, by_sector=True)

    # ----------------------------------------------------------------------------------------

    # GET TESS FLARES ------------------------------------------------------------------------

    df = pd.read_csv("../data/hip67522_tess_flares.csv")

    # ----------------------------------------------------------------------------------------



    # COMBINE ALL OBSERVED PHASES ------------------------------------------------------------

    lcs = tess_phases.copy()
    lcs.append(np.sort(cheops_phases))

    # ----------------------------------------------------------------------------------------


    # RUN AD AND KS TESTS -------------------------------------------------------------------

    pvals = []
    for i in range(100):
        # TESS FLARES
        tessps = ((df.tstart + 2457000 - midpoint + period ) % period / period ).values
        tesslc = df.qcs.values

        # CHEOPS FLARES
        cheopsps = cheopsflares["phase"].values
        cheopslc = [1,1,1,1]

        # combine the flares
        pss = np.concatenate([tessps, cheopsps])
        lcc = np.concatenate([tesslc, cheopslc])

        # sort the phases and get observed phases
        ps, bins = get_observed_phases(np.sort(pss), lcs, [2,2,2,10./60.], [11, 38, 64,1])

        # calculate the expected number of flares in the CHEOPS data based on TESS data
        F_k_cheops = 12 / ps[[11, 38, 64]].sum().sum() * ps[1].sum()

        # sum ps in each lc
        print("Obs time")
        print(ps.sum(axis=0))

        # init dataframe 
        dp = pd.DataFrame({"phases": pss, "qcs": lcc})

        # get the cumulative distributions
        n_i, n_exp, cum_n_exp, cum_n_i = get_cumulative_distributions(dp, ps, [11, 38, 64, 1], F_k_cheops=F_k_cheops)

        # plot the cumulative distribution
        plt.figure(figsize=(8,6))
        plt.plot(np.sort(pss), np.linspace(0,1,len(pss)), "k--")
        plt.savefig(f"../plots/adtest/measured_cumhist.png")

        # complete the cumulative distribution with 0 and 1 for the AD test
        p = np.sort(pss)
        p = np.insert(p,0,0)
        p = np.append(p,1)

        cum_n_exp = np.insert(cum_n_exp, 0, 0)
        cum_n_i = np.insert(cum_n_i, 0, 0)

        plt.figure(figsize=(8,6))
        plt.plot(p, cum_n_exp, "o", color="green")
        plt.savefig(f"../plots/adtest/expected_cumhist.png")

        # plt.plot(p, cum_n_i, "o", color="red")

        f = interpolate.interp1d(p, cum_n_exp, fill_value="extrapolate")
        ph = np.linspace(0,1,1000)
        plt.plot(ph, f(ph), c="green")


        N =20000
        # Make a diagnostic plot
        plt.figure(figsize=(8,6))

        # plt.plot(p,f(p), c="green")
        dsave = pd.DataFrame({"p":p, "f":f(p)})
        dsave.to_csv(f"../results/adtest/cumhist.csv",
                        index=False)
        cumsum =  np.cumsum(np.ones_like(p)) / len(p)
        plt.scatter(p, cumsum, c="r")
        plt.title(f"HIP 67522")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig(f"../plots/adtest/cumhist.png")
        # plt.close()

        # Finally, run the A-D test
        A2 = sample_AD_for_custom_distribution(f, p.shape[0], N)

        # This should go into the function above
        # select only the finite values
        print(np.isfinite(A2).sum(), A2.shape)
        A2 = A2[np.isfinite(A2)]

        plt.figure(figsize=(8,6))   
        plt.hist(A2, bins=50, histtype="step")

        # Calculate the p-value and A2 value using the distribution 
        # of A2 values
        pval, atest = get_pvalue_from_AD_statistic(p, f, A2)
        plt.axvline(atest, color="red")
        plt.savefig(f"../plots/adtest/adtest.png")
        print(pval, atest)
        pvals.append(pval)

        ks = kstest(p, f, args=(), N=1000, alternative='two-sided', mode='approx')
        print(ks)

    print("Mean and std of p-values: ")
    print(np.mean(pvals), np.std(pvals))
        

    # PLOT THE expected, measured and drawn distributions

    # read samples.npy
    samples = np.load("samples.npy")

    # plot a random subset of the samples as a cumulative histogram
    plt.figure(figsize=(8,6))
    for i in range(1000):
        plt.plot(np.sort(samples[i]), np.linspace(0,1,len(samples[i])), alpha=0.07, color="blue")
    plt.plot(np.sort(pss), np.linspace(0,1,len(pss)), "k--")

    # plot f(p) as a green line 
    f = interpolate.interp1d(p, cum_n_exp, fill_value="extrapolate")
    ph = np.linspace(0,1,1000)
    plt.plot(ph, f(ph), c="orange")

    # plot the 1-1 line
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), "k")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.ylabel("Cumulative distribution of flares")
    plt.xlabel("Orbital Phase of HIP 67522 b")

    # make legend handles with 1D line objects
    dashed_line = mlines.Line2D([], [], color='k', linestyle="--", label='Measured distribution')
    blue_line = mlines.Line2D([], [], color='blue', label='1000 draws from null distribution')
    green_line = mlines.Line2D([], [], color='orange', label='Expected distribution')
    black_line = mlines.Line2D([], [], color="k", label="1-1 line")
    plt.legend(handles=[dashed_line, blue_line, green_line], frameon=False, fontsize=13, loc=2)

    plt.savefig(f"../plots/paper/samples_cumhist.png")
