"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script defines Poisson models for the flare rate 
of HIP 67522 and fits them to the observed flare data.

MCMC chains, corner plots, and best fit parameters are plotted.

We also calculate the Bayesian Information Criterion (BIC),
the Akaike Information Criterion (AIC), and the chi-squared/dof
values for the best fit solutions.
"""


import numpy as np 
import pandas as pd

import emcee
import corner
from math import factorial
from scipy.optimize import minimize

import matplotlib.pyplot as plt 

if __name__ == "__main__":


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
    nbins = 201
    bins = np.linspace(0, 1, nbins)
    binmids= (bins[1:] + bins[:-1]) / 2

    # assign bin number to each observed phase
    arr = np.digitize(obs_phases, bins)

    # sum the observing times in each bin to binned weights
    # unit of entries in binned is [days]
    binned = np.array([np.sum(weights[arr==i]) for i in range(1, len(bins))]) 

    # define maximum flare rate we could possibly accept based on observations
    max_lambda0 = 2.
    max_lambda1 = 2.

    print(f"Prior on lambda0: Jeffrey's in [0, {max_lambda0}]")
    print(f"Prior on lambda1: Jeffrey's in [0, {max_lambda1}]")

    # observed numbers of flares per time bin
    hist, bins = np.histogram(phases, bins=bins)

    # define the factorials for the numbers in hist for the likelihood computation
    factorials = np.array([factorial(h) for h in hist])

    print(f"Number of flares: {len(phases)}")
    print(f"Number of bins: {len(binned)}")
    print(f"Sum of observing times: {np.sum(binned)}")

    # make sure you didn't mess up the data somewhere else somehow
    assert len(phases) == 15
    assert 73 < np.sum(binned) < 74
    assert len(binned) == nbins - 1

    print("Data loaded and preprocessed successfully.")

    # Poisson statistics ---------------------------------------------------------
    # define the two models we want to compare

    # modulated model
    def modulated_model(binmids, lambda0, lambda1, phase0, dphase, weight=binned):

        # make a mask for the elevated flare rate
        mask = (binmids > phase0) & (binmids < (phase0 + dphase)%1)

        # initialize the rate values with zeros
        result = np.zeros_like(binmids)

        # multiply by weight to get lambda_0*t_i and lambda_1*t_i
        result[~mask] = lambda0 * weight[~mask]
        result[mask] = lambda1 * weight[mask]

        return result # rate of observed flares per bin

    # unmodulated model
    def unmodulated_model(lambda0, weight=binned):
        return lambda0 * weight # rate of observed flares per bin

    # ----------------------------------------------------------------------------

    # Poisson log-likelihood function
    def log_likelihood_poisson(rate, hist, factorials):
        # Poisson log-likelihood for each bin
        # rate = lambda_0 * t_i or lambda_1 * t_i
        # hist = n_i
        # factorials = n_i!
        logs = -rate - np.log(factorials) + np.log(rate) * hist

        # sum the log-likelihoods for all bins
        return np.sum(logs)

    # log-likelihood for the modulated model
    def log_likelihood_mod(params):

        # get lambda_0 * t_i and lambda_1 * t_i
        rate = modulated_model(binmids, *params, weight=binned)

        # return the log-likelihood
        return log_likelihood_poisson(rate, hist, factorials)


    # minimize neg. log-likelihood for a starting point in the mcmc
    params =  [0.1, .7,  0.30, 0.07]
    logmin = lambda x: -log_likelihood_mod(x)
    res = minimize(logmin, params)

    # plt.plot(binmids, modulated_model(binmids, *res.x, weight=binned), label="model")
    # plt.plot(binmids, hist, label="observed")
    # plt.xlim(0, 1)
    # plt.legend(frameon=False)   



    # define log-prior with Jeffrey's priors on Poisson rates
    def log_prior_mod(params):
        if ((params[0] > 0) & (params[1] > 0) & 
            (params[1] < max_lambda1) & ( params[0] < max_lambda0) &
            (params[2] > 0) & (params[2] < 1) & (params[3] > 0) & (params[3] < 1) ):
            p1 = np.log(1/np.sqrt(params[0]) / 2 / np.sqrt(max_lambda0)) 
            p2 = np.log(1/np.sqrt(params[1]) / 2 / np.sqrt(max_lambda1)) 
            return p1 + p2
        return -np.inf

    # define log-probability
    def log_probability_mod(params):
        lp = log_prior_mod(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_mod(params)

    # ---------------------------------------------------------------------------
    # MCMC MODULATED MODEL ------------------------------------------------------
    ndim = 4
    nwalkers = 32
    pos = params + 1e-4 * np.random.randn(nwalkers, ndim)

    mod_sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_mod)
    mod_sampler.run_mcmc(pos, 50000, progress=True)

    # ---------------------------------------------------------------------------



    # ----------------------------------------------------------------------------
    # MCMC CHAIN PLOTS -----------------------------------------------------------

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    mod_samples = mod_sampler.get_chain(discard=10000)
    labels = [r"$\lambda_0$", r"$\lambda_1$", r"$\phi_0$",r"$\Delta\phi$",]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(mod_samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(mod_samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.05, 0.5)

    axes[-1].set_xlabel("step number")

    plt.tight_layout()
    plt.savefig(f"../plots/poisson_model/{nbins}_chain_modulated.png", dpi=300)

    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # MCMC CORNER PLOT -----------------------------------------------------------

    # plot corner
    mod_flat_samples = mod_sampler.get_chain(discard=10000, thin=15, flat=True)

    # increase font size to 20
    plt.rcParams.update({'font.size': 18})

    fig = corner.corner(mod_flat_samples, labels=labels)
    plt.tight_layout()

    plt.savefig(f"../plots/poisson_model/{nbins}_corner_modulated.png", dpi=300)

    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # GET BEST FIT PARAMETERS ----------------------------------------------------

    # calculate the 16th, 50th and 84th percentiles for all parameters
    lambda0_mcmc = np.percentile(mod_flat_samples[:, 0], [16, 50, 84])
    lambda1_mcmc = np.percentile(mod_flat_samples[:, 1], [16, 50, 84])
    phi0_mcmc = np.percentile(mod_flat_samples[:, 2], [16, 50, 84])
    dphi_mcmc = np.percentile(mod_flat_samples[:, 3], [16, 50, 84])

    print("lambda0:", lambda0_mcmc[1], "+", lambda0_mcmc[2] - lambda0_mcmc[1], "-", lambda0_mcmc[1] - lambda0_mcmc[0])
    print("lambda1:", lambda1_mcmc[1], "+", lambda1_mcmc[2] - lambda1_mcmc[1], "-", lambda1_mcmc[1] - lambda1_mcmc[0])
    print("phi0:", phi0_mcmc[1], "+", phi0_mcmc[2] - phi0_mcmc[1], "-", phi0_mcmc[1] - phi0_mcmc[0])
    print("dphi:", dphi_mcmc[1], "+", dphi_mcmc[2] - dphi_mcmc[1], "-", dphi_mcmc[1] - dphi_mcmc[0])

    mod_best_median = [lambda0_mcmc[1], lambda1_mcmc[1], phi0_mcmc[1], dphi_mcmc[1]]

    # put in pandas dataframe with 16th, 50th and 84th percentiles as indices
    df = pd.DataFrame({"lambda0": lambda0_mcmc, "lambda1": lambda1_mcmc, "phi0": phi0_mcmc, "dphi": dphi_mcmc})
    df.index = ["16th", "50th", "84th"]

    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # Log L, PRIOR, POSTERIOR UNMODULATED MODEL ----------------------------------

    # define log-likelihood, prior, and probability
    def log_likelihood_unmod(params):
        lambda0 = params[0]

        # get lambda_0 * t_i
        rate = unmodulated_model(lambda0, weight=binned)

        # return the log-likelihood
        return log_likelihood_poisson(rate, hist, factorials)
    
    def log_prior_unmod(params):
        if ((params[0] > 0) & (params[0] < max_lambda0)):
            return np.log(1 / np.sqrt(params[0]) / 2 / np.sqrt(max_lambda0))
        return -np.inf

    def log_probability_unmod(params):
        lp = log_prior_unmod(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_unmod(params)
    
    # ---------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # MCMC -----------------------------------------------------------------------
    ndim = 1
    nwalkers = 32
    pos = [0.5] + 1e-4 * np.random.randn(nwalkers, ndim)

    unmod_sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_unmod)
    unmod_sampler.run_mcmc(pos, 10000, progress=True)

    # ---------------------------------------------------------------------------
    # PLOT CHAIN ----------------------------------------------------------------

    # plot chains
    fig, axes = plt.subplots(ndim, figsize=(10, 3), sharex=True)
    unmod_samples = unmod_sampler.get_chain()
    unmod_label = labels[0]
    axes.plot(unmod_samples[:,:,  0], "k", alpha=0.3)
    axes.set_xlim(0, len(unmod_samples))
    axes.set_ylabel(unmod_label)
    axes.yaxis.set_label_coords(-0.05, 0.5)
    axes.set_xlabel("step number")
    plt.tight_layout()
    plt.savefig(f"../plots/poisson_model/{nbins}_chain_unmodulated.png", dpi=300)

    # ---------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # GET BEST FIT PARAMETERS ----------------------------------------------------

    # Unmodulated model best_fit result and histogram
    unmod_flat_samples = unmod_sampler.get_chain(discard=4000, thin=15, flat=True)

    # calculate the 16th, 50th and 84th percentiles for all parameters
    lambda0_mcmc = np.percentile(unmod_flat_samples[:, 0], [16, 50, 84])

    unmod_best_median = [lambda0_mcmc[1]]

    print("lambda0:", lambda0_mcmc[1] ,lambda0_mcmc[2]-lambda0_mcmc[1],  lambda0_mcmc[1]-lambda0_mcmc[0])

    # save to file
    df["lambda0_unmod"] = lambda0_mcmc
    df.to_csv(f"../results/bestfit_parameters.csv")

    # ----------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------
    # CORNER PLOT ----------------------------------------------------------------

    # corner plot
    fig = corner.corner(unmod_flat_samples, labels=[unmod_label])
    plt.tight_layout()
    plt.savefig(f"../plots/poisson_model/{nbins}_corner_unmodulated.png", dpi=300)

    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # PLOT BOTH MODELS -----------------------------------------------------------

    # make the fontsize small again
    plt.rcParams.update({'font.size': 12})

    plt.figure(figsize=(6, 5))
    plt.plot(binmids, unmodulated_model(*unmod_best_median), 
             label=r"$M_{\text{unmod}}$ - without clustering", c="black", linestyle=":")
    plt.plot(binmids, modulated_model(binmids, *mod_best_median), 
             label=r"$M_{\text{mod}}$ - with clustering", c="black", linestyle="--")
    plt.plot(binmids, hist, label="observed", c="magenta", zorder=-10)
    plt.legend(frameon=False, loc=2)
    plt.xlabel("Orbital phase")
    plt.xlim(0,1)
    plt.ylabel("Number of flares")
    plt.tight_layout()
    plt.savefig(f"../plots/poisson_model/{nbins}_best_median_fit.png", dpi=300)

    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # CHI-SQUARE -----------------------------------------------------------------

    # compute the chisq for each of the best fit solutions
    chisq_mod = np.sum((modulated_model(binmids, *mod_best_median) - hist)**2 / 
                        modulated_model(binmids, *mod_best_median)) 
    chisq_unmod = np.sum((unmodulated_model(*unmod_best_median) - hist)**2 / 
                          unmodulated_model(binmids, *unmod_best_median))

    # divide by degrees of freedom
    dof = len(hist) - 4
    chisq_mod /= dof
    chisq_unmod /= dof

    # n bins, chisq/dof_mod, chisq/dof_unmod
    # 30: (0.864040631852652, 14.331350204160442)
    # 200: (0.6923462715743284, 1.0254345758724361)
    # 400: (0.7665232907316073, 0.5100476296357096)

    print(f"Chisq/dof modulated: {chisq_mod}")
    print(f"Chisq/dof unmodulated: {chisq_unmod}")

    # -----------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------
    # BIC AND AIC -----------------------------------------------------------------------

    # get the log likelihoods for the best fit solutions
    logmod, logunmod = log_likelihood_mod(mod_best_median), log_likelihood_unmod(unmod_best_median)

    BICmod = -2 * logmod + 4 * np.log(len(hist))
    BICunmod = -2 * logunmod + 2 * np.log(len(hist))


    AICmod = -2 * logmod + 2 * 4
    AICunmod = -2 * logunmod + 2 * 1


    AICmod, AICunmod, BICmod, BICunmod, np.exp((AICmod - AICunmod) / 2), logmod, logunmod

    print(f"AIC modulated: {AICmod}")
    print(f"AIC unmodulated: {AICunmod}")
    if AICmod < AICunmod:
        print("\nModulated model is preferred!\n")
        print(f"Delta AIC: {AICunmod - AICmod}")
        print(f"exp((AICmod - AICunmod) / 2): {np.exp((AICmod - AICunmod) / 2)}\n")
    else:
        print("Unmodulated model is preferred!")
        print(f"Delta AIC: {AICmod - AICunmod}")

    print(f"BIC modulated: {BICmod}")
    print(f"BIC unmodulated: {BICunmod}")
    if BICmod < BICunmod:
        print("\nModulated model is preferred!\n")
        print(f"Delta BIC: {BICunmod - BICmod}")
    else:
        print("Unmodulated model is preferred!")
        print(f"Delta BIC: {BICmod - BICunmod}")

    # -----------------------------------------------------------------------------------

