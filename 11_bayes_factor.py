"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


Calculate the Bayes Factor
"""

import numpy as np
import pandas as pd
import concurrent.futures

from math import factorial

import os

def compute_integral(num, phi0n, dphin, lambda0n_split, lambda0n, lambda0max, 
                        lambda1n, lambda1max, min_lambda0, min_lambda1, log_probability_mod):
    
    phi0 = np.linspace(0, 1, phi0n)
    dphi = np.linspace(0, max_dphi, dphin)
    # set what range of lambda to cover in this split
    lambda0s = np.linspace(min_lambda0 + num/lambda0n*(lambda0max - min_lambda0), 
                            min_lambda0 + (num+lambda0n_split)/lambda0n*(lambda0max - min_lambda0), 
                            lambda0n_split)
    lambda1s = np.linspace(min_lambda1, lambda1max, lambda1n)
    logls = np.zeros((lambda0n_split, lambda1n, phi0n, dphin))
    
    print("lambda0: ", lambda0s)

    for i in range(lambda0n_split): 
        print(i, nbins)
        for j in range(lambda1n):
            for k in range(phi0n):
                for l in range(dphin):
                    logl = log_probability_mod([lambda0s[i], lambda1s[j], phi0[k], dphi[l]])
                    logls[i, j, k, l] = logl
    
    # Perform integration using np.trapz
    integral = np.trapz(np.trapz(np.trapz(np.trapz(np.exp(logls), dphi), phi0), lambda1s), lambda0s)
    
    return integral

def main(nums, phi0n, dphin, lambda0n_split, lambda0n, lambda0max,
            lambda1n, lambda1max, min_lambda0, min_lambda1, log_probability_mod):
    integrals = 0
    # Use ProcessPoolExecutor for parallel computation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks for each l in the nums list
        futures = [executor.submit(compute_integral, num, phi0n, dphin, 
                                    lambda0n_split, lambda0n, lambda0max, lambda1n, 
                                    lambda1max,  min_lambda0, min_lambda1,log_probability_mod) for l, num in enumerate(nums)]
        
        # Wait for all tasks to finish and collect the results
        for future in concurrent.futures.as_completed(futures):
            integrals += future.result()
        print(integrals)
    return integrals

if __name__ == "__main__":

    # decide if you want to run the real deal or fake random flaring
    fake = False

    # read phases from file
    tess_phases = np.loadtxt("results/tess_phases.txt")
    cheops_phases = np.loadtxt("results/cheops_phases.txt")

    # weigh by observing cadence
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. / 60. / 60. / 24., 
                              np.ones_like(tess_phases) * 2. / 60. / 24.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])

    # flare phases

    flares = pd.read_csv("results/hip67522_flares.csv")
    flares = flares.sort_values("mean_bol_energy", ascending=True).iloc[1:] # exclude the smallest flare
    phases = flares.orb_phase.values

    # shift by 0.5
    obs_phases = (obs_phases + 0.5) % 1
    phases = (phases + 0.5) % 1

    # define binning
    # read nbins from command line argument 1
    nbins = int(os.sys.argv[1])
    bins = np.linspace(0, 1, nbins)
    binmids= (bins[1:] + bins[:-1]) / 2

    # bin the phases
    arr = np.digitize(obs_phases, bins)

    # sum the observing times in each bin to binned weights
    # unit of entries in binned is [days]
    binned = np.array([np.sum(weights[arr==i]) for i in range(1, len(bins))]) 

    # define the two models we want to compare
    def modulated_model(binmids, lambda0, lambda1, phase0, dphase, weight=binned):
        mask = (binmids > phase0) & (binmids < (phase0 + dphase)%1)
        result = np.zeros_like(binmids)

        # multiply by weight because that modifies the measured rate
        result[~mask] = lambda0 * weight[~mask]
        result[mask] = lambda1 * weight[mask]

        return result #number of observed flares per bin

    def unmodulated_model(lambda0, weight=binned):
        return lambda0 * weight #number of observed flares per bin
    
    max_lambda0 = 2.
    max_lambda1 = 2.
    min_lambda0 = 0.000
    min_lambda1 = 0.000

    max_dphi = 0.5 # otherwise we're probing the same parameter space twice


    if fake:

        fake_l = 0.2 # set a fake flare rate
        rate = unmodulated_model(fake_l) # get flare rates for the unmodulated case

        hist = np.array([np.random.poisson(r) for r in rate]) # make histogram of fake observed flares

    else:
        hist, bins = np.histogram(phases, bins=bins)
        # define the factorials for the numbers in hist for the likelihood computation


    # pre-calculated factorials to save time
    factorials = np.array([factorial(h) for h in hist])

    # Poisson log-likelihood function
    def log_likelihood_poisson(rate, hist, factorials):
        logs = -rate - np.log(factorials) + np.log(rate) * hist
        return np.sum(logs)

    # log-likelihood for the modulated model
    def log_likelihood_mod(params):

        rate = modulated_model(binmids, *params, weight=binned)

        return log_likelihood_poisson(rate, hist, factorials)

    def log_prior_mod(params):
        if ((params[0] > min_lambda0) & (params[1] > min_lambda1) & (params[1] <= max_lambda1) & ( params[0] <= max_lambda0) &
            (params[2] > 0) & (params[2] <= 1) & (params[3] > 0) & (params[3] <= max_dphi) ):
            return (np.log(1 / np.sqrt(params[0]) / 2 / (np.sqrt(max_lambda0) - np.sqrt(min_lambda0))) + 
                    np.log(1 / np.sqrt(params[1]) / 2 / (np.sqrt(max_lambda1) - np.sqrt(min_lambda1))) + np.log(1/max_dphi))
        return -np.inf
    
    # flat prior
    # def log_prior_mod(params):
    #     if ((params[0] > min_lambda0) & (params[1] > min_lambda1) & (params[1] <= max_lambda1) & ( params[0] <= max_lambda0) &
    #         (params[2] > 0) & (params[2] <= 1) & (params[3] > 0) & (params[3] <= max_dphi) ):
    #         return np.log(1/8)
    #     return -np.inf

    # log-probability
    def log_probability_mod(params):
        lp = log_prior_mod(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_mod(params)

    # define log-likelihood, prior, and probability
    def log_likelihood_unmod(params):
        lambda0 = params[0]
        rate = unmodulated_model(lambda0, weight=binned)
        return log_likelihood_poisson(rate, hist, factorials)
    
    def log_prior_unmod(params):
        if ((params[0] > min_lambda0) & (params[0] <= max_lambda0)):
            return np.log(1 / np.sqrt(params[0]) / 2 / (np.sqrt(max_lambda0) - np.sqrt(min_lambda0)))
        return -np.inf

    # flat prior    
    # def log_prior_unmod(params):
    #     if ((params[0] > min_lambda0) & (params[0] <= max_lambda0)):
    #         return np.log(1/4)
    #     return -np.inf

    def log_probability_unmod(params):
        lp = log_prior_unmod(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_unmod(params)
    

    # NOW CALCULATE THE BAYES FACTOR USING PARALLEL COMPUTATION:

    lambda0max, lambda1max, lambda0n, lambda1n, phi0n, dphin = max_lambda0, max_lambda1, 400, 400, 100, 50

    lambda0n_split = 10 # split the 400 grid in 10 x 40: will need 40 cores to work

    # set up iterator
    nums = np.arange(0, lambda0n, lambda0n_split)
    print(nums)

    # get the marginal likelihood for the modulated model
    integralmod = main(nums, phi0n, dphin, lambda0n_split, lambda0n, lambda0max,
                    lambda1n, lambda1max,  min_lambda0, min_lambda1, log_probability_mod)

    # get the marginal likelihood for the unmodulated model
    logsunmod = np.zeros(lambda0n)
    lambda0s = np.linspace(min_lambda0, lambda0max, lambda0n)
    print(lambda0s)
    for i in range(lambda0n):
        logl = log_probability_unmod([lambda0s[i]])
        logsunmod[i] = logl
    integralunmod = np.trapz(np.exp(logsunmod), lambda0s)

    bayes_factor = integralmod / integralunmod


    with open("results/bayes_factor.txt", "a") as f:
        string = f"{len(phases)},{nbins},{lambda0max},{lambda1max}," \
                 f"{lambda0n},{lambda1n},{phi0n},{dphin},{bayes_factor}," \
                 f"{integralmod},{integralunmod},{min_lambda0},{min_lambda1}\n"
        f.write(string)
        print(string)

