
import numpy as np
import concurrent.futures

from math import factorial

if __name__ == "__main__":

    # read phases from file
    tess_phases = np.loadtxt("tess_phases.txt")
    cheops_phases = np.loadtxt("cheops_phases.txt")

    # weigh by observing cadence
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. / 60. / 60. / 24., 
                              np.ones_like(tess_phases) * 2. / 60. / 24.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])

    # flare phases
    phases = np.array([0.61248919, 0.81165721, 0.01788908, 0.0296636,  0.05760315, 0.04067287,
    0.73005547, 0.94878914, 0.11323833, 0.20031473, 0.15087211, 0.04514247,
    0.02527212, 0.05657772, 0.06247738, ]) 


    # shift by 0.5
    obs_phases = (obs_phases + 0.5) % 1
    phases = (phases + 0.5) % 1

    # define binning
    nbins = 100
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
    
    max_lambda0 = 1.
    max_lambda1 = 1.

    min_lambda0 = 0.0001
    min_lambda1 = 0.0001

    # fake_l = 0.2
    # rate = unmodulated_model(fake_l)

    # hist = np.array([np.random.poisson(r) for r in rate])

    # save rate and hist to fake_bayes_factors_samples.txt
    # with open("fake_bayes_factors_samples.txt", "a") as f:
    #     string = f"{rate} {hist}\n"
    #     f.write(string)
    #     print(string)

    # print(rate)

    # print(hist)
    # observed:
    hist, bins = np.histogram(phases, bins=bins)
    # define the factorials for the numbers in hist for the likelihood computation
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
        if ((params[0] > min_lambda0) & (params[1] > min_lambda1) & (params[1] < max_lambda1) & ( params[0] < max_lambda0) &
            (params[2] > 0) & (params[2] < 1) & (params[3] > 0) & (params[3] < 1) ):
            return (np.log(1 / np.sqrt(params[0]) / 2 / (np.sqrt(max_lambda0) - np.sqrt(min_lambda0))) + 
                    np.log(1 / np.sqrt(params[1]) / 2 / (np.sqrt(max_lambda1) - np.sqrt(min_lambda1))))
        return -np.inf

    # define log-probability
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
        if ((params[0] > min_lambda0) & (params[0] < max_lambda0)):
            return np.log(1 / np.sqrt(params[0]) / 2 / (np.sqrt(max_lambda0) - np.sqrt(min_lambda0)))
        return -np.inf

    def log_probability_unmod(params):
        lp = log_prior_unmod(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_unmod(params)

    ranged =  [max_lambda0, max_lambda1, 500, 500, 100, 100]

    integrals = 0
    lambda0max, lambda1max, lambda0n, lambda1n, phi0n, dphin = ranged

    lambda0n_split = 20
    nums = np.arange(0, lambda0n, lambda0n_split)
    print(nums)

    def compute_integral(num, phi0n, dphin, lambda0n_split, lambda0n, lambda0max, lambda1n, lambda1max, min_lambda0, min_lambda1, log_probability_mod):
        phi0 = np.linspace(0, 1, phi0n)
        dphi = np.linspace(0, 1, dphin)
        lambda0s = np.linspace(min_lambda0 + num/lambda0n*(lambda0max - min_lambda0), min_lambda0 + (num+lambda0n_split)/lambda0n*(lambda0max - min_lambda0), lambda0n_split)
        lambda1s = np.linspace(min_lambda1, lambda1max, lambda1n)
        logls = np.zeros((lambda0n_split, lambda1n, phi0n, dphin))
        integral = 1
        print("lambda0: ", lambda0s)
        # print("lambda1: ", lambda1s)
        # print("phi0: ", phi0)
        # print("dphi: ", dphi)
        for i in range(lambda0n_split): 
            print(i)
            for j in range(lambda1n):
                for k in range(phi0n):
                    for l in range(dphin):
                        logl = log_probability_mod([lambda0s[i], lambda1s[j], phi0[k], dphi[l]])
                        logls[i, j, k, l] = logl
        
        # Perform integration using np.trapz
        integral = np.trapz(np.trapz(np.trapz(np.trapz(np.exp(logls), dphi), phi0), lambda1s), lambda0s)
        
        return integral

    def main(nums, phi0n, dphin, lambda0n_split, lambda0n, lambda0max, lambda1n, lambda1max, min_lambda0, min_lambda1, log_probability_mod):
        integrals = 0
        # Use ProcessPoolExecutor for parallel computation
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit tasks for each l in the nums list
            futures = [executor.submit(compute_integral, num, phi0n, dphin, lambda0n_split, lambda0n, lambda0max, lambda1n, lambda1max,  min_lambda0, min_lambda1,log_probability_mod) for l, num in enumerate(nums)]
            
            # Wait for all tasks to finish and collect the results
            for future in concurrent.futures.as_completed(futures):
                integrals += future.result()
            print(integrals)
        return integrals

    # Usage example (ensure to pass appropriate parameters):
    integrals = main(nums, phi0n, dphin, lambda0n_split, lambda0n, lambda0max, lambda1n, lambda1max,  min_lambda0, min_lambda1,log_probability_mod)





    logsunmod = np.zeros(lambda0n)
    lambda0s = np.linspace(0, lambda0max, lambda0n)
    print(lambda0s)
    for i in range(lambda0n):
        logl = log_probability_unmod([lambda0s[i]])
        logsunmod[i] = logl
    integralunmod = np.trapz(np.exp(logsunmod), lambda0s)

    bayes_factor = integrals / integralunmod


    with open("bayes_factor.txt", "a") as f:
        string = f"{len(phases)},{nbins},{lambda0max},{lambda1max},{lambda0n},{lambda1n},{phi0n},{dphin},{bayes_factor},{integrals},{integralunmod}\n"
        f.write(string)
        print(string)

