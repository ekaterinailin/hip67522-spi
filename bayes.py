
import numpy as np 

from math import factorial

if __name__ == "__main__":

    # read phases from file
    tess_phases = np.loadtxt("../data/tess_phases.txt")
    cheops_phases = np.loadtxt("../data/cheops_phases.txt")

    # weigh by observing cadence
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. / 60. / 60. / 24., np.ones_like(tess_phases) * 2. / 60. / 24.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])

    # flare phases
    phases = np.array([0.61248919, 0.81165721, 0.01788908, 0.0296636,  0.05760315, 0.04067287,
    0.73005547, 0.94878914, 0.11323833, 0.20031473, 0.15087211, 0.04514247,
    0.02527212, 0.05657772, 0.06247738, ]) 

    # shift by 0.5
    obs_phases = (obs_phases + 0.5) % 1
    phases = (phases + 0.5) % 1

    # define binning
    nbins = 200
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
        if ((params[0] > 0) & (params[1] > 0) & (params[1] < 5) & ( params[0] < 5) &
            (params[2] > 0) & (params[2] < 1) & (params[3] > 0) & (params[3] < 1) ):
            return np.log(1/np.sqrt(params[0])) + np.log(1/np.sqrt(params[1])) 
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
        if ((params[0] > 0) & (params[0] < 10)):
            return np.log(1/np.sqrt(params[0]))
        return -np.inf

    def log_probability_unmod(params):
        lp = log_prior_unmod(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_unmod(params)

    ranged =  [5, 10, 500, 500, 100, 100]
    # ranged =  [5, 10, 400, 200, 50, 50]
    # ranged = [5, 10, 300, 200, 50, 50]
    integrals = 0
    lambda0max, lambda1max, lambda0n, lambda1n, phi0n, dphin = ranged
    for l, num in enumerate([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250,
                             275, 300, 325, 350, 375, 400, 425, 450, 475]):
        print("\n")
        print(l)

        
        lambda0n_split = 25
        phi0 = np.linspace(0, 1, phi0n)
        dphi = np.linspace(0, 1, dphin)
        lambda0s = np.linspace(l, l + 1, lambda0n_split)
        lambda1s = np.linspace(0, lambda1max, lambda1n)
        logls = np.zeros((lambda0n_split,lambda1n,phi0n, dphin))
        for i in range(lambda0n_split):
            print(i)
            for j in range(lambda1n):
                for k in range(phi0n):
                    for l in range(dphin):
                        logl = log_probability_mod([lambda0s[i], lambda1s[j], phi0[k], dphi[l]])
                        logls[i, j, k, l] = logl
        integral = np.trapz(np.trapz(np.trapz(np.trapz(np.exp(logls), dphi), phi0), lambda1s), lambda0s) # trapz integrates over the last axis
        integrals += integral


    logsunmod = np.zeros(lambda0n)
    lambda0s = np.linspace(0, lambda0max, lambda0n)
    for i in range(lambda0n):
        logl = log_probability_unmod([lambda0s[i]])
        logsunmod[i] = logl
    integralunmod = np.trapz(np.exp(logsunmod), lambda0s)

    bayes_factor = integrals / integralunmod


    with open(f"../results/bayes_factor_{nbins}.txt", "a") as f:
        string = f"{nbins},{lambda0max},{lambda1max},{lambda0n},{lambda1n},{phi0n},{dphin},{bayes_factor},{integrals},{integralunmod}\n"
        f.write(string)
        print(string)

