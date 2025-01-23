"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


Extracts the flare light curve from a CHEOPS imagette light curve.
Fits a flare model of choice and calculates the equivalent duration and bolometric flare energy.

"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import emcee
import corner

from altaipony.fakeflares import flare_model
from altaipony.altai import equivalent_duration
from altaipony.flarelc import FlareLightCurve

from astropy import units as u

from funcs.flares import flare_factor

if __name__ == "__main__":

    # if there is no results/cheops_flares.csv file, create it and write the header
    if not os.path.exists("results/cheops_flares.csv"):
        with open("results/cheops_flares.csv", "w") as f:
            f.write("date,med_flux,amplitude,t_peak_BJD,dur_d,ED,"
                    "EDerr,mean_bol_energy,std_bol_energy,tmin,tmax,"
                    "parametrization\n")

    # GET THE IMAGETTE LC -----------------------------------------------------------
    
    # read file string from command line
    file = os.sys.argv[1]
    pi = os.sys.argv[2]

    # read the light curve from file
    folder = "results/cheops"

    # read the light curve
    lc = pd.read_csv(f"{folder}/HIP67522_{file}{pi}_detrended_lc.csv")

    # READ THE MODEL PARAMETERS --------------------------------------------------------------
    # read input from cheops_flares_input.csv
    input_file = "data/cheops_flares_input.csv"
    df = pd.read_csv(input_file)

    # get the row for the file
    row = df[df["newpipe"] == f"{file}{pi}"].iloc[0]

    # get the parameters
    tmin = row.tmin
    tmax = row.tmax
    t_peak = row.t_peak
    parametrization = "mendoza2022"
    

    # extend the flare region using tmin and tmax
    extra_flag = (lc.time.values >= tmin) & (lc.time.values <= tmax)
    flag = np.zeros(len(lc))
    flag[extra_flag] = 1

    # -----------------------------------------------------------------------------------------

    newmed = lc.flux[flag==0].median()
    f_flare = lc.flux[flag==1].values - newmed
    t_flare = lc.time[flag==1].values
    ferr_flare = np.std(lc.flux[flag==0].values)
    print(f"Flare region flux std: {ferr_flare:.2e}")


    # WRITE THE FLARE LIGHT CURVE TO A CSV FILE ------------------------------------------------

    # flarelc = pd.DataFrame({"t": t_flare, "f": f_flare, "ferr": ferr_flare})
    # flarelc.to_csv(f"results/cheops/{}{}/hip67522_flare_lc_{file}{pi}.csv", index=False)

    # PLOT THE FLARE LIGHT CURVE --------------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(t_flare, f_flare+newmed, ".", markersize=1, color="black")
    
    plt.axhline(newmed, color="red", lw=1)
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Flare region flux after subtraction of quiescent model")
    
    plt.savefig(f"plots/diagnostic/{file}{pi}/hip67522_flare_lc_{file}{pi}.png", dpi=300)
    # -----------------------------------------------------------------------------------------

    # GET A FIRST GUESS OF THE FLARE PARAMETERS ------------------------------------------------

    # same dur and ampl for everyone
    dur = 0.01
    ampl = 0.01*newmed

    # define flare model using the parametrization
    def flare_fit_model(t, t_peak, dur, ampl):
        return flare_model(parametrization, t, t_peak, dur, ampl)
    
    # fit the flare model to the flare region
    popt, pcov = curve_fit(flare_fit_model, t_flare, f_flare, 
                           p0=[t_peak, dur, ampl], 
                           bounds=([tmin, 0, 0], [tmax,np.inf,np.inf]))

    # get the fitted flare model
    print("Fitted flare parameters:")
    print(popt)


    # USE MCMC TO GET THE FLARE PARAMETERS ---------------------------------------------------

    # define the log likelihood, prior, and probability
    def log_likelihood(theta, t, f, ferrstd):
        t_peak, dur, ampl = theta
        model = flare_fit_model(t, t_peak, dur, ampl)
        return -0.5 * np.sum((f - model)**2 / ferrstd**2)

    def log_prior(theta):
        t_peak, dur, ampl = theta
        if (t_flare[0] < t_peak < t_flare[-1]) and (0.0 < dur < 0.2) and (0.001 < ampl < newmed):
            return 0.0
        return -np.inf

    def log_probability(theta, t, f, ferrstd):
        lp = log_prior(theta)
        logl = log_likelihood(theta, t, f, ferrstd)
        if (not np.isfinite(lp)) or (not np.isfinite(logl)):
            return -np.inf
        return lp + logl

    # define the number of dimensions, walkers, and steps
    ndim = 3
    nwalkers = 32
    nsteps = 30000

    # define the initial position of the walkers
    pos = popt + 1e-4 * np.random.randn(nwalkers, ndim)

    # define the standard deviation of the flux in the flare region using the non-flare region
    ferrstd = np.std(lc.flux[lc.flag==0].values)

    # init the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(t_flare, f_flare, ferrstd))

    # run the sampler
    sampler.run_mcmc(pos, nsteps, progress=True)

    # PLOT THE CHAINS ----------------------------------------------------------------

    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$t_p$", "FWHM", r"$a$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(f"plots/diagnostic/{file}{pi}/hip67522_oneflare_model_mcmc_chains.png")
    # ---------------------------------------------------------------------------------

    # SHOW THE CORNER PLOT -----------------------------------------------------------

    # get the flat samples
    flat_samples = sampler.get_chain(discard=nsteps//10*9, thin=15, flat=True)

    # plot the corner plot
    fig = corner.corner(flat_samples, labels=labels)
    plt.savefig(f"plots/diagnostic/{file}{pi}/hip67522_oneflare_model_mcmc_corner.png")

    # ---------------------------------------------------------------------------------

    # SAMPLE SOLUTIONS FROM THE CHAIN AND PLOT THEM ------------------------------------

    # get median values
    t_peak, dur, ampl = np.median(flat_samples, axis=0)

    # init a light curve for interpolation
    ndat = int((t_flare.max() - t_flare.min() ) * 24 * 60 * 6)
    t_interpolate = np.linspace(t_flare.min(), t_flare.max() , ndat)

    # initif figure
    plt.figure(figsize=(10, 5))

    # get the equivalent durations, too
    eds = []

    # loop over the samples 500 times
    for i in np.random.randint(len(flat_samples), size=500):

        # get sample
        sample = flat_samples[i]

        # get the interpolated flux model
        f_interpolate = flare_fit_model(t_interpolate, *sample) + newmed

        # plot the interpolated flux model
        plt.plot(t_interpolate, f_interpolate,  ".", color="red", alpha=0.01, markersize=1)

        # get the equivalent duration
        flc = FlareLightCurve(time=t_interpolate, flux = f_interpolate, flux_err=ferrstd)
        flc.it_med = newmed
        flc.detrended_flux = f_interpolate
        ed = equivalent_duration(flc, 0, len(t_interpolate)-1)  
        
        eds.append(ed)

    # plot the flare region on top
    plt.plot(t_flare, f_flare+newmed, ".", markersize=1, color="black")

    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Best fit one-flare model")
    plt.xlim(t_flare.min(), t_flare.max())

    plt.savefig(f"plots/diagnostic/{file}{pi}/hip67522_oneflare_model_bestfit.png")
    # ---------------------------------------------------------------------------------

    # PLOT THE RESIDUALS ---------------------------------------------------------------

    # subtract model from f_flare
    residuals = f_flare - flare_fit_model(t_flare, t_peak, dur, ampl) 

    plt.figure(figsize=(10, 5))
    plt.plot(t_flare, residuals+newmed, ".", markersize=3)
    plt.plot(lc.time[lc.flag==0], lc.flux[lc.flag==0], ".", markersize=1)
    plt.axhline(newmed, color="red", lw=1)
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Residuals of the one-flare model")
    plt.savefig(f"plots/diagnostic/{file}{pi}/hip67522_oneflare_model_residuals.png")
    # ---------------------------------------------------------------------------------

   

    # GET AND PLOT EQUIVALENT DURATIONS ------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.hist(eds, histtype="step", bins=50, color="black")
    plt.xlabel("Equivalent Duration [s]")
    plt.savefig(f"plots/diagnostic/{file}{pi}/hip67522_model_posterior_ED.png")

    ED = np.median(eds)
    EDerr = np.std(eds)

    print(f"Equivalent duration: {ED} +/- {EDerr}")

    # ---------------------------------------------------------------------------------


    # GET BOLOMETRIC FLARE ENERGY ------------------------------------------------------
    
    # read CHEOPS response function
    cheops_resp = pd.read_csv("data/CHEOPS_bandpass.csv")
    wav, resp = cheops_resp.WAVELENGTH.values, cheops_resp.THROUGHPUT.values

    # effective temperature of HIP 67522 from Rizzuto et al. 2020
    teff = 5650 # +- 75

    # radius of HIP 67522 from Rizzuto et al. 2020
    radius = 1.392 # +- 0.05

    print(f"Effective temperature: {teff:.0f} K")
    print(fr"Radius: {radius:.3f} solar radii")

    tflare = 10000.  # K

    # calculate bolometric flare energy
    print("\nWe use an 10000K flare temperature.")
    bol_energy = flare_factor(teff, radius, wav, resp,  tflare=tflare) * ED * u.s


    print("\nBolometric flare energy in ergs:")
    print(f"{bol_energy:.2e}")
    
    # ---------------------------------------------------------------------------------

    # GET POSTERIOR DISTRIBUTION OF THE FLARE ENERGY ------------------------------------

    # sample the posterior distribution of the flare energy using the MCMC samples on ED, and teff and radius with Gaussian errors
    teff = np.random.normal(5650, 75, 500)
    radius = np.random.normal(1.392, 0.05, 500)

    # calculate the bolometric flare energy for each sample
    ffactor = flare_factor(teff.reshape((500,1)), radius.reshape((500,1)), wav, resp,  tflare=tflare)
    bol_energies = ffactor * np.random.choice(eds, 500) * u.s

    # calculate the mean and standard deviation of the bolometric flare energy
    mean_bol_energy = np.mean(bol_energies).value
    std_bol_energy = np.std(bol_energies).value
    mean_ffactor = np.mean(ffactor).value

    # plot the distribution of the bolometric flare energy
    plt.figure(figsize=(6, 5))
    plt.hist(bol_energies.value.flatten(), histtype="step", bins=50, color="black")
    plt.xlabel("Bolometric Flare Energy [ergs]")
    plt.savefig(f"plots/diagnostic/{file}{pi}/hip6752_model_posterior_bolometric_energy.png")



    # WRITE THE RESULTS TO A CSV FILE ---------------------------------------------------
    with open(f"results/cheops_flares.csv", "a") as f:

        f.write(f"{file}{pi},{newmed},{ampl},{t_peak},{dur},{ED:.2e}," +
                f"{EDerr:.2e},{mean_bol_energy},{std_bol_energy}," + 
                f"{tmin},{tmax},{parametrization}\n")
        
    # ---------------------------------------------------------------------------------










