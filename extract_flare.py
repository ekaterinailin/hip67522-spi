"""
Python 3.11.7, UTF-8

Extracts the flare light curve from a CHEOPS imagette light curve.
Fits a flare model of choice and calculates the equivalent duration and bolometric flare energy.

"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import curve_fit

import emcee
import corner

from altaipony.fakeflares import flare_model
from altaipony.altai import equivalent_duration
from altaipony.flarelc import FlareLightCurve

from astropy import units as u

from funcs.flares import flare_factor

if __name__ == "__main__":

    # GET THE IMAGETTE LC -----------------------------------------------------------
    
    # read file string from command line
    file = os.sys.argv[1]

    # file = "20240415114752"
    IMG = f'../data/hip67522/CHEOPS-products-{file}/Outdata/00000/hip67522_CHEOPS-products-{file}_im.fits'
    hdulist = fits.open(IMG)
    print(f"Imagette file found for {file}:\n {IMG}\n")

    # get the image data
    image_data = hdulist[1].data

    # get the time, flux, flux error, and roll angle
    t, f, ferr, roll = image_data["BJD_TIME"], image_data["FLUX"], image_data["FLUXERR"], image_data["ROLL"]

    # make sure the data is in fact 10s cadence
    assert np.diff(t).min() * 24 * 60 * 60 < 10.05, "Time series is not 10s cadence"

    # big endian to little endian
    t = t.byteswap().newbyteorder()
    f = f.byteswap().newbyteorder()
    ferr = ferr.byteswap().newbyteorder()
    roll = roll.byteswap().newbyteorder()
    # ----------------------------------------------------------------------------------------

    # READ THE MODEL PARAMETERS --------------------------------------------------------------
    # read input from cheops_flares_input.csv
    input_file = "../data/cheops_flares_input.csv"
    df = pd.read_csv(input_file)

    # get the row for the file
    row = df[df["file"] == int(file)].iloc[0]

    # get the parameters
    egress = row.egress
    ingress = row.ingress 
    tmin = row.tmin
    tmax = row.tmax
    t_peak = row.t_peak
    parametrization = row.parametrization
    two_flare = row.two_flare   

    # mask the data: outliers and transit
    if ~np.isnan(ingress):
        mask = (f < 2.9e6) & (f > 2.43e6) & (t < ingress - 0.02) 
    elif ~np.isnan(egress):
        mask = (f < 2.9e6) & (f > 2.43e6) & (t > egress + 0.02)

    t = t[mask]
    f = f[mask]
    roll = roll[mask]
    ferr = ferr[mask]

    print(f"Initial mask: {mask.sum()} data points")

    print(f"Using the {parametrization} flare model")

    print(f"Egress mask: {mask.sum()} data points")

    print(f"Tmin, Tpeak, Tmax: {tmin, t_peak, tmax}")
    # ----------------------------------------------------------------------------------------


    # PLOT THE INITIAL LIGHT CURVE -----------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t, f, ".", markersize=1)
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Initial light curve, masking transit and outliers")
    plt.savefig(f"../plots/{file}/flares/hip67522_initial_lc.png")
    # ----------------------------------------------------------------------------------------


    # MASK THE FLARE REGION ------------------------------------------------------------------
    flare_mask = np.zeros_like(f, dtype=bool)

    flare_mask[(t > tmin) & (t < tmax)] = True
    # ----------------------------------------------------------------------------------------


    # PLOT THE MASKED LIGHT CURVE -----------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t[flare_mask], f[flare_mask], ".", markersize=2, color="blue")
    plt.plot(t[~flare_mask], f[~flare_mask], ".", markersize=1, color="red")
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Initial light curve, masking flare region")
    plt.savefig(f"../plots/{file}/flares/hip67522_initial_lc_flare_mask.png")
    # ----------------------------------------------------------------------------------------


    # LEAST-SQUARE FIT A 5th ORDER POLYNOMIAL TO THE LIGHT CURVE -----------------------------

    # define the offset of the full lightcurve to make fitting easier
    offset2 = t[0]

    # use that to define the polynomial function
    def func(x, a, b, c, d, e, f, offset):
        return (f * (x - offset2 + offset)**5 + 
                e * (x - offset2 + offset)**4 + 
                a * (x - offset2 + offset)**3 + 
                b * (x - offset2 + offset)**2 + 
                c * (x - offset2 + offset) + d)

    # fit the polynomial, hard code the initial guess
    popt, pcov = curve_fit(func, t[~flare_mask], f[~flare_mask],
                           p0=[-1.45888787e+04, -1.41685433e+08, -1.03596058e+09,  1.00000000e+00,
                                1.19292031e-02, -2.42900480e-09,  8.42088604e-01])

    # get the fitted light curve
    fitted = func(t, *popt)


    # FIRST POLYNOMIAL FIT TO THE LIGHT CURVE ------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(t, fitted, color="blue", lw=1)
    plt.plot(t, f, ".", markersize=1, color="black")
    plt.plot(t[flare_mask], f[flare_mask], ".", markersize=2, color="red")

    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("First polynomial fit to the light curve w/o flare")
    plt.savefig(f"../plots/{file}/flares/hip67522_polyfit_init.png")
    # -----------------------------------------------------------------------------------------


    # SUBTRACT THE POLYNOMIAL FIT FROM THE LIGHT CURVE ----------------------------------------

    # get the median of the light curve
    med = np.median(f[~flare_mask])

    # subtract fitted from f and add the median
    f_sub = f - fitted + med

    # mask outliers in the subtracted light curve above and below the median
    mask_outliers = (f_sub[~flare_mask] > 1.0025 * med) | (f_sub[~flare_mask] < 0.995 * med)
    print(f"Outliers: {mask_outliers.sum()}")


    # PLOT THE SUBTRACTED LIGHT CURVE --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t, f, ".", markersize=1)
    plt.plot(t, f_sub, ".", markersize=1)
    plt.axhline(med, color="red", lw=1)

    # plot masked outliers
    plt.plot(t[~flare_mask][mask_outliers], f_sub[~flare_mask][mask_outliers],
             ".", markersize=10, color="red")
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Subtracting the polynomial fit and masking outliers")
    plt.savefig(f"../plots/{file}/flares/hip67522_subtract_polyfit_init.png")
    # -----------------------------------------------------------------------------------------

    # FIT A POLYNOMIAL TO THE LIGHT CURVE W/O FLARE AND MASKED OUTLIERS -----------------------
    
    # apply new outlier mask
    tnew = t[~flare_mask][~mask_outliers]
    fnew = f[~flare_mask][~mask_outliers]

    # fit the polynomial, hard code the initial guess
    popt, pcov = curve_fit(func, tnew, fnew, p0=popt)

    # get the fitted light curve
    newfitted = func(t, *popt)

    # get the new median
    newmed = np.median(f[~flare_mask][~mask_outliers])

    # subtract fitted from f and add the median
    newf_sub = f - fitted + med

    # get the number of total data points and observing time in days
    print(f"Total data points: {len(t)}")

    tot_obs_time_d = len(t) * 10. / 60. / 60. / 24.


    # PLOT THE DIFFERENCE BETWEEN FIRST AND SECOND POLYNOMIAL FIT -----------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(t, fitted - newfitted, color="blue", lw=1)
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Difference between first and second polynomial fit")
    plt.savefig(f"../plots/{file}/flares/hip67522_polyfit_diff.png")

    # -----------------------------------------------------------------------------------------


    # PLOT THE FINAL POLYNOMIAL FIT TO THE LIGHT CURVE ----------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(t, f, ".", markersize=1, color="grey")
    plt.plot(t[flare_mask], f[flare_mask], ".", markersize=2, color="red")
    plt.plot(t, newfitted, ".", markersize=1, color="black")
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Second polynomial fit to the light curve w/o flare")
    plt.savefig(f"../plots/{file}/flares/hip67522_polyfit_final.png")

    # -----------------------------------------------------------------------------------------

    # PLOT FLUX AGAINST ROLL ANGLE -----------------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(roll[~flare_mask], newf_sub[~flare_mask], ".", markersize=1)
    plt.xlabel("Roll angle [deg]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.savefig(f"../plots/{file}/flares/hip67522_roll_flux.png")

    # -----------------------------------------------------------------------------------------


    # GET A QUIESCENT FLUX MODEL FOR THE FLARE REGION -----------------------------------------
    # approximate the flux at the roll values in the flare region with 
    # the flux at the closest roll value in the non-flare region

    # flux and roll in the flare region
    f_sub_flare = newf_sub[flare_mask]
    roll_flare = roll[flare_mask]

    # flux and roll in the non-flare region
    f_sub_no_flare = newf_sub[~flare_mask]
    roll_no_flare = roll[~flare_mask]

    # init the quiescent model
    f_sub_flare_approx = np.zeros_like(f_sub_flare)

    # loop over all data point in the flare region and approximate the flux
    for i, r in enumerate(roll_flare):

        # find the nearest data points within 1 deg range
        idx = [np.argmin(np.abs(roll_no_flare - r-delt)) for delt in np.linspace(-1, 1, 50)]

        # get the median of the flux at the nearest data points
        f_sub_flare_approx[i] = np.mean(f_sub_no_flare[idx])


    # PLOT THE FLARE REGION FLUX VS QUIESCENT MODEL ------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t[flare_mask], f_sub_flare_approx, ".", markersize=1, 
             color="red", label="quiescent model")
    plt.plot(t[flare_mask], f_sub_flare, ".", markersize=1, 
             color="blue", label="flare region")
    plt.plot(t[~flare_mask], f_sub_no_flare, ".", markersize=1, 
             color="black", label="non-flare region")
    plt.axhline(newmed, color="black", lw=1)
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e^${-}$/s]")
    plt.legend(loc=0, frameon=False)
    plt.title("Flare region flux vs quiescent model")
    plt.savefig(f"../plots/{file}/flares/hip67522_flare_vs_quiescent_model.png")
    # -----------------------------------------------------------------------------------------

    # GET THE FLARE LIGHT CURVE --------------------------------------------------------------

    # subtract the quiescent model from the flare region and add the median
    f_flare = f_sub_flare - f_sub_flare_approx + newmed


    # WRITE THE FLARE LIGHT CURVE TO A CSV FILE ------------------------------------------------

    flarelc = pd.DataFrame({"t": t[flare_mask], "f": f_flare - newmed, "ferr": ferr[flare_mask]})
    flarelc.to_csv(f"../results/hip67522_flare_lc_{file}.csv", index=False)

    # PLOT THE FLARE LIGHT CURVE --------------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(t[flare_mask], f_flare, ".", markersize=1, color="black")
    
    plt.axhline(newmed, color="red", lw=1)
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Flare region flux after subtraction of quiescent model")
    
    plt.savefig(f"../plots/{file}/flares/hip67522_flare_extracted.png")
    # -----------------------------------------------------------------------------------------

    # GET A FIRST GUESS OF THE FLARE PARAMETERS ------------------------------------------------

    # same dur and ampl for everyone
    dur = 0.01
    ampl = 0.01*newmed

    # define flare model using the parametrization
    def flare_fit_model(t, t_peak, dur, ampl):
        return flare_model(parametrization, t, t_peak, dur, ampl)
    
    # fit the flare model to the flare region
    popt, pcov = curve_fit(flare_fit_model, t[flare_mask], f_flare-newmed, 
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
        if (t[flare_mask][0] < t_peak < t[flare_mask][-1]) and (0.0 < dur < 0.1) and (0.001 < ampl < 1e7):
            return 0.0
        return -np.inf

    def log_probability(theta, t, f, ferrstd):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, t, f, ferrstd)

    # define the number of dimensions, walkers, and steps
    ndim = 3
    nwalkers = 32
    nsteps = 4000

    # define the initial position of the walkers
    pos = popt + 1e-4 * np.random.randn(nwalkers, ndim)

    # define the standard deviation of the flux in the flare region using the non-flare region
    ferrstd = np.std(newf_sub[~flare_mask])

    # init the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(t[flare_mask], f_flare-newmed, ferrstd))

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
    plt.savefig(f"../plots/{file}/flares/hip67522_oneflare_model_mcmc_chains.png")
    # ---------------------------------------------------------------------------------

    # SHOW THE CORNER PLOT -----------------------------------------------------------

    # get the flat samples
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

    # plot the corner plot
    fig = corner.corner(flat_samples, labels=labels)
    plt.savefig(f"../plots/{file}/flares/hip67522_oneflare_model_mcmc_corner.png")

    # ---------------------------------------------------------------------------------

    # SAMPLE SOLUTIONS FROM THE CHAIN AND PLOT THEM ------------------------------------

    # get median values
    t_peak, dur, ampl = np.median(flat_samples, axis=0)

    # init a light curve for interpolation
    ndat = int((t[flare_mask].max() - t[flare_mask].min()) * 24 * 60 * 6)
    t_interpolate = np.linspace(t[flare_mask].min(), t[flare_mask].max(), ndat)

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
    plt.plot(t[flare_mask], f_flare, ".", markersize=1, color="black")

    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Best fit one-flare model")

    plt.savefig(f"../plots/{file}/flares/hip67522_oneflare_model_bestfit.png")
    # ---------------------------------------------------------------------------------

    # PLOT THE RESIDUALS ---------------------------------------------------------------

    # subtract model from f_flare
    residuals = f_flare - flare_fit_model(t[flare_mask], t_peak, dur, ampl) 

    plt.figure(figsize=(10, 5))
    plt.plot(t[flare_mask], residuals, ".", markersize=3)
    plt.plot(t[~flare_mask], newf_sub[~flare_mask], ".", markersize=1)
    plt.axhline(newmed, color="red", lw=1)
    plt.xlabel("Time [BJD]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.title("Residuals of the one-flare model")
    plt.savefig(f"../plots/{file}/flares/hip67522_oneflare_model_residuals.png")
    # ---------------------------------------------------------------------------------

    # DO THE SAME FOR A TWO-FLARE MODEL -------------------------------------------------
    if two_flare:

        # now fit a two component model to the flare light curve
        def flare_fit_model(t, t_peak, dur, ampl, t_peak2, dur2, ampl2):
            return (flare_model(parametrization, t, t_peak, dur, ampl) + 
                    flare_model(parametrization, t, t_peak2, dur2, ampl2))

        # get the initial guess
        tpeak2 = t_peak + 0.01
        dur2 = 0.02
        ampl2 = 0.04*newmed

        # fit the two-flare model
        popt, pcov = curve_fit(flare_fit_model, t[flare_mask], f_flare-newmed, p0=[t_peak, dur, ampl, t_peak, dur, ampl])

        print("Fitted two-flare parameters:")
        print(popt)

        # define the log likelihood, prior, and probability
        def log_likelihood(theta, t, f, ferrstd):
            t_peak, dur, ampl, t_peak2, dur2, ampl2 = theta
            model = flare_fit_model(t, t_peak, dur, ampl, t_peak2, dur2, ampl2)
            return -0.5 * np.sum((f - model)**2 / ferrstd**2)

        def log_prior(theta):
            t_peak, dur, ampl, t_peak2, dur2, ampl2 = theta
            if  ((t[flare_mask][0] < t_peak < t[flare_mask][-1]) and 
                (0.0 < dur < 0.1) and 
                (0.01 < ampl < 1e7) and 
                (t[flare_mask][0] < t_peak2 < t[flare_mask][-1]) and 
                (0.0 < dur2 < 0.1) and 
                (0.01 < ampl2 < 1e7)):
                return 0.0
            return -np.inf

        def log_probability(theta, t, f, ferrstd):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, t, f, ferrstd)

        # define the number of dimensions, walkers, and steps
        ndim = 6
        nwalkers = 32
        nsteps = 20000

        # get std of the flux outside the flare region
        ferrstd = np.std(newf_sub[~flare_mask])

        # define the initial position of the walkers
        pos = popt + 1e-4 * np.random.randn(nwalkers, ndim)

        # init the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(t[flare_mask], f_flare-newmed, ferrstd))

        # run the sampler
        sampler.run_mcmc(pos, nsteps, progress=True)

        # PLOT THE CHAINS ----------------------------------------------------------------

        fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["t_peak", "dur", "ampl", "t_peak2", "dur2", "ampl2"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig(f"../plots/{file}/flares/hip67522_twoflare_model_mcmc_chains.png")
        # ---------------------------------------------------------------------------------

        # PLOT THE CORNER PLOT -----------------------------------------------------------

        flat_samples = sampler.get_chain(discard=5000, thin=15, flat=True)

        fig = corner.corner(flat_samples, labels=labels, truths=popt)

        plt.savefig(f"../plots/{file}/flares/hip67522_twoflare_model_mcmc_corner.png")

        # SAMPLE SOLUTIONS FROM THE CHAIN AND PLOT THEM -----------------------------------

        # median
        t_peak, dur, ampl, t_peak2, dur2, ampl2 = np.median(flat_samples, axis=0)

        # init light curve for interpolation
        # make linspace such that each data point is 10s

        ndat = int((t[flare_mask].max() - t[flare_mask].min()) * 24 * 60 * 6)
        t_interpolate = np.linspace(t[flare_mask].min(), t[flare_mask].max(), ndat)

        f_interpolate = flare_fit_model(t_interpolate, t_peak, dur, ampl, t_peak2, dur2, ampl2) + newmed

        # get the equivalent durations, too
        eds = []

        # init figure
        plt.figure(figsize=(10, 5))

        # iterate over a 500 random samples from the chain
        for i in np.random.randint(len(flat_samples), size=500):

            # get sample
            sample = flat_samples[i]

            # get the interpolated flux model
            f_interpolate = flare_fit_model(t_interpolate, *sample) + newmed

            # plot the interpolated flux model
            plt.plot(t_interpolate, f_interpolate, color="black", alpha=0.01, lw=1)

            # get the equivalent duration
            flc = FlareLightCurve(time=t_interpolate, flux=f_interpolate, flux_err=ferrstd)
            flc.detrended_flux = f_interpolate
            flc.detrended_flux_err = ferrstd
            flc.it_med = newmed
            ed = equivalent_duration(flc, 0, len(f_interpolate)-1)  
            eds.append(ed)


        plt.plot(t_interpolate, f_interpolate, color="black", alpha=0.01, lw=1, label="samples from posterior distribution")

        plt.plot(t_interpolate, f_interpolate, color="red", lw=1, label="best-fit model")

        plt.plot(t[flare_mask], f_flare, ".", markersize=1, color="black", label="de-trended flare light curve")
        plt.xlim(t[flare_mask].min(), t[flare_mask].max())
        plt.axhline(newmed, color="black", lw=1)
        plt.xlabel("Time [BJD]")
        plt.ylabel("Flare flux [e-/s]")

        plt.legend(loc=0, frameon=False)
        plt.title("Two-flare model")
        plt.savefig(f"../plots/{file}/flares/hip67522_twoflare_model_posterior.png")
        # ---------------------------------------------------------------------------------

        # PLOT THE RESIDUALS ---------------------------------------------------------------

        # get the residuals
        residuals = f_flare - flare_fit_model(t[flare_mask], t_peak, dur, ampl, t_peak2, dur2, ampl2)

        plt.figure(figsize=(10, 5))
        plt.plot(t[flare_mask], residuals-newmed, ".", markersize=3)

        plt.axhline(0, color="red", lw=1)
        plt.xlabel("Time [BJD]")
        plt.ylabel(r"Flux [e$^{-}$/s]")
        plt.title("Residuals of the two-flare model")
        plt.savefig(f"../plots/{file}/flares/hip67522_twoflare_model_residuals.png")
        # ---------------------------------------------------------------------------------


    # GET AND PLOT EQUIVALENT DURATIONS ------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.hist(eds, histtype="step", bins=50, color="black")
    plt.xlabel("Equivalent Duration [s]")
    plt.savefig(f"../plots/{file}/flares/hip67522_model_posterior_ED.png")

    ED = np.median(eds)
    EDerr = np.std(eds)

    print(f"Equivalent duration: {ED} +/- {EDerr}")

    # ---------------------------------------------------------------------------------


    # GET BOLOMETRIC FLARE ENERGY ------------------------------------------------------
    
    # read CHEOPS response function
    cheops_resp = pd.read_csv("../data/CHEOPS_bandpass.csv",)
    wav, resp = cheops_resp.WAVELENGTH.values, cheops_resp.THROUGHPUT.values

    # effective temperature of HIP 67522 from Rizzuto et al. 2020
    teff = 5650 # +- 75

    # radius of HIP 67522 from Rizzuto et al. 2020
    radius = 1.392 # +- 0.05

    print(f"Effective temperature: {teff:.0f} K")
    print(fr"Radius: {radius:.3f} solar radii")

    tflare = 10000.  # K

    # calculate bolometric flare energy
    print("\n We use an 10000K flare temperature.")
    bol_energy = flare_factor(teff, radius, wav, resp,  tflare=tflare) * ED * u.s


    print("\nBolometric flare energy in ergs:")
    print(f"{bol_energy:.2e}")
    
    # ---------------------------------------------------------------------------------

    # GET POSTERIOR DISTRIBUTION OF THE FLARE ENERGY ------------------------------------

    # sample the posterior distribution of the flare energy using the MCMC samples on ED, and teff and radius with Gaussian errors
    teff = np.random.normal(5650, 75, 500)
    radius = np.random.normal(1.392, 0.05, 500)

    # calculate the bolometric flare energy for each sample
    bol_energies = flare_factor(teff.reshape((500,1)), radius.reshape((500,1)), wav, resp,  tflare=tflare) * np.random.choice(eds, 500) * u.s

    # calculate the mean and standard deviation of the bolometric flare energy
    mean_bol_energy = np.mean(bol_energies)
    std_bol_energy = np.std(bol_energies)

    # plot the distribution of the bolometric flare energy
    plt.figure(figsize=(6, 5))
    plt.hist(bol_energies.value.flatten(), histtype="step", bins=50, color="black")
    plt.xlabel("Bolometric Flare Energy [ergs]")
    plt.savefig(f"../plots/{file}/flares/hip6752_model_posterior_bolometric_energy.png")

    # ---------------------------------------------------------------------------------


    # get total masked data points
    print(f"Total masked data points: {mask.sum()}")


    # WRITE THE RESULTS TO A CSV FILE ---------------------------------------------------
    with open(f"../results/cheops_flares.csv", "a") as f:
        # f.write("date,med_flux,amplitude,t_peak_BJD,dur_d,amplitude2,t_peak_BJD2,dur_d2,ED,"
        #         "EDerr,mean_bol_energy,std_bol_energy,ingress,egress,tmin,tmax,parametrization,tot_obs_time_d\n")
        if two_flare:
            f.write(f"{file},{newmed},{ampl},{t_peak},{dur},{ampl2},{t_peak2},{dur2},{ED:.2e}," +
                    f"{EDerr:.2e},{mean_bol_energy},{std_bol_energy},{ingress},{egress}," + 
                    f"{tmin},{tmax},{parametrization},{tot_obs_time_d}\n")
        else:
            f.write(f"{file},{newmed},{ampl},{t_peak},{dur},,,,{ED:.2e}," +
                    f"{EDerr:.2e},{mean_bol_energy},{std_bol_energy},{ingress},{egress}," + 
                    f"{tmin},{tmax},{parametrization},{tot_obs_time_d}\n")
        
    # ---------------------------------------------------------------------------------










