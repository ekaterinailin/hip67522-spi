"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


Extracts the flare light curve from a TESS light curve.
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

    # init the results table
    with open(f"../results/tess_flares.csv", "a") as f:
        f.write("sector,flare,med_flux,amplitude,t_peak_BJD,dur_d,ED,"
                "EDerr,mean_bol_energy,std_bol_energy,parametrization\n")

    parametrization = "mendoza2022"#"davenport2014"

    location = "../data/hip67522_tess_flares.csv"

    flares = pd.read_csv(location)

    for i, row in flares.iterrows():

        if (i == 3) or (i == 4):
            continue 
        sector = row.qcs
        tstart, tstop = row.tstart, row.tstop
        t_peak = (tstop + tstart) / 2

        print(f"Processing sector {sector} from {tstart} to {tstop}")


        lc = pd.read_csv(f"../data/tess/HIP67522_detrended_lc_{i}_{sector}.csv")


        newmed = lc.flux[(lc.time.values < tstart)].median()
        t_flare, f_flare = lc.time.values, lc.flux.values - newmed



        # PLOT THE FLARE LIGHT CURVE --------------------------------------------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(t_flare, f_flare+newmed, ".", markersize=1, color="black")

        plt.axhline(newmed, color="red", lw=1)
        plt.xlabel("Time [BJD]")
        plt.ylabel(r"Flux [e$^{-}$/s]")
        plt.title("Flare region flux after subtraction of quiescent model")

        plt.savefig(f"../plots/tess/hip67522_flare_detrended_{i}_{sector}.png", dpi=300)
        # -----------------------------------------------------------------------------------------

        # GET A FIRST GUESS OF THE FLARE PARAMETERS ------------------------------------------------

        # same dur and ampl for everyone
        dur = 0.02
        ampl = 0.01*newmed

        if i == 2:

            dur = 0.01

            p1, p2, p3 = [(flares.iloc[k].tstart + flares.iloc[k].tstop) / 2 for k in [2,3,4]]

            def flare_fit_model(t, p1, p2, p3, dur1, ampl1, dur2, ampl2, dur3, ampl3):
                return (flare_model(parametrization, t, p1, dur1, ampl1) +
                        flare_model(parametrization, t, p2, dur2, ampl2) + 
                        flare_model(parametrization, t, p3, dur3, ampl3))
            
            p0 = [p1, p2, p3, dur, ampl, dur, ampl, dur, ampl]
            bounds = ([p1-0.05, p2-0.05, p3-0.05, 0, 0, 0, 0, 0, 0],
                      [p1+0.05, p2+0.05, p3+0.05, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        else:
        
            # define flare model using the parametrization
            def flare_fit_model(t, t_peak, dur, ampl):
                return flare_model(parametrization, t, t_peak, dur, ampl)
            
            p0 = [t_peak, dur, ampl]
            bounds = ([tstart-0.05, 0, 0], [tstop+0.05,np.inf,np.inf])
        
        # fit the flare model to the flare region
        popt, pcov = curve_fit(flare_fit_model, t_flare, f_flare, 
                            p0=p0, bounds=bounds)

        # get the fitted flare model
        print("Fitted flare parameters:")
        print(popt)


        # USE MCMC TO GET THE FLARE PARAMETERS ---------------------------------------------------


        if i == 2:
            # define the log likelihood, prior, and probability
            def log_likelihood(theta, t, f, ferrstd):
                p1, p2, p3, dur1, ampl1, dur2, ampl2, dur3, ampl3 = theta
                model = flare_fit_model(t, p1, p2, p3, dur1, ampl1, dur2, ampl2, dur3, ampl3)
                return -0.5 * np.sum((f - model)**2 / ferrstd**2)
            
            def log_prior(theta):
                p1, p2, p3, dur1, ampl1, dur2, ampl2, dur3, ampl3 = theta
                if ((t_flare[0] < p1 < t_flare[-1]) & 
                    (t_flare[0] < p2 < t_flare[-1]) & 
                    (t_flare[0] < p3 < t_flare[-1]) & 
                    (0.0 < dur1 < 0.2) & 
                    (0.001 < ampl1 < 1e7) &
                    (0.0 < dur2 < 0.2) & 
                    (0.001 < ampl2 < 1e7) &
                    (0.0 < dur3 < 0.2) & 
                    (0.001 < ampl3 < 1e7)):
                    return 0.0
                return -np.inf
            
        else:

            # define the log likelihood, prior, and probability
            def log_likelihood(theta, t, f, ferrstd):
                t_peak, dur, ampl = theta
                model = flare_fit_model(t, t_peak, dur, ampl)
                return -0.5 * np.sum((f - model)**2 / ferrstd**2)

            def log_prior(theta):
                t_peak, dur, ampl = theta
                if (t_flare[0] < t_peak < t_flare[-1]) and (0.0 < dur < 0.2) and (0.001 < ampl < 1e7):
                    return 0.0
                return -np.inf

        def log_probability(theta, t, f, ferrstd):
            lp = log_prior(theta)
            logl = log_likelihood(theta, t, f, ferrstd)
            if (not np.isfinite(lp)) | (not np.isfinite(logl)):
                return -np.inf
            return lp + logl

        # define the number of dimensions, walkers, and steps
        if i == 2:
            ndim = 9
        else:
            ndim = 3

      
        nwalkers = 32
        nsteps = 4000

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

        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = [r"$t_p$", "FWHM", r"$a$"]
        if ndim == 9:
            labels = [r"$t_1$", r"$t_2$", r"$t_2$", r"FWHM$_1$", r"$a_1$", r"FWHM$_2$", r"$a_2$", r"FWHM$_3$", r"$a_3$"]
        for j in range(ndim):
            ax = axes[j]
            ax.plot(samples[:, :, j], "k", alpha=0.3)
            ax.set_ylabel(labels[j])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig(f"../plots/tess/hip67522_oneflare_model_mcmc_chains_{i}_{sector}.png")
        # ---------------------------------------------------------------------------------

        # SHOW THE CORNER PLOT -----------------------------------------------------------

        # get the flat samples
        flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

        # plot the corner plot
        fig = corner.corner(flat_samples, labels=labels)
        plt.savefig(f"../plots/tess/hip67522_oneflare_model_mcmc_corner_{i}_{sector}.png")

        # ---------------------------------------------------------------------------------

        # SAMPLE SOLUTIONS FROM THE CHAIN AND PLOT THEM ------------------------------------

        if i == 2:

            # get median values
            t_peak1, t_peak2, t_peak3, dur1, ampl1, dur2, ampl2, dur3, ampl3 = np.median(flat_samples, axis=0)

        else:

            # get median values
            t_peak, dur, ampl = np.median(flat_samples, axis=0)

        # init a light curve for interpolation
        ndat = int((t_flare.max() - t_flare.min()) * 24 * 60 * 6)
        t_interpolate = np.linspace(t_flare.min(), t_flare.max(), ndat)

        # initif figure
        plt.figure(figsize=(10, 5))

        # get the equivalent durations, too
        eds = []

        # loop over the samples 500 times
        for j in np.random.randint(len(flat_samples), size=500):

            # get sample
            sample = flat_samples[j]


            if i == 2:
                # get the equivalent duration from a model of each individual flare
                def flare_fit_model_temp(t, t_peak, dur, ampl):  
                    return flare_model(parametrization, t, t_peak, dur, ampl)
                

                ed3 = []
                for k in range(3):
                    flarelc = flare_fit_model_temp(t_interpolate, sample[k], sample[2*k+3], sample[2*k+4])
                    flc = FlareLightCurve(time=t_interpolate, flux = flarelc + newmed, flux_err=ferrstd)
                    flc.it_med = newmed
                    flc.detrended_flux = flarelc + newmed
                    ed = equivalent_duration(flc, 0, len(t_interpolate)-1)  
                    ed3.append(ed)
                    plt.plot(t_interpolate, flarelc + newmed,  ".", 
                         color="red", alpha=0.01, markersize=1)
                    
                    plt.plot(t_interpolate, flarelc + newmed, ".", markersize=1, color="black")

                eds.append(np.array(ed3))


            else:


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

        if i==2:
            plt.title("Best fit two-flare model")
        else:
            plt.title("Best fit one-flare model")

        # plot the flare region on top
        plt.plot(t_flare, f_flare+newmed, ".", markersize=1, color="black")

        # newmed
        plt.axhline(newmed, color="grey", lw=1, linestyle="--")    

        plt.xlabel("Time [BJD]")
        plt.ylabel(r"Flux [e$^{-}$/s]")
        plt.xlim(t_flare.min(), t_flare.max())

        plt.savefig(f"../plots/tess/hip67522_flare_model_bestfit_{i}_{sector}.png")

        # ---------------------------------------------------------------------------------

        # PLOT THE RESIDUALS ---------------------------------------------------------------

        # subtract model from f_flare
        if i==2:
            residuals = f_flare - flare_fit_model(t_flare, t_peak1, t_peak2, t_peak3, dur1, ampl1, dur2, ampl2, dur3, ampl3)
            title = "Residuals of the three-flare model"
        else:
            residuals = f_flare - flare_fit_model(t_flare, t_peak, dur, ampl) 
            title = "Residuals of the one-flare model"

        plt.figure(figsize=(10, 5))
        plt.plot(t_flare, residuals+newmed, ".", markersize=3)
        plt.plot(lc.time[lc.flag==0], lc.flux[lc.flag==0], ".", markersize=1)
        plt.axhline(newmed, color="red", lw=1)
        plt.xlabel("Time [BJD]")
        plt.ylabel(r"Flux [e$^{-}$/s]")
        plt.title(title)
        plt.savefig(f"../plots/tess/hip67522_oflare_model_residuals_{i}_{sector}.png")
        # ---------------------------------------------------------------------------------


        # GET AND PLOT EQUIVALENT DURATIONS ------------------------------------------------
        plt.figure(figsize=(6, 5))
        if i ==2:


            eds = np.array(eds).T

            print(eds.shape)
            
            EDs, EDerrs = [], []
            for ed in eds:
                plt.hist(ed, histtype="step", bins=50, color="black")
                print(len(ed))
                EDs.append(np.median(ed))
                EDerrs.append(np.std(ed))
 
                print(f"Equivalent duration: {np.median(ed)} +/- {np.std(ed)}")
        else:
            plt.hist(eds, histtype="step", bins=50, color="black")

            ED = np.median(eds)
            EDerr = np.std(eds)

            print(f"Equivalent duration: {ED} +/- {EDerr}")
        plt.xlabel("Equivalent Duration [s]")
        plt.savefig(f"../plots/tess/hip67522_model_posterior_ED_{i}_{sector}.png")



        # ---------------------------------------------------------------------------------


        # GET BOLOMETRIC FLARE ENERGY ------------------------------------------------------
        
        # read TESS response function
        tess_resp = pd.read_csv("../data/tess-response-function-v2.0.csv", skiprows=7, 
                                header=None, names=["Wavelength", "Throughput"])
        
        wav, resp = tess_resp.Wavelength.values, tess_resp.Throughput.values

        # effective temperature of HIP 67522 from Rizzuto et al. 2020
        teff = 5650 # +- 75

        # radius of HIP 67522 from Rizzuto et al. 2020
        radius = 1.392 # +- 0.05

        print(f"Effective temperature: {teff:.0f} K")
        print(fr"Radius: {radius:.3f} solar radii")

        tflare = 10000.  # K

        # calculate bolometric flare energy
        print("\n We use an 10000K flare temperature.")
        if i == 2:
            bol_energy = flare_factor(teff, radius, wav, resp,  tflare=tflare) * EDs * u.s
        else:
            bol_energy = flare_factor(teff, radius, wav, resp,  tflare=tflare) * ED * u.s


        print("\nBolometric flare energy in ergs:")
        if i == 2:
            print(f"{bol_energy[0]:.2e}")
            print(f"{bol_energy[1]:.2e}")
            print(f"{bol_energy[2]:.2e}")
        else:
            print(f"{bol_energy:.2e}")
        
        # ---------------------------------------------------------------------------------

        # GET POSTERIOR DISTRIBUTION OF THE FLARE ENERGY ------------------------------------

        # sample the posterior distribution of the flare energy using the MCMC samples on ED, and teff and radius with Gaussian errors
        teff = np.random.normal(5650, 75, 500)
        radius = np.random.normal(1.392, 0.05, 500)

        # calculate the bolometric flare energy for each sample
        ffactor = flare_factor(teff.reshape((500,1)), radius.reshape((500,1)), wav, resp,  tflare=tflare)
        plt.figure(figsize=(6, 5))

        if i == 2:
            meens, estd, mean_ffactors = [], [], []
            for ed in eds:
                bol_energies = ffactor * np.random.choice(ed, (500)) * u.s
            
                # calculate the mean and standard deviation of the bolometric flare energy
                mean_bol_energy = np.mean(bol_energies)
                std_bol_energy = np.std(bol_energies)
                mean_ffactor = np.mean(ffactor)

                meens.append(mean_bol_energy)
                estd.append(std_bol_energy)
                mean_ffactors.append(mean_ffactor)




        else:    
            bol_energies = ffactor * np.random.choice(eds, 500) * u.s

            # calculate the mean and standard deviation of the bolometric flare energy
            mean_bol_energy = np.mean(bol_energies)
            std_bol_energy = np.std(bol_energies)
            mean_ffactor = np.mean(ffactor)

            # plot the distribution of the bolometric flare energy
            
        if i == 2:
            for j in range(3):
                plt.hist(bol_energies[:,j].value.flatten(), histtype="step", bins=50, color="black")
        else:
            plt.hist(bol_energies.value.flatten(), histtype="step", bins=50, color="black")
        plt.xlabel("Bolometric Flare Energy [ergs]")
        plt.savefig(f"../plots/tess/hip6752_model_posterior_bolometric_energy_{i}_{sector}.png")



        # WRITE THE RESULTS TO A CSV FILE ---------------------------------------------------
        with open(f"../results/tess_flares.csv", "a") as f:
            if i ==2:
                tpeaks = [t_peak1, t_peak2, t_peak3]
                amps = [ampl1, ampl2, ampl3]
                durs = [dur1, dur2, dur3]
                for k in range(3):
                    f.write(f"{sector},{i+k},{newmed},{amps[k]},{tpeaks[k]},{durs[k]},{EDs[k]},"
                        f"{EDerrs[k]},{meens[k].value},{estd[k].value},{parametrization}\n")

            else:
                f.write(f"{sector},{i},{newmed},{ampl},{t_peak},{dur},{ED},{EDerr},"
                        f"{mean_bol_energy.value},{std_bol_energy.value},{parametrization}\n")            
        # ---------------------------------------------------------------------------------










