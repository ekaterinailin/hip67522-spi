# -*- coding: utf-8 -*-
"""
@author: Ekaterina Ilin, 2025, ilin@astron.nl

This script is used to detrend segments of light curves around flares
found in TESS light curves, to get a baseline for the flare fitting.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from funcs.pipe import extract

import batman

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# capture the screen output to file
import sys



flux_label = r"Flux [e$^{-}$/s]"
time_label = "Time [BJD]"




def metafunc(offset2, transit):
    """Defines a polynomial function with a time offset and a known transit included.
    
    Parameters
    ----------
    offset2 : float
        Time offset. Use the last time stamp in light curve.
    transit : array
        Transit model. Use the batman model.

    Returns
    -------
    func : function
        Function that can be used to fit the light curve.
    
    """
    def func(x, a, b, c, d, e, f, offset):
        return (f * (x - offset2 + offset)**5 + 
                e * (x - offset2 + offset)**4 + 
                a * (x - offset2 + offset)**3 + 
                b * (x - offset2 + offset)**2 + 
                c * (x - offset2 + offset) + d + 
                transit)

    return func
    

if __name__ == '__main__':


    # flares identified in Ilin+2024
    location = "data/tess_flares_input.csv"

    flares = pd.read_csv(location)

    for i, row in flares.iterrows():

        # drop these flares because we will treat them together will flare 2 which is close in time to both
        if i == 3 or i == 4:
            continue

        sector = row.qcs
        tstart, tstop = row.tstart, row.tstop

        print(f"Processing sector {sector} from {tstart} to {tstop}")
        

        # get LC data
        hdu = fits.open(f"data/tess/tess_hip67522_{sector}.fits")

        t = hdu[1].data["TIME"]
        f = hdu[1].data["PDCSAP_FLUX"]
        ferr = hdu[1].data["PDCSAP_FLUX_ERR"]
        flag = hdu[1].data["QUALITY"]

        # make sure the data is in fact 2-min cadence
        assert np.diff(t[np.isfinite(t)]).min() * 24 * 60 * 60 < 121, "Time series is not 2min cadence"

        buffer = 30 / 60 / 24 # 30 minutes
        total_mask_buffer = 0.2 # d

        # define big mask spanning all three flares under indices 2,3,4
        if i == 2:
            big_init_mask = ((flag==0) &
                                np.isfinite(t) &
                                np.isfinite(f) &
                                np.isfinite(ferr) &
                                (~np.isnan(t)) &
                                (~np.isnan(f)) &
                                (~np.isnan(ferr)) &
                                (t > flares.tstart[2] - total_mask_buffer + 0.05) & # bit of tinkering here to a wide enough
                                (t < flares.tstop[4] + total_mask_buffer + 0.2 )) # but not too wide mask

            print(f"Big initial mask: {np.where(~big_init_mask)[0].shape[0]} data points")

            flare_mask = ((t > flares.tstart[2] - buffer * 1.45) &
                              (t < flares.tstop[4] + buffer * 4) &
                             ~((t>2335.05) & (t<2335.16)))
            print(f"Big flare mask: {np.where(flare_mask)[0].shape[0]} data points")

            # define flare light curve
            tflare, fflare, ferrflare, flagflare = [arr[flare_mask & big_init_mask] for arr in [t, f, ferr, flag]]

            # apply the mask
            t, f, ferr, flag = [arr[big_init_mask & ~flare_mask] for arr in [t, f, ferr, flag]]

        else:

            # initial mask

            init_mask = ((flag==0) & 
                        np.isfinite(t) &
                        np.isfinite(f) & 
                        np.isfinite(ferr) &
                        (~np.isnan(t)) &  
                            (~np.isnan(f)) &
                            (~np.isnan(ferr)) &
                        (t > tstart - total_mask_buffer) & 
                        (t < tstop + total_mask_buffer))
            
            print(f"Initial mask: {np.where(~init_mask)[0].shape[0]} data points")

            # define flare mask

            if row.ampl_rec < 0.03:
                factor_end = 2
            else:
                factor_end = 8
            flare_mask = (t > tstart - buffer) & (t < tstop + buffer*factor_end) 
            print(f"Flare mask: {np.where(flare_mask)[0].shape[0]} data points")

            # define flare light curve
            tflare, fflare, ferrflare, flagflare = [arr[flare_mask & init_mask] for arr in [t, f, ferr, flag]]

            # apply the mask
            t, f, ferr, flag = [arr[init_mask & ~flare_mask] for arr in [t, f, ferr, flag]]

        t = t.astype(float)
        f = f.astype(float)
        ferr = ferr.astype(float)
        flag = flag.astype(int)
       

        # PLOT THE INITIAL LIGHT CURVE -------------------------------------------------

        plt.figure(figsize=(10, 5))
        plt.plot(t, f, ".", markersize=1)
        plt.plot(tflare, fflare, ".", markersize=6, color="red")

        plt.xlabel(time_label)
        plt.ylabel(flux_label)
        plt.title("Initial light curve, masking outliers")
        plt.savefig(f"plots/diagnostic/tess/hip67522_flare_{i}_{sector}.png")

        # -----------------------------------------------------------------------------


        # DEFINE A TRANSIT MODEL USING BARBER ET AL. 2024 PARAMETERS -------------------

        # use batman to create a transit model
        params = batman.TransitParams()

        params.t0 = 1604.02344             #time of inferior conjunction in BTJD
        params.per = 6.9594738               #orbital period
        params.rp = 0.0668                      #planet radius (in units of stellar radii)
        params.a = 11.74                       #semi-major axis (in units of stellar radii)
        params.inc = 89.46                     #orbital inclination (in degrees)
        params.ecc = 0.053                      #eccentricity
        params.w = 199.1                       #longitude of periastron (in degrees)
        params.u = [0.22, 0.27]                #limb darkening coefficients [u1, u2]
        params.limb_dark = "quadratic"       #limb darkening model

        m = batman.TransitModel(params, t)    #initializes model

        transit = m.light_curve(params)          #calculates light curve

        # make a mask of the transit defined a everything below 1
        transit_mask = transit < 1

        transit = (transit - 1) * np.median(f) # scale to the median flux


        # -----------------------------------------------------------------------------

        # FIT A POLYNOMIAL MODEL -------------------------------------------------------

        # 5th degree polynomial with a time offset and a transit
        modelfunc = metafunc(t[-1], transit)

        # fit the model to the light curve
        
        popt, pcov = curve_fit(modelfunc, t, f,p0=[-1.45888787e+04, -1.41685433e+08, -1.03596058e+09,  1.00000000e+00,
            1.19292031e-02, -2.42900480e-09,  8.42088604e-01])
        # print the fitted parameters
        print("Fitted parameters for the first iteration: ", popt)

        # get the fitted model
        fitted = modelfunc(t, *popt)

        # PLOT THE FITTED MODEL -------------------------------------------------------

        plt.figure(figsize=(10, 5))

        # flux
        plt.plot(t, f, ".", markersize=2, color="red")
        # fitted model
        plt.plot(t, fitted, color="blue", lw=1)

        plt.xlabel(time_label)
        plt.ylabel(flux_label)
        plt.title("First polynomial fit to the light curve w/o flare")
        plt.savefig(f"plots/diagnostic/tess/hip67522_polyfit_init_{i}.png")

        # -----------------------------------------------------------------------------

        # SUBTRACT THE FITTED MODEL  ---------------------------------------------------

        # median
        med = np.median(f)

        # subtract the fitted model
        f_sub = f - fitted + med

        # get a new median
        newmed = np.median(f_sub)


        # MASK OUTLIERS ----------------------------------------------------------------

        # mask out the outliers
        outlier_mask = (f_sub < newmed + 3 * np.std(f_sub)) & (f_sub > newmed - 3 * np.std(f_sub))

        print(f"Masking outliers: {np.where(~outlier_mask)[0].shape[0]} data points")

        plt.figure(figsize=(10, 5))
        plt.plot(t, f, ".", markersize=1)
        plt.plot(t[outlier_mask], f_sub[outlier_mask], ".", markersize=1)
        plt.plot(t[~outlier_mask], f_sub[~outlier_mask], ".", markersize=6, color="red")
        plt.axhline(med, color="red", lw=1)

        plt.xlabel(time_label)
        plt.ylabel(flux_label)
        plt.title("Subtracting the polynomial fit and masking outliers")
        plt.savefig(f"plots/diagnostic/tess//hip67522_subtract_polyfit_init_{i}_{sector}.png")

        # -----------------------------------------------------------------------------


        # UPDATE TRANSIT MODEL WITH SUBTRACTED LIGHT CURVE -----------------------------
        m = batman.TransitModel(params, t[outlier_mask])    #initializes model
        transit = m.light_curve(params)          #calculates light curve

        # make a mask of the transit defined a everything below 1
        transit_mask = transit < 1

        transit = (transit - 1) * np.median(f_sub[outlier_mask])


        # FIT A SECOND POLYNOMIAL MODEL ------------------------------------------------

        # define a new model function with the new transit model
        newmodelfunc = metafunc(t[outlier_mask][-1], transit)

        # fit the new model to the subtracted light curve
        popt, pcov = curve_fit(newmodelfunc, t[outlier_mask], f[outlier_mask], p0=popt)

        # print the fitted parameters
        print("Fitted parameters for the second iteration: ", popt)

        # get the new fitted model but use the full array
        newfitted = modelfunc(t, *popt)


        # subtract the new fitted model
        newf_sub = f - newfitted + newmed

        # PLOT THE NEW FITTED MODEL RESIDUALS WITH THE OLD FITTED MODEL  ----------------

        plt.figure(figsize=(10, 5))
        plt.plot(t, fitted-newfitted, color="blue", lw=1)
        plt.xlabel(time_label)
        plt.ylabel(flux_label)
        plt.title("Difference between first and second polynomial fit")
        plt.savefig(f"plots/diagnostic/tess/hip67522_polyfit_diff_{i}_{sector}.png")

        # --------------------------------------------------------------------------------

        # PLOT THE FINAL LIGHT CURVE WITH MODEL ------------------------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(t, f, ".", markersize=1, color="grey")
        plt.plot(t, newfitted, ".", markersize=1, color="black")
        plt.xlabel(time_label)
        plt.ylabel(flux_label)
        plt.title("Second polynomial fit to the light curve w/o flare")
        plt.savefig(f"plots/diagnostic/tess/hip67522_polyfit_final_{i}_{sector}.png")

        # --------------------------------------------------------------------------------



        # SMOOTH THE RESIDUALS WITH A SAVITZKY-GOLAY FILTER -----------------------------

        # smooth the light curve
        smoothed = savgol_filter(newf_sub, len(newf_sub)//5, 3)

        # PLOT THE SMOOTHED LIGHT CURVE -------------------------------------------------

        plt.figure(figsize=(10, 5))
        plt.plot(t, newf_sub, ".", markersize=1, color="grey")
        plt.plot(t, smoothed, ".",  color="black")
        plt.xlabel(time_label)
        plt.ylabel(flux_label)

        plt.title("Savitzky-Golay smoothed light curve")
        plt.savefig(f"plots/diagnostic/tess/hip67522_savgol_model_{i}_{sector}.png")

        # --------------------------------------------------------------------------------

        ff = newf_sub - smoothed + np.median(newf_sub)

        # PLOT THE SAVGOL DE-TRENDED LIGHT CURVE ----------------------------------------------------

        plt.figure(figsize=(10, 5))
        plt.plot(t, newf_sub, ".", markersize=1, color="grey")
        plt.xlabel(time_label)
        plt.ylabel(flux_label)
        plt.title("Savitzky-Golay de-trended light curve")
        plt.savefig(f"plots/diagnostic/tess/hip67522_savgol_detrended_lc_{i}_{sector}.png")


        # REINTRODUCE FLARES TO THE FINAL LIGHT CURVE -----------------------------------------

      
        notransitmodelfunc = metafunc(t[outlier_mask][-1], 0)

        # subtract the flare model and roll angle correction
        ff = np.append(ff[outlier_mask], fflare + newmed -notransitmodelfunc(tflare, *popt))

        t = np.append(t[outlier_mask], tflare)

        ferr = np.append(ferr[outlier_mask], ferrflare)
        transit_mask = np.append(transit_mask, np.zeros_like(tflare))
        

        print(flag.shape, transit_mask.shape)
        newfitted = np.append(newfitted[outlier_mask], notransitmodelfunc(tflare, *popt))

        # make a raw flux with the flare
        raw_f = np.append(f[outlier_mask], fflare)

        flare_mask = np.append(np.zeros_like(f[outlier_mask]), np.ones_like(tflare))

       
        #Print all array shapes
        print("Final array shapes:")
        print("Time: ", t.shape)
        print("Flux: ", ff.shape)
        print("Model: ", newfitted.shape)
        print("Flux error: ", ferr.shape)
        print("Raw flux: ", raw_f.shape)
        print("Transit mask: ", transit_mask.shape)
        print("Flare mask: ", flare_mask.shape)


        # sort all arrays by time
        t, ff, newfitted, ferr, raw_f = [arr[np.argsort(t)] for arr in [t, ff, newfitted, ferr, raw_f]]


        # PLOT THE FINAL FLUX  ----------------------------------------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(t, ff, ".", markersize=1, color="red", label="quiescent model")

        plt.axhline(newmed, color="black", lw=1, label="median")

        plt.xlabel(time_label)
        plt.ylabel(flux_label)
        plt.title("Final de-trendend light curve")
        plt.savefig(f"plots/diagnostic/tess/hip67522_final_detrended_light_curve_{i}_{sector}.png")

        # --------------------------------------------------------------------------------


        # WRITE THE FINAL LIGHT CURVE TO A CSV FILE ------------------------------------------
        df = pd.DataFrame({"time": t, "flux": ff, "model" : newfitted, 
                        "flux_err": ferr, "masked_raw_flux": raw_f,
                        "transit_mask": transit_mask,
                        "flare_mask" : flare_mask})

        # new PIPE
        df.to_csv(f"results/tess/HIP67522_detrended_lc_{i}_{sector}.csv", index=False)

        # WRITE THE INITAL MASK TO A txt FILE ------------------------------------------

        np.savetxt(f"results/tess/HIP6752_{i}_{sector}_mask.txt", init_mask, fmt="%d")
