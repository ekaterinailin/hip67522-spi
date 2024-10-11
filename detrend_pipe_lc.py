# -*- coding: utf-8 -*-
"""
@author: Ekaterina Ilin, 2024, ilin@astron.nl

This script is used to detrend the light curves after 
the imagette data cubes have been reduced with PIPE.
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

    # GET THE IMAGE DATA -----------------------------------------------------------
    pi = sys.argv[1]
    file = sys.argv[2]

    sys.stdout = open(f"../plots/{file}{pi}/detrending.log", "w")

    # new PIPE resulst
    IMG = f'../data/hip67522/pipe_HIP67522/HIP67522{pi}_{file}_im.fits'

    # make a folder in the plots directory for each file
    import os
    if not os.path.exists(f"../plots/{file}{pi}/flares"):
        os.makedirs(f"../plots/{file}{pi}/flares")


    # open the fits file
    hdulist = fits.open(IMG)
    print(f"Imagette file found for {file}{pi}:\n {IMG}\n")

    # get the image data
    image_data = hdulist[1].data


    # get LC data
    t, f, ferr, roll, dT, flag, bg, xc, yc = [extract(image_data, stri) for stri in ["BJD_TIME", "FLUX", "FLUXERR",
                                                                                    "ROLL", "thermFront_2", "FLAG",
                                                                                    "BG", "XC", "YC"]]
    
    raw_f = f.copy()

    # make sure the data is in fact 10s cadence
    assert np.diff(t).min() * 24 * 60 * 60 < 10.05, "Time series is not 10s cadence"

    # initial mask
    init_mask = (f < 2.96e6) & (f > 2.3e6) & (flag==0) #& (t < 2460413.3) only for the 102ch flare fit bc the polynomial is slightly off there
    print(f"Initial mask: {np.where(~init_mask)[0].shape[0]} data points")


    # GET THE KNOWN FLARES --------------------------------------------------------

    flares = pd.read_csv(f"../results/cheops_flares.csv")

    # convert flares["date"] to string
    # flares["date"] = flares["date"].astype(str)
    flares["newpipe"] = flares["newpipe"].astype(str)

    name = f"{file}{pi}"
    if str(name) in flares["newpipe"].values:
        print(name)
        flare = flares[flares["newpipe"] == name]
        flare_mask = (t > flare["tmin"].values[0]) & (t < flare["tmax"].values[0])
        print(f"Flare mask: {np.where(flare_mask)[0].shape[0]} data points")
        fullinit_mask = init_mask & ~flare_mask

        # define the flare light curve
        tflare, fflare, ferrflare, rollflare, dTflare, flagflare, bgflare, xcflare, ycflare = [arr[flare_mask & init_mask] for arr in 
                                                                                           [t, f, ferr, roll, dT, flag, bg, xc, yc]]


        print(f"Masking flares: {np.where(flare_mask)[0].shape[0]} data points")

    else:
        fullinit_mask = init_mask

    # apply the mask
    t, f, ferr, roll, dT, flag, bg, xc, yc = [arr[fullinit_mask] for arr in [t, f, ferr, roll, dT, flag, bg, xc, yc]]

    # make a diagnostic plot of the residuals on the detector 
    # get_residual_image(file, index=664)

    # PLOT THE INITIAL LIGHT CURVE -------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(t, f, ".", markersize=1)

    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Initial light curve, masking outliers")
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_initial_lc.png")

    # -----------------------------------------------------------------------------


    # DEFINE A TRANSIT MODEL USING BARBER ET AL. 2024 PARAMETERS -------------------

    # use batman to create a transit model
    params = batman.TransitParams()

    params.t0 = 1604.02344 + 2457000.             #time of inferior conjunction in BJD
    params.per = 6.9594738               #orbital period
    params.rp = 0.0668                      #planet radius (in units of stellar radii)
    params.a = 11.74                       #semi-major axis (in units of stellar radii)
    params.inc = 89.46                     #orbital inclination (in degrees)
    params.ecc = 0.053                      #eccentricity
    params.w = 199.1                       #longitude of periastron (in degrees)
    params.u = [0.22, 0.27]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model

    # print the parameters
    print("Batman transit parameters from Barber et al. 2024:")
    print(vars(params))

    m = batman.TransitModel(params, t)    #initializes model
    transit = m.light_curve(params)          #calculates light curve

    transit = (transit - 1) * np.median(f) # scale to the median flux

    # -----------------------------------------------------------------------------

    # FIT A POLYNOMIAL MODEL -------------------------------------------------------

    # 5th degree polynomial with a time offset and a transit
    modelfunc = metafunc(t[-1], transit)

    # fit the model to the light curve
    
    popt, pcov = curve_fit(modelfunc, t, f, p0=[-1.45888787e+04, -1.41685433e+08, -1.03596058e+09,  1.00000000e+00,
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
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_polyfit_init.png")

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
    outlier_mask = (f_sub < newmed + 4 * np.std(f_sub)) & (f_sub > newmed - 4 * np.std(f_sub))

    print(f"Masking outliers: {np.where(~outlier_mask)[0].shape[0]} data points")

    plt.figure(figsize=(10, 5))
    plt.plot(t, f, ".", markersize=1)
    plt.plot(t[outlier_mask], f_sub[outlier_mask], ".", markersize=1)
    plt.plot(t[~outlier_mask], f_sub[~outlier_mask], ".", markersize=6, color="red")
    plt.axhline(med, color="red", lw=1)

    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Subtracting the polynomial fit and masking outliers")
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_subtract_polyfit_init.png")

    # -----------------------------------------------------------------------------


    # UPDATE TRANSIT MODEL WITH SUBTRACTED LIGHT CURVE -----------------------------
    m = batman.TransitModel(params, t[outlier_mask])    #initializes model
    transit = m.light_curve(params)          #calculates light curve

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
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_polyfit_diff.png")

    # --------------------------------------------------------------------------------

    # PLOT THE FINAL LIGHT CURVE WITH MODEL ------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(t, f, ".", markersize=1, color="grey")
    plt.plot(t, newfitted, ".", markersize=1, color="black")
    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Second polynomial fit to the light curve w/o flare")
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_polyfit_final.png")

    # --------------------------------------------------------------------------------

    # PLOT THE FINAL QUIET LIGHT CURVE AGAINST ROLL ANGLE ----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(roll, newf_sub, ".", markersize=1)
    plt.xlabel("Roll")
    plt.ylabel(flux_label)
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_roll_flux.png")

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
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_savgol_model.png")

    # --------------------------------------------------------------------------------

    newf_sub = newf_sub - smoothed + np.median(newf_sub)

    # PLOT THE SAVGOL DE-TRENDED LIGHT CURVE ----------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(t, newf_sub, ".", markersize=1, color="grey")
    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Savitzky-Golay de-trended light curve")
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_savgol_detrended_lc.png")

    # --------------------------------------------------------------------------------

    # ROLL ANGLE CORRECTION ---------------------------------------------------------

    # approximate the flux at the roll values in the flare region with 
    # the flux at the closest roll value in the non-flare region

    f_sub_no_flare_approx = np.zeros_like(newf_sub)
    for i, r in enumerate(roll):
        
        idx = [np.argmin(np.abs(roll - r - delt)) for delt in np.linspace(-2, 2, 100)]
        
        f_sub_no_flare_approx[i] = np.median(newf_sub[idx])


    # DEFINE THE FINAL FLUX ---------------------------------------------------------

    # final flux 
    ff = newf_sub - f_sub_no_flare_approx + newmed

    # REINTRODUCE FLARES TO THE FINAL LIGHT CURVE -----------------------------------------

    # if str(file) in flares["date"].values:
    if str(name) in flares["newpipe"].values:
        
        notransitmodelfunc = metafunc(t[outlier_mask][-1], 0)


        # do the roll angle correction for the flare region
        f_sub_flare_approx = np.zeros_like(fflare)
        for i, r in enumerate(rollflare):
            idx = [np.argmin(np.abs(roll - r - delt)) for delt in np.linspace(-2, 2, 100)]
            f_sub_flare_approx[i] = np.median(newf_sub[idx])

        # subtract the flare model and roll angle correction
        ff = np.append(ff[outlier_mask], fflare + 2 * newmed - f_sub_flare_approx -notransitmodelfunc(tflare, *popt))

        t = np.append(t[outlier_mask], tflare)
        bg = np.append(bg[outlier_mask], bgflare)
        xc = np.append(xc[outlier_mask], xcflare)
        yc = np.append(yc[outlier_mask], ycflare)
        dT = np.append(dT[outlier_mask], dTflare)
        roll = np.append(roll[outlier_mask], rollflare)
        ferr = np.append(ferr[outlier_mask], ferrflare)
        
        newfitted = np.append(newfitted[outlier_mask], notransitmodelfunc(tflare, *popt))

        # add a mask for the flare region
        flag = np.append(flag, np.ones_like(fflare))

        # make a raw flux with the flare
        raw_f = np.append(raw_f[fullinit_mask][outlier_mask], fflare)
    
    else:
        t = t[outlier_mask]
        newfitted = newfitted[outlier_mask]
        ff = ff[outlier_mask]
        bg = bg[outlier_mask]
        xc = xc[outlier_mask]
        yc = yc[outlier_mask]
        dT = dT[outlier_mask]
        roll = roll[outlier_mask]
        ferr = ferr[outlier_mask]
        flag = flag[outlier_mask]


        raw_f = raw_f[fullinit_mask][outlier_mask]

    #Print all array shapes
    print("Final array shapes:")
    print("Time: ", t.shape)
    print("Flux: ", ff.shape)
    print("Model: ", newfitted.shape)
    print("Flux error: ", ferr.shape)
    print("Roll: ", roll.shape)
    print("dT: ", dT.shape)
    print("Flag: ", flag.shape)
    print("Background: ", bg.shape)
    print("Xc: ", xc.shape)
    print("Yc: ", yc.shape)
    print("Raw flux: ", raw_f.shape)


    # sort all arrays by time
    t, ff, newfitted, ferr, roll, dT, flag, bg, xc, yc, raw_f = [arr[np.argsort(t)] for arr in [t, ff, newfitted, ferr, roll, dT, flag, bg, xc, yc, raw_f]]


    # PLOT THE FINAL FLUX  ----------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(t, ff, ".", markersize=1, color="red", label="quiescent model")

    plt.axhline(newmed, color="black", lw=1, label="median")

    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Final de-trendend light curve")
    plt.savefig(f"../plots/{file}{pi}/flares/hip67522_final_detrended_light_curve.png")

    # --------------------------------------------------------------------------------


    # WRITE THE FINAL LIGHT CURVE TO A CSV FILE ------------------------------------------
    
    df = pd.DataFrame({"time": t, "flux": ff, "model" : newfitted, "flux_err": ferr, "masked_raw_flux": raw_f,
                       "roll": roll, "dT": dT, "flag": flag, "bg": bg, "xc": xc, "yc": yc})

    # new PIPE
    df.to_csv(f"../data/hip67522/pipe_HIP67522/HIP67522_{file}{pi}_detrended_lc.csv", index=False)

    # WRITE THE INITAL MASK TO A txt FILE ------------------------------------------

    np.savetxt(f"../data/hip67522/pipe_HIP67522/HIP67522_{file}{pi}_mask.txt", fullinit_mask, fmt="%d")
