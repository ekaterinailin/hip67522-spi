import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from funcs.pipe import get_residual_image, extract


import batman


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
    file = sys.argv[1]

    # file name
    IMG = f'../data/hip67522/CHEOPS-products-{file}/Outdata/00000/hip67522_CHEOPS-products-{file}_im.fits'
    
    # new PIPE resulst
    IMG = f'../data/hip67522/pipe_HIP67522/HIP67522_{file}_im.fits'

    # make a folder in the plots directory for each file
    import os
    if not os.path.exists(f"../plots/{file}/flares"):
        os.makedirs(f"../plots/{file}/flares")


    # open the fits file
    hdulist = fits.open(IMG)
    print(f"Imagette file found for {file}:\n {IMG}\n")

    # get the image data
    image_data = hdulist[1].data


    # get LC data
    t, f, ferr, roll, dT, flag, bg, xc, yc = [extract(image_data, stri) for stri in ["BJD_TIME", "FLUX", "FLUXERR",
                                                                                    "ROLL", "thermFront_2", "FLAG",
                                                                                    "BG", "XC", "YC"]]

    # make sure the data is in fact 10s cadence
    assert np.diff(t).min() * 24 * 60 * 60 < 10.05, "Time series is not 10s cadence"

    # initial mask
    init_mask = (f < 2.96e6) & (f > 2.3e6) & (flag==0)
    print(f"Initial mask: {init_mask.sum()} data points")


    # GET THE KNOWN FLARES --------------------------------------------------------

    flares = pd.read_csv(f"../results/cheops_flares.csv")

    # convert flares["date"] to string
    # flares["date"] = flares["date"].astype(str)
    flares["newpipe"] = flares["newpipe"].astype(str)

    # if file is in the date of flares add it to the initial mask
    # if str(file) in flares["date"].values:
    #     flare = flares[flares["date"] == file]
    #     flare_mask = (t > flare["tmin"].values[0]) & (t < flare["tmax"].values[0])
    #     print(f"Flare mask: {flare_mask.sum()} data points")
    #     init_mask = init_mask & ~flare_mask

    if str(file) in flares["newpipe"].values:
        print(file)
        flare = flares[flares["newpipe"] == file]
        flare_mask = (t > flare["tmin"].values[0]) & (t < flare["tmax"].values[0])
        print(f"Flare mask: {flare_mask.sum()} data points")
        init_mask = init_mask & ~flare_mask

    print(f"Masking flares: {init_mask.sum()} data points")



    # apply the mask
    t, f, ferr, roll, dT, flag, bg, xc, yc = [arr[init_mask] for arr in [t, f, ferr, roll, dT, flag, bg, xc, yc]]

    # make a diagnostic plot of the residuals on the detector 
    # get_residual_image(file, index=664)

    # PLOT THE INITIAL LIGHT CURVE -------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(t, f, ".", markersize=1)

    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Initial light curve, masking outliers")
    plt.savefig(f"../plots/{file}/flares/hip67522_initial_lc.png")

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
    plt.savefig(f"../plots/{file}/flares/hip67522_polyfit_init.png")

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

    plt.figure(figsize=(10, 5))
    plt.plot(t, f, ".", markersize=1)
    plt.plot(t[outlier_mask], f_sub[outlier_mask], ".", markersize=1)
    plt.plot(t[~outlier_mask], f_sub[~outlier_mask], ".", markersize=6, color="red")
    plt.axhline(med, color="red", lw=1)

    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Subtracting the polynomial fit and masking outliers")
    plt.savefig(f"../plots/{file}/flares/hip67522_subtract_polyfit_init.png")

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
    plt.savefig(f"../plots/{file}/flares/hip67522_polyfit_diff.png")

    # --------------------------------------------------------------------------------

    # PLOT THE FINAL LIGHT CURVE WITH MODEL ------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(t, f, ".", markersize=1, color="grey")
    plt.plot(t, newfitted, ".", markersize=1, color="black")
    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Second polynomial fit to the light curve w/o flare")
    plt.savefig(f"../plots/{file}/flares/hip67522_polyfit_final.png")

    # --------------------------------------------------------------------------------

    # PLOT THE FINAL QUIET LIGHT CURVE AGAINST ROLL ANGLE ----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(roll, newf_sub, ".", markersize=1)
    plt.xlabel("Roll")
    plt.ylabel(flux_label)
    plt.savefig(f"../plots/{file}/flares/hip67522_roll_flux.png")

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
    plt.savefig(f"../plots/{file}/flares/hip67522_savgol_model.png")

    # --------------------------------------------------------------------------------

    newf_sub = newf_sub - smoothed + np.median(newf_sub)

    # PLOT THE SAVGOL DE-TRENDED LIGHT CURVE ----------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(t, newf_sub, ".", markersize=1, color="grey")
    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Savitzky-Golay de-trended light curve")
    plt.savefig(f"../plots/{file}/flares/hip67522_savgol_detrended_lc.png")

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
    if str(file) in flares["newpipe"].values:
        
        notransitmodelfunc = metafunc(t[outlier_mask][-1], 0)
        flarelc = pd.read_csv(f"../results/hip67522_flare_lc_{file}.csv")
        ff = np.append(ff, flarelc["f"].values + newmed)
        t = np.append(t, flarelc["t"].values)
        bg = np.append(bg, extract(image_data, "BG")[flare_mask])
        xc = np.append(xc, extract(image_data, "XC")[flare_mask])
        yc = np.append(yc, extract(image_data, "YC")[flare_mask])
        flag = np.append(flag, extract(image_data, "FLAG")[flare_mask])
        roll = np.append(roll, extract(image_data, "ROLL")[flare_mask])
        dT = np.append(dT, extract(image_data, "thermFront_2")[flare_mask])
        ferr = np.append(ferr, extract(image_data, "FLUXERR")[flare_mask])

        
        newfitted = np.append(newfitted, notransitmodelfunc(flarelc["t"].values, *popt))

    # sort t, ff, and newfitted by t
    t, ff, newfitted = [arr[np.argsort(t)] for arr in [t, ff, newfitted]]


    # PLOT THE FINAL FLUX  ----------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(t, ff, ".", markersize=1, color="red", label="quiescent model")

    plt.xlabel(time_label)
    plt.ylabel(flux_label)
    plt.title("Final de-trendend light curve")
    plt.savefig(f"../plots/{file}/flares/hip67522_final_detrended_light_curve.png")

    # --------------------------------------------------------------------------------


    # WRITE THE FINAL LIGHT CURVE TO A CSV FILE ------------------------------------------

    df = pd.DataFrame({"time": t, "flux": ff, "model" : newfitted, "flux_err": ferr,
                       "roll": roll, "dT": dT, "flag": flag, "bg": bg, "xc": xc, "yc": yc})
    
    # df.to_csv(f"../data/hip67522/CHEOPS-products-{file}/Outdata/00000/{file}_detrended_lc.csv", index=False)

    # new PIPE
    df.to_csv(f"../data/hip67522/pipe_HIP67522/HIP67522_{file}_detrended_lc.csv", index=False)

    # WRITE THE INITAL MASK TO A txt FILE ------------------------------------------

    # np.savetxt(f"../data/hip67522/CHEOPS-products-{file}/Outdata/00000/{file}_mask.txt", init_mask, fmt="%d")
    np.savetxt(f"../data/hip67522/pipe_HIP67522/HIP67522_{file}_mask.txt", init_mask, fmt="%d")
