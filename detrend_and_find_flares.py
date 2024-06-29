# open the fits image and plot it with matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from altaipony.flarelc import FlareLightCurve
import time
import glob
import os
import sys



def roll_angle_sys(phi, a_s, b_s):
    """Model of the roll angle systematics.
    
    Parameters
    ----------
    phi : float
        The roll angle in radians.
    a_s : list 
        List of sine coefficients.
    b_s : list 
        List of cosine coefficients.

    Returns
    -------
    sys : float
        The roll angle systematics.
    """
    
    sys = [a_s[i] * np.sin(i * phi) + b_s[i] * np.cos(i * phi) for i in range(len(a_s))]

    # sum over all the terms along the roll angle
    return np.sum(sys, axis=0)


def fit_roll_angle_sys(x, phi, flux, flux_err):
    """Fit the roll angle systematics to the data.
    
    Parameters
    ----------
    phi : list
        List of roll angles in radians.
    flux : list
        List of flux values.
    flux_err : list
        List of flux errors.
    a_s : list 
        List of sine coefficients.
    b_s : list 
        List of cosine coefficients.

    Returns
    -------
    chi2 : float
        The chi-squared value of the fit.
    """
    a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 = x
    
    a_s = [a1, a2, a3, a4, a5]
    b_s = [b1, b2, b3, b4, b5]

    # fit the roll angle systematics
    sys = roll_angle_sys(phi, a_s, b_s)
    
    # calculate the chi-squared value
    chi2 = np.sum((flux - sys)**2 / flux_err**2)
    
    return chi2


if __name__ == "__main__":
   


    # SET UP THE DATA STRUCTURE ---------------------------------------------------

    # get a timestamp YYYYMMDDHHMMSS
    timestamp = time.strftime("%Y%m%d%H")

    # open the fits file usinhg the command line argument
    file = sys.argv[1]

    # make a  directory in plots for the results
    os.makedirs(f"../plots/{file}", exist_ok=True)


    # COMPARE DRP TO IMAGETTE BASED LIGHTCURVE --------------------------------------   

    # # open the DRP LC
    DRP = glob.glob(f'../data/hip67522/CHEOPS-products-{file}/*_SCI_COR_Lightcurve-DEFAULT_V0300.fits')
    print(f"DRP file found for {file}:\n {DRP}\n")
    cor = fits.open(DRP[0])

    # get the correct columns
    t, f, ferr, s, roll = cor[1].data['BJD_TIME'], cor[1].data['FLUX'], cor[1].data['FLUXERR'], cor[1].data['STATUS'], cor[1].data['ROLL_ANGLE']

    # only use s==0
    mask = (s == 0)

    tdrp, fdrp, ferrdrp, rolldrp = t[mask], f[mask], ferr[mask], roll[mask]

    # GET THE IMAGE DATA -----------------------------------------------------------
    IMG = f'../data/hip67522/CHEOPS-products-{file}/Outdata/00000/hip67522_CHEOPS-products-{file}_im.fits'
    hdulist = fits.open(IMG)
    print(f"Imagette file found for {file}:\n {IMG}\n")


    # get the image data
    image_data = hdulist[1].data

    t, f, ferr, roll = image_data["BJD_TIME"], image_data["FLUX"], image_data["FLUXERR"], image_data["ROLL"]

    # make sure the data is in fact 10s cadence
    assert np.diff(t).min() * 24 * 60 * 60 < 10.05, "Time series is not 10s cadence"

    # big endian to little endian
    t = t.byteswap().newbyteorder()
    f = f.byteswap().newbyteorder()
    ferr = ferr.byteswap().newbyteorder()
    roll = roll.byteswap().newbyteorder()

    # PLOT THE DRP VS IMAGETTE DATA ---------------------------------------------------

    plt.figure(figsize=(12, 7))

    plt.scatter(t, f/np.median(f), s=1, label="Imagette based LC - 10s cadence", c="k")
    plt.scatter(tdrp, fdrp/np.median(fdrp) + 5 * np.std(f/np.median(f)), s=1, label="DRP LC - 30s cadence", c="r")

    # layout
    # make symbols in legend larger
    plt.legend(markerscale=5, frameon=False)

    t0 = min(t.min(), tdrp.min())
    tf = max(t.max(), tdrp.max())
    plt.xlim(t0,tf)

    # save the plot
    plt.savefig(f"../plots/{file}/{timestamp}_DRP_vs_Imagette.png", dpi=300)


    # FIT THE ROLL ANGLE SYSTEMATICS ---------------------------------------------------


    # split by roll angle diff 
    roll_diff = np.diff(roll)

    # index the data by where the roll angle change is positive
    index = np.where(roll_diff > 0)[0]

    # add an incremental index to each phase between the indices
    phase = np.zeros(len(roll))
    phase[:index[0]] = 0
    for i in range(len(index) - 1):
        phase[index[i]:index[i+1]] = i+1
    phase[index[-1]:] = len(index) 

    # make a Dataframe for easy indexing
    df = pd.DataFrame({"BJD": t, "FLUX": f, "FLUXERR": ferr, "ROLL": roll, "PHASE": phase})


    # open a diagnostic plot
    plt.figure(figsize=(12, 12))

    # init a model array
    model = np.array([])

    # go roll by roll
    for ro, g in df.groupby("PHASE"):

        # sort by roll angle
        g = g.sort_values("ROLL")

        # convert roll to radians
        rr = np.radians(g["ROLL"])

        # normalize the flux and fluxerr
        ff = g["FLUX"] / np.median(g["FLUX"])
        fferr = g["FLUXERR"] / np.median(g["FLUX"])

        # plot the data with an offset
        plt.scatter(rr, ff + ro/50. , s=1, c="grey")

        # clip 5 sigma outliers
        mask = np.abs(ff - np.median(ff)) < 5 * np.std(ff)

        # use the masked data for the fit
        rrm, ffm, fferrm = rr[mask], ff[mask], fferr[mask]

        # fit roll angle systematics to roll, flux, flux_err 
        # optimize a_s and b_s
        res = minimize(fit_roll_angle_sys, x0=np.ones(10)/10., args=(rrm, ffm, fferrm))

        # calculate the model light curve
        a_s = res.x[:5]
        b_s = res.x[5:]
        m = roll_angle_sys(rr, a_s, b_s)

        # plot the model light curve
        plt.plot(rr, m + ro/50., c='r', lw=1)

        # append model
        g["MODEL"] = m * np.median(g["FLUX"])

        # sort abck by BJD
        g = g.sort_values("BJD")
        model = np.append(model, g["MODEL"])


    # add model to the Dataframe
    df["MODEL"] = model

    # layout the diagnostic plot
    plt.xlabel("Roll angle [deg]")
    plt.ylabel("Normalized flux + offset")
    plt.savefig(f"../plots/{file}/{timestamp}_roll_angle_systematics.png", dpi=300)

    # DEFINE THE DETRENDED FLUX --------------------------------------------------

    df["DETRENDED"] = df["FLUX"] - df["MODEL"] + np.median(df["FLUX"])

    # mask 8 sigma outliers
    mask = np.abs(df["DETRENDED"] - np.median(df["DETRENDED"])) < 8 * np.std(df["DETRENDED"])

    df = df[mask]

    # DEFINE THE DETRENDED ERROR --------------------------------------------------

    # mask outliers
    mask = np.abs(df["DETRENDED"] - np.median(df["DETRENDED"])) < 5 * np.std(df["DETRENDED"])

    df["DETRENDED_ERR"] = df.loc[mask, "DETRENDED"].std()

    # MAKE DIAGNOSTIC FIGURE OF THE RESIDUALS ---------------------------------------------------

    plt.figure(figsize=(12, 5))

    # plot the detrended flux
    plt.errorbar(df["BJD"] - df["BJD"].iloc[0], df["DETRENDED"], yerr=df["DETRENDED_ERR"], fmt=".", markersize=1)

    # layout
    plt.xlabel(f"Time [BJD - {df['BJD'].iloc[0]}]")
    plt.ylabel("Detrended flux [e-/s]")

    plt.xlim(0, df["BJD"].iloc[-1] - df["BJD"].iloc[0])

    # save the plot
    plt.savefig(f"../plots/{file}/{timestamp}_detrended_flux.png", dpi=300)

    # FIND FLARES --------------------------------------------

    # define flare light curve
    flc = FlareLightCurve(time=df["BJD"], flux=df["DETRENDED"], flux_err=df["DETRENDED_ERR"])

    # define detrended flux
    flc.detrended_flux = df["DETRENDED"]
    flc.detrended_flux_err = df["DETRENDED_ERR"]

    # find flares
    flares = flc.find_flares().flares

    # delete unnecessary columns
    del flares["cstart"]
    del flares["cstop"]

    # add file number
    flares["file"] = file

    # add total number of valid data points to a metadata file, together with file date
    with open("../results/flare_metadata.txt", "a") as f:
        f.write(f"{timestamp},{file},{len(df)},{len(flares)}\n")

    # write out flare to file
    with open(f"../results/flares.csv", 'a') as f:
        flares.to_csv(f, header=False, index=False)

    # MAKE DIAGNOSTIC PLOT OF EACH FLARE --------------------------------------------

    # make a directory for the flare plots inside the file directory
    os.makedirs(f"../plots/{file}/flares", exist_ok=True)

    # plot each flare in its own figure
    for r, fl in flares.iterrows():
        plt.figure(figsize=(6, 5))
        plt.plot(flc.time.value, flc.detrended_flux, label="masked")
        plt.axvline(fl.tstart, c='grey', lw=1, linestyle='--')
        plt.xlim(fl.tstart - 0.2/24, fl.tstop + 0.2/24)
        plt.xlabel("Time [BJD]")
        plt.ylabel(r"Flux [e$^-$/s]")   
        plt.title(f"Flare at {fl.tstart:.9e} [BJD]")
        plt.savefig(f"../plots/{file}/flares/{timestamp}_flare_{fl.tstart:.9e}.png", dpi=300)