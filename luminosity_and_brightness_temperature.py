"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


This script calculates the radio luminosity of HIP 67522 based on 
the the ATCA observation, bot for quiescence and bursts.
"""

import pandas as pd
import numpy as np
import astropy.units as u
from astropy.constants import k_B, c, R_sun

def brightness_temperature(value, std, radius_rsun, radius_rsun_err, d, d_err):

    Sv = value * u.Jy * d**2 / (radius_rsun**2 * R_sun**2)
    Tb = (c**2 * Sv  / (np.pi * 2 * (2.1 * u.GHz)**2 * k_B )).to(u.K)   

    # error propagation
    r = radius_rsun * R_sun
    r_err = radius_rsun_err * R_sun
    
    Sv_err = np.sqrt((d**2 / r**2 * std * u.Jy)**2 + 
                    (2 * d * value * u.Jy / r**2 * d_err)**2 + 
                    (2 * value * u.Jy * d**2 / r**3 * r_err)**2)

    Tb_err = (c**2 * Sv_err  / (2 * np.pi * (2.1 * u.GHz)**2 * k_B )).to(u.K)

    return Tb.value, Tb_err.value

if __name__ == "__main__":

    # GET STELLAR PARAMETERS ----------------------------------------------------

    hip67522_params = pd.read_csv('../data/hip67522_params.csv')
    radius_rsun = hip67522_params.iloc[4].val
    radius_rsun_err = hip67522_params.iloc[4].err
    d = 124.7 * u.pc
    d_err = 0.3 * u.pc 

    # ---------------------------------------------------------------------------


    # QUIESCENT LUMINOSITY ------------------------------------------------------

    all = pd.read_csv('../data/atca_full_integration_time_series.csv')
    all = all[all["source_J_val"]] # exclude non-detection

    d = 124.7 * u.pc

    # calculate the quiescent flux density
    all_except = all[all["obsname"] != "2024-06-11"]
    all_except = all_except[all_except["obsname"] != "2024-05-11"]

    # get the mean and std of the quiescent flux density
    mean_quiescent = all_except.source_J.mean()

    # use the average background rms because we are not measuring the same state independently
    std_quiescent = all_except.bkg_rms_J.mean() 

    print(fr"Quiescent L band flux: {mean_quiescent*1e3:.3f} \pm {std_quiescent*1e3:.3f} mJy")

    # convert erg/s/Hz using distance
    l_mean_quiescent = (mean_quiescent * u.Jy * 4 * np.pi * d**2).to(u.erg / u.s / u.Hz)
    l_std_quiescent = (std_quiescent * u.Jy * 4 * np.pi * d**2).to(u.erg / u.s / u.Hz)

    print(fr"Quiescent L band luminosity: {l_mean_quiescent:.3e} \pm {l_std_quiescent:.3e}")

    # get the log10 of mean quiescent luminosity
    log10_l_mean_quiescent = np.log10(l_mean_quiescent.value)

    # get uncertainties in log10
    uncert_log10_l_mean_quiescent = np.log10((l_mean_quiescent + l_std_quiescent).value) - log10_l_mean_quiescent


    print(f"log10 of quiescent luminosity: {log10_l_mean_quiescent:.3f} +/- {uncert_log10_l_mean_quiescent:.3f}")

    # -------------------------------------------------------------------------


    # BURST LUMINOSITIES ------------------------------------------------------

    all = pd.read_csv("../data/all_timeseries.csv") 

    # the two bursts
    burstname1 = 20240511
    burstname2 = 20240611

    # detection threshold for a burst
    thresh = mean_quiescent + 3 * std_quiescent

    # is the May 11 burst above the threshold?
    n_above = np.where((all.loc[all["obsname"] == burstname1, "source_J"] > thresh).values)[0].shape[0]

    print(f"Number of points above threshold on May 11: {n_above}")

    # get the peak fluxes for May 11 and June 11
    maxmay = all.loc[all["obsname"] == burstname1, "source_J"].max()
    uncert_maxmay = all.loc[all["obsname"] == burstname1, "bkg_rms_J"].max()
    maxjune = all.loc[all["obsname"] == burstname2, "source_J"].max()
    uncert_maxjune = all.loc[all["obsname"] == burstname2, "bkg_rms_J"].max()
    print(f"Max flux in May: {maxmay*1e3:.3f} \pm {uncert_maxmay:.4f} mJy")
    print(f"Max flux in June: {maxjune*1e3:.3f} \pm {uncert_maxjune:.4f} mJy")

    # convert maxmay and maxjune to erg/s/Hz
    l_maxmay = (maxmay * u.Jy * 4 * np.pi * d**2).to(u.erg / u.s / u.Hz)
    l_maxjune = (maxjune * u.Jy * 4 * np.pi * d**2).to(u.erg / u.s / u.Hz)

    print(f"Max luminosity in May: {l_maxmay:.3e}")
    print(f"Max luminosity in June: {l_maxjune:.3e}")

    # get the log10 of max luminosities
    log10_l_maxmay = np.log10(l_maxmay.value)
    log10_l_maxjune = np.log10(l_maxjune.value)

    print(f"log10 of max luminosity in May: {log10_l_maxmay:.3f}")
    print(f"log10 of max luminosity in June: {log10_l_maxjune:.3f}")

    # -------------------------------------------------------------------------


    # BRIGHTNESS TEMPERATURE ---------------------------------------------------

    Tb, Tb_err = brightness_temperature(mean_quiescent, std_quiescent, radius_rsun, 
                                        radius_rsun_err, d, d_err)
    print(f"Quiescent brightness temperature: {Tb:.2e} \pm {Tb_err:.2e} K")

    # same for maxmay and maxjune
    Tb_maxmay, Tb_maxmay_err = brightness_temperature(maxmay, uncert_maxmay, radius_rsun, 
                                                    radius_rsun_err, d, d_err)
    Tb_maxjune, Tb_maxjune_err = brightness_temperature(maxjune, uncert_maxjune, radius_rsun, 
                                                        radius_rsun_err, d, d_err)

    print(f"Max brightness temperature in May: {Tb_maxmay:.2e} \pm {Tb_maxmay_err:.1e} K")
    print(f"Max brightness temperature in June: {Tb_maxjune:.2e} \pm {Tb_maxjune_err:.1e} K")

    # -------------------------------------------------------------------------
