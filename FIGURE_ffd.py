"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl


Calculate the flare energies for the TESS and CHEOPS flares of HIP 67522 and compare the FFDs.
Fit power laws to the FFDs in different orbital phase ranges.
Determine the detection thresholds for TESS and CHEOPS flares.
The TESS flares are from Ilin+2024, the CHEOPS flares are from this work.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from altaipony.ffd import FFD

from funcs.flares import flare_factor
import astropy.units as u   

from funcs.helper import get_tess_orbital_phases, get_cheops_orbital_phases

# set default matplotlib fontsize to 13
plt.rcParams.update({'font.size': 13})


def get_flare_energy(ed, ederr, teff, tefferr, radius, radiuserr, wav, resp):
    """Sample the posterior distribution of the flare energy using random samples on ED, 
    and teff and radius with Gaussian errors.

    Parameters
    ----------
    ed : float
        The ED of the flare.
    ederr : float
        The error on the ED of the flare.
    teff : float
        The effective temperature of the star in K.
    tefferr : float
        The error on the effective temperature of the star in K.
    radius : float
        The radius of the star in solar radii.
    radiuserr : float
        The error on the radius of the star in solar radii.
    wav : array
        The wavelengths of the TESS response function.
    resp : array
        The response of the TESS response function.

    Returns
    -------
    mean_bol_energy : float
        The mean bolometric flare energy in erg.
    std_bol_energy : float
        The standard deviation of the bolometric flare energy in erg.
    """
    teff = np.random.normal(teff, tefferr, 500)
    radius = np.random.normal(radius, radiuserr, 500)
    eds = np.random.normal(ed, ederr, 500)
    # calculate the bolometric flare energy for each sample
    ffactor = flare_factor(teff.reshape((500,1)), 
                                 radius.reshape((500,1)), wav, resp,  
                                 tflare=10000)
    
    bol_energies = (ffactor * np.random.choice(eds, 500) * u.s).value

    # calculate the mean and standard deviation of the bolometric flare energy
    mean_bol_energy = np.mean(bol_energies)
    std_bol_energy = np.std(bol_energies)
    mean_ffactor = np.mean(ffactor)  

    return mean_bol_energy, std_bol_energy, mean_ffactor


if __name__ == "__main__":

    # GET STELLAR AND PLANET PARAMETERS -----------------------------------------------------

    hip67522params = pd.read_csv("../data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]
    teff = hip67522params[hip67522params.param=="teff_K"].val.values[0]
    tefferr = hip67522params[hip67522params.param=="teff_K"].err.values[0]
    radius = hip67522params[hip67522params.param=="radius_rsun"].val.values[0]
    radiuserr = hip67522params[hip67522params.param=="radius_rsun"].err.values[0]

    # ----------------------------------------------------------------------------------------

    # GET TESS FLARES FROM ILIN+2024 AND CALCULATE ENERGIES -------------------------------------

    # read the TESS flares
    hip67522tess = pd.read_csv("../data/hip67522_tess_flares.csv")

    # read TESS response function
    tess_resp = pd.read_csv("../data/tess-response-function-v2.0.csv", skiprows=7, 
                            names=["wav", "resp"], header=None)
    wav, resp = tess_resp.wav.values, tess_resp.resp.values

    # calculate the flare energies
    res = hip67522tess.apply(lambda x: get_flare_energy(x["ed_rec"], x["ed_rec_err"], teff, 
                                                        tefferr, radius, radiuserr, wav, resp),
                                                        axis=1)

    # write res to two columns in a new dataframe using the right column names for the FFD object
    hip67522tessffd = pd.DataFrame(res.values.tolist(), columns=["ed_rec", "ed_rec_err", "ffactor"])   
    hip67522tessffd["phase"] = hip67522tess["phase"].values

    # --------------------------------------------------------------------------------------------

    # GET TESS OBSERVING TIME --------------------------------------------------------------------

    tessphases, ttess01, ttess09, tot_obs_time_d_tess = get_tess_orbital_phases(period, split=0.1)

    print(f"Observing time in first 10% of TESS light curve: {ttess01:.2f} days")
    print(f"Observing time in last 90% of TESS light curve: {ttess09:.2f} days")
    print(f"Total observing time of TESS light curve: {tot_obs_time_d_tess:.2f} days")

    # READ IN CHEOPS FLARES ----------------------------------------------------------------------

    cheopsflares = pd.read_csv('../results/cheops_flares.csv')

    # convert some columns to be readable for the FFD object
    cheopsflares["ed_rec"] = cheopsflares["mean_bol_energy"].values
    cheopsflares["ed_rec_err"] = cheopsflares["std_bol_energy"].values
    cheopsflares.ed_rec = cheopsflares.ed_rec.apply(lambda x: float(x[:-4]))
    cheopsflares.ed_rec_err = cheopsflares.ed_rec_err.apply(lambda x: float(x[:-4]))

    # --------------------------------------------------------------------------------------------

    # print minimum flare energies
    print(f"Minimum flare energy in TESS: {hip67522tessffd.ed_rec.min()}")
    print(f"Minimum flare energy in CHEOPS: {cheopsflares.ed_rec.min()}")


    # READ IN CHEOPS OBSERVING PHASES ------------------------------------------------------------

    cheopsphases, tcheops01, tcheops09, tot_obs_time_d_cheops = get_cheops_orbital_phases(period, midpoint, split=0.1)
    print(f"Observing time in first 10% of CHEOPS light curve: {tcheops01:.2f} days")
    print(f"Observing time in last 90% of CHEOPS light curve: {tcheops09:.2f} days")
    print(f"Total CHEOPS observation time in days: {tot_obs_time_d_cheops}")

    # --------------------------------------------------------------------------------------------

    # GET MEAN FLARE FACTORS TO ESTIMATE THE DIFFERENCE IN SENSITIVITY ---------------------------

    ffactortess = hip67522tessffd.ffactor.mean().value
    ffactorcheops = cheopsflares["flare_factor"].mean()
    print(f"Mean flare factor for CHEOPS: {ffactorcheops}")
    print(f"Mean flare factor for TESS: {ffactortess}")

    # INITIATE FFDS -----------------------------------------------------------------------------------

    tessffd = FFD(f=hip67522tessffd, tot_obs_time=tot_obs_time_d_tess, ID="HIP 67522")
    cheopsffd = FFD(f=cheopsflares, tot_obs_time=tot_obs_time_d_cheops, ID="HIP 67522")

    # --------------------------------------------------------------------------------------------

    # PLOT THE FFDs -----------------------------------------------------------------------------------

    plt.figure(figsize=(6.5, 5))
    color = ["b","r"]
    legend = ["TESS", "CHEOPS"]
    marker = ["o", "d"]
    for ffd, c, l, m in list(zip([tessffd, cheopsffd], 
                            color, legend, marker)):
        ed, freq, counts = ffd.ed_and_freq()
        plt.scatter(ed, freq, c=c, label=l, marker=m)
        
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bolometric flare energy [erg]")
    plt.ylabel("cumulative number of flares per day")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../plots/ffd/cheops_vs_tess_ffd.png", dpi=300) 

    # --------------------------------------------------------------------------------------------

    # GET FLARE PHASES and ENERGIES ----------------------------------------------------------------

    # CHEOPS flare phases
    cheopsflares["phase"] = ((cheopsflares["t_peak_BJD"] - midpoint) % period) / period

    # make a DataFrame with the cheops and tess flare energies and phases
    df = pd.concat([cheopsflares[["ed_rec", "ed_rec_err", "phase"]], hip67522tessffd[["ed_rec", "ed_rec_err", "phase"]]])

    # --------------------------------------------------------------------------------------------

    # PLOT FLARE ENERGIES against PHASE -----------------------------------------------------------

    plt.figure(figsize=(6.5, 5))

    plt.errorbar(cheopsflares["phase"], cheopsflares["ed_rec"], 
                yerr=cheopsflares['ed_rec_err'], c="r", label="CHEOPS", fmt="d" )
    plt.errorbar(hip67522tessffd["phase"], hip67522tessffd["ed_rec"], 
                yerr = hip67522tessffd["ed_rec_err"], c="b", label="TESS", fmt="o")

    plt.legend(frameon=False)
    plt.yscale("log")
    plt.xlabel("Orbital Phase")
    plt.ylabel("Bolometric Flare Energy [erg]")
    plt.tight_layout()
    plt.savefig("../plots/ffd/phase_vs_flare_energy.png", dpi=300)

    # --------------------------------------------------------------------------------------------

    # FIT A POWER LAW FOR EACH PHASE RANGE ----------------------------------------------------------

    df10 = df[df["phase"]<0.1]
    df90 = df[df["phase"]>0.1]
    obs10 =  ttess01 + tcheops01
    obs90 =  ttess09 + tcheops09 

    ffd10 = FFD(f=df10, tot_obs_time=obs10, ID="phases 0-0.1")
    ffd90 = FFD(f=df90, tot_obs_time=obs90, ID="phases 0.1-1")

    color = ["b", "olive"]

    bfas = []
    for ffd, c in list(zip([ffd10, ffd90], color)):

        ed, freq, counts = ffd.ed_and_freq()
        # fit power law to each
        bfas.append(ffd.fit_powerlaw("mcmc"))

    # --------------------------------------------------------------------------------------------



    # MAKE A FIGURE OF THE FFD FITS ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for ffd, c, BFA in list(zip([ffd10, ffd90], color, bfas)):
        ed, freq, counts = ffd.ed_and_freq()
        

        ffd.plot_mcmc_powerlaw(ax, BFA, c=c, custom_xlim=(4e33,1e36))

        ax.scatter(ed, freq, c="k", s=45, zorder=1000)
        ax.scatter(ed, freq, c=c, label=ffd.ID, s=25, zorder=1001)
    
    # add legend handles manually
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                        label='orbital phases 0.0 - 0.1', markerfacecolor='b', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', 
                            label='orbital phases 0.1 - 1.0', markerfacecolor='olive', markersize=10),
                plt.Line2D([0], [0], color='k', 
                            label='Posterior draws from power law fit')]

    ax.legend(handles=handles, frameon=False, loc=3)
        
    plt.xscale("log")   
    plt.yscale("log")
    plt.xlabel("Bolometric Flare Energy [erg]")
    plt.ylabel("cumulative number of flares per day")
    plt.xlim(4e33, 1e36)
    plt.ylim(4e-4, 2)

    plt.tight_layout()

    plt.savefig("../plots/ffd/ffd_vs_phases.png")

    # --------------------------------------------------------------------------------------------

    # DETECTION THRESHOLD CALCULATIONS -----------------------------------------------------------

    # from cheops_tess_noise_levels.py
    noisetess, noisecheops = 0.000780, 0.000810

    # print the detection thresholds
    print("Detection thresholds:")
    print("TESS:", noisetess * 4 * 120 * 3 * ffactortess)
    print("CHEOPS:", noisecheops * 4 * 10 * 3 * ffactorcheops)
