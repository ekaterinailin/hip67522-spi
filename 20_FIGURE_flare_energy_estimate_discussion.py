"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

Answer two questions about the flare energies:
1. How sensitive is the flare energy we derive with the Shibayama+2013 method on flare temperature?
2. How much does the NUV/FUV continuum contribute to the total flare energy?
"""

import numpy as np 
import pandas as pd

from astropy.modeling import models
import astropy.units as u

from funcs.flares import flare_factor

import matplotlib.pyplot as plt

# set plt fontsize to 13
plt.rcParams.update({'font.size': 13})

if __name__ == "__main__":

    # HOW MUCH DOES THE FLARE ENERGY WE DERIVE DEPEND ON THE CHOSEN FLARE TEMPERATURE?

    # HIP 67522 temperature and radius in solar radii
    teff = 5650
    radius = 1.392

    # wild range of flare temperatures
    tflares = np.linspace(4000, 14000, 100)

    # read CHEOPS response function
    cheops_resp = pd.read_csv("data/CHEOPS_bandpass.csv")
    wav, resp = cheops_resp.WAVELENGTH.values, cheops_resp.THROUGHPUT.values

    # compare 10000 and 9000, and 11000 and 10000
    f10k = flare_factor(teff, radius, wav, resp,  tflare=10000).value
    ratio_9000_10000_cheops = (f10k - flare_factor(teff, radius, wav, resp,  tflare=9000).value) / f10k
    ratio_10000_11000_cheops = (f10k - flare_factor(teff, radius, wav, resp,  tflare=11000).value) / f10k

    # calculate the bolometric flare energy for each sample
    ffactor_cheops = np.array([flare_factor(teff, radius, wav, resp,  tflare=tf).value for tf in tflares]) 

    # read TESS response function
    tess_resp = pd.read_csv("data/tess-response-function-v2.0.csv", skiprows=7, 
                            header=None, names=["Wavelength", "Throughput"])
    wav, resp = tess_resp.Wavelength.values, tess_resp.Throughput.values

    # compare 10000 and 9000, and 11000 and 10000
    f10k = flare_factor(teff, radius, wav, resp,  tflare=10000).value
    ratio_9000_10000_tess = (f10k - flare_factor(teff, radius, wav, resp,  tflare=9000).value) / f10k
    ratio_10000_11000_tess = (f10k - flare_factor(teff, radius, wav, resp,  tflare=11000).value) / f10k

    ffactor_tess = np.array([flare_factor(teff, radius, wav, resp,  tflare=tf).value for tf in tflares])

    print(f"Reduce flare temperature from 10k to 9k in CHEOPS: {ratio_9000_10000_cheops:.2f}")
    print(f"Increase flare temperature from 10k to 9k in CHEOPS: {ratio_10000_11000_cheops:.2f}")
    print(f"Reduce flare temperature from 10k to 9k in TESS: {ratio_9000_10000_tess:.2f}")
    print(f"Increase flare temperature from 10k to 9k in TESS: {ratio_10000_11000_tess:.2f}")


    # PLOT the temperature dependence of the flare factor
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(tflares, ffactor_cheops, color='navy', alpha=0.9, label='CHEOPS')
    plt.plot(tflares, ffactor_tess, color='peru', alpha=0.9, label='TESS')
    plt.xlabel('Flare BB temperature [K]')
    plt.ylabel('Flare energy per second equivalent duration [erg/s]')
    plt.xlim(7000, 14000)
    plt.legend(frameon=False)   
    plt.tight_layout()
    plt.savefig('plots/flare_energy_temperature.png', dpi=300)

    # ----------------------------------------------------------------------------------    

    # HOW MUCH DOES THE NUV&FUV CONTINUUM CONTRIBUTE TO THE TOTAL FLARE ENERGY?

    # blackbody
    t1, t2 = 9000, 15000
    scale =  1 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
    bb1 = models.BlackBody(temperature=t1 * u.K, scale=scale)
    bb2 = models.BlackBody(temperature=t2 * u.K, scale=scale) 

    # wild wavelength range
    wav = np.linspace(1000, 15000, 20000)

    # blackbody flux in TESS band
    bbwavs1 = bb1(wav * u.AA)

    # assume an oom increase of NUV flux over the BB 9kK expectation
    NUV_enhance = 10

    print(f"Assume NUV flux  at 2000 AA higher than 9kK BB expectation by a factor of {NUV_enhance}.")

    # derive the scaling factor for the 15kK BB to match the NUV flux
    a = NUV_enhance * bb1(2000*u.AA) / bb2(2000*u.AA)

    # calculate the scaled BB curve
    bbwavs2 = a * bb2(wav * u.AA)

    # get the fluxes in for each  component 
    fluxs1 = np.trapz(bbwavs1.value, wav)
    fluxs2 = np.trapz(bbwavs2.value, wav)

    # calculate the total flux ratio
    flux_ratio = fluxs2 / fluxs1
    print(f"Total energy ratio 15kK vs 9kK component: {flux_ratio:.2f}")

    # take a look at FUV only
    # FUV continua as defined by Loyd+2018
    ranges = [(1173.65,1198.49), (1201.71,1212.16), (1219.18,1274.04), (1329.25,1354.49), (1356.71,1357.59), (1359.51,1428.90)]
    # integrate the flux in each range
    flux1fuv = np.sum([np.trapz(bb1(np.linspace(r[0], r[1], 50) * u.AA).value, np.linspace(r[0], r[1], 50)) for r in ranges])
    flux2fuv = np.sum([np.trapz(bb2(np.linspace(r[0], r[1], 50) * u.AA).value, np.linspace(r[0], r[1], 50)) for r in ranges])
    # total flux in FUV
    fluxfuv = flux1fuv + flux2fuv

    # calculate the FUV underestimate
    ratiofuv = fluxfuv / flux1fuv
    print(f"Energy underestimate in FUV130 continuum by a single 9kK blackbody: {ratiofuv:.2f}")

    # PLOT THE TWO COMPONENT FOR VISUAL PLEASURE
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(wav, bbwavs1, c='navy', label=fr"T={t1} K, F={fluxs1:.2e} erg/s/cm$^2$/sr")
    plt.plot(wav, bbwavs2, c='peru', label=fr"T={t2} K, F={fluxs2:.2e} erg/s/cm$^2$/sr")

    plt.legend(frameon=False, loc=1, fontsize=12)

    plt.xlim(1000, 10000)
    plt.ylim(0,)
    plt.ylabel(r'Flux [erg/s/cm$^2$/$\AA$/sr]')
    plt.xlabel('Wavelength [nm]')
    plt.tight_layout()
    plt.savefig('plots/two_flare_energy_components.png', dpi=300)

    # ----------------------------------------------------------------------------------
