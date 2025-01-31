"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

Fit the spectra from L band observations, weed out the non-detections, then

1. individually (there's trends that suggest there low flux densities are biased)
2. confirm the trend
3. fit with one spectral index for all spectra
4. double check that this is not a calibration error by fitting the phase calibrator 
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from funcs.helper import COLORS

# make font size bigger
plt.rcParams.update({'font.size': 12})

def spectrum(nu, alpha, offset):
    return np.log10(nu) * alpha + offset

def spectrum_err(nu, alpha, offset, errvals):
    alphaerr, offseterr = errvals
    val = np.log10(nu) * alpha + offset
    valerr = np.sqrt(np.log10(nu)**2 * alphaerr**2 + offseterr**2)
    return valerr


if __name__ == "__main__":

    N = 10

    # get the spectra, note that 1,2,3,4 is backwards in L band!
    df = pd.read_csv('data/atca/Stokes_I_4bins.csv')
    df = df.sort_values(by='obsname')

    # weed out the non-detections, same criterion as for the rest of the paper
    for i in range(1,5):
        mask = (df[f"source_J_{i}"] < 4 * df[f"bkg_rms_J_{i}"]) | (df[f"source_J_{i}"] < df[f"bkg_max_J_{i}"])
        df[f"source_J_{i}"][mask] = np.nan
        df[f"bkg_rms_J_{i}"][mask] = np.nan


    # make a list of three frequencies that are the midpoints of the three bins between 1.1 and 3.1 GHz
    freqs = np.linspace(1.1, 3.1, 5) # GHz
    freqs = (freqs[1:] + freqs[:-1]) / 2 * 1e9 # to Hz

    # FIT EACH SPECTRUM INDIVIDUALLY ------------------------

    alphas, alphaerrs, minfluxes, betas = [],[],[],[]

    plt.figure(figsize=(6, 4))

    colors = COLORS.copy() 

    # plot each spectrum
    for i, row in df.iterrows():

        c = colors.pop()

        J = np.array([row[f"source_J_{i}"] for i in [4,3,2,1]])
        Jerr = np.array([row[f"bkg_rms_J_{i}"] for i in [4,3,2,1]])

        logJ = np.log10(J)
        logJerr = np.log10(J + Jerr) - logJ

        # drop nans
        mask = ~np.isnan(logJ)
        f = freqs[mask]
        logJ, logJerr = logJ[mask], logJerr[mask]
        J, Jerr = J[mask], Jerr[mask]

        # fit the spectrum
        popt, pcov = curve_fit(spectrum, f, logJ, sigma=logJerr)

        # plot the fit
        plt.plot(f, 10**spectrum(f, *popt), c='gray', alpha=0.5)

        # get uncertainty in the fit
        perr = np.sqrt(np.diag(pcov))

        # get legend label
        _ = str(row['obsname'])
        date = _[:4] + "/" + _[4:6] + "/" + _[6:8]

        # plot the data
        plt.errorbar(f, J, yerr=Jerr, label=date + ' ' + fr"$\alpha={popt[0]:.2f}$", 
                    fmt='o', c=c)

        # add result to lists
        alphas.append(popt[0])
        minfluxes.append(np.mean(J))
        alphaerrs.append(perr[0])  
        betas.append(popt[1]) 
        
    plt.ylabel('Flux density [Jy]')
    plt.xlabel('Frequency [GHz]')


    # add a legend handel for the fit
    plt.plot([], [], c='gray', alpha=0.5, label=r'best fit to $S_\nu \propto \nu^\alpha$')
    plt.legend(loc=(1.01, 0), fontsize=8.4)

    # save figure
    plt.tight_layout()
    plt.savefig('plots/atca/four_bin_spectra_individual_fits.png', dpi=300)

    # ----------------------------------------------------------------

    # CONFIRM THAT THERE IS CORRELATION BETWEEN FLUX AND SPECTRAL INDEX

    # check that there is indeed a correlation between alpha and flux
    plt.figure(figsize=(4, 3))
    plt.errorbar(minfluxes, alphas, yerr=alphaerrs, fmt='o')
    plt.xlabel('Mean flux density [Jy]')
    plt.ylabel(r'Spectral index $\alpha$')
    plt.tight_layout()
    plt.savefig('plots/atca/four_bin_spectra_individual_fits_alpha_vs_flux.png', dpi=300)


    # NOW FIT ALL SPECTRA WITH DIFFERENT OFFSETS BUT THE SAME ALPHA

    # prepare the data for the fit

    # log10 the frequencies
    logfreqs = np.log10(freqs)

    # select the fluxes and errors
    sel = df[["source_J_4","source_J_3","source_J_2","source_J_1"]].values
    error = df[["bkg_rms_J_4","bkg_rms_J_3","bkg_rms_J_2","bkg_rms_J_1"]].values

    # mask the nans
    mask = np.isnan(sel)

    # get the error in log space
    log_err = np.log10(sel + error) - np.log10(sel)
    # log_err = np.log10(sel) - np.log10(sel - error)

    # prepare the initial guess for the offsets and alpha
    offset = np.full_like(betas, np.median(betas))
    p0 = np.concatenate([[1.6], offset])

    # define the fit function with the mask included
    def full_spec(logfreqs, alpha, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10):

        logfreqs = np.tile(logfreqs, N)
        offset = np.repeat([o1,o2,o3,o4,o5,o6,o7,o8,o9,o10], 4)

        logfreqs = logfreqs[~mask.flatten()]
        offset = offset[~mask.flatten()]
        
        return logfreqs * alpha + offset


    # fit the full spectrum
    popt, pcov = curve_fit(full_spec, logfreqs, np.log10(sel[~mask].flatten()), p0=p0, 
                        sigma=log_err[~mask].flatten(), absolute_sigma=True)


    # get the model values

    # use an unmasked version of the frequencies
    masked_vals = full_spec(logfreqs, *popt)

    # fill a 10,4 shape with nans
    vals = np.full((N,4), np.nan)

    # fill in vals into masked range
    vals[~mask] = masked_vals


    # CALCULATE THE QUIESCENT BETA VALUE ----------------------------

    # sort betas by value
    betas = popt[1:]
    sort_betas = np.sort(betas)[:-2] # drop the two highest values that correspond to the bursts
    
    mean_beta = np.mean(sort_betas)
    std_beta = np.std(sort_betas)

    std_beta = np.diag(pcov)[1:-2].mean()

    print(fr"Mean power law offset in quiescence: ${mean_beta:.5f} \pm {std_beta:.5f}")
    
    # save the mean beta and std beta to file
    with open('results/atca/spectral_offset.txt', 'w') as f:
        f.write(fr"{mean_beta},{std_beta}")

    # sanity check flux at 2.1 GHz
    specval = spectrum(2.1e9, popt[0], mean_beta)
    print(f"Flux at 2.1 GHz: {10**specval:.2e} Jy")

    # sanity check uncertainty on flux at 2.1 GHz
    specerr = spectrum_err(2.1e9, popt[0], mean_beta, errvals=[np.diag(pcov)[0], std_beta])
    
    print(f"Upper uncertainty on flux at 2.1 GHz: {10**(specval + specerr) - 10**(specval):.2e} Jy")
    print(f"Lower uncertainty on flux at 2.1 GHz: {10**(specval) - 10**(specval - specerr):.2e} Jy")




    # PLOT THE RESULTS ----------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 4.5))

    freqs /= 1e9

    # loop through the rows in vals
    color = COLORS.copy()
    for val in vals.reshape(N,4):
        m = np.isnan(val)
        plt.plot(freqs[~m], 10**val[~m], c=color.pop(),zorder=-20)


    # loop through the data
    color = COLORS.copy()
    for row, errorrow, label in list(zip(sel, error, df["obsname"].values)):
        label = str(label)
        label = label[:4] + "/" + label[4:6] + "/" + label[6:8]
        m = np.isnan(row)
        c = color.pop()
        plt.errorbar(freqs[~m], row[~m], yerr=errorrow[~m], fmt='d', c=c, markersize =8, zorder=-10)
        plt.scatter(freqs[~m], row[~m], c="k", s=80, marker='d')
        plt.scatter(freqs[~m], row[~m], marker='d', c=c, s =30)#, label=label)


    # layout
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Flux density [mJy]')

    # plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-4,1.4e-3)

    # make the y-axis have 4 y tick labels
    ax.yaxis.set_ticks([1e-4,0.2e-3, 0.5e-3, 1e-3, 1.4e-3])      
    ax.yaxis.set_ticklabels([r"$0.1$",r"$0.2$", r"$0.5$", r"$1.0$",r"$1.4$"])  

    plt.tight_layout()

    # save the figure
    plt.savefig("plots/atca/four_bin_spectra.png", dpi=300)

    print(fr"Spectral index: ${popt[0]:.3f} \pm {np.diag(pcov)[0]:.3f}$")

    # save spectral index to file
    with open('results/atca/spectral_index.txt', 'w') as f:
        f.write(fr"{popt[0]},{np.diag(pcov)[0]}")

    # CONSISTENCY CHECK ----------------------------

    # consistency check on 1424-418 phase calibrator -- what is the spectral index doing?

    # values from the imaging with 4 bands on May 15
    pcal = np.array([6.041965, 5.658123, 5.2, 4.72577])[::-1]

    plt.figure(figsize=(5, 4))
    plt.scatter(freqs, pcal, c='r')

    # fit a spectrum   
    popt, pcov = curve_fit(spectrum, freqs, np.log10(pcal))
    plt.plot(freqs, 10**spectrum(freqs, *popt), c='r', label=fr"Phase calibrator $\alpha={popt[0]:.2f}$")
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Flux density [Jy]')
    plt.legend(frameon=False)
    plt.savefig('plots/atca/1424-418_spectral_index.png')

    print(rf"Phase calibrator $\alpha$ measured on May 15, 2024: {popt[0]:.2f}")
    # https://www.narrabri.atnf.csiro.au/calibrators/calibrator_database_viewcal?source=1424-418
    print(r"Phase calibrator $\alpha$ measured independently with ATCA on June 7, 2024: 0.21")

    # ------------------------------------------------

