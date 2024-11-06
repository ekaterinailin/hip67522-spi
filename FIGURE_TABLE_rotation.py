"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script calculates the Lomb-Scargle periodogram 
of the TESS, CHEOPS and ATCA data of HIP 67522,
together with the window function.

It also calculates the false alarm probability of the 
highest peak in the periodogram for ATCA.

The script saves the periodograms as plots and the
periods as a table in LaTeX format.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import lightkurve as lk

from astropy.timeseries import LombScargle

# set font size to 13
plt.rcParams.update({'font.size': 13})


def periodogram(t, y, dy, fmin=0.2, fmax=5, fap_levels=[1e-5], norm="standard", fap_method='baluev'):
    freq = np.linspace(fmin, fmax, 10000)
    power = LombScargle(t, y, dy, normalization=norm).power(freq)

    # make a lomb scargle of the window function
    window = np.ones_like(t) * 10
    window_power = LombScargle(t, window,  normalization=norm, center_data=False).power(freq)

    # period at maximum power
    period = 1/freq[np.argmax(power)]

    # false alarm probability
    fap = LombScargle(t, y, dy,normalization=norm).false_alarm_probability(power.max(), method=fap_method)

    fap_values = [LombScargle(t, y, dy,normalization=norm).false_alarm_level(fap_level, method=fap_method) for fap_level in fap_levels]

    return freq, power, window_power, period, fap, fap_values, fap_levels, fmin, fmax


if __name__ == "__main__":

    # initialize the list of periods
    periods = {}

    # ----------------------------------------------------------
    # TESS -----------------------------------------------------

    # GET DATA -------------------------------------------------

    # Get all the light curves
    lcs = lk.search_lightcurve('HIP 67522', author='SPOC', exptime="120").download_all()
    lcs = lcs[[0,1,3]] # Remove the one that is not at 120s cadence

    # get TESS flares
    flares = pd.read_csv('../data/hip67522_tess_flares.csv')


    # mask out the flares
    for i, lc in enumerate(lcs):
        time = lc.time.value
        flux = lc.flux.value
        flux_err = lc.flux_err.value
        mask = np.ones_like(time, dtype=bool)
        for _, flare in flares.iterrows():
            mask &= ~((time > flare.tstart) & (time < flare.tstop))
        lcs[i] = lk.LightCurve(time[mask], flux[mask], flux_err[mask])
        lcs[i].meta["sector"] = lc.sector
        print(f"Masked out {np.sum(~mask)} points from light curve {i}")

    # mask the NaNs
    for i, lc in enumerate(lcs):
        mask = ~np.isnan(lc.flux)
        lcs[i] = lc[mask]
        print(f"Masked out {np.sum(~mask)} NaNs from light curve {i}")


    # PERIODOGRAM + PLOTS -----------------------------------------

    # plot the periodogram
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), sharex=True, sharey=True)

    for lc, ax in zip(lcs, axes):


        # mask the nan

        t = lc.time.value
        y = lc.flux.value / np.nanmedian(lc.flux.value)
        
        # use rolling std to estimate the noise
        dy = pd.Series(y).rolling(50).std().fillna(method='bfill').values

        
        freq, power, window_power, period, fap, fap_values, fap_levels, fmin, fmax = periodogram(t, y, dy)

        periods[f"TESS {lc.sector}"] = period

        # plot the periodogram
        ax.plot(1/freq, power, label=f'Sector {lc.sector}', color='olive')
        # plot the window function
        ax.plot(1/freq, window_power, linestyle='--', color='blue', alpha=0.5, label='Window function')

        # plot the period
        ax.axvline(period, color='black', linestyle='-', label=f'Period = {period:.3f} d')

        # layout
        ax.set_ylabel('Power')
        ax.set_xlim(1/fmax, 1/fmin)


        ax.legend(frameon=False, loc=4)

    ax.set_xlabel('Period [d]')
    plt.tight_layout()
    plt.yscale('log')
    # reduce the vertical space between the plots
    plt.subplots_adjust(hspace=0.1)

    # add more subticks
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))

    # add labels to 0.5 interval subticks
    plt.gca().xaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    # save the plot
    plt.savefig(f'../plots/paper/tess_periodogram.png', dpi=300)

    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # CHEOPS ---------------------------------------------------

    # GET DATA -------------------------------------------------

    # load the file names
    files = np.loadtxt("files.txt", dtype=str)

    # read the flare table
    flares = pd.read_csv("../results/cheops_flares.csv")

    # read in all the de-trended light curves
    dlcs = []
    for pi, file in files:
        location = f"../data/hip67522/pipe_HIP67522/HIP67522_{file}{pi}_detrended_lc.csv"
        dlcs.append(pd.read_csv(location))

    #sort the list of light curves by the first time stamp
    dlcs = sorted(dlcs, key=lambda x: x["time"].min())

    # now loop over the light curves and calculate the mean and std of each detrended light curve
    stds, meds = [], []
    for i, dlc in enumerate(dlcs):
        # check if there are flares in the light curve
        if len(flares[(flares["tmin"] > dlcs[i]["time"].min()) & (flares["tmax"] < dlcs[i]["time"].max())]) == 0:
            print("No flares in this light curve")
            
        else:
            for i, flare in flares[(flares["tmin"] > dlcs[i]["time"].min()) & (flares["tmax"] < dlcs[i]["time"].max())].iterrows():
                # mask flare
                dlcs[i] = dlcs[i][((dlcs[i]["time"] < flare["tmin"]) | (dlcs[i]["time"] > flare["tmax"]))]
                print("Flare masked")

    # stitch the light curves together
    t = np.concatenate([dlc["time"].values for dlc in dlcs])
    y = np.concatenate([dlc["masked_raw_flux"].values for dlc in dlcs])

    dy = np.mean(np.array([dlc["flux"].std() for dlc in dlcs]))

    # PERIODOGRAM ----------------------------------------------

    # make a periodogram of the CHEOPS data with astropy
    freq, power, window_power, period, fap, fap_values, fap_levels, fmin, fmax = periodogram(t, y, dy)

    periods["CHEOPS"] = period

    # PLOT -----------------------------------------------------

    # plot the periodogram
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # plot the periodogram
    ax.plot(1/freq, power, label='CHEOPS', color='olive')
    # plot the window function
    ax.plot(1/freq, window_power, linestyle='--', color='blue', label='Window function', alpha=0.5)

    ax.axvline(period, color='k', linestyle='-', label=f'Period = {period:.3f} d')


    # inset ------------------------------------------------
    axin = ax.inset_axes([0.65, 0.47, 0.3, 0.3])
    axin.plot(1/freq, power, color='olive')
    axin.plot(1/freq, window_power, linestyle='--', 
            color='blue', label='Window function', alpha=0.5)
    axin.set_xlim(period*0.9, period*1.1)   
    axin.set_ylim(0, .7)
    axin.axvline(period, color='k', linestyle='-', 
                label=f'Period = {period:.3f} d')
    # -------------------------------------------------------

    ax.axvline(6.9596/2, c="r")

    # layout
    ax.set_xlabel('Period [d]')
    ax.set_ylabel('Power')
    ax.set_xlim(1/fmax, 1/fmin)
    # ax.set_yscale('log')
    ax.legend(frameon=False)
    plt.tight_layout()

    # save the plot
    plt.savefig(f'../plots/paper/cheops_periodogram.png', dpi=300)

    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # ATCA -----------------------------------------------------

    # GET DATA -------------------------------------------------

    df = pd.read_csv('../data/all_timeseries.csv')
    df = df[df["source_J"] < 5e-4] # exclude the burst
    df = df[df["source_J_val"]]  #exclude the upper limits

    t = df['jd'].values
    y = df['source_J'].values
    dy = df['bkg_rms_J'].values

    # PERIODOGRAM ----------------------------------------------

    freq, power, window_power, period, fap, fap_values, fap_levels, fmin, fmax = periodogram(t, y, dy, fap_levels=[1e-2, 1e-3, 1e-4])#

    periods["ATCA"] = period

    # PLOT -----------------------------------------------------

    # plot the periodogram
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # plot the periodogram
    ax.plot(1/freq, power, label='ATCA', color='olive')
    # plot the window function
    ax.plot(1/freq, window_power, linestyle='--', color='blue', label='Window function', alpha=0.5)

    ax.axvline(period, color='k', linestyle='-', label=f'Period = {period:.3f} d', zorder=-10)


    periods["ATCA"] = "-"

    # inset ------------------------------------------------
    axin = ax.inset_axes([0.66, 0.46, 0.3, 0.3])
    axin.plot(1/freq, power, color='olive')
    axin.plot(1/freq, window_power, linestyle='--', 
            color='blue', label='Window function', alpha=0.5)
    axin.set_xlim(period*0.95, period*1.05)
    axin.set_ylim(1e-5, .5)
    # axin.set_yscale('log')
    axin.axvline(period, color='k', linestyle='-', 
                label=f'Period = {period:.3f} d')
    # plot FAP values
    for fap_value, fap_level in zip(fap_values, fap_levels):
        axin.axhline(fap_value, color='grey', linestyle=':')
    # make font size smaller on tick labels
    axin.tick_params(axis='both', which='major', labelsize=10)
    # -------------------------------------------------------

    # layout
    ax.set_xlabel('Period [d]')
    ax.set_ylabel('Power')
    ax.set_xlim(1/fmax, 1/fmin)
    # ax.set_yscale('log')
    ax.legend(frameon=False)
    plt.tight_layout()

    # save the plot
    plt.savefig(f'../plots/paper/atca_periodogram.png', dpi=300)

    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # PERIOD TABLE ---------------------------------------------


    # make a list of the periods
    vals = list(periods.values())[:-1]

    # double the aliased period
    vals[0] *= 2.

    # add the mean+std of all periods
    periods["all"] = fr"{np.mean(vals):.4f} \pm {np.std(vals):.4f}"

    print(periods["all"])

    print("Coherence time scale:")
    print(np.mean(vals)**2 / np.std(vals))

    # make a pandas dataframe and convert to latex
    ptab = pd.DataFrame(periods, index=["LS period [d]"]).T

    # add comments
    ptab["comment"] = [r"1/2 $P$", "", "", "" , r"no sign. $P$",""]
 
    # convert ptab to latex
    table = ptab.to_latex(index=True, escape=False, column_format="lll")

    # replace toprule and bottomrule with hline
    table = table.replace("\\toprule", "\\hline")
    table = table.replace("\\bottomrule", "\\hline")
    table = table.replace("\\midrule", "\\hline")
    table = table.replace("all", "\\textbf{all}")
    table = table.replace("\pm", "$\pm$")

    # save the table
    with open("../tables/rotation_periods.tex", "w") as f:
        f.write(table)

    # ----------------------------------------------------------
