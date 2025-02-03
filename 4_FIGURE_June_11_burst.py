"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

Read in the ATCA time series data and plot the flux 
density vs time for the 30 min cadence data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# turn off warnings
import warnings


# set default matplotlib fontsize to 13
plt.rcParams.update({'font.size': 12})

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # Compile from CASA results

    # Load the data
    # df1 = pd.read_csv('/home/ilin/Documents/2024_04_HIP67522_ATCA/results/20240611/timeseries/timeseries.csv')

    # # read tstart with np.loadtxt
    # tstart = str(np.loadtxt('/home/ilin/Documents/2024_04_HIP67522_ATCA/results/20240611/timeseries30/tstart.txt', dtype=str))

    # # convert num which is in hours  to a  column with a time stamp 
    # df1['time'] = [pd.to_datetime('2024-06-11 ' + f"{num:02d}" + tstart[2:]) for num in df1['num'].values] 

    # df1['jd'] = df1['time'].apply(lambda x: x.to_julian_date())

    # # read in the 30 min cadence data
    # df30 = pd.read_csv('/home/ilin/Documents/2024_04_HIP67522_ATCA/results/20240611/timeseries30/timeseries.csv').iloc[:-1]

    # # convert num which is in hours  to a  column with a time stamp
    # df30['time'] = pd.to_datetime("2024-06-11 " + df30['num'].str.replace("_", ":") + ":" + tstart[-2:])

    # # convert datetime to jd
    # df30['jd'] = df30['time'].apply(lambda x: x.to_julian_date())

    # # convert jd to seconds
    # df30['time_seconds'] = (df30['jd'] - df30['jd'].min()) * u.day.to(u.s)

    # df30.to_csv("results/20240611_30_min.csv", index=False)

    # ----------------------------

    df30 = pd.read_csv("results/20240611_30_min.csv")

    # make a comparison plot
    plt.figure(figsize=(6.5, 5))

    # make a 30 min timedelta object
    thirty_min = 30 / 60 / 24

    # make a 15 min timedelta object
    fifteen_min = 15 / 60 / 24

    # rewrite as plt.errorbar with bkg_rms_J
    plt.errorbar(df30['jd'], df30['source_J']*1e3, yerr=df30['bkg_rms_J']*1e3, xerr=fifteen_min, fmt='o-', label='30 min cadence', color='blue')

    plt.xlabel('Time [JD]')
    plt.ylabel('Flux density [mJy]')
    plt.xlim(df30['jd'].min() - fifteen_min, df30['jd'].max() + fifteen_min)

    plt.tight_layout()

    plt.savefig("plots/atca/20240611_burst.png", dpi=300)


