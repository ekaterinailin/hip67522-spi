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

# set default matplotlib fontsize to 13
plt.rcParams.update({'font.size': 12})

if __name__ == "__main__":

    # Load the data
    df1 = pd.read_csv('/home/ilin/Documents/2024_04_HIP67522_ATCA/results/20240611/timeseries/timeseries.csv')

    # read tstart with np.loadtxt
    tstart = str(np.loadtxt('/home/ilin/Documents/2024_04_HIP67522_ATCA/results/20240611/timeseries30/tstart.txt', dtype=str))

    # convert num which is in hours  to a  column with a time stamp 
    df1['time'] = [pd.to_datetime('2024-06-11 ' + f"{num:02d}" + tstart[2:]) for num in df1['num'].values] 

    df1['jd'] = df1['time'].apply(lambda x: x.to_julian_date())

    # read in the 30 min cadence data
    df30 = pd.read_csv('/home/ilin/Documents/2024_04_HIP67522_ATCA/results/20240611/timeseries30/timeseries.csv').iloc[:-1]

    # convert num which is in hours  to a  column with a time stamp
    df30['time'] = pd.to_datetime("2024-06-11 " + df30['num'].str.replace("_", ":") + ":" + tstart[-2:])

    # convert datetime to jd
    df30['jd'] = df30['time'].apply(lambda x: x.to_julian_date())

    # convert jd to seconds
    df30['time_seconds'] = (df30['jd'] - df30['jd'].min()) * u.day.to(u.s)

    # make a comparison plot
    plt.figure(figsize=(6.5, 5))

    # make a 30 min timedelta object
    thirty_min = 30 / 60 / 24

    # make a 15 min timedelta object
    fifteen_min = 15 / 60 / 24

    # rewrite as plt.errorbar with bkg_rms_J
    # plt.errorbar(df1["jd"], df1['source_J'], yerr=df1['bkg_rms_J'], xerr=thirty_min, fmt='x', label='1 h cadence', color='olive')
    plt.errorbar(df30['jd'], df30['source_J'], yerr=df30['bkg_rms_J'], xerr=fifteen_min, fmt='o-', label='30 min cadence', color='blue')

    plt.xlabel('Time [JD]')
    plt.ylabel('Flux density [Jy]')
    plt.xlim(df30['jd'].min() - fifteen_min, df30['jd'].max() + fifteen_min)

    plt.legend(frameon=False, loc=2)   

    plt.title('2024-06-11', fontsize=12)

    plt.tight_layout()

    plt.savefig("../plots/paper/20240611_burst.png", dpi=300)


