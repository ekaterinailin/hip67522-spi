"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


This script comnibes the segmented reduction of the CHEOPS light curve
102 ch after they were detrended individually

"""

import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # get the individual light curves
    df1 = pd.read_csv("results/cheops/HIP67522_102ch_detrended_lc1.csv")
    df2 = pd.read_csv("results/cheops/HIP67522_102ch_detrended_lc2.csv")

    # diagnostic plot
    plt.figure()
    plt.plot(df1['time'], df1['masked_raw_flux'], 'r.')
    plt.plot(df2['time'], df2['masked_raw_flux'], 'b.')
    plt.plot(df1['time'], df1['model'], 'g.')
    plt.plot(df2['time'], df2['model'], 'k.')
    plt.xlabel("Time (BJD)")
    plt.ylabel(r"Flux (e$^{-}$/s)")
    plt.savefig("plots/diagnostic/HIP67522_102ch_detrended_lc.png")


    # use df1 until 2460413.25 and df2 after that, combine in a new light curve
    df = pd.concat([df1[df1['time'] < 2460413.25], df2[df2['time'] >= 2460413.25]])

    # calculate the new median flux as the average of the two flux means in the datasets
    mean = (df1['flux'].mean() + df2['flux'].mean()) / 2

    # scale the flux in the two datasets to the new mean
    df1['flux'] = df1['flux'] * mean / df1['flux'].mean()
    df2['flux'] = df2['flux'] * mean / df2['flux'].mean()

    df = pd.concat([df1[df1['time'] < 2460413.25], df2[df2['time'] >= 2460413.25]])

    # diagnostic plot
    plt.figure()
    plt.plot(df['time'], df['masked_raw_flux'], 'r.', markersize= 1)
    plt.plot(df['time'], df['model'], 'g', linewidth= 1)
    plt.plot(df['time'], df['flux'], 'k.', markersize= 1)
    plt.xlabel("Time (BJD)")
    plt.ylabel(r"Flux (e$^{-}$/s)")
    plt.savefig("plots/diagnostic/HIP67522_102ch_detrended_lc_combined.png")

    # write df to a new csv file
    df.to_csv("results/cheops/HIP67522_102ch_detrended_lc.csv", index=False)
