
"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

Gather and tidy the ATCA results for HIP 67522, put them in two files:
- atca_full_integration_time_series.csv
- atca_all_timeseries.csv

"""


import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time

# set default font size in matplotlib
plt.rcParams.update({'font.size': 12})

colors = [
    "cyan",  # Blue
    "#D95F0E",  # Vermilion
    "#009E73",  # Teal
    "maroon",  # maroon
    "#CC79A7",  # Pink
    "#56B4E9",  # Sky Blue
    "#FF7F00",  # Orange
    "olive",   # Dark Red
    "#FF4500",   # Orange Red
    "#1F78B4" # Blue
        ]*2

if __name__ == "__main__":


    # read HIP 67522 params 
    hip67522_params = pd.read_csv('../data/hip67522_params.csv')
    period = hip67522_params.loc[hip67522_params.param == "orbper_d","val"].values[0]
    midpoint = hip67522_params.loc[hip67522_params.param == "midpoint_BJD","val"].values[0]
    prot = 1.4145 # mean val from TESS AND CHEOPS



    # read in the full integration fluxes ---------------------------------------------

    full_integration_fluxes = pd.read_csv('../data/Stokes_I_fluxes.csv')

    # calculate the mean time between tstart and tstop, first convert to datetime
    full_integration_fluxes['tstart'] = pd.to_datetime(full_integration_fluxes['tstart'], format='%H:%M:%S')
    full_integration_fluxes['tstop'] = pd.to_datetime(full_integration_fluxes['tstop'], format='%H:%M:%S')

    # calculate the mean time between tstart and tstop
    timedelta = full_integration_fluxes['tstop'] - full_integration_fluxes['tstart']
    full_integration_fluxes['mean_time'] = full_integration_fluxes['tstart'] + timedelta / 2

    # convert obsname YYYYMMDD  and mean_time combined to datetime
    full_integration_fluxes['obsname'] = pd.to_datetime(full_integration_fluxes['obsname'], format='%Y%m%d')
    full_integration_fluxes['datetime'] = (full_integration_fluxes['obsname'] +
                                        pd.to_timedelta(full_integration_fluxes['mean_time'].dt.strftime('%H:%M:%S')))

    # delete mean_time column
    full_integration_fluxes = full_integration_fluxes.drop(columns=['mean_time'])

    # convert tstart and tstop to just times
    full_integration_fluxes['tstart'] = full_integration_fluxes['tstart'].dt.strftime('%H:%M:%S')
    full_integration_fluxes['tstop'] = full_integration_fluxes['tstop'].dt.strftime('%H:%M:%S')

    # convert datetime to jd
    full_integration_fluxes['jd'] = Time(full_integration_fluxes["datetime"], scale='utc').jd


    # convert jd to phase for full_integration_fluxes
    full_integration_fluxes['phase'] = np.mod(full_integration_fluxes['jd'] - midpoint, period) / period
    full_integration_fluxes['rot_phase'] = np.mod(full_integration_fluxes['jd'] - midpoint, prot) / prot
    full_integration_fluxes['source_J_val'] = ((full_integration_fluxes["source_J"] > (5 * full_integration_fluxes["bkg_rms_J"])) &
                                            (full_integration_fluxes["source_J"] > (full_integration_fluxes["bkg_max_J"])) )

    # save the full_integration_fluxes to a csv file
    full_integration_fluxes.to_csv('../data/atca_full_integration_time_series.csv', index=False)

    # read in the 1hr integration fluxes ---------------------------------------------
    

    # search /home/ilin/Documents/2024_04_HIP67522_ATCA/results for all timeseries.csv files, including subdirectories for lcs
    files = glob.glob('/home/ilin/Documents/2024_04_HIP67522_ATCA/results/**/timeseries/timeseries.csv', recursive=True)

    # write all files to a single dataframe
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # search the same folder for all instances of start.txt
    start_files = glob.glob('/home/ilin/Documents/2024_04_HIP67522_ATCA/results/**/timeseries/tstart.txt', recursive=True)

    # for each tstart.txt file read in the start time and the obsname from the file path and put them in a dictionary
    obs_dict = {}
    for file in start_files:
        with open(file, 'r') as f:
            obsname = file.split('/')[-3]
            start = f.read().strip()
            obs_dict[int(obsname)] = start


    # add the start time to the dataframe
    df['tstart'] = df['obsname'].map(obs_dict)

    # convert obsname and num to date and time column
    df['date'] = pd.to_datetime(df['obsname'], format='%Y%m%d')


    df['time'] = pd.to_datetime(df['num'].astype(str) + df["tstart"].apply(lambda x: x[2:])).dt.time

    # combine date and time into a single column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))

    # convert the UTC to JD
    df['jd'] = Time(df['datetime'], scale="utc").jd

    # if source_J is < than 3 * bkg_rms_J, set it to 3 * bkg_rms_J and upper limit the error bar
    df['source_J_val'] = df["source_J"] > (3 * df["bkg_rms_J"]) 

    # convert JD to orbital phase
    df['phase'] = np.mod(df['jd'] - midpoint, period) / period
    df['rot_phase'] = np.mod(df['jd'] - midpoint, prot) / prot

    # save the 1hr_integration_fluxes to a csv file
    df.to_csv('../data/atca_all_timeseries.csv', index=False)


