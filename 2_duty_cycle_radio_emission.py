
"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script reads the ATCA data and calculates the duty cycle of the detection.

"""

import pandas as pd
import numpy as np

if __name__ == "__main__":

    SN = 4

    df = pd.read_csv('data/atca/atca_all_timeseries.csv')
    min_detected = df.loc[df["source_J_val"]==True, "source_J"].min()
    min_detected_err_loc = df.loc[df["source_J_val"]==True, "source_J" ].argmin()
    min_detected_err = df.loc[min_detected_err_loc, "bkg_rms_J"]

    # undetected with senstive limits
    off_resolved = df.loc[(df["source_J_val"]==False) & (df["bkg_rms_J"]*4 < min_detected)]
    off_resolved = off_resolved.shape[0] / 24 # duration in days

    # detected
    on_resolved = df.loc[(df["source_J_val"]==True)].shape[0] / 24


    # add the full-integration non-detection duration
    f = pd.read_csv('data/atca/atca_full_integration_time_series.csv')

    # define duration of observations in each non-detection run

    f["duration"] = (pd.to_datetime(f["tstop"]) - pd.to_datetime(f["tstart"]))
    f["duration"] = f["duration"].dt.total_seconds() / 60 / 60 / 24 # in days

    # use non-detections with sensitive limits
    f = f.loc[(f["source_J_val"]==False) & (f["bkg_rms_J"]* 4 * np.sqrt(f["duration"]*24) < min_detected)]

    off_unresolved = f['duration'].sum()    

    on = on_resolved
    off = off_resolved + off_unresolved

    print(f"Minimum detected flux: {min_detected*1e3:.2f} +/- {min_detected_err*1e3:.2f} mJy")

    print(f"Duty cycle: {on / (on + off):.2f}")
