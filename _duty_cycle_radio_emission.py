
"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script reads the ATCA data and calculates the duty cycle of the detection.

"""

import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('../data/atca_all_timeseries.csv')
    min_detected = df.loc[df["source_J_val"]==True, "source_J"].min()
    min_detected_err_loc = df.loc[df["source_J_val"]==True, "source_J" ].argmin()
    min_detected_err = df.loc[min_detected_err_loc, "bkg_rms_J"]
    off_resolved = df.loc[(df["source_J_val"]==False) & (df["bkg_rms_J"]*5<min_detected)]
    off_resolved = off_resolved.shape[0] / 24
    on_resolved = df.loc[(df["source_J_val"]==True)].shape[0] / 24

    on = on_resolved
    off = off_resolved

    print(f"Minimum detected flux: {min_detected*1e3:.2f} +/- {min_detected_err*1e3:.2f} mJy")

    print(f"Duty cycle: {on / (on + off):.2f}")
