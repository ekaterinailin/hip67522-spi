
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

    # Load the data
    f = pd.read_csv('../data/atca_full_integration_time_series.csv')
    f["duration"] = pd.to_datetime(f["tstop"]) - pd.to_datetime(f["tstart"])
    f["duration"] = f["duration"].dt.total_seconds() / 60 / 60 / 24

    min_detected = f.loc[f["source_J_val"]==True, "source_J"].min()
    f[f["source_J_val"]==True].sort_values(by= "source_J")

    off = f.loc[(f["source_J_val"]==False) & (f["source_J"]<min_detected), "duration"].sum()
    on = f.loc[(f["source_J_val"]==True), "duration"].sum()

    df = pd.read_csv('../data/atca_all_timeseries.csv')
    off_resolved = df.loc[(df["source_J_val"]==False) & (df["source_J"]<min_detected)]
    off_resolved = off_resolved.shape[0] / 24

    on = on - off_resolved
    off = off + off_resolved

    print(f"Duty cycle: {on / (on + off):.2f}")


    print("Individual detections below the threshold:")
    print(df.loc[(df["source_J_val"]==True) & (df["source_J"]<min_detected)])
