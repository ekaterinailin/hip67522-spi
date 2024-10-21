"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

Calculate the duty cycle of the quiescent emission of HIP 67522. Define a minimum flux density,
and on and off states based on that, including non-detections.

"""

import pandas as pd


if __name__ == "__main__":


    # get full integration image fluxes
    f = pd.read_csv('../data/atca_full_integration_time_series.csv')

    # define duration of observations in each run
    f["duration"] = pd.to_datetime(f["tstop"]) - pd.to_datetime(f["tstart"])
    f["duration"] = f["duration"].dt.total_seconds() / 60 / 60 / 24 # in days

    # define minimum flux density
    min_detected = f.loc[f["source_J_val"]==True, "source_J"].min()
    
    # define off state as non-detections with upper limits below the minimum flux density
    off = f.loc[(f["source_J_val"]==False) & (f["source_J"]<min_detected), "duration"].sum()

    # define on state as detections above the minimum flux density
    on = f.loc[(f["source_J_val"]==True), "duration"].sum()

    # from on state subtract the time of non-detections from the time series
    df = pd.read_csv('../data/atca_all_timeseries.csv')
    off_resolved = df.loc[(df["source_J_val"]==False) & (df["source_J"]<min_detected)]
    off_resolved = off_resolved.shape[0] / 24 # in days

    # shift the resolved time from the on to the off state
    on = on - off_resolved
    off = off + off_resolved    

    # calculate the duty cycle
    duty_cycle = on / (on + off)

    print(f"The duty cycle of the quiescent emission of HIP 67522 is {duty_cycle:.2f}.")

    # for double checking:
    
    # how far is the resolved data away from the unresolved data? only 1h! the following should give one row:
    # df.loc[(df["source_J_val"]==True) & (df["source_J"]<min_detected)]