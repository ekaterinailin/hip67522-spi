"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script reads the observing log of HIP 67522 and converts it to a latex table.

"""

import pandas as pd

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    measurements = pd.read_csv("data/atca/atca_full_integration_time_series.csv")
    measurements["Stokes_I"] = measurements.apply(lambda x: f"{x.source_J*1e3:.2f} [{x.bkg_rms_J*1e3:.2f}]", axis=1)
    measurements.loc[measurements["source_J_val"]==False, "Stokes_I"] = measurements.loc[measurements["source_J_val"]==False, "bkg_rms_J"].apply(lambda x: f"<{x*4*1e3:.2f}")


    df = pd.read_csv("data/atca/hip67522_atca_obs_summary.csv")

    total_obs_dur = (df["nrows"] / 90.  / 60.).sum().round(1)
    df["Duration"] = (df["nrows"] / 90.  / 60.).round(1).apply(lambda x: f"{x:.1f}")

    # convert to datetime
    measurements["obsname"] = pd.to_datetime(measurements["obsname"])
    # convert obs_date to datetime
    df["obs_date"] = pd.to_datetime(df["obs_date"])

    # merge log with observations
    df = pd.merge(df, measurements[["obsname", "Stokes_I"]], left_on="obs_date", right_on="obsname")

    # sort by obs_date
    df = df.sort_values("obs_date")
    # drop nrows
    df = df.drop(columns=["nrows"])
    # round tstart and tstop to nearest minute, but first convert to time object without a date

    df["tstart"] = pd.to_datetime(df["tstart"], format="%H:%M:%S.%f")
    df["tstop"] = pd.to_datetime(df["tstop"], format="%H:%M:%S.%f")
    # round to nearest minute, but really round, don't cut off and keep as time object
    df["tstart"] = (df["tstart"].dt.round("min"))
    df["tstop"] = (df["tstop"].dt.round("min"))

    # reformat to hh:mm
    df["tstart"] = df["tstart"].dt.strftime("%H:%M")
    df["tstop"] = df["tstop"].dt.strftime("%H:%M")


    configs = pd.read_csv("data/atca/atca_array_configs.csv")

    configs["date"] = pd.to_datetime(configs["date"], format="%d/%m/%Y")

    # merge the two dataframes
    df = pd.merge_asof(df, configs, left_on="obs_date", right_on="date")

    # drop the date column
    df = df.drop(columns=["date"])
    df = df.drop(columns=["obsname"])

    # reformat the obs_date to show only the date as a string
    df["obs_date"] = df["obs_date"].dt.strftime("%Y-%m-%d")

    # rename  columns
    df = df.rename(columns={"obs_date":"Obs. Date", 
                            "array_config": "Array", 
                            "tstart": "Obs. Start", 
                            "tstop": "Obs. Stop",
                            "Stokes_I": "Stokes I"})

    # put units and UTC into the second row
    df.loc[-1] = ["[UTC]","[UTC]","[UTC]", "[h]", "[mJy]","Config."]
    df.index = df.index + 1
    df = df.sort_index()

    # add a row for total duration at the bottom of the table
    df.loc[len(df)] = ["", "", "", "$\\Sigma = $" + f"{total_obs_dur:.1f}", "",""]

    # convert to latex
    string = df.to_latex(index=False, escape=False)
    # replace midrule, top and bottom rule with hline
    string = string.replace("\\toprule", "\\hline")
    string = string.replace("\\bottomrule", "\\hline")

    # remove midrule
    string = string.replace("\\midrule", "")

    # put hline after [h] & \\
    string = string.replace("[mJy] & Config. \\\\", "[mJy] &  Config. \\\\ \\hline")

    # write to file
    with open("tables/atca_observing_log.tex", "w") as f:
        f.write(string)
        
    print(string)