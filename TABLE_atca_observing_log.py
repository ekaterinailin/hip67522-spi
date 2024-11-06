"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script reads the observing log of HIP 67522 and converts it to a latex table.

"""

import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("../data/hip67522_atca_obs_summary.csv")

    total_obs_dur = (df["nrows"] / 90.  / 60.).sum().round(1)
    df["Duration"] = (df["nrows"] / 90.  / 60.).round(1).apply(lambda x: f"{x:.1f}")


    print(total_obs_dur)
    # convert obs_date to datetime
    df["obs_date"] = pd.to_datetime(df["obs_date"])
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


    configs = pd.read_csv("../data/atca_array_configs.csv")

    configs["date"] = pd.to_datetime(configs["date"], format="%d/%m/%Y")

    # merge the two dataframes
    df = pd.merge_asof(df, configs, left_on="obs_date", right_on="date")

    # drop the date column
    df = df.drop(columns=["date"])

    # reformat the obs_date to show only the date as a string
    df["obs_date"] = df["obs_date"].dt.strftime("%Y-%m-%d")

    # rename  columns
    df = df.rename(columns={"obs_date":"Obs. Date", "array_config": "Array", "tstart": "Obs. Start", "tstop": "Obs. Stop"})

    # put units and UTC into the second row
    df.loc[0] = ["[UTC]","[UTC]","[UTC]", "[h]", "Config."]

    # add a row for total duration at the bottom of the table
    df.loc[len(df)] = ["", "", "", "$\\Sigma = $" + f"{total_obs_dur:.1f}", ""]


    # convert to latex
    string = df.to_latex(index=False, escape=False)
    # replace midrule, top and bottom rule with hline
    string = string.replace("\\toprule", "\\hline")
    string = string.replace("\\bottomrule", "\\hline")

    # remove midrule
    string = string.replace("\\midrule", "")

    # put hline after [h] & \\
    string = string.replace("[h] &  \\\\", "[h] &  \\\\ \\hline")


    # write to file
    with open("../tables/atca_observing_log.tex", "w") as f:
        f.write(string)
        
    print(string)