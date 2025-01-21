"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl


Extracts the flare light curve from a TESS light curve.
Fits a flare model of choice and calculates the equivalent duration and bolometric flare energy.

"""

import pandas as pd 
import numpy as np

if __name__ == "__main__":

    # number of bins
    n_bins = 100

    # load observed phases
    tess_phases = np.loadtxt("results/tess_phases.txt")
    cheops_phases = np.loadtxt("results/cheops_phases.txt")

    # attach exposure times: 10 sec for CHEOPS, 120 sec for TESS
    weights = np.concatenate([np.ones_like(cheops_phases) * 10. , 
                                np.ones_like(tess_phases) * 120.] )
    obs_phases = np.concatenate([cheops_phases, tess_phases])

    # create a column with respective mission names
    mission = len(cheops_phases) * ["CHEOPS"] + len(tess_phases) * ["TESS"]

    # create a pandas dataframe
    df = pd.DataFrame({"Mission": mission, 
                    "Orb. phase": obs_phases, 
                    "Exposure time [s]": weights})

    # get total observing time per bin, in 200 bins running from 0 to 1
    sum = df.groupby(pd.cut(df["Orb. phase"], np.linspace(0, 1, n_bins+1))).sum()

    # cast observing time in second as tring with 0 decimal places
    df["Exposure time [s]"] = df["Exposure time [s]"].apply(lambda x: f"{x:.0f}")

    # sort by phase
    df = df.sort_values(by="Orb. phase")


    # PRINT THE HEAD OF THE TABLE
    # get only the first 10 rows
    dfhead = df.head(10)

    # to latex string
    string = dfhead.to_latex(index=False, escape=False, column_format='@{}lll@{}')

    # replace bottomrule with botrule
    string = string.replace("\\bottomrule", "\\botrule")

    print("\nExtended Data Table: Observed phases and exposure times\n")
    print(string)

    # SAVE RAW PHASES
    df.to_csv("results/observing_times.csv", index=False)
    print("Saved raw observing phases to observing_times.csv\n")

    # BINNED TABLE
    sumd  = sum[['Exposure time [s]']] / 60 / 60 / 24   
    sumd = sumd.rename(columns={"Exposure time [s]": "Exposure time [d]"})
    sumd = sumd.reset_index()
    assert sumd["Exposure time [d]"].sum() > 73.2
    assert sumd["Exposure time [d]"].sum() < 73.3


    # split the table into two columns
    second_col = sumd.iloc[50:]
    sumd = sumd.iloc[:50]



    sumd["Orb. phase 9 9 9 "] = second_col["Orb. phase"].values
    sumd["Exposure time [d] 9 9 9 "] = second_col["Exposure time [d]"].values

    print(sumd.head())


    string = sumd.to_latex(column_format='@{}llll@{}', index=False, escape=False)

    # remove 9 9 9 from column names
    string = string.replace(" 9 9 9 ", "")


    print("\nExtended Data Table: Phase coverage\n")
    print(string)
