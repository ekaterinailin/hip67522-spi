"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script reads the observing log of CHEOPS observations of HIP 67522 and converts it to a latex table.

"""



import pandas as pd
import numpy as np


if __name__ == "__main__":


    cheops = pd.read_csv("../data/cheops_all_timeseries.csv")

    # count all the data points
    print("Total number of data points: ", cheops.shape[0])

    total_covered = cheops.shape[0] * 10 / 60 / 60

    print(f"Total observed time: {total_covered:.2f} days")


    df = pd.read_csv("../data/CHEOPS_observing_log.csv")

    df["File Key"] = ("CH\_PR" + df["Visit ID"].astype(str).apply(lambda x: x[:6]) + 
            "\_TG" + df["Visit ID"].astype(str).apply(lambda x: x[6:]) + 
                "\_V" + df["Revision"].astype(str).apply(lambda x: x.zfill(2)) + "00" )

    df["Obs. baseline [h]"] = ((pd.to_datetime(df["Obs Stop"].astype(str).apply(lambda x: x.strip("T")), format="mixed") - 
                        pd.to_datetime(df["Obs Start"].astype(str).apply(lambda x: x.strip("T")), format="mixed"))).values.astype(float) / 3600 / 1e9

    total_duration = df["Obs. baseline [h]"].sum()

    # round duration to 2 decimal places
    df["Obs. baseline [h]"] = df["Obs. baseline [h]"].apply(lambda x: np.round(x, 2))

    # convert to string and remove all decimals beyond 2
    df["Obs. baseline [h]"] = df["Obs. baseline [h]"].astype(str).apply(lambda x: x[:x.index(".") + 3])


    rename_dict = {"Obs Start": "Start Date [UTC]"}
    df = df.rename(columns=rename_dict)

    df["Start Date [UTC]"] = pd.to_datetime(df["Start Date [UTC]"].astype(str).apply(lambda x: x.strip("T")), format="mixed").dt.strftime("%Y-%m-%d %H:%M")

    cols = ["OBSID", "File Key", "Start Date [UTC]", "Obs. baseline [h]"]
    sel = df[cols]

    # add a last row with the total duration
    append = ["", "", "", r"$\Sigma_{\rm baseline}=$ " + f"{total_duration:.1f}" ]
    sel.loc[len(sel)] = append
    append = ["", "", "", r"$\Sigma_{\rm observed}=$ " + f"{total_covered:.1f}" ]
    sel.loc[len(sel)] = append


    print("Duty cycle: ")   
    print(total_covered / total_duration)

    # convert sel to latex
    table = sel.to_latex(index=False, escape=False, column_format="llll")

    # replace toprule and bottomrule with hline
    table = table.replace("\\toprule", "\\hline")
    table = table.replace("\\bottomrule", "\\hline")
    table = table.replace("\\midrule", "\\hline")

    print(table)

    # save table to file
    with open("../tables/cheops_observing_log.tex", "w") as f:
        f.write(table)

        

