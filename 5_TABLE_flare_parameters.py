"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

Aggregates the flare data for HIP 67522.

"""

import pandas as pd
import numpy as np

if __name__ == "__main__":


    # CHEOPS data ---------------------
    cheops = pd.read_csv('results/cheops_flares.csv')
    cheops["rel_amp"] = cheops['amplitude'] / cheops['med_flux']
    cheops["mission"] = "CHEOPS"


    # TESS data -----------------------
    tess = pd.read_csv('results/tess_flares.csv')

    tess["rel_amp"] = tess['amplitude'] / tess['med_flux']
    tess["mission"] = "TESS"
    tess["t_peak_BJD"] = tess["t_peak_BJD"] + 2457000.



    # MERGE ---------------------------
    columns = [ 'mission', 't_peak_BJD', 'rel_amp',
            'mean_bol_energy', 'std_bol_energy']
    df = pd.concat([cheops[columns], tess[columns]], ignore_index=True)


    # ORBITAL PHASE --------------------
    hip67522params = pd.read_csv("data/hip67522_params.csv")
    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]
    df['orb_phase'] = (df['t_peak_BJD'] - midpoint) / period % 1


    # SAVE TO FILE ---------------------
    df.to_csv('results/hip67522_flares.csv', index=False)

    # LOG10 the flare energy -----------
    df['log10_mean_bol_energy'] = np.log10(df['mean_bol_energy'])
    df['log10_std_bol_energy_down'] =  df['log10_mean_bol_energy'] - np.log10(df['mean_bol_energy'] - df['std_bol_energy']) 
    df['log10_std_bol_energy_up'] = np.log10(df['mean_bol_energy'] + df['std_bol_energy']) - df['log10_mean_bol_energy'] 

    # EXCLUDED FLARES -------------------
    df["~"] = " "
    df.loc[df['log10_mean_bol_energy'] < 33.5, '~'] = r"excluded$^1$"

    # FORMAT STRINGS --------------------
    energy = df['log10_mean_bol_energy'].apply(lambda x: f"{x:.1f}")
    err_energy_down = df['log10_std_bol_energy_down'].apply(lambda x: f"{x:.2f}")
    err_energy_up = df['log10_std_bol_energy_up'].apply(lambda x: f"{x:.2f}")
    df['energy_str'] = energy + r"$_{-" + err_energy_down + r"}^{+" + err_energy_up + r"}$"

    # format new column r"$a$" relative amplitude, with 4 significant digits
    df['rel_amp'] = df['rel_amp'].apply(lambda x: f"{x:.3f}")

    # format t_peak_BJD to $t_{\rm peak}$ [BJD] with 2 significant digits
    df['t_peak_BJD'] = df['t_peak_BJD'].apply(lambda x: f"{x:.3f}")

    # select columns
    cols = ['mission', 't_peak_BJD', 'rel_amp', 'energy_str', 'orb_phase', '~']

    # rename columns
    newdf = df[cols].rename(columns={'mission': 'Mission',
                                    't_peak_BJD': r"$t_{\rm peak}$ [BJD]",
                                    'rel_amp': r"$a$",
                                    'energy_str': r"$\log_{10} E$ [erg]",
                                    'orb_phase': r"orb. phase"})


    # CONVERT TO LATEX -----------------
    table = newdf.to_latex(index=False, escape=False, column_format='@{}llllll@{}')
    table = table.replace("\\bottomrule", "\\botrule")

    print(table)