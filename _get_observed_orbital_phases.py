"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl

This script compiles the observed orbital phases of the TESS and CHEOPS missions for HIP 67522.

"""

import numpy as np 
import pandas as pd

from funcs.helper import get_tess_orbital_phases, get_cheops_orbital_phases

if __name__ == "__main__":

    hip67522params = pd.read_csv("../data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]
    teff = hip67522params[hip67522params.param=="teff_K"].val.values[0]
    tefferr = hip67522params[hip67522params.param=="teff_K"].err.values[0]
    radius = hip67522params[hip67522params.param=="radius_rsun"].val.values[0]
    radiuserr = hip67522params[hip67522params.param=="radius_rsun"].err.values[0]


    # ----------------------------------------------------------------------------------------


    cheops_phases, tcheops01, tcheops09, tot_obs_time_d_cheops = get_cheops_orbital_phases(period, midpoint, split=0.1, usemask=False)


    # GET OBSERVED TESS PHASES ---------------------------------------------------------------

    tess_phases, ttess01, ttess09, ttess = get_tess_orbital_phases(period, split=0.1, by_sector=False, usemask=False) 

    # write phases to file
    np.savetxt("../data/tess_phases.txt", tess_phases)
    np.savetxt("../data/cheops_phases.txt", cheops_phases)
