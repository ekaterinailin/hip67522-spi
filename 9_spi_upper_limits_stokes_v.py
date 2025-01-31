"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl

Read ATCA Stokes V data, and calculate

- orbital phase coverage
- duration of a 1 deg wide cone burst
- upper limit on the Stokes V burst flux
- efficiency of the SPI assuminng all other effects are irrelevant (e.g. beaming, etc.)

"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time

def flux(power, efficiency =1, omega=0.1, d=124.7*u.pc, B=2100*u.G, frac_at_pole = 0.2):
    """ECME flux given power of interaction.
    
    Parameters
    ----------
    power : astropy.Quantity
        Power of interaction in erg/s
    efficiency : float
        Efficiency of the emission, from 0 to 1
    omega : float
        Solid angle of the emission in steradians
    d : astropy.Quantity
        Distance to the source in parsecs
    B : astropy.Quantity
        Average surface magnetic field in Gauss
    frac_at_pole : float
        Fraction of surface magnetic field at the pole
    """
    flux = (power / (omega * d**2)).to("erg * s^-1 * cm^-2") # flux at distance through the solid angle
    flux_per_Hz = flux / (2.8e6 * (B * frac_at_pole) * u.Hz) # per bandwidth
    fluxinband = flux_per_Hz.value * 1e23 * efficiency * 1e3 # in mJy with some efficiency 
    return fluxinband



if __name__ == "__main__":  

    # read in the ATCA data in Stokes I
    df = pd.read_csv('../data/atca_full_integration_time_series.csv')

    # convert uUTCc to JD
    df['datetime'] = (df['obsname'] + ' ' + df['tstart']).apply(lambda x: Time(x, format='iso', scale='utc').datetime)
    df['datetime_end'] = (df['obsname'] + ' ' + df['tstop']).apply(lambda x: Time(x, format='iso', scale='utc').datetime)
    df["jd"] = df['datetime'].apply(lambda x: Time(x).jd)
    df["jd_end"] = df['datetime_end'].apply(lambda x: Time(x).jd)

    # calculate the duration of each observation
    df['duration'] = (df['datetime_end'] - df['datetime']).apply(lambda x: x.total_seconds()  / 3600) 

    # get Stokes V flux upper limits
    stokesV = pd.read_csv('../data/Stokes_V_fluxes.csv')

    # merge the two dataframes
    df["obsname"] = df["obsname"].str.replace("-", "").astype(int)
    merged = pd.merge(df, stokesV, on='obsname', suffixes=('', '_stokesV'))

    # read in stellar parameters
    hip67522params = pd.read_csv("../data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0] * 24
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]


    # PHASE COVERAGE -------------------------------------------------------------

    # calculate the phase coverage of each observation
    df['start_phase'] = ((df['jd'] - midpoint) % period) / period
    df["orb_coverage"] = df["duration"] / period
    df['end_phase'] = (df['start_phase'] + df['orb_coverage']) 

    phases = np.linspace(0, 1, 1000)
    coverage = np.zeros_like(phases)

    for i, row in df.iterrows():
        coverage[(phases > row['start_phase']) & (phases < row["end_phase"])] = 1

        if row["end_phase"] > 1:
            coverage[(phases > 0) & (phases < row["end_phase"] - 1)] = 1

    phase_сoverage = np.sum(coverage) / len(coverage)  

    print(f"Phase coverage of ATCA observations: {phase_сoverage:.2f}")

    # -----------------------------------------------------------------------------

    # BURST DURATION --------------------------------------------------------------

    # for a 1 deg wide cone what would be a typical duration 
    # of the observation given the orbital period in hours?
    dur_burst = period / 360 
    print(f"Duration of a 1 deg wide cone burst: {dur_burst:.2f} hours")

    # -----------------------------------------------------------------------------

    # FLUX UPPER LIMIT CALCULATION -------------------------------------------------

    merged["upperlimit_burst_mJy"] = merged.bkg_rms_J_stokesV * 4 / dur_burst * merged.duration * 1e3

    upperlimit = np.mean(merged["upperlimit_burst_mJy"])

    print(f"Average upper limit on the Stokes V burst flux: {upperlimit:.2f} mJy")

    # -----------------------------------------------------------------------------

    # WHAT IS THE EFFICIENCY OF THE SPI? -------------------------------------------

    # solid angle of cone with 1 deg width
    omega = np.cos(89*np.pi/180) * 4 * np.pi

    # range of SPI powers
    powers = np.logspace(20,26,100) * u.erg / u.s

    # calculate fluxes
    fluxes = np.array([flux(power, omega=omega) for power in powers])

    # plot
    # plt.plot(powers,fluxes)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.axhline(upperlim_mJ)

    power_at_upperlim = powers[np.argmin(np.abs(fluxes-upperlimit))]

    fraction_of_predicted_power = power_at_upperlim / (6.3e25 * u.erg / u.s) # from Ilin+2024

    print(f"Power of SPI at measured upper limit: {power_at_upperlim:.2e}")
    print(f"... which is a fraction of predicted power of 6.3e25 erg/s: {fraction_of_predicted_power:.3f}")
