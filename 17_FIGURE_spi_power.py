"""
UTF-8, Python 3.11.7

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


Calculate the power of star-planet interactions for the HIP 67522 system,
and compare it to the observed SPI flux from flares, and LX.

Functions are the same as in Ilin+2024
"""


import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.constants import R_sun, R_jup

import numpy as np
import astropy.units as u
from astropy.constants import R_jup, R_sun, sigma_sb, L_sun


def b_from_lx_reiners(lx, r, error=False, lx_err=None, r_err=None):
    """Reiners et al. 2022 Eq. 4 inverted to get magnetic field in Gauss.
    
    Parameters
    ----------
    lx : float
        X-ray luminosity in erg/s.
    r : float
        Stellar radius in R_sun.
    error : bool
        If True, return the error in the magnetic field by
        estimating it from the scatter in the relation, and the
        error in the stellar radius and X-ray luminosity.
    lx_err : float
        Error in X-ray luminosity in erg/s.
    r_err : float
        Error in stellar radius in R_sun.

    Returns
    -------
    B : float
        Magnetic field in Gauss.
    B_err : float
        Error in magnetic field in Gauss.
    """
    if np.isnan(lx) | np.isnan(r) | (lx == 0) | (r == 0):
        B = np.nan
        highB, lowB = np.nan, np.nan
        if error:
            return B, highB, lowB
        else:
            return B
    else:
        # convert stellar radius to cm
        rcm = (r * R_sun).to(u.cm).value

        # constants from Reiners et al. 2022 Eq. 4    
        a, b = 1 / 3.28e-12, 1.58

        # error on b from Reiners et al. 2022 Eq. 4
        b_err = 0.06

        # calculate magnetic field from inverse of Reiners et al. 2022 Eq. 4
        get_B = lambda lx, rcm, a, b: (a * lx)**(1/b) / (4. * np.pi * rcm**2)

        B =  get_B(lx, rcm, a, b) 

        if error:

            # convert error in radius
            rcm_err = (r_err * R_sun).to(u.cm).value

            # calculate upper and lower 1-sigma range on magnetic field
            highB = get_B(lx+lx_err, rcm+rcm_err, a, b - b_err)
            lowB = get_B(lx-lx_err, rcm-rcm_err, a, b + b_err)

            return B, highB, lowB
        else:
            return B

    

def calculate_relative_velocity(a_au, orbper, rotper, error=False,
                                a_au_err=None, orbper_err=None, rotper_err=None):
    """Calculate the relative velocity between stellar rotation at the planetary orbit
    and the orbital distance of the planet in km/s.

    Parameters
    ----------
    au : float
        Semi-major axis in AU.
    orbper : float
        Orbital period in days.
    rotper : float
        Stellar rotation period in days.
    error : bool
        If True, return the error in the relative velocity from
        error propagation.
    a_au_err : float
        Error in semi-major axis in AU.
    orbper_err : float
        Error in orbital period in days.
    rotper_err : float
        Error in stellar rotation period in days.

    Returns
    -------
    v_rel_km_s : float
        Relative velocity between stellar rotation at the planetary orbit
        and the orbital velocity of the planet in km/s.
    """
    
    # return in km/s
    # 1AU = 149597870.700 km
    # 1day = 24h * 3600s

    # convet au to km
    a_au = a_au * 149597870.700

    if (np.isnan(a_au) | np.isnan(orbper) | np.isnan(rotper) | 
        (a_au == 0) | (orbper == 0) | (rotper == 0)):
        v_rel = np.nan
        v_rel_err = np.nan

    else:
        # convert orbper to s
        orbper = orbper * 24. * 3600.

        # convert rotper to s
        rotper = rotper * 24. * 3600.
        
        v_rel = 2 * np.pi * a_au * (1/orbper - 1/rotper)

        if error:
            # get error in km/s
            dv_dau = 2 * np.pi * (1/orbper - 1/rotper)
            dv_dorbper = -2 * np.pi * a_au * (1/orbper**2)
            dv_drotper = 2 * np.pi * a_au * (1/rotper**2)

            # convert a_au_err to km
            a_au_err = a_au_err * 149597870.700

            # convert orbper_err to s
            orbper_err = orbper_err * 24. * 3600.

            # convert rotper_err to s
            rotper_err = rotper_err * 24. * 3600.

            # quadratic error propagation
            v_rel_err = np.sqrt((dv_dau * a_au_err)**2 +
                                (dv_dorbper * orbper_err)**2 +
                                (dv_drotper * rotper_err)**2)

    if error:

        return v_rel, v_rel_err

    else:
        return v_rel

def b_from_ro_reiners2022(Ro, error=False, Ro_high=None, Ro_low=None):
    """Calculate the manetic field from the Rossby number.
    Based on Reiners et al. (2022), Table 2.
    
    Parameters
    ----------
    Ro : float
        Rossby number.
    error : bool
        If True, return the error in the magnetic field from
        scatter in relation.
    Ro_high : float
        Upper 1-sigma limit in Rossby number.
    Ro_low : float
        Lower 1-sigma limit in Rossby number.

    Returns
    -------
    B : float
        Magnetic field in Gauss.
    """

    # slow rotator
    if Ro > 0.13:
        B = 199 * Ro**(-1.26)#pm .1
        if error:
            if Ro>=1:
                B_high = 199 * Ro_low**(-1.26 + 0.1)
                B_low = 199 * Ro_high**(-1.26 - 0.1)
            else:
                B_low = 199 * Ro_high**(-1.26 + 0.1)
                B_high = 199 * Ro_low**(-1.26 - 0.1)
            
    # fast rotator
    elif Ro < 0.13:
        B = 2050 * Ro**(-0.11) #pm 0.03
        if error:
            B_low = 2050 * Ro_low**(-0.11 + 0.03)
            B_high = 2050 * Ro_high**(-0.11 - 0.03)
    else:
        B, B_high, B_low = np.nan, np.nan, np.nan

    if error:
        return B, B_high, B_low
    else:
        return B



def pspi_kavanagh2022(Rp, B, vrel, a,  R, Bp=1., error=False, Rphigh=None, Bphigh=1.,
                      Bhigh=None, vrelhigh=None, alow=None, Rplow=None, Bplow=1., 
                      Blow=None, vrellow=None, ahigh=None, Rhigh=None, Rlow=None, frho=1):
    """Power of star-plaet interactions following the
    Saur et al. 2013 model, put in a scaling law by
    Kavanagh et al. 2022.
    
    Parameters
    ----------
    Rp : float
        Planet radius in Jupiter radii
    B : float
        Stellar magnetic field in G
    vrel : float
        Relative velocity in km/s
    a : float
        Orbital separation in AU
    R : float
        Stellar radius in R_sun
    Bp : float
        Planet magnetic field in G
    error : bool
        If True, return the error on the pspi
    Rphigh : float
        Upper limit on planet radius in Jupiter radii
    Bphigh : float
        Upper limit on planet magnetic field in G
    Bhigh : float   
        Upper limit on stellar magnetic field in G
    vrelhigh : float
        Upper limit on relative velocity in km/s
    alow : float    
        Lower limit on orbital separation in AU
    Rplow : float
        Lower limit on planet radius in Jupiter radii
    Bplow : float
        Lower limit on planet magnetic field in G
    Blow : float
        Lower limit on stellar magnetic field in G
    vrellow : float
        Lower limit on relative velocity in km/s
    ahigh : float
        Upper limit on orbital separation in AU
    Rhigh : float
        Upper limit on stellar radius in R_sun
    Rlow : float
        Lower limit on stellar radius in R_sun
    frho : float
        Factor to scale the stellar wind density. Default is 1, 
        which is the solar value.

    Returns
    -------
    pspi : float
        prop. to power of star-planet interaction in erg/s
    """
    # set rho star to fixed value of Alvarado-Gomez et al. 2020, and similar to solar value
    # use mass of proton in kg for ionized  hydrogen
    rho_star = frho * 2e22 * u.km**(-3) * 1.6726e-27 * u.kg

    # convert a from AU to km
    a = (a * u.AU).to(u.km)

    # convert Rp from Jupiter radii to km
    Rp = Rp * R_jup.to(u.km)

    # convert R from R_sun to km
    R = R * R_sun.to(u.km)

    gauss_cgs = u.g**(0.5) * u.cm**(-0.5) * u.s**(-1)
    vrel_squared_unit = u.km**2 / u.s**2
    unit_add = gauss_cgs * vrel_squared_unit


    if Bp > 0:

        constant_factor = np.pi**0.5 * 2**(-2/3)

        pspi = constant_factor * np.sqrt(rho_star) * Rp**2 * Bp**(2/3) * B**(1/3) * R**2 * vrel**2 * a**(-2) * unit_add

        if error:
            # convert alow and ahigh from AU to km
            alow = (alow * u.AU).to(u.km)
            ahigh = (ahigh * u.AU).to(u.km)

            # convert Rplow and Rphigh from Jupiter radii to km
            Rplow = Rplow * R_jup.to(u.km)
            Rphigh = Rphigh * R_jup.to(u.km)

            # convert R_high and R_low from R_sun to km
            Rhigh = Rhigh * R_sun.to(u.km)
            Rlow = Rlow * R_sun.to(u.km)

            pspi_high = constant_factor * np.sqrt(rho_star) * Rphigh**2 * Bphigh**(2/3) * Bhigh**(1/3) * Rhigh**2 * vrelhigh**2 * alow**(-2) * unit_add
            pspi_low = constant_factor *  np.sqrt(rho_star) * Rplow**2 * Bplow**(2/3) * Blow**(1/3) * Rlow**2 * vrellow**2 * ahigh**(-2) * unit_add
        
        
            return (pspi.decompose().to(u.erg/u.s).value, 
                    pspi_high.decompose().to(u.erg/u.s).value, 
                    pspi_low.decompose().to(u.erg/u.s).value)

        else:

            return pspi.decompose().to(u.erg/u.s).value

    elif Bp==0:

        constant_factor = np.pi**0.5
              
        pspi = np.sqrt(rho_star) * Rp**2 * R**4 * B * vrel**2 * a**(-4) * unit_add

        if error:
            # convert alow and ahigh from AU to km
            alow = (alow * u.AU).to(u.km)
            ahigh = (ahigh * u.AU).to(u.km)

            # convert Rplow and Rphigh from Jupiter radii to km
            Rplow = Rplow * R_jup.to(u.km)
            Rphigh = Rphigh * R_jup.to(u.km)

            # convert R_high and R_low from R_sun to km
            Rhigh = Rhigh * R_sun.to(u.km)
            Rlow = Rlow * R_sun.to(u.km)

            pspi_high = np.sqrt(rho_star) * Rphigh**2 * Rhigh**4 * Bhigh * vrelhigh**2 * alow**(-4) * unit_add
            pspi_low = np.sqrt(rho_star) * Rplow**2 * Rlow**4 * Blow * vrellow**2 * ahigh**(-4) * unit_add
        
            return (pspi.decompose().to(u.erg/u.s).value, 
                    pspi_high.decompose().to(u.erg/u.s).value, 
                    pspi_low.decompose().to(u.erg/u.s).value)
        else:

            return pspi.decompose().to(u.erg/u.s).value


def rossby_reiners2014(Lbol, Prot, error=False, Lbol_high=None, 
                       Lbol_low=None, Prot_high=None, Prot_low=None):
    """Calculate the Rossby number for a given bolometric luminosity.
    Based on Reiners et al. (2014), as noted in Reiners et al. (2022).
    
    Parameters
    ----------
    Lbol : float
        Bolometric luminosity in solar units.
    Prot : float
        Rotation period in days.
    error : bool
        If True, return the error in the Rossby number from
        error propagation.
    Lbol_high : float
        Upper 1-sigma error in bolometric luminosity in solar units.
    Lbol_low : float
        Lower 1-sigma error in bolometric luminosity in solar units.
    Prot_high : float
        Upper 1-sigma error in rotation period in days.
    Prot_low : float    
        Lower 1-sigma error in rotation period in days.


    Returns
    -------
    Ro : float
        Rossby number.
    """
    # convective turnover time
    tau = 12.3 / (Lbol**0.5)

    if error:
        tau_high = 12.3 / (Lbol_low**0.5)
        tau_low = 12.3 / (Lbol_high**0.5)

    # Rossby number
    Ro = Prot / tau

    if error:
        Ro_high = Prot_high / tau_low
        Ro_low = Prot_low / tau_high

    if error:
        return Ro, Ro_high, Ro_low
    else:
        return Ro



# increase font size
plt.rcParams.update({'font.size': 13})

if __name__ == "__main__":

    # GET STELLAR AND PLANET PARAMETERS -----------------------------------------------------

    hip67522params = pd.read_csv("data/hip67522_params.csv")

    period = hip67522params[hip67522params.param=="orbper_d"].val.values[0]
    prot = hip67522params[hip67522params.param=="rotper_d"].val.values[0]
    midpoint = hip67522params[hip67522params.param=="midpoint_BJD"].val.values[0]
    teff = hip67522params[hip67522params.param=="teff_K"].val.values[0]
    tefferr = hip67522params[hip67522params.param=="teff_K"].err.values[0]
    radius = hip67522params[hip67522params.param=="radius_rsun"].val.values[0]
    radiuserr = hip67522params[hip67522params.param=="radius_rsun"].err.values[0]


    # flares table 
    flares = pd.read_csv("results/hip67522_flares.csv").sort_values("mean_bol_energy", ascending=True)



    # maximum flare energy in sample
    Emax = flares.mean_bol_energy.max() * u.erg
    # threshold energy
    Emin = flares.mean_bol_energy.iloc[1] * u.erg

    # flux in excess of lambda0
    lambd = 0.5 / u.d

    # power law exponent of FFD read from results/ffd_full_sample_alpha.txt
    alpha = np.loadtxt("results/ffd_full_sample_alpha.txt").astype(float)
    print(f"Power law exponent of full sample: {alpha:.2f}")

    # planet radius in units of stellar radii
    rp_in_stars = 0.0668

    # semi major axis in units of stellar radii
    a = 11.7

    # X-ray luminosity fromo Maggio et al. 2024
    lx, lxerr =  3e30, 0.2e30  

    # Calculate the stellar magnetic field ------------------------------------------------

    # ... from X-ray luminosity
    Bs = b_from_lx_reiners(lx, radius, error=True, lx_err=lxerr, r_err=radiuserr)
    Blx, Bhigh, Blow = Bs

    print(f"Magnetic field strength from X-ray luminosity: {Blx:.0f} G")

    # ... from Rossby number
    Lbol = (4 * np.pi * (radius * R_sun.to(u.cm))**2 * sigma_sb * (teff *u.K)**4 / L_sun).decompose()
    ro = rossby_reiners2014(Lbol, prot)
    Bro = b_from_ro_reiners2022(ro)

    print(f"Magnetic field strength from Rossby number: {Bro:.0f} G")


    # Power of star-planet interactions -----------------------------------------------------

    # orbital distance in AU
    a_au = (a * radius * R_sun).to(u.AU).value

    # relative velocity in km/s
    vrel = np.abs(calculate_relative_velocity(a_au, period, prot))

    # planet radius in Jupiter radii
    rp = (rp_in_stars * radius * R_sun / R_jup).decompose().value    

    # range for the planet magnetic field 0.01-100 G
    bp = np.logspace(-2, 2, 30)

    # initialize min and max
    minspi, maxspi = np.inf, 0

    # range for stellar wind density at the base of the corona
    for frho in np.logspace(-1, 2, 10):
        # range of stellar magnetic fields roughly covering the range from X-ray luminosity and Rossby number
        for b in np.linspace(Bro-500, Blx+500, 10):
            # get power of interaction for each combination of parameters
            pspi = [pspi_kavanagh2022(rp,b, vrel, a_au, radius, Bp=bp_, frho=frho) for bp_ in bp]
            # plot as a function of planet magnetic field
            plt.plot(bp, pspi)

            if np.min(pspi) < minspi:
                minspi = np.min(pspi)
            if np.max(pspi) > maxspi:
                maxspi = np.max(pspi)

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Planet magnetic field [G]")
    plt.ylabel("Power of star-planet interactions [erg/s]")
    plt.savefig("plots/pspi_aw.png")


    print(f"Minimum power of star-planet interactions: {minspi:.2e} erg/s")
    print(f"Maximum power of star-planet interactions: {maxspi:.2e} erg/s")



    # plot the total flaring SPI flux as a function of maximum flare energy --------------------------------

    plt.figure(figsize=(8, 6))


    # define a range of maximum flare energies
    Emaxs = np.logspace(35, 38, 100) 

    # calculate beta of power law based on rate at threshold
    beta = lambd * (alpha -1) / (Emin**(1-alpha)) 

    Es = np.logspace(32, 33, 100)

    # show that the lower energy range does not matter unless it's close to Emax
    for E in Es:

        # integral over FFD
        tot_flux =  beta.value * (Emaxs**(-alpha + 2) - E**(-alpha + 2))  / (-alpha + 2)

        # convert to erg/s
        tot_flux = (tot_flux * u.erg / u.d).to("erg/s").value

        # plot
        plt.plot(Emaxs, tot_flux, alpha=0.4, color="navy")

    # add legend handle for observed SPI flux
    plt.plot([], [], color="navy", alpha=0.7, label="Planet-induced flare flux from observations") 

    # fill between min and max expected SPI flux
    plt.fill_between(Emaxs, minspi, maxspi, color="steelblue", alpha=0.6, label="Star-planet interaction flux (Saur et al. 2013)")    

    # add line for maximum flare energy in sample
    plt.axvline(Emax.value, color="black", linestyle="--", label="Max. flare energy in clustered region")

    # add line for upper limit on SPI flux based on LX
    plt.axhline(lx, color="black", linestyle=":", label=r"Upper limit $L_{\rm SPI} < L_{\rm X}$")

    # plt.axhline(4.8e29) #-- this where the power quoted in the paper is
    # calculate tot_flux closest to Emax
    spi_flux = tot_flux[np.argmin(np.abs(Emaxs-Emax.value))]
    print(f"SPI flux: {spi_flux:.2e} erg/s")


    # layout
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Power [erg/s]")
    plt.xlabel(r"Maximum planet-induced flare energy $E_{\rm max}$ [erg]")
    plt.xlim(Emaxs[0], Emaxs[-1])
    plt.legend(loc=(0.33, 0.05), frameon=False, fontsize=11.5)  
    plt.savefig("plots/paper/SPI_flux_vs_Emax.png", dpi=300)

    # what is the frequency of flares above Emax * X?
    X = 1
    f = - beta * ((Emax * X)**(-alpha + 1)) / (-alpha + 1)
    fr = f.to("1/year").value
    print(f"Flares above {Emax*X:.2e} occur {fr:.1f} times per year")