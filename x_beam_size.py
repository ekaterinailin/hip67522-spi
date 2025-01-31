
"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


This script calculates the radio luminosity of HIP 67522 based on 
the the ATCA observation, bot for quiescence and bursts.
"""

import numpy as np
import astropy.units as u

beam_major = 7.04 * u.arcsec
beam_minor = 3.74 * u.arcsec

beam_size = np.pi * beam_major * beam_minor / (4 * np.log(2))
beam_size.to(u.arcsec**2)