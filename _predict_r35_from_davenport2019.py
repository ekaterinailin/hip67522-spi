
"""
UTF-8, Python 3

------------
HIP 67522
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl


Compute the r35 flare frequency from the Davenport 2019 model.
"""


import numpy as np

if __name__ == "__main__":
    a1, a2, a3 = -0.07, 0.79, -1.06
    b1, b2, b3 = 2.01, -25.15, 33.99
    mass_sun = 1.2
    age_Myr = 16.5
    a = a1 * np.log10(age_Myr) + a2 * mass_sun  + a3
    b = b1 * np.log10(age_Myr) + b2 * mass_sun  + b3 
    log10e = 35
    lognu = a * log10e + b
    
    print("log10(nu) = ", lognu)