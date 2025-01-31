### Searching for planet-induced radio signal from a young Hot Jupiter host star

The code in this repository reproduces all figures and results in the manuscript of the same title. Dynamic spectra are produced using the method by Bloot et al. (2024), hosted under  [SBloot/radio-dynspec]{https://github.com/SBloot/radio-dynspec}. 

### Structure of this repository

The scripts in each part are enumerated in the order in which they should be executed to self-consistently reproduce all figures. Scripts with additional functionalities and checks are prefixed with ``x_``. The numbered scripts are

0. Make a table for the ATCA observing log.
1. Format flux values extracted from images in CASA into digestible csv files.
2. Calculate the duty cycle of HIP 67522 radio emission.
3. Make a large figure showing all L band light curves.
4. Make an extra figure showing the June 11, 2024 burst, at 30-min cadence.
5. Calculate radio luminosities and brightness temperatures for quiescence and bursts.
6. Fit a joint power law to all spectra, and make a figure.
7. XSPEC script to extrapolate the X-ray luminosity from Maggio et al. 2024 to 0.2-2.0 keV band.
8. Extrapolate the radio flux to VLA band, and check for consistency with GBR.
9. Estimate the efficiency of SPI, assuming the Stokes V upper limits are informative.


### Installation

Clone this repository with git

``git clone https://github.com/ekaterinailin/hip67522-spi.git``
``git checkout radio-spi``

and install the dependencies found under ``requirements.txt``. All required Python packages can be installed with pip or conda (see their respective documentations). Installation time is oom minutes.

### Requirements

All scripts except Ubuntu 22.04.5 LTS, 16 GB RAM, Python 3.11.7. 



