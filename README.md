## Close-in planet induces flares on its host star 

The code in this repository reproduces all results in the manuscript of the same title. The TESS light curves required for the analysis are publicly avaiable via the Mikulski Archive for Space Telescopes (MAST). The CHEOPS light curve files can be provided upon request. Raw CHEOPS light curves will go public by the end of 2025, and will be accessible via the CHEOPS Archive using on the observing log information (see _data/cheops/_). Reduced CHEOPS light curves will be made available on Zenodo at the time of publication, or upon request.

### Structure of this repository

The analysis consists of three parts, summarized in three bash scripts (``PART1``, ``PART2``, ``PART3``). The scripts in each part are enumerated. Only script 11 is computationally heavy (see **Requirements**). Below each part is listed as described in the bash scripts.

#### PART 1: De-trending and flare characterization

0. Make a table for the CHEOPS observing log
1. Detrending of TESS data
2. Detrending of reduced CHEOPS data
3. Extraction of flares from CHEOPS data
4. Extraction of flares from TESS data
5. Table with flare parameters for paper

#### PART 2: Flare clustering analysis

8. Get observed phases for TESS and CHEOPS, i.e. the phase coverage
9. Make a table with the observed phases for the paper
10. Clustering analysis -- find best fit for the flare parameters with 100 bins
11. Calculate the bayes factor (_CAUTION_: make sure to have enough RAM, see **Requirements**)
12. Make the best fit figure for the paper
13. Make the Bayes Factor and AIC figure for the paper
14. Double check the bin effect on the best fit (produces a figure showing average best-fit parameters)

#### PART 3: Energetics

15. Calculate the power law exponent for the flare energy distribution
16. Make the figure that puts the FFD in one figure with the Tu et al sample
17. Calculate power of SPI and plot it
18. Calculate energy limited escape driven by CMEs

#### Other notable scripts and data provided

- Figures 1 and 2 can be reproduced with ``FIGURE_illustration_of_system.py`` and ``FIGURE_polar_histogram.py``.
- The **CHEOPS** observing log table is produced with ``TABLE_cheops_observing_log.py``
- Scripts with additional functionalities and checks are prefixed with ``x_``, including the script that runs the CHEOPS reduction pipeline.
- To run the statistics and energetic only, the flare tables are provided under _results/_.

### Installation

Clone this repository with git

``git clone https://github.com/ekaterinailin/hip67522-spi.git``
``git checkout flaring-spi``

and install the dependencies found under ``requirements.txt``. All required Python packages can be installed with pip or conda (see their respective documentations). Installation time is oom minutes.

### Requirements

All scripts except script 11 were run on 

- Ubuntu 22.04.5 LTS, 16 GB RAM, Python 3.11.7 

Script 11 was run on 

- Ubuntu 20.04.5 LTS, 64 GB RAM, Python 3.8.10. 

Script 11 is optimized to 40 cores, but can be adjusted to run on fewer or more.






