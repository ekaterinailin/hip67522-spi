### Searching for planet-induced radio signal from a young Hot Jupiter host star

The code in this repository reproduces all figures and results in the manuscript of the same title. Dynamic spectra are produced using the method by Bloot et al. (2024), hosted under  [SBloot/radio-dynspec]{https://github.com/SBloot/radio-dynspec}. 

The analysis consists of three parts, contained in three bash scripts (PART1, PART2, PART3). The underlying scripts in each task are enumerated 1 through 18. Only script 11 is computationally heavy, and needs to be run parallelized.

Figures 1 and 2 can be reproduced with FIGURE_illustration_of_system.py and FIGURE_polar_histogram.py.

The CHEOPS observing log table is produced with TABLE_cheops_observing_log.py

Scripts with additional functionalities and checks are prefixed with "x", including the script that runs the CHEOPS reduction pipeline.

If you believe my flares and want to run the statistics and energetic only, the flare tables are provided under results/.

The analysis was run entirely with Python 3.11.7, package requirements can be found under requirements.txt