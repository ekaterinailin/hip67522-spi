## Close-in planet induces flares on its host star 

The code in this repository reproduces all results in the manuscript of the same title. The light curve files from CHEOPS can be provided upon request, but will go public by the end of 2025, and can be accessed based on the observing log information (see _data/cheops/_). Reduced light curves will be made available on Zenodo at the time of publication.

### Structure of this repository

The analysis consists of three parts, summarized in three bash scripts (``PART1``, ``PART2``, ``PART3``). The scripts in each part are enumerated. Only script 11 is computationally heavy (see **Requirements**).

Figures 1 and 2 can be reproduced with 

``FIGURE_illustration_of_system.py`` and 
``FIGURE_polar_histogram.py``.

The **CHEOPS** observing log table is produced with ``TABLE_cheops_observing_log.py``

Scripts with additional functionalities and checks are prefixed with ``x_``, including the script that runs the CHEOPS reduction pipeline.

To run the statistics and energetic only, the flare tables are provided under _results/_.

### Requirements

Software package requirements can be found under requirements.txt. All required Python packages can be installed with pip or conda (see their respective documentations). Installation time oom minutes.

All scripts except script 11 were run on 

- Ubuntu 22.04.5 LTS, 16 GB RAM, Python 3.11.7 

Script 11 was run on 

- Ubuntu 20.04.5 LTS, 64 GB RAM, Python 3.8.10. 

Script 11 is optimized to 40 cores, but can be adjusted to run on fewer or more.






