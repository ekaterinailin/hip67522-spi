# FIRST PART OF ANALYSIS:
# 0. Make a table for the CHEOPS observing log
# 1. Detrending of TESS data
# 2. Detrending of CHEOPS data
# 3. Extraction of flares from CHEOPS data
# 4. Extraction of flares from TESS data
# 5. Table with flare parameters for paper
python3 0_TABLE_cheops_observing_log.py
python3 1_detrend_tess.py
bash 2_detrend_cheops.sh
bash 3_extract_flare_cheops.sh
python3 4_extract_flare_tess.py 
python3 5_TABLE_flare_parameters.py