# SECOND PART OF ANALYSIS:
# 8. Get observed phases for TESS and CHEOPS
# 9. Make a table with the observed phases for the Paper
# 10. Clustering analysis -- find best fit for the flare parameters with 100 bins
# 11. Calculate the bayes factor (CAUTION -- large calculation, parallelize)
python3 8_get_observed_orbital_phases.py
python3 9_TABLE_phase_coverage_raw_and_binned.py
python3 10_model_fit_with_elevated_phase_range.py 101
# OR run for a range of bin sizes
bash 10_model_fit_with_elevated_phase_range.sh
# python3 11_bayes_factor.py 101
# OR RUN FOR A RANGE OF BIN SIZES
# bash 11_bayes_factor.sh 