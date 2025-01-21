# SECOND PART OF ANALYSIS:
# 8. Get observed phases for TESS and CHEOPS
# 9. Make a table with the observed phases for the Paper
# 10. Clustering analysis -- find best fit for the flare parameters with 100 bins
# 11. Calculate the bayes factor (CAUTION -- large calculation, parallelized to 40 cores)
# 12. Make the best fit figure for the paper
# 13. Make the bayes factor and AIC figure for the paper
# 14. Double check the bin effect on the best fit (produces a figure)
python3 8_get_observed_orbital_phases.py
python3 9_TABLE_phase_coverage_raw_and_binned.py
python3 10_model_fit_with_elevated_phase_range.py 101 0 # number of bin edges, diagniostics: 1 to produce diagnostic plots, 0 to not produce them
# OR run for a range of bin sizes
bash 10_model_fit_with_elevated_phase_range.sh
# python3 11_bayes_factor.py 101
# OR RUN FOR A RANGE OF BIN SIZES
# bash 11_bayes_factor.sh 
python3 12_FIGURE_best_fit.py
python3 13_FIGURE_bayes_factor_and_aic.py
python3 14_double_check_bin_effect_on_best_fit.py