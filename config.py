'''
config.py
Autor: Bryson Sanders
Creation Date: 06/21/2025
Last modified: 06/21/2025
Purpose: Contains standardized values or lists
'''
#contains the collumn names of the dataset.csv
features = [
    "mean", "var", "std", "len_weighted", "gaps_squared", "n_peaks",
    "smooth10_n_peaks", "smooth20_n_peaks", "var_div_duration", "var_div_len",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "kurtosis", "skew",
] #removed len and duration as recomended by authors