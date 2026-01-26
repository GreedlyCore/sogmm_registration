import sys
import numpy as np
from pathlib import Path


# BASE FILE PATH:
# ./wet/src/gmm_d2d_registration_examples/python/print_profiling.py

# Usage examples (profiling folder is automatically appended):
# python print_profiling.py kitti_sequence_04/200_components
# python print_profiling.py kitti_sequence_00/100_components
# python print_profiling.py kitti_sequence_00/adaptive_bw10_components


folder = Path(sys.argv[1])

# Automatically add 'profiling' subdirectory if not already present
if folder.name != 'profiling':
    folder = folder / 'profiling'

is_adaptive = 'adaptive_' in str(folder)

if is_adaptive:
    gmm_files = list(folder.glob('gmm_stats_*.csv'))
    ms_files = list(folder.glob('ms_stats_*.csv'))

    gmm_mstep_means = []
    gmm_estep_means = []
    gmm_fit_means = []
    ms_fit_means = []

    for csv_file in gmm_files:
        data = np.genfromtxt(csv_file, delimiter=',', names=True, dtype=None, encoding='utf-8')
        # Handle single-row CSVs (convert 0-d array to 1-d array)
        data = np.atleast_1d(data)
        for row in data:
            if row['Key'] == 'mStep':
                gmm_mstep_means.append(row['Mean'])
            elif row['Key'] == 'eStep':
                gmm_estep_means.append(row['Mean'])
            elif row['Key'] == 'fit':
                gmm_fit_means.append(row['Mean'])

    for csv_file in ms_files:
        data = np.genfromtxt(csv_file, delimiter=',', names=True, dtype=None, encoding='utf-8')
        # Handle single-row CSVs (convert 0-d array to 1-d array)
        data = np.atleast_1d(data)
        for row in data:
            if row['Key'] == 'fit':
                ms_fit_means.append(row['Mean'])

    print(" GMM (EM) Profiling ")
    print(f"mStep: {np.mean(gmm_mstep_means)*1000:.3f} (ms)")
    print(f"eStep: {np.mean(gmm_estep_means)*1000:.3f} (ms)")
    print(f"fit: {np.mean(gmm_fit_means)*1000:.3f} (ms)")
    print(" MeanShift Profiling ")
    print(f"fit: {np.mean(ms_fit_means)*1000:.3f} (ms)")
    print(f"Total: {(np.mean(gmm_fit_means) + np.mean(ms_fit_means))*1000:.3f} (ms)")
else:
    csv_files = list(folder.glob('gmm_stats_*.csv'))

    mstep_means = []
    estep_means = []
    fit_means = []

    for csv_file in csv_files:
        data = np.genfromtxt(csv_file, delimiter=',', names=True, dtype=None, encoding='utf-8')
        # Handle single-row CSVs (convert 0-d array to 1-d array)
        data = np.atleast_1d(data)
        for row in data:
            if row['Key'] == 'mStep':
                mstep_means.append(row['Mean'])
            elif row['Key'] == 'eStep':
                estep_means.append(row['Mean'])
            elif row['Key'] == 'fit':
                fit_means.append(row['Mean'])

    print(" GMM Profiling ")
    print(f"mStep: {np.mean(mstep_means)*1000:.3f} (ms)")
    print(f"eStep: {np.mean(estep_means)*1000:.3f} (ms)")
    print(f"fit: {np.mean(fit_means)*1000:.3f} (ms)")
