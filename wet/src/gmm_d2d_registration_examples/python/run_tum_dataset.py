#!/usr/bin/env python
import argparse
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
from utils.run_dataset import run_dataset
from utils.plot_results import plot_results
from utils.gmm_folder import detect_gmm_range
from utils.metrics import compute_metrics


# python create_and_save_gmm_tum.py --bandwidth 0.05 --every_n_filter --every_n 5 --voxel_filter --voxel_size 0.04

# python create_and_save_gmm_tum.py --n_components 200 --every_n_filter --every_n 5 --voxel_filter --voxel_size 0.04
# --> around 7k points per 7s estimation required

# python run_tum_dataset.py --gmm_dir /home/sonieth2/thesis/gira3d-registration/wet/src/gmm_d2d_registration_examples/python/runs/tum_rgbd_dataset_freiburg3_long_office_household/adaptive_bw1_components_17021334
# python run_tum_dataset.py --gmm_dir /home/sonieth2/thesis/gira3d-registration/wet/src/gmm_d2d_registration_examples/python/runs/tum_rgbd_dataset_freiburg3_long_office_household/100_components_17021309

# official confirmed
# python3 run_tum_dataset.py --gmm_dir /home/sonieth2/thesis/gira3d-registration/data/rgbd_dataset_freiburg3_long_office_household/100_components
# RMSE  translation : 0.003313 m
# RMSE  rotation    : 0.361263 deg  (0.006305 rad)
# OE    translation : 0.002763 m
# OE    rotation    : 0.314164 deg  (0.005483 rad)


# bw = 0.04
# Detected scan range: 0 to 284 (step=1)
# GMM dir: /home/sonieth2/thesis/gira3d-registration/wet/src/gmm_d2d_registration_examples/python/runs/tum_rgbd_dataset_freiburg3_long_office_household/bw4_components_25021452
# Pairs: 284 (scans 0 to 284)
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 284/284 [06:10<00:00,  1.31s/it]
# RMSE  translation : 0.045729 m
# RMSE  rotation    : 0.955009 deg  (0.016668 rad)
# OE    translation : 0.010835 m
# OE    rotation    : 0.355172 deg  (0.006199 rad)

# Components: min=279, max=453, avg=358.5

# bw = 0.03
# Detected scan range: 0 to 103 (step=1)
# GMM dir: /home/sonieth2/thesis/gira3d-registration/wet/src/gmm_d2d_registration_examples/python/runs/tum_rgbd_dataset_freiburg3_long_office_household/bw3_components_25021631
# Pairs: 103 (scans 0 to 103)
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [04:50<00:00,  2.82s/it]
# RMSE  translation : 0.074818 m
# RMSE  rotation    : 1.485714 deg  (0.025931 rad)
# OE    translation : 0.014420 m
# OE    rotation    : 0.390609 deg  (0.006817 rad)

# Components: min=472, max=714, avg=566.5


# bw = 0.02
# Detected scan range: 0 to 102 (step=1)
# GMM dir: /home/sonieth2/thesis/gira3d-registration/wet/src/gmm_d2d_registration_examples/python/runs/tum_rgbd_dataset_freiburg3_long_office_household/bw2_components_25021707
# Pairs: 102 (scans 0 to 102)
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 102/102 [18:42<00:00, 11.01s/it]
# RMSE  translation : 0.075293 m
# RMSE  rotation    : 1.496459 deg  (0.026118 rad)
# OE    translation : 0.015291 m
# OE    rotation    : 0.412750 deg  (0.007204 rad)

# Components: min=894, max=1313, avg=1040.5

# bw = 0.01
# RMSE translation : 0.073217 m
# RMSE rotation:1.458541 deg (0.025456 rad)
# OE translation: 0.015027 m
# OE rotation: 0.410095 deg (0.007157 rad)


# bw = 0.05
# Pairs: 100 (scans 0 to 100)
#  40%|████████████████████████████████████████████████▊                                                                         | 40/100 [00:17<00:24,  2.41it/s]100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:50<00:00,  1.97it/s]
# RMSE  translation : 0.075887 m
# RMSE  rotation    : 1.502656 deg  (0.026226 rad)
# OE    translation : 0.014380 m
# OE    rotation    : 0.374038 deg  (0.006528 rad)

# Components: min=184, max=307, avg=239.1



# TODO: 5 4 3 2 1 создаём
# python3 create_and_save_gmm_tum.py --bw 0.03 --every-n 5 --voxel 0.05
# 100 scans for each one

def main():
    parser = argparse.ArgumentParser(description='Run TUM dataset registration evaluation')

    parser.add_argument('--dataset', type=str,
                        default='rgbd_dataset_freiburg3_long_office_household',
                        help='TUM dataset folder name')
    parser.add_argument('--gmm_dir', type=str, required=True,
                        help='Path to GMM directory')
    parser.add_argument('--n_components', type=int, default=100,
                        help='Number of components (default: 100)')
    parser.add_argument('--prefix', type=str, default='isoplanarhybrid_',
                        help='Results file prefix (default: isoplanarhybrid_)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting results')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx',   type=int, default=None)

    args = parser.parse_args()

    detected_first, detected_last, detected_skip = detect_gmm_range(args.gmm_dir)
    if detected_first is None:
        raise ValueError(f'No .gmm files found in {args.gmm_dir}')
    first_scan = args.start_idx if args.start_idx is not None else detected_first
    last_scan  = args.end_idx   if args.end_idx   is not None else detected_last
    print(f'Detected scan range: {first_scan} to {last_scan} (step={detected_skip})')

    # Ensure gmm_dir ends with /
    gmm_dir = args.gmm_dir
    if not gmm_dir.endswith('/'):
        gmm_dir = gmm_dir + '/'

    # run_dataset expects FIRST_SCAN/LAST_SCAN as pair range:
    # it iterates i in [FIRST_SCAN, LAST_SCAN) and reads files i+1, i+2
    # so FIRST_SCAN = first_gmm - 1, LAST_SCAN = last_gmm - 1
    pair_first = first_scan - 1
    pair_last = last_scan - 1

    print(f'GMM dir: {args.gmm_dir}')
    print(f'Pairs: {pair_last - pair_first} (scans {first_scan} to {last_scan})')

    transforms, ground_truths = run_dataset(
        args.dataset, pair_first, pair_last,
        args.prefix, args.n_components, GMM_DIR=gmm_dir)

    compute_metrics(transforms, ground_truths)

    # Print component count stats from GMM files
    comp_counts = []
    for f in sorted(os.listdir(gmm_dir)):
        if f.endswith('.gmm'):
            n_lines = sum(1 for _ in open(os.path.join(gmm_dir, f)))
            comp_counts.append(n_lines)
    if comp_counts and min(comp_counts) != max(comp_counts):
        print(f'\nComponents: min={min(comp_counts)}, max={max(comp_counts)}, avg={np.mean(comp_counts):.1f}')

    if not args.no_plot:
        plot_results(transforms, ground_truths, pair_first, pair_last)


if __name__ == '__main__':
    main()
