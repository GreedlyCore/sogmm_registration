#!/usr/bin/env python
import argparse
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('qtagg')
from utils.run_dataset import run_dataset
from utils.plot_results import plot_results
from utils.metrics import compute_metrics

PREFIX = 'isoplanarhybrid_'
DATASET = 'mine_001_part3'

# For full sequence:
# RMSE  translation : 0.342150 m
# RMSE  rotation    : 29.237599 deg  (0.510292 rad)
# OE    translation : 0.289102 m
# OE    rotation    : 13.076585 deg  (0.228229 rad)

# From proposed paper: FIRST_SCAN = 0 && LAST_SCAN = 320
# python3 run_mine_dataset.py --gmm_dir ~/thesis/sogmm_registration/data/mine_001_part3/100_components

# return a first/last scan choice 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gmm_dir', default=None,
                        help='Path to GMM folder (e.g. runs/mine_001_part3/100_components_24020218)')
    parser.add_argument('--load', action='store_true',
                        help='Skip odometry — reload cached results from {gmm_dir}/odom/ and replot')
    args = parser.parse_args()

    NUM_COMPONENTS = 100

    # resolve paths
    if args.gmm_dir is not None:
        odom_dir = os.path.join(args.gmm_dir, 'odom')
    else:
        odom_dir = f'./runs/{DATASET}/odom'

    cache_path = os.path.join(odom_dir, 'results.pkl')
    plot_path  = os.path.join(odom_dir, 'trajectory_plot.png')

    # ── load cache ────────────────────────────────────────────────────────────
    if args.load:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f'No cache found at {cache_path}')
        with open(cache_path, 'rb') as f:
            transforms, ground_truths, FIRST_SCAN, LAST_SCAN = pickle.load(f)
        print(f'Loaded cache: {cache_path}  ({len(transforms)} pairs)')
        compute_metrics(transforms, ground_truths, label=DATASET)
        plot_results(transforms, ground_truths, FIRST_SCAN, LAST_SCAN, save_path=plot_path)
        return

    # ── run odometry ──────────────────────────────────────────────────────────
    if args.gmm_dir is not None:
        gmm_files = sorted(Path(args.gmm_dir).glob('*.gmm'), key=lambda p: int(p.stem))
        if len(gmm_files) < 2:
            raise ValueError(f'Need at least 2 GMM files in {args.gmm_dir}')
        
        # FIRST_SCAN = int(gmm_files[0].stem) - 1
        # LAST_SCAN  = int(gmm_files[-1].stem) - 1
        
        FIRST_SCAN = 0
        LAST_SCAN = 320
        
        meta_path = os.path.join(args.gmm_dir, 'meta.yaml')
        if os.path.exists(meta_path):
            import yaml
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            NUM_COMPONENTS = meta.get('n_components', NUM_COMPONENTS)

        print(f'Auto-detected: {len(gmm_files)} GMM files, '
              f'FIRST_SCAN={FIRST_SCAN}, LAST_SCAN={LAST_SCAN}, K={NUM_COMPONENTS}')
    else:
        FIRST_SCAN = 0
        LAST_SCAN  = 320

    transforms, ground_truths = run_dataset(
        DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS, GMM_DIR=args.gmm_dir)

    os.makedirs(odom_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((transforms, ground_truths, FIRST_SCAN, LAST_SCAN), f)
    print(f'Cache saved → {cache_path}')

    compute_metrics(transforms, ground_truths)
    plot_results(transforms, ground_truths, FIRST_SCAN, LAST_SCAN, save_path=plot_path)


if __name__ == '__main__':
    main()
