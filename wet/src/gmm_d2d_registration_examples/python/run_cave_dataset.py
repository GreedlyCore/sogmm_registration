#!/usr/bin/env python
import numpy as np
import os

import gmm_d2d_registration_py
from utils.run_dataset import run_dataset
from utils.plot_results import plot_results
from utils.metrics import compute_metrics


# From proposed paper dataset (certified):
# python3 run_cave_dataset.py 
# RMSE  translation : 0.033008 m
# RMSE  rotation    : 0.347886 deg  (0.006072 rad)
# OE    translation : 0.027799 m
# OE    rotation    : 0.277151 deg  (0.004837 rad)

def main():

    FIRST_SCAN = 499
    LAST_SCAN = 890
    PREFIX = 'isoplanarhybrid_'
    DATASET = 'cave'
    NUM_COMPONENTS = 100

    transforms, ground_truths = run_dataset(
        DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS)

    compute_metrics(transforms, ground_truths)

    # Save the plot to results directory
    RESULTS_DIR = "./wet/src/gmm/data"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f'{PREFIX}{NUM_COMPONENTS}_trajectory_plot_{DATASET}.png')

    plot_results(transforms, ground_truths, FIRST_SCAN, LAST_SCAN, save_path=save_path)


if __name__ == '__main__':
    main()
