#!/usr/bin/env python
import copy
import numpy as np
import os
from os.path import expanduser

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have Qt installed
from tqdm import tqdm
import gmm_d2d_registration_py
from utils.RToZYX import RToZYX
from utils.ZYXToR import ZYXToR
from utils.QuatToR import QuatToR
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse
from utils.run_dataset import run_dataset
from utils.plot_results import plot_results


def main():

    FIRST_SCAN = 0
    LAST_SCAN = 320
    PREFIX = 'isoplanarhybrid_'
    DATASET = 'mine_001_part3'
    NUM_COMPONENTS = 100

    transforms, ground_truths = run_dataset(
        DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS)

    # Save the plot to results directory
    cwd = os.getcwd()
    # SANDBOX_NAME = 'gira3d-registration'
    # matches = cwd.split(SANDBOX_NAME)
    # GIRA3D_REGISTRATION_SANDBOX = matches[0] + SANDBOX_NAME
    # RESULTS_DIR = GIRA3D_REGISTRATION_SANDBOX + '/data/' + DATASET + '/results/'
    
    RESULTS_DIR = "./wet/src/gmm/data"
    save_path = os.path.join(RESULTS_DIR, f'{PREFIX}{NUM_COMPONENTS}_trajectory_plot.png')

    plot_results(transforms, ground_truths, FIRST_SCAN, LAST_SCAN, save_path=save_path)


if __name__ == '__main__':
    main()
