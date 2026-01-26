#!/usr/bin/env python
import copy
import numpy as np
import os
from os.path import expanduser
import pickle
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

    FIRST_SCAN = 499
    LAST_SCAN = 890
    PREFIX = 'isoplanarhybrid_'
    DATASET = 'cave'
    NUM_COMPONENTS = 100

    transforms, ground_truths = run_dataset(
        DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS)

    # Save the plot to results directory
    RESULTS_DIR = "./wet/src/gmm/data"
    save_path = os.path.join(RESULTS_DIR, f'{PREFIX}{NUM_COMPONENTS}_trajectory_plot_{DATASET}.png')

    plot_results(transforms, ground_truths, FIRST_SCAN, LAST_SCAN, save_path=save_path)


if __name__ == '__main__':
    main()
