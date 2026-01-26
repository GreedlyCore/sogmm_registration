#!/usr/bin/env python
import pickle
import os
from utils.plot_results import plot_results

# Configuration
DATASET = 'cave'
FIRST_SCAN = 499
LAST_SCAN = 890
PREFIX = 'isoplanarhybrid_'
NUM_COMPONENTS = 100

# Load results
DATA_DIR = './data/' + DATASET + '/results/'
with open(DATA_DIR + PREFIX + str(NUM_COMPONENTS) + '_results.pkl', 'rb') as handle:
    transforms, ground_truths, errors = pickle.load(handle)

print(f'Loaded results for {DATASET} dataset')
print(f'Transforms shape: {transforms.shape}')
print(f'Ground truths shape: {ground_truths.shape}')

# Save the plot
RESULTS_DIR = "./wet/src/gmm/data"
save_path = os.path.join(RESULTS_DIR, f'{PREFIX}{NUM_COMPONENTS}_trajectory_plot_{DATASET}.png')

plot_results(transforms, ground_truths, FIRST_SCAN, LAST_SCAN, save_path=save_path)
