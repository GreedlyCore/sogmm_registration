#!/usr/bin/env python
"""
Compare GMM (classic EM, wet/src/gmm) vs fsogmm (fixed SOGMM, self_organizing_gmm)
on the same TUM scan. Saves each to a separate folder with meta.yaml.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import yaml
from datetime import datetime

from sklearn.mixture import GaussianMixture
from utils.save_gmm import save

# python create_and_save_gmm_compare.py

# Just compare:
# python compare_gmm_results.py ./100_components_24021707_gmm ./100_components_24021707_fsogmm

# Compare same methods, just two separated runs:
# python compare_gmm_results.py ./100_components_24021707_gmm ./100_components_24021725_gmm
# OR
# python compare_gmm_results.py ./100_components_24021725_fsogmm ./100_components_24021707_fsogmm

def create_and_save_compare():
    cwd = os.getcwd()
    SANDBOX_NAME = 'sogmm_registration'
    matches = cwd.split(SANDBOX_NAME)
    GIRA3D_REGISTRATION_SANDBOX = matches[0] + SANDBOX_NAME
    PCLD_DIR = GIRA3D_REGISTRATION_SANDBOX + '/data/rgbd_dataset_freiburg3_long_office_household/pointclouds/'

    idx = 1260
    n_components = 100

    data = np.loadtxt(PCLD_DIR + str(idx) + '.txt', delimiter=',')
    n_samples = data.shape[0]

    timestamp = datetime.now().strftime("%d%m%H%M")

    # ------------------------------------------------------------------ gmm --
    # Using sklearn GaussianMixture: the C++ SKLearn from wet/src/gmm has a
    # numerical bug in estimate_log_gaussian_prob (writes into resp_T_ as scratch
    # mid-computation, corrupting EM state with float32 precision).
    gmm_dir = f'./{n_components}_components_{timestamp}_gmm'
    os.makedirs(gmm_dir, exist_ok=True)

    g_gmm = GaussianMixture(n_components)
    g_gmm.fit(data)
    save(os.path.join(gmm_dir, f'{idx}.gmm'), g_gmm)

    meta_gmm = {
        'type': 'gmm',
        'dataset': 'TUM',
        'scan_idx': idx,
        'n_components': n_components,
    }
    with open(os.path.join(gmm_dir, 'meta.yaml'), 'w') as f:
        yaml.dump(meta_gmm, f, default_flow_style=False, sort_keys=False)

    print(f'gmm    -> {gmm_dir}')

    # --------------------------------------------------------------- fsogmm --
    from gmm_py import GMMf3CPU
    from kinit_py import KInitf3CPU

    fsogmm_dir = f'./{n_components}_components_{timestamp}_fsogmm'
    os.makedirs(fsogmm_dir, exist_ok=True)

    data32 = data.astype(np.float32)
    kinit = KInitf3CPU()
    _, indices = kinit.resp_calc(data32, n_components)
    resp = np.zeros((n_samples, n_components), dtype=np.float32)
    resp[indices, np.arange(n_components)] = 1

    g_fsogmm = GMMf3CPU(n_components)
    g_fsogmm.fit(data32, resp)
    save(os.path.join(fsogmm_dir, f'{idx}.gmm'), g_fsogmm)

    meta_fsogmm = {
        'type': 'fsogmm',
        'dataset': 'TUM',
        'scan_idx': idx,
        'n_components': n_components,
    }
    with open(os.path.join(fsogmm_dir, 'meta.yaml'), 'w') as f:
        yaml.dump(meta_fsogmm, f, default_flow_style=False, sort_keys=False)

    print(f'fsogmm -> {fsogmm_dir}')


if __name__ == '__main__':
    create_and_save_compare()
