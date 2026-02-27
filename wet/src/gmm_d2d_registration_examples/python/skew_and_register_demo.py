#!/usr/bin/env python
"""
Demo: apply a known mock rigid transform to one GMM component set,
save it as a new folder with '_skewed' suffix, then run registration
to recover the transform and compare with ground truth.

Usage:
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx 0.3 --ty 0.1 --tz 0.05 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx 0.8 --ty 0.2 --tz 0.5 --roll 0.05 --pitch 0.02 --yaw 0.1
    
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx 0.2 --ty 0.2 --tz 0.2 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx 0.3 --ty 0.3 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
    
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx 0.5 --ty 0.5 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx 0.5 --ty 0.5 --tz 0.5 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx 0.5 --ty 0.5 --tz -0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx -0.5 --ty -0.5 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_fsogmm --tx -0.5 --ty -0.5 --tz -0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
    ---
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.5 --ty 0.5 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.5 --ty 0.5 --tz 0.5 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.5 --ty 0.5 --tz -0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx -0.5 --ty -0.5 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
    python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx -0.5 --ty -0.5 --tz -0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
"""

import argparse
import os
import shutil
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.open3d_visualizer import load_gmm_file
from utils.ZYXToR import ZYXToR

import gmm_d2d_registration_py


def apply_transform(gmm, R, t):
    """Apply rigid transform (R, t) to all GMM components."""
    new_means = (R @ gmm['means'].T).T + t          # (K, 3)
    new_covs  = R @ gmm['covariances'] @ R.T        # broadcast: (K,3,3) @ (3,3)
    return new_means, new_covs, gmm['weights']


def save_gmm_csv(filepath, means, covariances, weights):
    n = len(weights)
    data = np.zeros((n, 13))
    data[:, 0:3]  = means
    data[:, 3:12] = covariances.reshape(n, 9)
    data[:, 12]   = weights
    np.savetxt(filepath, data, fmt='%.15f', delimiter=',')


def build_T(tx, ty, tz, roll, pitch, yaw):
    """Build 4x4 from ZYX Euler angles (roll=Rx, pitch=Ry, yaw=Rz)."""
    T = np.eye(4)
    T[0:3, 0:3] = ZYXToR(np.array([roll, pitch, yaw]))
    T[0:3, 3]   = [tx, ty, tz]
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gmm_dir', required=True,
                        help='Source GMM folder (contains *.gmm + meta.yaml)')
    parser.add_argument('--tx',    type=float, default=0.5)
    parser.add_argument('--ty',    type=float, default=0.2)
    parser.add_argument('--tz',    type=float, default=0.1)
    parser.add_argument('--roll',  type=float, default=0.05,  help='rad')
    parser.add_argument('--pitch', type=float, default=0.02,  help='rad')
    parser.add_argument('--yaw',   type=float, default=0.1,   help='rad')
    args = parser.parse_args()

    src_dir = args.gmm_dir.rstrip('/')
    dst_dir = src_dir + '_skewed'

    # ------------------------------------------------------------------ #
    # 1. Find .gmm files
    # ------------------------------------------------------------------ #
    gmm_files = sorted(
        f for f in os.listdir(src_dir) if f.endswith('.gmm')
    )
    if not gmm_files:
        print(f'No .gmm files in {src_dir}')
        sys.exit(1)

    # Pick the first one as the component we'll skew
    gmm_filename = gmm_files[0]
    src_gmm_path = os.path.join(src_dir, gmm_filename)
    print(f'Source GMM: {src_gmm_path}')

    # ------------------------------------------------------------------ #
    # 2. Build mock transform
    # ------------------------------------------------------------------ #
    T_gt = build_T(args.tx, args.ty, args.tz, args.roll, args.pitch, args.yaw)
    R_gt = T_gt[0:3, 0:3]
    t_gt = T_gt[0:3, 3]
    print(f'\n--- Ground-truth transform ---')
    print(f'  translation : {t_gt}')
    print(f'  rotation (R):\n{R_gt}')

    # ------------------------------------------------------------------ #
    # 3. Load, transform, save into _skewed folder
    # ------------------------------------------------------------------ #
    os.makedirs(dst_dir, exist_ok=True)

    # Copy meta.yaml if present
    meta_src = os.path.join(src_dir, 'meta.yaml')
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(dst_dir, 'meta.yaml'))

    gmm = load_gmm_file(src_gmm_path)
    new_means, new_covs, new_weights = apply_transform(gmm, R_gt, t_gt)

    dst_gmm_path = os.path.join(dst_dir, gmm_filename)
    save_gmm_csv(dst_gmm_path, new_means, new_covs, new_weights)
    print(f'\nSkewed GMM saved to: {dst_gmm_path}')

    # ------------------------------------------------------------------ #
    # 4. Run registration:  source = skewed,  target = original
    # ------------------------------------------------------------------ #
    print('\n--- Running registration ---')
    Tinit = np.eye(4)
    # isoplanar pass
    t0 = time.perf_counter()
    out = gmm_d2d_registration_py.isoplanar_registration(
        Tinit, dst_gmm_path, src_gmm_path)
    t_iso = time.perf_counter() - t0

    # anisotropic refinement
    t1 = time.perf_counter()
    ret  = gmm_d2d_registration_py.anisotropic_registration(
        out[0], dst_gmm_path, src_gmm_path)
    t_aniso = time.perf_counter() - t1

    T_est = ret[0]
    R_est = T_est[0:3, 0:3]
    t_est = T_est[0:3, 3]

    print(f'\n--- Timing ---')
    print(f'  isoplanar   : {t_iso*1e3:.1f} ms')
    print(f'  anisotropic : {t_aniso*1e3:.1f} ms')
    print(f'  total       : {(t_iso+t_aniso)*1e3:.1f} ms')

    print(f'\n--- Estimated transform ---')
    print(f'  translation : {t_est}')
    print(f'  rotation (R):\n{R_est}')

    # ------------------------------------------------------------------ #
    # 5. Error
    # registration returns skewed→original = T_gt^{-1}, so compare against that
    # ------------------------------------------------------------------ #
    T_gt_inv  = np.linalg.inv(T_gt)
    R_gt_inv  = T_gt_inv[0:3, 0:3]
    t_gt_inv  = T_gt_inv[0:3, 3]

    t_err = np.linalg.norm(t_gt_inv - t_est)
    R_err = np.arccos(np.clip((np.trace(R_gt_inv.T @ R_est) - 1) / 2, -1, 1))
    print(f'\n--- Expected (T_gt^-1) ---')
    print(f'  translation : {t_gt_inv}')
    print(f'\n--- Error (vs T_gt^-1) ---')
    print(f'  |t_expected - t_est|  = {t_err:.6f} m')
    print(f'  rotation error        = {np.degrees(R_err):.4f} deg')


if __name__ == '__main__':
    main()
