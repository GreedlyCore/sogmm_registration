#!/usr/bin/env python
"""
Run odometry estimation on MAI City dataset GMMs and evaluate against ground truth.
GT is loaded from KITTI-format pose file (bin/poses/00.txt).
No bag file needed at runtime — GT is already in velodyne frame.

Usage:
    python run_mai_dataset.py \
        --gmm_dir runs/mai_00/150_components_XXXXXXXX \
        --poses_file ~/thesis/mai_city/bin/poses/00.txt
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('qtagg')
import numpy as np
from tqdm import tqdm

import gmm_d2d_registration_py
from utils.RToZYX import RToZYX
from utils.ZYXToR import ZYXToR
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse
from utils.plot_results import plot_results
from utils.kitti_loader import load_kitti_poses
from utils.metrics import compute_metrics


# python run_mai_dataset.py --gmm_dir runs/mai_00/150_components_19021854

DEFAULT_POSES_FILE = os.path.expanduser('~/thesis/mai_city/bin/poses/00.txt')
MAI_LIDAR_HZ = 10.0
MAI_START_TIME = 10800.0  # 03:00:00 #TODO: who said that???


def run_mai_dataset(gmm_dir, poses_file=DEFAULT_POSES_FILE):
    # Load meta to get skip_scans
    meta_path = os.path.join(gmm_dir, 'meta.yaml')
    skip_scans = 1
    if os.path.exists(meta_path):
        import yaml
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        skip_scans = meta.get('skip_scans', 1)
        print(f'Meta: skip_scans={skip_scans}, n_components={meta.get("n_components")}')

    # Sorted GMM files by scan index (filenames encode bag scan index)
    gmm_files = sorted(
        [p for p in Path(gmm_dir).glob('*.gmm')],
        key=lambda p: int(p.stem)
    )
    n_scans = len(gmm_files)
    print(f'Found {n_scans} GMM files in {gmm_dir}')

    if n_scans < 2:
        raise ValueError('Need at least 2 GMM files to run odometry')

    first_scan = int(gmm_files[0].stem)
    last_scan = int(gmm_files[-1].stem)

    # Load KITTI GT poses and slice to GMM range
    all_poses = load_kitti_poses(poses_file)
    gt_poses = all_poses[first_scan: last_scan + 1: skip_scans]
    print(f'Using GT poses [{first_scan}:{last_scan + 1}:{skip_scans}] → {len(gt_poses)} poses')

    if len(gt_poses) < n_scans:
        print(f'WARNING: only {len(gt_poses)} GT poses for {n_scans} GMMs — truncating')
        n_scans = len(gt_poses)
        gmm_files = gmm_files[:n_scans]

    # Synthetic timestamps at MAI_LIDAR_HZ (KITTI format has no timestamps)
    timestamps = [MAI_START_TIME + first_scan / MAI_LIDAR_HZ + i / MAI_LIDAR_HZ
                  for i in range(n_scans)]

    # BIN/GT frame: x-forward, y-left, z-up  (KITTI convention)
    # BAG/GMM frame: z-forward, x-left, y-up  (ROS velodyne topic convention)
    # Cyclic permutation: BAG = R * BIN,  BIN = R.T * BAG
    R_bin_to_bag = np.array([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]], dtype=np.float64)

    n_pairs = n_scans - 1
    transforms = np.zeros((n_pairs, 6))
    ground_truths = np.zeros((n_pairs, 6))

    # Seed with expected per-frame motion (1 m forward in BAG z-axis)
    # so the optimizer starts inside the convergence basin; warm-start takes over after frame 0
    Tinit = np.eye(4)
    Tinit[2, 3] = 1.0
    for i in tqdm(range(n_pairs)):
        target_file = str(gmm_files[i])
        source_file = str(gmm_files[i + 1])

        output = gmm_d2d_registration_py.isoplanar_registration(Tinit, source_file, target_file)
        ret = gmm_d2d_registration_py.anisotropic_registration(output[0], source_file, target_file)
        Tout = ret[0]

        translation = Tout[0:3, 3]
        rotation = RToZYX(Tout[0:3, 0:3])
        transforms[i] = np.concatenate([translation, rotation])
        # warm-start: only propagate if output is non-trivial, else keep 1m seed
        if np.linalg.norm(Tout[0:3, 3]) > 0.1:
            Tinit = Tout
        else:
            Tinit = np.eye(4)
            Tinit[2, 3] = 1.0

        # GT is in BIN frame → convert to BAG frame for fair comparison
        Tc1c2_bin = pose_compose(pose_inverse(gt_poses[i]), gt_poses[i + 1])
        Tc1c2 = np.eye(4)
        Tc1c2[0:3, 0:3] = R_bin_to_bag @ Tc1c2_bin[0:3, 0:3] @ R_bin_to_bag.T
        Tc1c2[0:3, 3]   = R_bin_to_bag @ Tc1c2_bin[0:3, 3]
        dgt = np.concatenate([Tc1c2[0:3, 3], RToZYX(Tc1c2[0:3, 0:3])])
        ground_truths[i] = dgt

    return transforms, ground_truths, gt_poses, timestamps


def main():
    parser = argparse.ArgumentParser(description='Run MAI City odometry evaluation')
    parser.add_argument('--gmm_dir', type=str, required=True, help='Path to GMM folder')
    parser.add_argument('--poses_file', type=str, default=DEFAULT_POSES_FILE,
                        help=f'Path to KITTI poses .txt file (default: {DEFAULT_POSES_FILE})')
    args = parser.parse_args()

    args.gmm_dir = os.path.expanduser(args.gmm_dir)
    args.poses_file = os.path.expanduser(args.poses_file)

    transforms, ground_truths, gt_poses, timestamps = run_mai_dataset(
        args.gmm_dir, args.poses_file)

    compute_metrics(transforms, ground_truths)

    gmm_label = Path(args.gmm_dir).name
    seq_name = Path(args.poses_file).stem
    results_dir = os.path.join(os.path.dirname(args.gmm_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save pkl
    results_pkl = os.path.join(results_dir, f'mai_{seq_name}_{gmm_label}_results.pkl')
    with open(results_pkl, 'wb') as f:
        pickle.dump([transforms, ground_truths, transforms - ground_truths], f)
    print(f'Results saved to: {results_pkl}')

    # Build absolute trajectories in BAG frame
    R_bin_to_bag = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
    gt_poses_bag = []
    for T in gt_poses:
        Tb = np.eye(4)
        Tb[0:3, 0:3] = R_bin_to_bag @ T[0:3, 0:3] @ R_bin_to_bag.T
        Tb[0:3, 3]   = R_bin_to_bag @ T[0:3, 3]
        gt_poses_bag.append(Tb)

    odom_poses = [gt_poses_bag[0]]
    for i in range(len(transforms)):
        T = np.eye(4)
        T[:3, :3] = ZYXToR(transforms[i, 3:6])
        T[:3, 3] = transforms[i, :3]
        odom_poses.append(pose_compose(odom_poses[-1], T))


    save_path = os.path.join(results_dir, f'mai_{seq_name}_{gmm_label}_trajectory.png')
    plot_results(transforms, ground_truths, 0, len(transforms), save_path=save_path)


if __name__ == '__main__':
    main()
