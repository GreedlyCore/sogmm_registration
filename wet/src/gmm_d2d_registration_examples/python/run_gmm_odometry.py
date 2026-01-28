#!/usr/bin/env python
"""
Run GMM D2D registration on LiDAR datasets (KITTI, VIRAL, etc.)
"""
import os
import sys
import glob
import argparse
import pickle
import time
import json

import numpy as np
import yaml
from tqdm import tqdm

import gmm_d2d_registration_py
from utils.RToZYX import RToZYX
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse
from utils.plot_results import plot_results
from utils.kitti_loader import load_kitti_ground_truth, load_kitti_calib
from utils.viral_loader import load_viral_ground_truth, load_viral_lidar_config


def find_gmm_dir(base_dir, pattern, timestamp=None):
    """
    Find GMM directory, checking for exact match first, then timestamped versions.

    Args:
        base_dir: Base directory to search in
        pattern: Pattern to match (e.g., '100_components' or 'adaptive_bw10_components')
        timestamp: Specific timestamp suffix. If None, uses most recent.

    Returns:
        Path to GMM directory
    """
    if timestamp:
        return os.path.join(base_dir, f'{pattern}_{timestamp}')

    exact_path = os.path.join(base_dir, pattern)
    if os.path.exists(exact_path):
        return exact_path

    candidates = glob.glob(os.path.join(base_dir, f'{pattern}_*'))
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    return exact_path


def parse_meta_file(gmm_folder):
    """
    Parse meta.yaml from GMM folder to extract dataset configuration.

    Returns:
        dict with keys: dataset, sequence (kitti), bag_file (viral), skip_scans (viral)
    """
    meta_path = os.path.join(gmm_folder, 'meta.yaml')
    config = {}

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Infer dataset from parent folder if not in meta.yaml
    if 'dataset' not in config:
        parent_folder = os.path.basename(os.path.dirname(gmm_folder))
        if parent_folder.startswith('kitti_'):
            config['dataset'] = 'KITTI'
            parts = parent_folder.split('_')
            if len(parts) >= 3:
                config['sequence'] = parts[-1]
        elif parent_folder.startswith('viral_'):
            config['dataset'] = 'VIRAL'
            config['bag_file'] = '_'.join(parent_folder.split('_')[1:])

    return config


def compute_relative_transform(pose_i, pose_i1, Tr=None):
    """
    Compute relative transformation between two poses.

    For KITTI: poses are in camera frame, Tr converts velodyne->camera
    For VIRAL: poses are already in body/lidar frame, Tr=None

    Args:
        pose_i: 4x4 pose at time i
        pose_i1: 4x4 pose at time i+1
        Tr: Velodyne to camera calibration (KITTI only)

    Returns:
        4x4 relative transformation from i to i+1
    """
    if Tr is not None:
        # KITTI: transform from camera to velodyne frame
        # T_vel_i_to_vel_i1 = inv(Tr) @ inv(Pose_i1) @ Pose_i @ Tr
        T_rel = pose_compose(
            pose_compose(pose_inverse(Tr), pose_inverse(pose_i1)),
            pose_compose(pose_i, Tr)
        )
    else:
        # VIRAL: poses already in lidar/body frame
        T_rel = pose_compose(pose_inverse(pose_i1), pose_i)

    return T_rel


def run_gmm_odometry(gmm_folder, first_scan, last_scan,
                     kitti_dir=None, viral_dir=None,
                     enable_profiling=False):
    """
    Run GMM D2D registration on a dataset.

    Args:
        gmm_folder: Path to folder containing .gmm files
        first_scan: First scan index
        last_scan: Last scan index
        kitti_dir: KITTI dataset root directory
        viral_dir: VIRAL dataset root directory
        enable_profiling: Save timing data

    Returns:
        transforms: Estimated transformations
        ground_truths: Ground truth transformations
    """
    # Validate inputs
    if not os.path.exists(gmm_folder):
        raise FileNotFoundError(f'GMM folder not found: {gmm_folder}')

    # Parse meta.txt for dataset configuration
    meta = parse_meta_file(gmm_folder)
    dataset = meta.get('dataset', '').lower()

    if not dataset:
        raise ValueError('Cannot determine dataset type. Check meta.txt or folder naming.')

    # Resolve dataset directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dataset_dir = os.path.abspath(os.path.join(script_dir, '../../../../dataset'))

    if dataset == 'kitti':
        kitti_sequence = meta.get('sequence')
        if kitti_sequence is None:
            raise ValueError('Cannot determine KITTI sequence from meta.yaml or folder name')
        if kitti_dir is None:
            kitti_dir = os.path.join(repo_dataset_dir, 'kitti')
    elif dataset == 'viral':
        bag_name = meta.get('bag_file')
        if bag_name is None:
            raise ValueError('Cannot determine bag_file from meta.yaml')
        if viral_dir is None:
            viral_dir = os.path.join(repo_dataset_dir, 'viral')
        viral_bag = os.path.join(viral_dir, bag_name, f'{bag_name}.bag')
        viral_skip_scans = meta.get('skip_scans', 10)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    # Load ground truth
    print(f'Dataset: {dataset.upper()}')
    print(f'GMM folder: {gmm_folder}')

    Tr = None  # Calibration matrix (KITTI only)

    if dataset == 'kitti':
        poses, Tr = load_kitti_ground_truth(
            kitti_sequence, kitti_dir, start_idx=first_scan, end_idx=last_scan
        )
    elif dataset == 'viral':
        config_topic, T_body_lidar = load_viral_lidar_config(viral_bag)
        lidar_topic = config_topic or '/os1_cloud_node1/points'

        poses, _ = load_viral_ground_truth(
            viral_bag, lidar_topic=lidar_topic,
            skip_scans=viral_skip_scans, max_scans=last_scan + 1
        )
        poses = poses[first_scan:last_scan + 1]

    print(f'Scans: {first_scan} to {last_scan}')
    print(f'Total pairs: {last_scan - first_scan}')
    print()

    # Create results directory
    results_dir = os.path.join(os.path.dirname(gmm_folder), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Initialize arrays
    n_pairs = last_scan - first_scan
    transforms = np.zeros((n_pairs, 6))
    ground_truths = np.zeros((n_pairs, 6))
    errors = np.zeros((n_pairs, 6))

    if enable_profiling:
        profiling_data = {
            'dataset': dataset,
            'gmm_folder': os.path.basename(gmm_folder),
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS', 'not set'),
            'timings': [],
            'scan_indices': []
        }
        total_time_start = time.time()

    # Process each pair
    for i in tqdm(range(n_pairs), desc='Processing scans'):
        scan_i = first_scan + i
        scan_i1 = scan_i + 1

        # GMM files: 0-based for VIRAL, 1-based for KITTI
        if dataset == 'kitti':
            source_file = os.path.join(gmm_folder, f'{scan_i + 1}.gmm')
            target_file = os.path.join(gmm_folder, f'{scan_i + 2}.gmm')
        else:
            source_file = os.path.join(gmm_folder, f'{scan_i}.gmm')
            target_file = os.path.join(gmm_folder, f'{scan_i + 1}.gmm')

        if not os.path.exists(source_file) or not os.path.exists(target_file):
            print(f'\nWARNING: Missing GMM files for scan {scan_i}')
            continue

        # Run registration
        if enable_profiling:
            reg_start = time.time()

        Tinit = np.eye(4)
        output = gmm_d2d_registration_py.isoplanar_registration(
            Tinit, source_file, target_file)
        ret = gmm_d2d_registration_py.anisotropic_registration(
            output[0], source_file, target_file)
        Tout = ret[0]

        if enable_profiling:
            profiling_data['timings'].append(time.time() - reg_start)
            profiling_data['scan_indices'].append(scan_i)

        # Extract estimated transformation
        rotation = Tout[0:3, 0:3]
        translation = Tout[0:3, 3]
        dpose = np.concatenate([translation, RToZYX(rotation)])

        # Compute ground truth transformation
        pose_i = poses[i]
        pose_i1 = poses[i + 1]
        T_gt = compute_relative_transform(pose_i, pose_i1, Tr)

        rotation_gt = T_gt[0:3, 0:3]
        translation_gt = T_gt[0:3, 3]
        dground_truth = np.concatenate([translation_gt, RToZYX(rotation_gt)])

        transforms[i, :] = dpose
        ground_truths[i, :] = dground_truth
        errors[i, :] = dpose - dground_truth

    # Save results
    gmm_name = os.path.basename(gmm_folder)
    result_file = os.path.join(results_dir, f'{dataset}_{gmm_name}_results.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump([transforms, ground_truths, errors], f)
    print(f'\nResults saved to: {result_file}')

    # Save plot
    plot_file = os.path.join(results_dir, f'{dataset}_{gmm_name}_trajectory.png')
    plot_results(transforms, ground_truths, first_scan, last_scan, save_path=plot_file)
    print(f'Plot saved to: {plot_file}')

    # Print statistics
    translation_errors = np.linalg.norm(errors[:, :3], axis=1)
    rotation_errors = np.linalg.norm(errors[:, 3:], axis=1)

    translation_rmse = np.sqrt(np.mean(translation_errors**2))
    rotation_rmse = np.sqrt(np.mean(rotation_errors**2))

    print(f'\nTranslation error: {np.mean(translation_errors):.4f} ± '
          f'{np.std(translation_errors):.4f} m (RMSE: {translation_rmse:.4f})')
    print(f'Rotation error: {np.mean(rotation_errors):.4f} ± '
          f'{np.std(rotation_errors):.4f} rad (RMSE: {rotation_rmse:.4f})')

    # Save profiling data
    if enable_profiling:
        total_time = time.time() - total_time_start
        timings = np.array(profiling_data['timings'])

        profiling_data['total_time'] = round(total_time, 3)
        profiling_data['mean_time'] = round(np.mean(timings), 3)
        profiling_data['std_time'] = round(np.std(timings), 3)
        profiling_data['min_time'] = round(np.min(timings), 3)
        profiling_data['max_time'] = round(np.max(timings), 3)
        profiling_data['total_pairs'] = len(timings)
        profiling_data['timings'] = [round(t, 3) for t in timings.tolist()]

        profiling_dir = os.path.join(gmm_folder, 'profiling')
        os.makedirs(profiling_dir, exist_ok=True)
        profile_file = os.path.join(profiling_dir,
                                    f'registration_{first_scan}to{last_scan}.json')
        with open(profile_file, 'w') as f:
            json.dump(profiling_data, f, indent=2)

        print(f'\n=== PROFILING ===')
        print(f'Total time: {total_time:.2f}s')
        print(f'Per pair: {profiling_data["mean_time"]:.3f} ± {profiling_data["std_time"]:.3f}s')
        print(f'Throughput: {len(timings)/total_time:.2f} pairs/s')

    return transforms, ground_truths


def main():
    parser = argparse.ArgumentParser(
        description='Run GMM D2D registration on LiDAR datasets')

    parser.add_argument('--gmm_folder', type=str, required=True,
                        help='Path to folder containing .gmm files')
    parser.add_argument('--first_scan', type=int, default=0,
                        help='First scan index (default: 0)')
    parser.add_argument('--last_scan', type=int, default=100,
                        help='Last scan index (default: 100)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling')
    parser.add_argument('--kitti_dir', type=str, default=None,
                        help='KITTI dataset root directory (auto-detected)')
    parser.add_argument('--viral_dir', type=str, default=None,
                        help='VIRAL dataset root directory (auto-detected)')

    args = parser.parse_args()

    run_gmm_odometry(
        gmm_folder=args.gmm_folder,
        first_scan=args.first_scan,
        last_scan=args.last_scan,
        kitti_dir=args.kitti_dir,
        viral_dir=args.viral_dir,
        enable_profiling=args.profile
    )


if __name__ == '__main__':
    main()
