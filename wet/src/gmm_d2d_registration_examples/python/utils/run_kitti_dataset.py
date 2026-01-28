#!/usr/bin/env python
"""
Run GMM D2D registration on KITTI odometry dataset.
"""
import os
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import time
import json

import gmm_d2d_registration_py
from utils.RToZYX import RToZYX
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse
from utils.plot_results import plot_results
from utils.kitti_loader import load_kitti_poses, load_kitti_calib


def run_kitti_dataset(sequence, first_scan, last_scan, num_components,
                     kitti_dir='./dataset/kitti',
                     enable_profiling=False, bandwidth=None, timestamp=None):
    """
    Run GMM D2D registration on KITTI sequence.

    Args:
        sequence: KITTI sequence number (e.g., '04')
        first_scan: First scan index
        last_scan: Last scan index
        num_components: Number of GMM components (-1 for adaptive)
        kitti_dir: Root directory of KITTI dataset
        enable_profiling: If True, save timings
        bandwidth: Bandwidth value for adaptive mode (e.g., 0.1, 0.8, 1.0)
                   If provided, uses adaptive_bw{bandwidth*100}_components folder
        timestamp: Specific timestamp suffix for GMM folder (e.g., '21012004').
                   If not provided, uses the most recent timestamped folder.

    Returns:
        transforms: Estimated transformations
        ground_truths: Ground truth transformations
    """
    # Expand ~ to home directory
    kitti_dir = os.path.expanduser(kitti_dir)

    # Construct paths for the downloaded KITTI dataset
    velodyne_dir = os.path.join(kitti_dir, 'data_odometry_velodyne',
                                'dataset', 'sequences', sequence, 'velodyne')
    pose_file = os.path.join(kitti_dir, 'data_odometry_poses', f'{sequence}.txt')
    calib_file = os.path.join(kitti_dir, 'data_odometry_calib',
                             'dataset', 'sequences', sequence, 'calib.txt')

    # GMM files are stored in kitti_sequence_{ID} directory (created by create_and_save_gmm_kitti.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sequence_gmm_dir = os.path.join(script_dir, 'runs', f'kitti_sequence_{sequence}')

    def find_gmm_dir(base_dir, pattern, timestamp=None):
        """Find GMM directory, checking for exact match first, then timestamped versions.

        Args:
            base_dir: Base directory to search in
            pattern: Pattern to match (e.g., '100_components' or 'adaptive_bw10_components')
            timestamp: Specific timestamp suffix (e.g., '21012004'). If provided, looks for
                      pattern_timestamp. If not provided, uses the most recent timestamped folder.
        """
        # If timestamp is provided, look for that specific folder
        if timestamp:
            timestamped_path = os.path.join(base_dir, f'{pattern}_{timestamp}')
            if os.path.exists(timestamped_path):
                return timestamped_path
            # Fall through to error message if not found
            return timestamped_path

        # Check for exact match first (no timestamp)
        exact_path = os.path.join(base_dir, pattern)
        if os.path.exists(exact_path):
            return exact_path

        # Look for timestamped versions (e.g., pattern_21012004)
        import glob
        candidates = glob.glob(os.path.join(base_dir, f'{pattern}_*'))
        if candidates:
            # Sort by modification time (newest first) and return the latest
            candidates.sort(key=os.path.getmtime, reverse=True)
            return candidates[0]

        return exact_path  # Return expected path for error message

    # Handle adaptive GMM mode (negative num_components means adaptive)
    is_adaptive = (num_components < 0) or (bandwidth is not None)
    if is_adaptive:
        # If bandwidth is provided, use bandwidth-specific adaptive folder
        if bandwidth is not None:
            # Convert bandwidth to folder name format (e.g., 0.1 -> bw10, 0.8 -> bw8, 1.0 -> bw10)
            bw_value = int(bandwidth * 100)
            pattern = f'adaptive_bw{bw_value}_components'
            component_label = f'adaptive_bw{bw_value}'
        else:
            # Default adaptive folder (backward compatibility)
            pattern = 'adaptive_components'
            component_label = 'adaptive'
        gmm_dir = find_gmm_dir(sequence_gmm_dir, pattern, timestamp)
    else:
        pattern = f'{num_components}_components'
        component_label = str(num_components)
        gmm_dir = find_gmm_dir(sequence_gmm_dir, pattern, timestamp)

    results_dir = os.path.join(sequence_gmm_dir, 'results')

    # Check if paths exist
    if not os.path.exists(velodyne_dir):
        print(f'ERROR: Velodyne directory not found: {velodyne_dir}')
        print('\nMake sure you have downloaded the KITTI velodyne data.')
        raise FileNotFoundError(f'Velodyne directory not found: {velodyne_dir}')

    if not os.path.exists(calib_file):
        print(f'ERROR: Calibration file not found: {calib_file}')
        print('\nMake sure you have downloaded the KITTI calibration data.')
        raise FileNotFoundError(f'Calibration file not found: {calib_file}')

    if not os.path.exists(pose_file):
        print(f'ERROR: Pose file not found: {pose_file}')
        print('\nMake sure you have downloaded the KITTI poses data.')
        raise FileNotFoundError(f'Pose file not found: {pose_file}')

    if not os.path.exists(gmm_dir):
        print(f'ERROR: GMM directory not found: {gmm_dir}')
        print('\nYou need to run create_and_save_gmm_kitti.py first to convert')
        print('the velodyne scans to GMM format.')
        raise FileNotFoundError(f'GMM directory not found: {gmm_dir}')

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Load calibration and poses
    Tr = load_kitti_calib(calib_file)  # Velodyne to camera
    poses_cam = load_kitti_poses(pose_file)  # Camera to world poses

    print(f'KITTI Sequence: {sequence}')
    print(f'Scans: {first_scan} to {last_scan}')
    if is_adaptive:
        if bandwidth is not None:
            print(f'GMM components: adaptive (bandwidth={bandwidth})')
        else:
            print(f'GMM components: adaptive')
    else:
        print(f'GMM components: {num_components}')
    print(f'GMM directory: {os.path.relpath(gmm_dir)}')
    print(f'Total pairs: {last_scan - first_scan}')
    print()

    # Initialize arrays
    n_pairs = last_scan - first_scan
    transforms = np.zeros((n_pairs, 6))
    ground_truths = np.zeros((n_pairs, 6))
    errors = np.zeros((n_pairs, 6))

    # Profiling data
    if enable_profiling:
        profiling_data = {
            'sequence': sequence,
            'num_components': component_label,
            'is_adaptive': is_adaptive,
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS', 'not set'),
            'timings': [],
            'scan_indices': []
        }
        total_time_start = time.time()

    # Process each pair
    for i in tqdm(range(first_scan, last_scan), desc='Processing scans'):
        # GMM files use 1-based indexing (MATLAB convention)
        # Register scan i to scan i+1 (to match ground truth direction)
        source_file = os.path.join(gmm_dir, f'{i+1}.gmm')
        target_file = os.path.join(gmm_dir, f'{i+2}.gmm')

        if not os.path.exists(source_file) or not os.path.exists(target_file):
            print(f'\nWARNING: Missing GMM files for scan {i}')
            continue

        # Run registration with timing
        if enable_profiling:
            reg_start = time.time()

        Tinit = np.eye(4)
        output = gmm_d2d_registration_py.isoplanar_registration(
            Tinit, source_file, target_file)
        ret = gmm_d2d_registration_py.anisotropic_registration(
            output[0], source_file, target_file)
        Tout = ret[0]

        if enable_profiling:
            reg_time = time.time() - reg_start
            profiling_data['timings'].append(reg_time)
            profiling_data['scan_indices'].append(i)

        # Extract transformation
        rotation = Tout[0:3, 0:3]
        translation = Tout[0:3, 3]
        dpose = np.concatenate([translation, RToZYX(rotation)])

        # Compute ground truth transformation
        # According to KITTI: Pose[i] transforms from camera_i to world
        # Tr transforms from velodyne to camera
        # Therefore: T_world_vel = Pose[i] @ Tr
        Pose_i = poses_cam[i]      # Camera i to world
        Pose_i1 = poses_cam[i + 1]  # Camera i+1 to world

        # Compute relative transformation in velodyne frame
        # T_vel_i_to_vel_i1 = inv(Tr) @ inv(Pose_i1) @ Pose_i @ Tr
        Tv1_v2 = pose_compose(
            pose_compose(pose_inverse(Tr), pose_inverse(Pose_i1)),
            pose_compose(Pose_i, Tr)
        )

        # Extract ground truth pose
        rotation_gt = Tv1_v2[0:3, 0:3]
        translation_gt = Tv1_v2[0:3, 3]
        dground_truth = np.concatenate([translation_gt, RToZYX(rotation_gt)])

        transforms[i - first_scan, :] = dpose
        ground_truths[i - first_scan, :] = dground_truth
        errors[i - first_scan, :] = dpose - dground_truth

    # Save results
    prefix = 'isoplanarhybrid_'
    result_file = os.path.join(results_dir,
                              f'{prefix}{component_label}_seq{sequence}_results.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump([transforms, ground_truths, errors], f)

    print(f'\nResults saved to: {os.path.relpath(result_file)}')

    # Save plot to results directory (next to .pkl file)
    plot_file = os.path.join(results_dir,
                            f'{prefix}{component_label}_seq{sequence}_trajectory.png')
    plot_results(transforms, ground_truths, first_scan, last_scan,
                save_path=plot_file)
    print(f'Plot saved to: {os.path.relpath(plot_file)}')

    # Print statistics
    translation_errors = np.linalg.norm(errors[:, :3], axis=1)
    rotation_errors = np.linalg.norm(errors[:, 3:], axis=1)

    # Calculate RMSE
    translation_rmse = np.sqrt(np.mean(translation_errors**2))
    rotation_rmse = np.sqrt(np.mean(rotation_errors**2))

    print(f'\nTranslation error (mean): {np.mean(translation_errors):.4f} +/- '
          f'{np.std(translation_errors):.4f} m')
    print(f'Translation RMSE: {translation_rmse:.4f} m')
    print(f'\nRotation error (mean): {np.mean(rotation_errors):.4f} +/- '
          f'{np.std(rotation_errors):.4f} rad')
    print(f'Rotation RMSE: {rotation_rmse:.4f} rad')

    # Save profiling data
    if enable_profiling:
        total_time = time.time() - total_time_start
        timings = np.array(profiling_data['timings'])

        profiling_data['total_time'] = total_time
        profiling_data['mean_time_per_pair'] = np.mean(timings)
        profiling_data['std_time_per_pair'] = np.std(timings)
        profiling_data['min_time'] = np.min(timings)
        profiling_data['max_time'] = np.max(timings)
        profiling_data['total_pairs'] = len(timings)

        # Save to gmm_dir/profiling subfolder
        profiling_dir = os.path.join(gmm_dir, 'profiling')
        os.makedirs(profiling_dir, exist_ok=True)
        profile_file = os.path.join(profiling_dir,
                                   f'profiling_registration_seq{sequence}_{first_scan}to{last_scan}.json')
        with open(profile_file, 'w') as f:
            # Convert numpy arrays to lists and round to 3 decimal places
            save_data = profiling_data.copy()
            save_data['timings'] = [round(t, 3) for t in timings.tolist()]
            save_data['scan_indices'] = profiling_data['scan_indices']
            save_data['total_time'] = round(save_data['total_time'], 3)
            save_data['mean_time_per_pair'] = round(save_data['mean_time_per_pair'], 3)
            save_data['std_time_per_pair'] = round(save_data['std_time_per_pair'], 3)
            save_data['min_time'] = round(save_data['min_time'], 3)
            save_data['max_time'] = round(save_data['max_time'], 3)
            json.dump(save_data, f, indent=2)

        print(f'\n=== PROFILING RESULTS ===')
        print(f'OpenMP threads: {profiling_data["omp_num_threads"]}')
        print(f'Total time: {total_time:.2f} seconds')
        print(f'Time per pair (mean): {profiling_data["mean_time_per_pair"]:.4f} ± '
              f'{profiling_data["std_time_per_pair"]:.4f} seconds')
        print(f'Time per pair (min/max): {profiling_data["min_time"]:.4f} / '
              f'{profiling_data["max_time"]:.4f} seconds')
        print(f'Throughput: {profiling_data["total_pairs"]/total_time:.2f} pairs/second')
        print(f'Profiling data saved to: {os.path.relpath(profile_file)}')

    return transforms, ground_truths


def main():
    parser = argparse.ArgumentParser(
        description='Run GMM D2D registration on KITTI dataset')

    parser.add_argument('--sequence', type=str, default='04',
                       help='KITTI sequence number (default: 04)')
    parser.add_argument('--first_scan', type=int, default=0,
                       help='First scan index (default: 0)')
    parser.add_argument('--last_scan', type=int, default=100,
                       help='Last scan index (default: 100)')
    parser.add_argument('--num_components', type=int, default=100,
                       help='Number of GMM components (default: 100). Use -1 for adaptive mode.')
    parser.add_argument('--bandwidth', type=float, default=None,
                       help='Bandwidth value for adaptive mode (e.g., 0.1, 0.8, 1.0). '
                            'If provided, uses adaptive_bw{bandwidth*100}_components folder.')
    parser.add_argument('--kitti_dir', type=str,
                       default='./dataset/kitti',
                       help='Path to KITTI dataset root')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling (saves timing data to adaptive_components or X_components folder)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp suffix for GMM folder (e.g., 21012004). '
                            'If not provided, uses the most recent timestamped folder.')

    args = parser.parse_args()

    # Run registration (plot is automatically saved to results directory)
    run_kitti_dataset(
        sequence=args.sequence,
        first_scan=args.first_scan,
        last_scan=args.last_scan,
        num_components=args.num_components,
        kitti_dir=args.kitti_dir,
        enable_profiling=args.profile,
        bandwidth=args.bandwidth,
        timestamp=args.timestamp
    )


if __name__ == '__main__':
    main()
