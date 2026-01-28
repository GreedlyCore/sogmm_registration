#!/usr/bin/env python
"""
Script to convert KITTI Velodyne point clouds to GMM format.
Supports voxel filtering/radius filtering to reduce point count.
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import yaml
from tqdm import tqdm

from utils.kitti_loader import load_kitti_velodyne
from utils.pcl_filters import voxel_filter, radius_filter, min_distance_filter, remove_plane_ransac, every_n_filter
from utils.save_gmm import save_sogmm

from sogmm_py import SOGMM
from gmm_py import GMMf4CPU
from kinit_py import KInitf4CPU


def convert_kitti_to_gmm(sequence_dir, output_dir, sequence, n_components=None,
                         use_voxel_filter=False, voxel_size=0.1,
                         use_radius_filter=False, radius=5.0,
                         use_plane_removal=False, ransac_distance=0.3, ransac_iters=1000,
                         use_every_n_filter=False, every_n=5,
                         start_idx=0, end_idx=None, implementation='fsogmm',
                         bandwidth=None):
    """
    Convert KITTI velodyne scans to GMM format.

    Args:
        sequence_dir: Path to KITTI sequence (e.g., .../sequences/04/)
        output_dir: Directory to save GMM files
        sequence: KITTI sequence number (e.g., '04')
        n_components: Number of Gaussian components
        use_voxel_filter: Whether to apply voxel filtering
        voxel_size: Voxel size for filtering (meters)
        use_radius_filter: Whether to remove points beyond radius
        radius: Maximum distance from origin (meters)
        use_plane_removal: Whether to apply RANSAC ground removal
        ransac_distance: RANSAC distance threshold
        ransac_iters: RANSAC iterations
        use_every_n_filter: Whether to apply every-N downsampling
        every_n: Take every Nth point (reduces GMM init time)
        start_idx: First scan index to process
        end_idx: Last scan index to process (None = all)
        implementation: 'fsogmm' or 'sogmm'
        bandwidth: Bandwidth for SOGMM
    """
    if implementation == 'fsogmm' and n_components is None:
        raise ValueError("fsogmm requires n_components parameter")
    if implementation == 'sogmm' and bandwidth is None:
        raise ValueError("sogmm requires bandwidth parameter")

    velodyne_dir = os.path.join(sequence_dir, 'velodyne')
    if not os.path.exists(velodyne_dir):
        print(f'ERROR: Velodyne directory not found: {velodyne_dir}')
        print('\nDownload KITTI velodyne data (80 GB):')
        print('https://www.cvlibs.net/datasets/kitti/eval_odometry.php')
        sys.exit(1)

    bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    if end_idx is None:
        end_idx = len(bin_files)
    bin_files = bin_files[start_idx:end_idx]

    # Print config
    print(f'Converting {len(bin_files)} scans to GMM format...')
    if implementation == 'sogmm':
        print(f'Implementation: {implementation} (adaptive), bandwidth={bandwidth}')
    else:
        print(f'Implementation: {implementation}, components={n_components}')
    print(f'Every-N filter: {use_every_n_filter} (every {every_n}th point)')
    print(f'Voxel filter: {use_voxel_filter} (size={voxel_size}m)')
    print(f'Radius filter: {use_radius_filter} (r={radius}m)')
    print(f'Plane removal: {use_plane_removal} (dist={ransac_distance}m, iters={ransac_iters})')

    # Create output directory
    timestamp = datetime.now().strftime("%d%m%H%M")
    if implementation == 'sogmm':
        gmm_output_dir = os.path.join(output_dir, f"adaptive_bw{int(bandwidth*100)}_components_{timestamp}")
    else:
        gmm_output_dir = os.path.join(output_dir, f'{n_components}_components_{timestamp}')
    os.makedirs(gmm_output_dir, exist_ok=True)

    # Write metadata
    meta = {
        'dataset': 'KITTI',
        'sequence': sequence,
        'implementation': implementation,
        'n_components': n_components if implementation == 'fsogmm' else None,
        'bandwidth': bandwidth if implementation == 'sogmm' else None,
        'every_n_filter': use_every_n_filter,
        'every_n': every_n,
        'voxel_filter': use_voxel_filter,
        'voxel_size': voxel_size,
        'radius_filter': use_radius_filter,
        'radius': radius,
        'plane_removal': use_plane_removal,
        'ransac_distance': ransac_distance,
        'ransac_iters': ransac_iters,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'n_scans': len(bin_files),
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    meta_path = os.path.join(gmm_output_dir, 'meta.yaml')
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    # Process scans
    for bin_file in tqdm(bin_files, desc='Processing scans'):
        filepath = os.path.join(velodyne_dir, bin_file)
        points_4d = load_kitti_velodyne(filepath)

        # Apply filters
        if use_every_n_filter:
            points_4d = every_n_filter(points_4d, n=every_n)
        if use_radius_filter:
            points_4d = radius_filter(points_4d, radius)
        if use_voxel_filter:
            points_4d = voxel_filter(points_4d, voxel_size)
        if use_plane_removal:
            points_4d = remove_plane_ransac(points_4d, ransac_distance, 3, ransac_iters)

        # Remove points too close (vehicle)
        points_4d = min_distance_filter(points_4d, min_dist=2.0)

        if implementation == 'fsogmm' and len(points_4d) < n_components * 3:
            print(f'\nWARNING: {bin_file} has only {len(points_4d)} points')

        # Fit GMM
        if implementation == 'fsogmm':
            n_samples = points_4d.shape[0]
            kinit = KInitf4CPU()
            _, indices = kinit.resp_calc(points_4d, n_components)
            resp = np.zeros((n_samples, n_components), dtype=np.float32)
            resp[indices, np.arange(n_components)] = 1

            stats_dir = os.path.join(gmm_output_dir, 'profiling')
            os.makedirs(stats_dir, exist_ok=True)
            stats_file = f'gmm_stats_{bin_file.split(".")[0]}.csv'

            local_model = GMMf4CPU(n_components, True, stats_dir, stats_file)
            success = local_model.fit(points_4d, resp)

            if not success:
                print(f'WARNING: EM fitting failed for {bin_file}')
                continue
            gmm_4d = local_model

        elif implementation == 'sogmm':
            stats_dir = os.path.join(gmm_output_dir, 'profiling')
            os.makedirs(stats_dir, exist_ok=True)
            stats_file_prefix = bin_file.split(".")[0]

            sg = SOGMM(bandwidth, save_stats=True, stats_dir=stats_dir, stats_file_prefix=stats_file_prefix)
            local_model = sg.fit(points_4d)
            print(f"SOGMM fitted with {local_model.n_components_} components")
            gmm_4d = local_model

        # Save GMM (1-based indexing for MATLAB compatibility)
        idx = int(bin_file.split('.')[0]) + 1
        output_path = os.path.join(gmm_output_dir, f'{idx}.gmm')
        save_sogmm(output_path, gmm_4d)


def main():
    parser = argparse.ArgumentParser(description='Convert KITTI velodyne scans to GMM format')

    parser.add_argument('--sequence', type=str, required=True,
                        help='KITTI sequence number (e.g., 00, 04, 07)')
    parser.add_argument('--n_scans', type=int, required=True,
                        help='Number of scans to convert')
    parser.add_argument('--kitti_dir', type=str, default='./dataset/kitti',
                        help='Path to KITTI dataset root')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: ./kitti_sequence_{ID})')
    parser.add_argument('--n_components', type=int, default=None,
                        help='Number of GMM components for fsogmm')
    parser.add_argument('--bandwidth', type=float, default=None,
                        help='Bandwidth for sogmm')
    parser.add_argument('--every_n_filter', action='store_true',
                        help='Enable every-N downsampling (reduces GMM init time)')
    parser.add_argument('--every_n', type=int, default=5,
                        help='Take every Nth point (default: 5)')
    parser.add_argument('--voxel_filter', action='store_true',
                        help='Enable voxel filtering')
    parser.add_argument('--voxel_size', type=float, default=0.1,
                        help='Voxel size in meters (default: 0.1)')
    parser.add_argument('--radius_filter', action='store_true',
                        help='Enable radius filtering')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='Max distance from origin (default: 5.0)')
    parser.add_argument('--remove_plane', action='store_true',
                        help='Enable RANSAC ground plane removal')
    parser.add_argument('--ransac_distance', type=float, default=0.3,
                        help='RANSAC distance threshold (default: 0.3)')
    parser.add_argument('--ransac_iters', type=int, default=1000,
                        help='RANSAC iterations (default: 1000)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='First scan index (default: 0)')

    args = parser.parse_args()

    # Detect implementation
    if args.bandwidth is not None and args.n_components is not None:
        print('ERROR: Specify either --bandwidth OR --n_components, not both')
        sys.exit(1)
    elif args.bandwidth is not None:
        implementation = 'sogmm'
        print(f'Using SOGMM (adaptive) with bandwidth={args.bandwidth}')
    elif args.n_components is not None:
        implementation = 'fsogmm'
        print(f'Using fsogmm (fixed) with n_components={args.n_components}')
    else:
        print('ERROR: Must specify either --bandwidth or --n_components')
        sys.exit(1)

    # Resolve paths
    if args.kitti_dir == './dataset/kitti':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dataset = os.path.join(script_dir, '../../../../dataset/kitti')
        if os.path.exists(repo_dataset):
            args.kitti_dir = repo_dataset

    args.kitti_dir = os.path.expanduser(args.kitti_dir)
    sequence_dir = os.path.join(args.kitti_dir, 'data_odometry_velodyne',
                                'dataset', 'sequences', args.sequence)

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, f'kitti_sequence_{args.sequence}')

    if not os.path.exists(sequence_dir):
        print(f'ERROR: Sequence directory not found: {sequence_dir}')
        sys.exit(1)

    end_idx = args.start_idx + args.n_scans

    print(f'KITTI Sequence: {args.sequence}')
    print(f'Sequence dir: {sequence_dir}')
    print(f'Output dir: {args.output_dir}')
    print(f'Scans: {args.start_idx} to {end_idx-1} ({args.n_scans} total)\n')

    convert_kitti_to_gmm(
        sequence_dir=sequence_dir,
        output_dir=args.output_dir,
        sequence=args.sequence,
        n_components=args.n_components,
        use_every_n_filter=args.every_n_filter,
        every_n=args.every_n,
        use_voxel_filter=args.voxel_filter,
        voxel_size=args.voxel_size,
        use_radius_filter=args.radius_filter,
        radius=args.radius,
        use_plane_removal=args.remove_plane,
        ransac_distance=args.ransac_distance,
        ransac_iters=args.ransac_iters,
        start_idx=args.start_idx,
        end_idx=end_idx,
        implementation=implementation,
        bandwidth=args.bandwidth
    )


if __name__ == '__main__':
    main()
