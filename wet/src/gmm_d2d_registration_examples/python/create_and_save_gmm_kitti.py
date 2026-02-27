#!/usr/bin/env python
"""
Script to convert KITTI Velodyne point clouds to GMM format.
Supports voxel filtering/radius filtering to reduce point count.
"""
import os
import sys
import argparse

import numpy as np
import yaml
from tqdm import tqdm

from utils.kitti_loader import load_kitti_velodyne
from utils.pcl_filters import voxel_filter, radius_filter, remove_plane_ransac, every_n_filter
from utils.save_gmm import save_sogmm
from utils.gmm_fitter import fit_gmm, make_output_dir, detect_implementation


def convert_kitti_to_gmm(sequence_dir, output_dir, sequence, n_components=None,
                         voxel=None, radius=None,
                         use_plane_removal=False, ransac_distance=0.3, ransac_iters=1000,
                         every_n=None,
                         start_idx=0, end_idx=None, implementation='fixed',
                         bandwidth=None, mahal_distance=None, redux_kmeans=None):
    """
    Convert KITTI velodyne scans to GMM format.

    Args:
        sequence_dir: Path to KITTI sequence (e.g., .../sequences/04/)
        output_dir: Directory to save GMM files
        sequence: KITTI sequence number (e.g., '04')
        n_components: Number of Gaussian components
        voxel: Voxel size (m), None to skip
        radius: Maximum distance from origin (meters)
        use_plane_removal: Whether to apply RANSAC ground removal
        ransac_distance: RANSAC distance threshold
        ransac_iters: RANSAC iterations
        every_n: Take every Nth point (reduces GMM init time)
        start_idx: First scan index to process
        end_idx: Last scan index to process (None = all)
        implementation: 'fixed' or 'sogmm'
        bandwidth: Bandwidth for SOGMM
        mahal_distance: Mahalanobis distance bound for EM (None = disabled)
        redux_kmeans: Use every Nth point for KMeans++ init only, then assign all points (None = disabled)
    """
    if implementation == 'fixed' and n_components is None:
        raise ValueError("fixed requires n_components parameter")
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
        print(f'Implementation: {implementation} , bandwidth={bandwidth}')
    else:
        print(f'Implementation: {implementation}, components={n_components}')
    print(f'Every-N filter: {every_n is not None} (every {every_n}th point)')
    print(f'Voxel filter: {voxel is not None} (size={voxel}m)')
    print(f'Radius filter: {radius is not None} (r={radius}m)')
    print(f'Plane removal: {use_plane_removal} (dist={ransac_distance}m, iters={ransac_iters})')
    print(f'Mahalanobis bound: {mahal_distance is not None} (λ={mahal_distance})')
    print(f'Redux KMeans++: {redux_kmeans is not None} (every {redux_kmeans}th for init)')

    gmm_output_dir = make_output_dir(output_dir, implementation, n_components, bandwidth)

    # Write metadata
    meta = {
        'dataset': 'KITTI',
        'sequence': sequence,
        'implementation': implementation,
        'n_components': n_components if implementation == 'fixed' else None,
        'bandwidth': bandwidth if implementation == 'sogmm' else None,
        'every_n': every_n,
        'voxel': voxel,
        'radius': radius,
        'plane_removal': use_plane_removal,
        'ransac_distance': ransac_distance,
        'ransac_iters': ransac_iters,
        'mahal_distance': mahal_distance,
        'redux_kmeans': redux_kmeans,
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
        if every_n is not None:
            points_4d = every_n_filter(points_4d, n=every_n)
        if radius is not None:
            points_4d = radius_filter(points_4d, 0.5, radius)
        if voxel is not None:
            points_4d = voxel_filter(points_4d, voxel)
        if use_plane_removal:
            points_4d = remove_plane_ransac(points_4d, ransac_distance, 3, ransac_iters)

        if implementation == 'fixed' and len(points_4d) < n_components * 3:
            print(f'\nWARNING: {bin_file} has only {len(points_4d)} points')

        scan_name = bin_file.split('.')[0]
        stats_dir = os.path.join(gmm_output_dir, 'profiling')
        gmm_4d = fit_gmm(
            points_4d, implementation,
            n_components=n_components, bandwidth=bandwidth,
            redux_kmeans=redux_kmeans, mahal_distance=mahal_distance,
            stats_dir=stats_dir, scan_name=scan_name,
        )
        if gmm_4d is None:
            print(f'WARNING: EM fitting failed for {bin_file}')
            continue
        if implementation == 'sogmm':
            print(f"SOGMM fitted with {gmm_4d.n_components_} components")

        # Save GMM (1-based indexing for MATLAB compatibility)
        idx = int(scan_name) + 1
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
                        help='Number of GMM components for fixed')
    parser.add_argument('--bw', type=float, default=None,
                        help='Bandwidth for sogmm')
    parser.add_argument('--every-n', type=int, default=None, dest='every_n')
    parser.add_argument('--voxel',   type=float, default=None)
    parser.add_argument('--radius',  type=float, default=None)
    parser.add_argument('--remove_plane', action='store_true',
                        help='Enable RANSAC ground plane removal')
    parser.add_argument('--ransac_distance', type=float, default=0.3,
                        help='RANSAC distance threshold (default: 0.3)')
    parser.add_argument('--ransac_iters', type=int, default=1000,
                        help='RANSAC iterations (default: 1000)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='First scan index (default: 0)')
    parser.add_argument('--mahal_distance', type=float, default=None,
                        help='Mahalanobis distance bound for EM (default: disabled)')
    parser.add_argument('--redux_kmeans', type=int, default=None,
                        help='Use every Nth point for KMeans++ init only (default: disabled)')

    args = parser.parse_args()

    implementation, bandwidth, n_components = detect_implementation(args.bw, args.n_components)

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
        args.output_dir = os.path.join(script_dir, 'runs', f'kitti_sequence_{args.sequence}')

    if not os.path.exists(sequence_dir):
        print(f'ERROR: Sequence directory not found: {sequence_dir}')
        sys.exit(1)

    end_idx = args.start_idx + args.n_scans

    print(f'KITTI Sequence: {args.sequence}')
    print(f'Output dir: {args.output_dir}')
    print(f'Scans: {args.start_idx} to {end_idx-1} ({args.n_scans} total)\n')

    convert_kitti_to_gmm(
        sequence_dir=sequence_dir,
        output_dir=args.output_dir,
        sequence=args.sequence,
        n_components=n_components,
        every_n=args.every_n,
        voxel=args.voxel,
        radius=args.radius,
        use_plane_removal=args.remove_plane,
        ransac_distance=args.ransac_distance,
        ransac_iters=args.ransac_iters,
        start_idx=args.start_idx,
        end_idx=end_idx,
        implementation=implementation,
        bandwidth=bandwidth,
        mahal_distance=args.mahal_distance,
        redux_kmeans=args.redux_kmeans
    )


if __name__ == '__main__':
    main()
