#!/usr/bin/env python
"""
Script to convert VIRAL rosbag point clouds to GMM format.
Uses rosbags library to read ROS1 bags directly (no ROS environment needed).
"""
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from utils.viral_loader import load_viral_lidar_config, apply_transform, pointcloud2_to_numpy
from utils.pcl_filters import voxel_filter, radius_filter, every_n_filter
from utils.save_gmm import save_sogmm

from sogmm_py import SOGMM
from gmm_py import GMMf4CPU
from kinit_py import KInitf4CPU


def convert_viral_to_gmm(bag_path, output_dir, n_components=None,
                         lidar_topic='/os1_cloud_node1/points',
                         T_body_lidar=None,
                         use_voxel_filter=False, voxel_size=0.1,
                         use_radius_filter=False, radius=30.0,
                         use_every_n_filter=False, every_n=5,
                         max_scans=None, skip_scans=10, first_scan=0,
                         implementation='fsogmm', bandwidth=None,
                         mahal_distance=None, redux_kmeans=None):
    """
    Convert VIRAL rosbag point clouds to GMM format.

    Args:
        bag_path: Path to .bag file
        output_dir: Directory to save GMM files
        n_components: Number of Gaussian components (for fsogmm)
        lidar_topic: ROS topic name for point clouds
        T_body_lidar: 4x4 transform from lidar to body frame
        use_voxel_filter: Whether to apply voxel filtering
        voxel_size: Voxel size for filtering (meters)
        use_radius_filter: Whether to remove points beyond radius
        radius: Maximum distance from origin (meters)
        use_every_n_filter: Whether to apply every-N downsampling
        every_n: Take every Nth point (reduces GMM init time)
        max_scans: Maximum number of scans to process
        skip_scans: Process every Nth scan
        first_scan: First scan index to start processing from
        implementation: 'fsogmm' or 'sogmm'
        bandwidth: Bandwidth for SOGMM
        mahal_distance: Mahalanobis distance bound for EM (None = disabled)
        redux_kmeans: Use every Nth point for KMeans++ init only, then assign all points (None = disabled)
    """
    if implementation == 'fsogmm' and n_components is None:
        raise ValueError("fsogmm requires n_components parameter")
    if implementation == 'sogmm' and bandwidth is None:
        raise ValueError("sogmm requires bandwidth parameter")

    if not os.path.exists(bag_path):
        print(f'ERROR: Bag file not found: {bag_path}')
        sys.exit(1)

    # Print config
    print(f'Reading bag file: {bag_path}')
    print(f'LiDAR topic: {lidar_topic}')
    if implementation == 'sogmm':
        print(f'Implementation: {implementation} , bandwidth={bandwidth}')
    else:
        print(f'Implementation: {implementation}, components={n_components}')
    print(f'Every-N filter: {use_every_n_filter} (every {every_n}th point)')
    print(f'Voxel filter: {use_voxel_filter} (size={voxel_size}m)')
    print(f'Radius filter: {use_radius_filter} (r={radius}m)')
    print(f'Skip scans: {skip_scans}, first_scan: {first_scan}')
    print(f'Mahalanobis bound: {mahal_distance is not None} (λ={mahal_distance})')
    print(f'Redux KMeans++: {redux_kmeans is not None} (every {redux_kmeans}th for init)')

    # Create output directory
    timestamp = datetime.now().strftime("%d%m%H%M")
    if implementation == 'sogmm':
        gmm_output_dir = os.path.join(output_dir, f"adaptive_bw{int(bandwidth*100)}_components_{timestamp}")
    else:
        gmm_output_dir = os.path.join(output_dir, f'{n_components}_components_{timestamp}')
    os.makedirs(gmm_output_dir, exist_ok=True)

    # Write metadata
    bag_name = Path(bag_path).stem
    meta = {
        'dataset': 'VIRAL',
        'bag_file': bag_name,
        'lidar_topic': lidar_topic,
        'lidar_transform': T_body_lidar is not None,
        'implementation': implementation,
        'n_components': n_components if implementation == 'fsogmm' else None,
        'bandwidth': bandwidth if implementation == 'sogmm' else None,
        'every_n_filter': use_every_n_filter,
        'every_n': every_n,
        'voxel_filter': use_voxel_filter,
        'voxel_size': voxel_size,
        'radius_filter': use_radius_filter,
        'radius': radius,
        'skip_scans': skip_scans,
        'first_scan': first_scan,
        'mahal_distance': mahal_distance,
        'redux_kmeans': redux_kmeans,
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    meta_path = os.path.join(gmm_output_dir, 'meta.yaml')
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    # Process bag file
    scan_idx = 0
    saved_count = 0
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == lidar_topic]

        if not connections:
            print(f'ERROR: Topic {lidar_topic} not found in bag file')
            print('Available topics:')
            for c in reader.connections:
                print(f'  {c.topic} ({c.msgtype})')
            sys.exit(1)

        print(f'\nProcessing point clouds from {lidar_topic}...')

        for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections)):
            if scan_idx < first_scan:
                scan_idx += 1
                continue
            if (scan_idx - first_scan) % skip_scans != 0:
                scan_idx += 1
                continue

            if max_scans is not None and saved_count >= max_scans:
                break

            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            points_4d = pointcloud2_to_numpy(msg)
    
            points_4d = apply_transform(points_4d, T_body_lidar)
        
            if use_every_n_filter:
                points_4d = every_n_filter(points_4d, n=every_n)
            if use_radius_filter:
                min_radius, max_radius = 0.5, radius
                points_4d = radius_filter(points_4d, min_radius, max_radius)
            if use_voxel_filter:
                points_4d = voxel_filter(points_4d, voxel_size)

            if len(points_4d) < 100:
                print(f'\nToo sparse: Scan {scan_idx} has only {len(points_4d)} points, skipping')
                scan_idx += 1
                continue

            if implementation == 'fsogmm' and len(points_4d) < n_components * 3:
                print(f'\nToo sparse: Scan {scan_idx} has only {len(points_4d)} points')

            # Fit GMM
            if implementation == 'fsogmm':
                n_samples = points_4d.shape[0]
                kinit = KInitf4CPU()

                if redux_kmeans is not None:
                    # Redux: use subsampled points for KMeans++ init
                    points_sub = points_4d[::redux_kmeans]
                    centers, _ = kinit.resp_calc(points_sub, n_components)
                    # Assign ALL points to nearest center
                    dists = kinit.euclidean_dists(points_4d, centers)
                    assignments = np.argmin(dists, axis=1)
                    resp = np.zeros((n_samples, n_components), dtype=np.float32)
                    resp[np.arange(n_samples), assignments] = 1
                else:
                    # Standard: KMeans++ on all points
                    _, indices = kinit.resp_calc(points_4d, n_components)
                    resp = np.zeros((n_samples, n_components), dtype=np.float32)
                    resp[indices, np.arange(n_components)] = 1

                stats_dir = os.path.join(gmm_output_dir, 'profiling')
                os.makedirs(stats_dir, exist_ok=True)
                stats_file = f'gmm_stats_scan_{saved_count:06d}.csv'

                local_model = GMMf4CPU(n_components, True, stats_dir, stats_file)
                if mahal_distance is not None:
                    success = local_model.fit_mahal(points_4d, resp, mahal_distance)
                else:
                    success = local_model.fit(points_4d, resp)

                if not success:
                    print(f'\nWARNING: EM fitting failed for scan {scan_idx}')
                    scan_idx += 1
                    continue
                gmm_4d = local_model

            elif implementation == 'sogmm':
                stats_dir = os.path.join(gmm_output_dir, 'profiling')
                os.makedirs(stats_dir, exist_ok=True)
                stats_file_prefix = f'scan_{saved_count:06d}'

                sg = SOGMM(bandwidth, save_stats=True, stats_dir=stats_dir, stats_file_prefix=stats_file_prefix)
                local_model = sg.fit(points_4d)
                print(f"SOGMM fitted with {local_model.n_components_} components")
                gmm_4d = local_model

            # Save GMM
            output_path = os.path.join(gmm_output_dir, f'{scan_idx}.gmm')
            save_sogmm(output_path, gmm_4d)

            saved_count += 1
            scan_idx += 1

    print(f'\nSaved {saved_count} GMM files to: {gmm_output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Convert VIRAL rosbag point clouds to GMM format')

    parser.add_argument('--bag', type=str, required=True,
                        help='Path to .bag file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: ./viral_{bag_name})')
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
    parser.add_argument('--voxel_size', type=float, default=0.2,
                        help='Voxel size in meters (default: 0.2)')
    parser.add_argument('--radius_filter', action='store_true',
                        help='Enable radius filtering')
    parser.add_argument('--radius', type=float, default=30.0,
                        help='Max distance from origin (default: 30.0)')
    parser.add_argument('--max_scans', type=int, default=None,
                        help='Maximum number of scans to process')
    parser.add_argument('--skip_scans', type=int, default=1,
                        help='Process every Nth scan (default: 1)')
    parser.add_argument('--first_scan', type=int, default=0,
                        help='First scan index to start processing from (default: 0)')
    parser.add_argument('--mahal_distance', type=float, default=None,
                        help='Mahalanobis distance bound for EM (default: disabled)')
    parser.add_argument('--redux_kmeans', type=int, default=None,
                        help='Use every Nth point for KMeans++ init only (default: disabled)')

    args = parser.parse_args()

    # Detect implementation
    if args.bandwidth is not None and args.n_components is not None:
        print('ERROR: Specify either --bandwidth OR --n_components, not both')
        sys.exit(1)
    elif args.bandwidth is not None:
        implementation = 'sogmm'
        print(f'Using SOGMM  with bandwidth={args.bandwidth}')
    elif args.n_components is not None:
        implementation = 'fsogmm'
        print(f'Using fsogmm (fixed) with n_components={args.n_components}')
    else:
        print('ERROR: Must specify either --bandwidth or --n_components')
        sys.exit(1)

    # Resolve paths
    args.bag = os.path.expanduser(args.bag)

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bag_name = Path(args.bag).stem
        args.output_dir = os.path.join(script_dir, 'runs', f'viral_{bag_name}')

    # Load lidar config (topic + transform)
    config_topic, T_body_lidar = load_viral_lidar_config(args.bag)
    lidar_topic = config_topic or '/os1_cloud_node1/points'

    print(f'Bag file: {args.bag}')
    print(f'Output dir: {args.output_dir}\n')

    convert_viral_to_gmm(
        bag_path=args.bag,
        output_dir=args.output_dir,
        n_components=args.n_components,
        lidar_topic=lidar_topic,
        T_body_lidar=T_body_lidar,
        use_every_n_filter=args.every_n_filter,
        every_n=args.every_n,
        use_voxel_filter=args.voxel_filter,
        voxel_size=args.voxel_size,
        use_radius_filter=args.radius_filter,
        radius=args.radius,
        max_scans=args.max_scans,
        skip_scans=args.skip_scans,
        first_scan=args.first_scan,
        implementation=implementation,
        bandwidth=args.bandwidth,
        mahal_distance=args.mahal_distance,
        redux_kmeans=args.redux_kmeans
    )


if __name__ == '__main__':
    main()
