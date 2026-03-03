#!/usr/bin/env python
"""
Script to convert VIRAL rosbag point clouds to GMM format.
Uses rosbags library to read ROS1 bags directly (no ROS environment needed).
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from utils.viral_loader import load_viral_lidar_config, apply_transform, pointcloud2_to_numpy
from utils.pcl_filters import voxel_filter, radius_filter, every_n_filter
from utils.save_gmm import save_sogmm
from utils.gmm_fitter import fit_gmm, make_output_dir, detect_implementation


# python3 create_and_save_gmm_viral.py --every-n 5 --radius 15 --skip_scans 1 --first_scan 500 --bag ~/thesis/VIRAL/eee_03/eee_03.bag --n_components 150


#   take those sequences from viral dataset runs:\
#   python visualize/viral_gt.py --dataset nya_01 --start-id 1250 --final-id 2000
#   python visualize/viral_gt.py --dataset eee_03 --start-id 1250 --final-id 1500


# python3 create_and_save_gmm_viral.py --every-n 5 --radius 30 --skip_scans 1 --bag ~/thesis/VIRAL/eee_03/eee_03.bag --n_components 100 --start-id 1250 --final-id 2000
#   python run_viral_dataset.py \
#     --bag ~/thesis/VIRAL/eee_03/eee_03.bag \
#     --gmm_dir runs/viral_eee_03/150_components_19021747 \
#     --start-id 1250 --final-id 1500

"""
python3 create_and_save_gmm_viral.py --every-n 3 --radius 50 --skip_scans 1 --bag ~/thesis/VIRAL/eee_03/eee_03.bag --bandwidth 0.0003 --start-id 1250 --final-id 2000
"""

def convert_viral_to_gmm(bag_path, output_dir, n_components=None,
                         lidar_topic='/os1_cloud_node1/points',
                         T_body_lidar=None,
                         voxel=None, radius=None, every_n=None,
                         skip_scans=1, start_id=None, final_id=None,
                         implementation='fixed', bandwidth=None,
                         mahal_distance=None, redux_kmeans=None):
    """
    Convert VIRAL rosbag point clouds to GMM format.

    Args:
        bag_path: Path to .bag file
        output_dir: Directory to save GMM files
        n_components: Number of Gaussian components (for fixed)
        lidar_topic: ROS topic name for point clouds
        T_body_lidar: 4x4 transform from lidar to body frame
        voxel: Voxel size (m), None to skip
        radius: Maximum distance from origin (meters)
        every_n: Take every Nth point (reduces GMM init time)
        skip_scans: Process every Nth scan
        start_id: First scan index to process (bag iteration starts here)
        final_id: Last scan index to process (inclusive)
        implementation: 'fixed' or 'sogmm'
        bandwidth: Bandwidth for SOGMM
        mahal_distance: Mahalanobis distance bound for EM (None = disabled)
        redux_kmeans: Use every Nth point for KMeans++ init only, then assign all points (None = disabled)
    """
    if implementation == 'fixed' and n_components is None:
        raise ValueError("fixed requires n_components parameter")
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
    print(f'Every-N filter: {every_n is not None} (every {every_n}th point)')
    print(f'Voxel filter: {voxel is not None} (size={voxel}m)')
    print(f'Radius filter: {radius is not None} (r={radius}m)')
    print(f'Skip scans: {skip_scans}, start_id: {start_id}, final_id: {final_id}')
    print(f'Mahalanobis bound: {mahal_distance is not None} (λ={mahal_distance})')
    print(f'Redux KMeans++: {redux_kmeans is not None} (every {redux_kmeans}th for init)')

    gmm_output_dir = make_output_dir(output_dir, implementation, n_components, bandwidth)

    # Write metadata
    bag_name = Path(bag_path).stem
    meta = {
        'dataset': 'VIRAL',
        'bag_file': bag_name,
        'lidar_topic': lidar_topic,
        'lidar_transform': T_body_lidar is not None,
        'implementation': implementation,
        'n_components': n_components if implementation == 'fixed' else None,
        'bandwidth': bandwidth if implementation == 'sogmm' else None,
        'every_n': every_n,
        'voxel': voxel,
        'radius': radius,
        'skip_scans': skip_scans,
        'start_id': start_id,
        'final_id': final_id,
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
            if start_id is not None and scan_idx < start_id:
                scan_idx += 1
                continue
            if final_id is not None and scan_idx > final_id:
                break
            if (scan_idx - (start_id or 0)) % skip_scans != 0:
                scan_idx += 1
                continue

            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            points_4d = pointcloud2_to_numpy(msg)

            points_4d = apply_transform(points_4d, T_body_lidar)

            if every_n is not None:
                points_4d = every_n_filter(points_4d, n=every_n)
            if radius is not None:
                points_4d = radius_filter(points_4d, 0.5, radius)
            if voxel is not None:
                points_4d = voxel_filter(points_4d, voxel)

            if len(points_4d) < 100:
                print(f'\nToo sparse: Scan {scan_idx} has only {len(points_4d)} points, skipping')
                scan_idx += 1
                continue

            if implementation == 'fixed' and len(points_4d) < n_components * 3:
                print(f'\nToo sparse: Scan {scan_idx} has only {len(points_4d)} points')

            scan_name = f'scan_{saved_count:06d}'
            stats_dir = os.path.join(gmm_output_dir, 'profiling')
            gmm_4d = fit_gmm(
                points_4d, implementation,
                n_components=n_components, bandwidth=bandwidth,
                redux_kmeans=redux_kmeans, mahal_distance=mahal_distance,
                stats_dir=stats_dir, scan_name=scan_name,
            )
            if gmm_4d is None:
                print(f'\nWARNING: EM fitting failed for scan {scan_idx}')
                scan_idx += 1
                continue
            if implementation == 'sogmm':
                print(f"SOGMM fitted with {gmm_4d.n_components_} components")

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
                        help='Number of GMM components for fixed')
    parser.add_argument('--bw', type=float, default=None,
                        help='Bandwidth for sogmm')
    parser.add_argument('--every-n', type=int, default=None, dest='every_n')
    parser.add_argument('--voxel',   type=float, default=None)
    parser.add_argument('--radius',  type=float, default=None)
    parser.add_argument('--skip_scans', type=int, default=1,
                        help='Process every Nth scan (default: 1)')
    parser.add_argument('--start-id', type=int, default=None, dest='start_id',
                        help='First scan index to process (default: 0)')
    parser.add_argument('--final-id', type=int, default=None, dest='final_id',
                        help='Maximum scan index to save a GMM')
    parser.add_argument('--mahal_distance', type=float, default=None,
                        help='Mahalanobis distance bound for EM (default: disabled)')
    parser.add_argument('--redux_kmeans', type=int, default=None,
                        help='Use every Nth point for KMeans++ init only (default: disabled)')

    args = parser.parse_args()

    implementation, bandwidth, n_components = detect_implementation(args.bw, args.n_components)

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
        n_components=n_components,
        lidar_topic=lidar_topic,
        T_body_lidar=T_body_lidar,
        every_n=args.every_n,
        voxel=args.voxel,
        radius=args.radius,
        skip_scans=args.skip_scans,
        start_id=args.start_id,
        final_id=args.final_id,
        implementation=implementation,
        bandwidth=bandwidth,
        mahal_distance=args.mahal_distance,
        redux_kmeans=args.redux_kmeans
    )


if __name__ == '__main__':
    main()
