#!/usr/bin/env python
"""
Script to convert MAI City rosbag point clouds to GMM format.
Bag: ~/thesis/mai_city/bags/00.bag
Topic: /velodyne_points  (XYZI float32, point_step=16)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from utils.pcl_filters import voxel_filter, radius_filter, every_n_filter
from utils.save_gmm import save_sogmm
from utils.gmm_fitter import fit_gmm, make_output_dir, detect_implementation


# python create_and_save_gmm_mai.py --n_components 150 --radius 30.0 --voxel 0.2


MAI_PCL_DTYPE = np.dtype([
    ('x', np.float32), ('y', np.float32),
    ('z', np.float32), ('intensity', np.float32),
])


def pointcloud2_to_numpy(msg):
    points = np.frombuffer(msg.data, dtype=MAI_PCL_DTYPE)
    return np.column_stack([points['x'], points['y'], points['z'], points['intensity']])


def convert_mai_to_gmm(bag_path, output_dir, n_components=None,
                       lidar_topic='/velodyne_points',
                       voxel=None, radius=None, every_n=None,
                       first_scan=0, max_scans=None, skip_scans=1,
                       implementation='fixed', bandwidth=None,
                       mahal_distance=None, redux_kmeans=None):

    if implementation == 'fixed' and n_components is None:
        raise ValueError('fixed requires n_components')
    if implementation == 'sogmm' and bandwidth is None:
        raise ValueError('sogmm requires bandwidth')

    if not os.path.exists(bag_path):
        print(f'ERROR: Bag not found: {bag_path}')
        sys.exit(1)

    print(f'Bag: {bag_path}')
    print(f'Topic: {lidar_topic}')
    if implementation == 'sogmm':
        print(f'Implementation: sogmm, bandwidth={bandwidth}')
    else:
        print(f'Implementation: fixed, components={n_components}')
    print(f'Every-N filter: {every_n is not None} (n={every_n})')
    print(f'Voxel filter:   {voxel is not None} (size={voxel}m)')
    print(f'Radius filter:  {radius is not None} (r={radius}m)')
    print(f'first_scan={first_scan}, skip_scans={skip_scans}, max_scans={max_scans}')

    gmm_output_dir = make_output_dir(output_dir, implementation, n_components, bandwidth)

    bag_name = Path(bag_path).stem
    meta = {
        'dataset': 'MAI',
        'bag_file': bag_name,
        'lidar_topic': lidar_topic,
        'implementation': implementation,
        'n_components': n_components if implementation == 'fixed' else None,
        'bandwidth': bandwidth if implementation == 'sogmm' else None,
        'every_n': every_n,
        'voxel': voxel,
        'radius': radius,
        'first_scan': first_scan,
        'skip_scans': skip_scans,
        'mahal_distance': mahal_distance,
        'redux_kmeans': redux_kmeans,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    with open(os.path.join(gmm_output_dir, 'meta.yaml'), 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    scan_idx = 0
    saved_count = 0
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == lidar_topic]
        if not connections:
            print(f'ERROR: Topic {lidar_topic} not found')
            for c in reader.connections:
                print(f'  {c.topic}')
            sys.exit(1)

        for connection, timestamp_ns, rawdata in tqdm(reader.messages(connections=connections)):
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

            if every_n is not None:
                points_4d = every_n_filter(points_4d, n=every_n)
            if radius is not None:
                points_4d = radius_filter(points_4d, 0.5, radius)
            if voxel is not None:
                points_4d = voxel_filter(points_4d, voxel)

            if len(points_4d) < 100:
                print(f'\nToo sparse: scan {scan_idx} has {len(points_4d)} points, skipping')
                scan_idx += 1
                continue

            scan_name = f'scan_{saved_count:06d}'
            stats_dir = os.path.join(gmm_output_dir, 'profiling')
            gmm_4d = fit_gmm(
                points_4d, implementation,
                n_components=n_components, bandwidth=bandwidth,
                redux_kmeans=redux_kmeans, mahal_distance=mahal_distance,
                stats_dir=stats_dir, scan_name=scan_name,
            )
            if gmm_4d is None:
                print(f'\nWARNING: EM failed for scan {scan_idx}')
                scan_idx += 1
                continue

            save_sogmm(os.path.join(gmm_output_dir, f'{scan_idx}.gmm'), gmm_4d)
            saved_count += 1
            scan_idx += 1

    print(f'\nSaved {saved_count} GMM files to: {gmm_output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Convert MAI City bag to GMM format')
    parser.add_argument('--bag', type=str, default=os.path.expanduser('~/thesis/mai_city/bags/00.bag'))
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_components', type=int, default=None)
    parser.add_argument('--bw', type=float, default=None)
    parser.add_argument('--every-n', type=int, default=None, dest='every_n')
    parser.add_argument('--voxel',   type=float, default=None)
    parser.add_argument('--radius',  type=float, default=None)
    parser.add_argument('--first_scan', type=int, default=0)
    parser.add_argument('--skip_scans', type=int, default=1)
    parser.add_argument('--max_scans', type=int, default=None)
    parser.add_argument('--mahal_distance', type=float, default=None)
    parser.add_argument('--redux_kmeans', type=int, default=None)
    args = parser.parse_args()

    implementation, bandwidth, n_components = detect_implementation(args.bw, args.n_components)

    args.bag = os.path.expanduser(args.bag)
    if args.output_dir is None:
        bag_name = Path(args.bag).stem
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, 'runs', f'mai_{bag_name}')

    convert_mai_to_gmm(
        bag_path=args.bag,
        output_dir=args.output_dir,
        n_components=n_components,
        voxel=args.voxel,
        radius=args.radius,
        every_n=args.every_n,
        first_scan=args.first_scan,
        skip_scans=args.skip_scans,
        max_scans=args.max_scans,
        implementation=implementation,
        bandwidth=bandwidth,
        mahal_distance=args.mahal_distance,
        redux_kmeans=args.redux_kmeans,
    )


if __name__ == '__main__':
    main()
