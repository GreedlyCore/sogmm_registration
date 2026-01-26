#!/usr/bin/env python
"""
Script to convert VIRAL rosbag point clouds to GMM format.
Uses rosbags library to read ROS1 bags directly (no ROS environment needed).
Supports voxel filtering/radius filtering to reduce point count.
"""
import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
from pathlib import Path

#TODO: eee_03/lidar_vert.yaml or nya_01/lidar_vert.yaml --> add transform tf usage here 

# ROS bag reading --? #TODO: double check
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr
#pip install rosbags
    

from utils.save_gmm import save_sogmm
from sogmm_py import SOGMM
from gmm_py import GMMf4CPU
from kinit_py import KInitf4CPU


def pointcloud2_to_numpy(msg):
    """
    Convert ROS PointCloud2 message to numpy array.

    Args:
        msg: PointCloud2 message

    Returns:
        points: Nx4 array (x, y, z, intensity)
    """
    # Get point step and field offsets
    point_step = msg.point_step
    row_step = msg.row_step
    data = msg.data

    # VIRAL Ouster OS1 format: x, y, z, intensity, t, reflectivity, ring, ambient, range
    # We want x, y, z, intensity (first 4 fields)
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])

    # Parse binary data
    points = np.frombuffer(data, dtype=dtype)

    # Convert to Nx4 array
    return np.column_stack([points['x'], points['y'], points['z'], points['intensity']])


def voxel_filter(points, voxel_size=0.1):
    """
    Apply voxel grid to downsample point cloud.

    Args:
        points: Nx3 or Nx4 array of points
        voxel_size: Size of voxel grid (in meters)

    Returns:
        filtered_points: Downsampled point cloud
    """
    xyz = points[:, :3]

    # Compute voxel indices
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    # Get unique voxels and their first occurrence
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    filtered_points = points[unique_indices]

    print(f'Voxel filtering: {len(points)} -> {len(filtered_points)} points '
          f'({100.0 * len(filtered_points) / len(points):.1f}%)')

    return filtered_points


def radius_filter(points, radius=30.0):
    """
    Remove points farther than specified radius from origin.

    Args:
        points: Nx3 or Nx4 array of points
        radius: Maximum allowed distance (in meters)

    Returns:
        filtered_points: Points within radius
    """
    xyz = points[:, :3]
    distances = np.linalg.norm(xyz, axis=1)
    mask = distances < radius

    filtered_points = points[mask]

    print(f'Radius filtering (<{radius}m): {len(points)} -> {len(filtered_points)} points '
          f'({100.0 * len(filtered_points) / len(points):.1f}%)')

    return filtered_points


def convert_viral_to_gmm(bag_path, output_dir, n_components=None,
                        lidar_topic='/os1_cloud_node/points',
                        use_voxel_filter=False, voxel_size=0.1,
                        use_radius_filter=False, radius=30.0,
                        max_scans=None, skip_scans=10,
                        implementation='fsogmm', bandwidth=None):
    """
    Convert VIRAL rosbag point clouds to GMM format.

    Args:
        bag_path: Path to .bag file
        output_dir: Directory to save GMM files
        n_components: Number of Gaussian components (for fsogmm)
        lidar_topic: ROS topic name for point clouds
        use_voxel_filter: Whether to apply voxel filtering
        voxel_size: Voxel size for filtering (in meters)
        use_radius_filter: Whether to remove points beyond radius
        radius: Maximum distance from origin (in meters)
        max_scans: Maximum number of scans to process (None = all)
        skip_scans: Process every Nth scan (for speed)
        implementation: GMM implementation ('fsogmm' or 'sogmm')
        bandwidth: Bandwidth for SOGMM
    """
    # Validate parameters
    if implementation == 'fsogmm' and n_components is None:
        raise ValueError("fsogmm requires n_components parameter")
    if implementation == 'sogmm' and bandwidth is None:
        raise ValueError("sogmm requires bandwidth parameter")

    if not os.path.exists(bag_path):
        print(f'ERROR: Bag file not found: {bag_path}')
        sys.exit(1)

    print(f'Reading bag file: {bag_path}')
    print(f'LiDAR topic: {lidar_topic}')
    print(f'Converting scans to GMM format...')

    if implementation == 'sogmm':
        print(f'Implementation: {implementation} (adaptive)')
        print(f'Bandwidth: {bandwidth}')
    else:
        print(f'Implementation: {implementation}')
        print(f'Components: {n_components}')

    print(f'Voxel filtering: {use_voxel_filter} (voxel_size={voxel_size}m)')
    print(f'Radius filtering: {use_radius_filter} (radius={radius}m)')
    print(f'Skip scans: {skip_scans} (process every {skip_scans}th scan)')

    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%d%m%H%M")

    if implementation == 'sogmm':
        gmm_output_dir = os.path.join(output_dir, f"adaptive_bw{int(bandwidth*100)}_components_{timestamp}")
    else:
        gmm_output_dir = os.path.join(output_dir, f'{n_components}_components_{timestamp}')
    os.makedirs(gmm_output_dir, exist_ok=True)

    # Write metadata file
    bag_name = Path(bag_path).stem
    meta_path = os.path.join(gmm_output_dir, 'meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"# GMM Generation Metadata (VIRAL Dataset)\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"dataset: VIRAL\n")
        f.write(f"bag_file: {bag_name}\n")
        f.write(f"lidar_topic: {lidar_topic}\n")
        f.write(f"implementation: {implementation}\n")
        if implementation == 'sogmm':
            f.write(f"bandwidth: {bandwidth}\n")
        else:
            f.write(f"n_components: {n_components}\n")
        f.write(f"\n# Filtering parameters\n")
        f.write(f"voxel_filter: {use_voxel_filter}\n")
        f.write(f"voxel_size: {voxel_size}\n")
        f.write(f"radius_filter: {use_radius_filter}\n")
        f.write(f"radius: {radius}\n")
        f.write(f"skip_scans: {skip_scans}\n")

    # Read bag file
    scan_idx = 0
    saved_count = 0

    with Reader(bag_path) as reader:
        # Get connections for the lidar topic
        connections = [c for c in reader.connections if c.topic == lidar_topic]

        if not connections:
            print(f'ERROR: Topic {lidar_topic} not found in bag file')
            print('Available topics:')
            for c in reader.connections:
                print(f'  {c.topic} ({c.msgtype})')
            sys.exit(1)

        print(f'\nProcessing point clouds from {lidar_topic}...')

        for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections)):
            # Skip scans for faster processing
            if scan_idx % skip_scans != 0:
                scan_idx += 1
                continue

            # Stop if max_scans reached
            if max_scans is not None and saved_count >= max_scans:
                break

            # Deserialize message
            msg = deserialize_cdr(rawdata, connection.msgtype)

            # Convert to numpy
            points_4d = pointcloud2_to_numpy(msg)

            # Apply filtering
            if use_radius_filter:
                points_4d = radius_filter(points_4d, radius)
            if use_voxel_filter:
                points_4d = voxel_filter(points_4d, voxel_size)

            # Remove points too close (sensor itself)
            distances = np.linalg.norm(points_4d[:, :3], axis=1)
            points_4d = points_4d[distances > 1.0]

            if len(points_4d) < 100:
                print(f'\nWARNING: Scan {scan_idx} has only {len(points_4d)} points, skipping')
                scan_idx += 1
                continue

            # Warn if too few points for fixed component GMM
            if implementation == 'fsogmm' and len(points_4d) < n_components * 3:
                print(f'\nWARNING: Scan {scan_idx} has only {len(points_4d)} points, '
                      f'might be too few for {n_components} components')

            # Fit GMM
            if implementation == 'fsogmm':
                # Fixed component SOGMM
                n_samples = points_4d.shape[0]
                kinit = KInitf4CPU()
                _, indices = kinit.resp_calc(points_4d, n_components)
                resp = np.zeros((n_samples, n_components), dtype=np.float32)
                resp[indices, np.arange(n_components)] = 1

                stats_dir = os.path.join(gmm_output_dir, 'profiling')
                os.makedirs(stats_dir, exist_ok=True)
                stats_file = f'gmm_stats_scan_{saved_count:06d}.csv'

                local_model = GMMf4CPU(n_components, True, stats_dir, stats_file)
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

                # CPU SOGMM implementation
                sg = SOGMM(bandwidth, save_stats=True, stats_dir=stats_dir, stats_file_prefix=stats_file_prefix)
                local_model = sg.fit(points_4d)
                print(f"SOGMM fitted with {local_model.n_components_} components...")

                gmm_4d = local_model

            # Save GMM
            output_path = os.path.join(gmm_output_dir, f'{saved_count}.gmm')
            save_sogmm(output_path, gmm_4d)

            saved_count += 1
            scan_idx += 1

    print(f'\n✓ Saved {saved_count} GMM files to: {gmm_output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert VIRAL rosbag point clouds to GMM format')

    parser.add_argument('--bag', type=str, required=True,
                        help='Path to .bag file (e.g., dataset/viral/eee_03/eee_03.bag)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: ./viral_{bag_name})')
    parser.add_argument('--lidar_topic', type=str,
                        default='/os1_cloud_node/points',
                        help='ROS topic for LiDAR point clouds (default: /os1_cloud_node/points)')
    parser.add_argument('--n_components', type=int, default=None,
                        help='Number of GMM components for fsogmm (fixed components)')
    parser.add_argument('--bandwidth', type=float, default=None,
                        help='Bandwidth for sogmm (adaptive components)')
    parser.add_argument('--voxel_filter', action='store_true',
                        help='Enable voxel filtering to reduce point count')
    parser.add_argument('--voxel_size', type=float, default=0.2,
                        help='Voxel size in meters (default: 0.2)')
    parser.add_argument('--radius_filter', action='store_true',
                        help='Enable radius filtering to remove distant points')
    parser.add_argument('--radius', type=float, default=30.0,
                        help='Maximum distance from origin in meters (default: 30.0)')
    parser.add_argument('--max_scans', type=int, default=None,
                        help='Maximum number of scans to process (default: all)')
    parser.add_argument('--skip_scans', type=int, default=10,
                        help='Process every Nth scan (default: 10)')

    args = parser.parse_args()

    # Auto-detect implementation based on provided parameters
    if args.bandwidth is not None and args.n_components is not None:
        print('ERROR: Specify either --bandwidth (for sogmm) OR --n_components (for fsogmm), not both')
        sys.exit(1)
    elif args.bandwidth is not None:
        args.implementation = 'sogmm'
        print(f'Using SOGMM (adaptive) with bandwidth={args.bandwidth}')
    elif args.n_components is not None:
        args.implementation = 'fsogmm'
        print(f'Using fsogmm (fixed) with n_components={args.n_components}')
    else:
        print('ERROR: Must specify either --bandwidth (for sogmm) OR --n_components (for fsogmm)')
        sys.exit(1)

    # Expand paths
    args.bag = os.path.expanduser(args.bag)

    # Set output directory
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bag_name = Path(args.bag).stem
        args.output_dir = os.path.join(script_dir, f'viral_{bag_name}')

    print(f'Bag file: {args.bag}')
    print(f'Output dir: {args.output_dir}')
    print()

    convert_viral_to_gmm(
        bag_path=args.bag,
        output_dir=args.output_dir,
        n_components=args.n_components,
        lidar_topic=args.lidar_topic,
        use_voxel_filter=args.voxel_filter,
        voxel_size=args.voxel_size,
        use_radius_filter=args.radius_filter,
        radius=args.radius,
        max_scans=args.max_scans,
        skip_scans=args.skip_scans,
        implementation=args.implementation,
        bandwidth=args.bandwidth
    )


if __name__ == '__main__':
    main()
