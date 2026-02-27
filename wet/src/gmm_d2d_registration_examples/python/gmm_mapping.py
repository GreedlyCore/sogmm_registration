#!/usr/bin/env python
"""
Mapping with GTSAM pose graph (do not tested, deprecated) #TODO: remove?
"""
import os
import numpy as np
import argparse
import pickle
import json
from tqdm import tqdm

from utils.slam.PoseGraph import PoseGraph
from utils.gmm_transform import load_gmm_file, transform_gmm_dict, merge_gmms
from utils.ZYXToR import ZYXToR
from utils.pose_compose import pose_compose


def dpose_to_matrix(dpose):
    """
    Convert 6-DOF pose (tx, ty, tz, roll, pitch, yaw) to 4x4 matrix.

    Args:
        dpose: (6,) array [tx, ty, tz, roll, pitch, yaw]

    Returns:
        T: (4, 4) SE(3) transformation matrix
    """
    T = np.eye(4)
    T[:3, 3] = dpose[:3]
    T[:3, :3] = ZYXToR(dpose[3:])
    return T


def load_registration_results(results_file):
    """
    Load registration results from pickle file.

    Args:
        results_file: Path to .pkl file from run_kitti_dataset.py

    Returns:
        transforms: (N, 6) relative transforms
        ground_truths: (N, 6) ground truth transforms
    """
    with open(results_file, 'rb') as f:
        transforms, ground_truths, errors = pickle.load(f)
    return transforms, ground_truths


def build_pose_graph(transforms):
    """
    Build GTSAM pose graph from relative transforms.

    Args:
        transforms: (N, 6) array of relative transforms [tx, ty, tz, roll, pitch, yaw]

    Returns:
        pose_graph: PoseGraph object with optimized poses
        world_poses: list of (4, 4) world pose matrices
    """
    pg = PoseGraph()
    world_poses = [np.eye(4)]  # First pose at origin

    for i, dpose in enumerate(tqdm(transforms, desc="Building pose graph")):
        # Convert 6-DOF to matrix
        T_rel = dpose_to_matrix(dpose)

        # Update odometry incrementally
        pg.update_odometry_incrementally(T_rel)

        # Create between factor
        pg.create_between_factor()
        pg.increment_key()

        # Get current world pose from GTSAM
        T_world = pg.values.atPose3(pg.key - 1).matrix()
        world_poses.append(T_world)

    return pg, world_poses


def build_gmm_map(gmm_dir, world_poses, first_scan=0, last_scan=None, subsample=1):
    """
    Transform GMMs to world frame and build map.

    Args:
        gmm_dir: Directory containing .gmm files
        world_poses: list of (4, 4) world pose matrices
        first_scan: First scan index
        last_scan: Last scan index (None = all)
        subsample: Only include every Nth scan in map (default: 1 = all)

    Returns:
        map_data: dict with combined GMM data and trajectory
    """
    n_poses = len(world_poses)
    if last_scan is None:
        last_scan = n_poses

    world_gmms = []
    trajectory = []
    scan_indices = []

    for i in tqdm(range(first_scan, min(last_scan, n_poses)), desc="Transforming GMMs"):
        # GMM files use 1-based indexing
        gmm_file = os.path.join(gmm_dir, f'{i + 1}.gmm')

        if not os.path.exists(gmm_file):
            print(f"Warning: GMM file not found: {gmm_file}")
            continue

        T_world = world_poses[i]
        trajectory.append(T_world[:3, 3].copy())

        # Only include GMM in map if subsample allows
        if i % subsample == 0:
            gmm = load_gmm_file(gmm_file)
            gmm_world = transform_gmm_dict(gmm, T_world)
            gmm_world['scan_index'] = i
            world_gmms.append(gmm_world)
            scan_indices.append(i)

    # Merge all GMMs
    if world_gmms:
        merged_gmm = merge_gmms(world_gmms)
    else:
        merged_gmm = None

    return {
        'gmm': merged_gmm,
        'trajectory': np.array(trajectory),
        'world_poses': world_poses,
        'scan_indices': scan_indices,
        'first_scan': first_scan,
        'last_scan': last_scan
    }


def save_map_data(map_data, output_file):
    """Save map data to pickle file."""
    with open(output_file, 'wb') as f:
        pickle.dump(map_data, f)
    print(f"Map data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Build GMM map from registration results using GTSAM pose graph'
    )

    parser.add_argument('--sequence', type=str, required=True,
                       help='KITTI sequence number (e.g., 00, 01)')
    parser.add_argument('--results_file', type=str, default=None,
                       help='Path to registration results .pkl file. '
                            'If not provided, searches in kitti_sequence_{seq}/results/')
    parser.add_argument('--gmm_dir', type=str, default=None,
                       help='Path to GMM directory. If not provided, uses most recent.')
    parser.add_argument('--bw', type=float, default=None,
                       help='Bandwidth value to find adaptive GMM folder')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp suffix for GMM folder')
    parser.add_argument('--first_scan', type=int, default=0,
                       help='First scan index (default: 0)')
    parser.add_argument('--last_scan', type=int, default=None,
                       help='Last scan index (default: all)')
    parser.add_argument('--subsample', type=int, default=1,
                       help='Include every Nth scan in map (default: 1 = all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output map file path')
    parser.add_argument('--visualize', action='store_true',
                       help='Launch visualization after building map')

    args = parser.parse_args()

    # Find paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sequence_dir = os.path.join(script_dir, 'runs', f'kitti_sequence_{args.sequence}')
    results_dir = os.path.join(sequence_dir, 'results')

    # Find results file
    if args.results_file is None:
        import glob
        pattern = os.path.join(results_dir, 'isoplanarhybrid_*_results.pkl')
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not candidates:
            print(f"No results files found in {results_dir}")
            print("Run run_kitti_dataset.py first to generate registration results.")
            return
        args.results_file = candidates[0]
        print(f"Using results file: {args.results_file}")

    # Find GMM directory
    if args.gmm_dir is None:
        import glob
        if args.bw is not None:
            bw_value = int(args.bw * 100)
            pattern = f'adaptive_bw{bw_value}_components'
        else:
            pattern = '*_components'

        if args.timestamp:
            search_pattern = os.path.join(sequence_dir, f'{pattern}_{args.timestamp}')
        else:
            search_pattern = os.path.join(sequence_dir, f'{pattern}_*')

        candidates = sorted(glob.glob(search_pattern), key=os.path.getmtime, reverse=True)
        if not candidates:
            print(f"No GMM directories found matching {search_pattern}")
            return
        args.gmm_dir = candidates[0]
        print(f"Using GMM directory: {args.gmm_dir}")

    # Load registration results
    print(f"\nLoading registration results from: {args.results_file}")
    transforms, ground_truths = load_registration_results(args.results_file)
    print(f"Loaded {len(transforms)} relative transforms")

    # Build pose graph
    print("\nBuilding GTSAM pose graph...")
    pose_graph, world_poses = build_pose_graph(transforms)
    print(f"Pose graph has {pose_graph.key - 1} poses")

    # Build GMM map
    print("\nBuilding GMM map...")
    map_data = build_gmm_map(
        args.gmm_dir,
        world_poses,
        first_scan=args.first_scan,
        last_scan=args.last_scan,
        subsample=args.subsample
    )

    if map_data['gmm'] is not None:
        print(f"Map contains {map_data['gmm']['n_components']} GMM components")
        print(f"Trajectory has {len(map_data['trajectory'])} poses")

    # Save map data
    if args.output is None:
        args.output = os.path.join(
            sequence_dir,
            f'gmm_map_seq{args.sequence}_{args.first_scan}to{args.last_scan or len(world_poses)}.pkl'
        )

    save_map_data(map_data, args.output)

    # Visualize
    if args.visualize:
        print("\nLaunching visualization...")
        os.system(f'python visualize_gmm_map.py --map_file "{args.output}"')


if __name__ == '__main__':
    main()
