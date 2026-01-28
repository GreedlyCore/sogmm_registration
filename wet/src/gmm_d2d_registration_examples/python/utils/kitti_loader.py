#!/usr/bin/env python
"""
KITTI dataset loading utilities.
"""
import os
import numpy as np


def load_kitti_velodyne(filepath):
    """
    Load KITTI velodyne point cloud from .bin file.

    Args:
        filepath: Path to .bin file

    Returns:
        Nx4 array (x, y, z, reflectance)
    """
    return np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)


def load_kitti_poses(pose_file):

    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T_full = np.eye(4)
            T_full[0:3, :] = T
            poses.append(T_full)
    return poses


def load_kitti_calib(calib_file):

    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip():
                key, value = line.split(':', 1)
                calib[key] = np.fromstring(value, dtype=np.float64, sep=' ')

    # Tr is the transformation from velodyne to camera 0
    Tr = calib['Tr'].reshape(3, 4)
    Tr_full = np.eye(4)
    Tr_full[0:3, :] = Tr

    return Tr_full


def load_kitti_ground_truth(sequence, kitti_dir, start_idx=0, end_idx=None):
    """
    Load KITTI ground truth poses in velodyne frame.

    Args:
        sequence: KITTI sequence number (e.g., '04')
        kitti_dir: Root directory of KITTI dataset
        start_idx: First scan index
        end_idx: Last scan index (exclusive)

    Returns:
        poses: List of 4x4 numpy arrays (velodyne frame poses)
        Tr: Velodyne to camera calibration matrix
    """
    kitti_dir = os.path.expanduser(kitti_dir)

    pose_file = os.path.join(kitti_dir, 'data_odometry_poses', f'{sequence}.txt')
    calib_file = os.path.join(kitti_dir, 'data_odometry_calib',
                              'dataset', 'sequences', sequence, 'calib.txt')

    if not os.path.exists(pose_file):
        raise FileNotFoundError(f'Pose file not found: {pose_file}')
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f'Calibration file not found: {calib_file}')

    # Load camera poses and calibration
    poses_cam = load_kitti_poses(pose_file)
    Tr = load_kitti_calib(calib_file)

    # Slice poses if needed
    if end_idx is not None:
        poses_cam = poses_cam[start_idx:end_idx + 1]
    else:
        poses_cam = poses_cam[start_idx:]

    print(f'Loaded {len(poses_cam)} KITTI poses')

    return poses_cam, Tr
