#!/usr/bin/env python
"""
Debug script to check coordinate transformations and indexing.
"""
import os
import numpy as np
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse
from utils.RToZYX import RToZYX


def load_kitti_poses(pose_file):
    """Load KITTI ground truth poses from file."""
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
    """Load KITTI calibration file."""
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip():
                key, value = line.split(':', 1)
                calib[key] = np.fromstring(value, dtype=np.float64, sep=' ')

    Tr = calib['Tr'].reshape(3, 4)
    Tr_full = np.eye(4)
    Tr_full[0:3, :] = Tr
    return Tr_full

# Load calibration and first few poses
kitti_dir = './dataset/kitti'
sequence = '00'

calib_file = os.path.join(kitti_dir, 'data_odometry_calib',
                         'dataset', 'sequences', sequence, 'calib.txt')
pose_file = os.path.join(kitti_dir, 'data_odometry_poses', f'{sequence}.txt')

Tvc = load_kitti_calib(calib_file)  # Velodyne to camera
Tcv = pose_inverse(Tvc)  # Camera to velodyne
poses_cam = load_kitti_poses(pose_file)  # World to camera poses

print("=" * 60)
print("KITTI Coordinate Frame Analysis")
print("=" * 60)

print("\nCalibration Matrix Tr (Velodyne to Camera 0):")
print(Tvc)

print("\nRotation part of Tr:")
print(Tvc[0:3, 0:3])

print("\n" + "=" * 60)
print("GMM File Indexing Analysis")
print("=" * 60)

# Check what the GMM registration is actually computing
print("\nFor loop iteration i=0:")
print("  source_file = f'{i+2}.gmm' = 2.gmm (corresponds to scan 1, bin file 000001.bin)")
print("  target_file = f'{i+1}.gmm' = 1.gmm (corresponds to scan 0, bin file 000000.bin)")
print("\n  GMM registration output: T that transforms source (scan 1) to target (scan 0)")
print("  This is: T_0_1 (from scan 1 frame to scan 0 frame)")

print("\n  Ground truth calculation:")
print("    poses_cam[i] = poses_cam[0] (world pose of scan 0)")
print("    poses_cam[i+1] = poses_cam[1] (world pose of scan 1)")
print("    Tv1_v2 = inv(Tw_v0) @ Tw_v1")
print("  This is: T_0_1 (from scan 0 frame to scan 1 frame)")

print("\n  ** MISMATCH DETECTED **")
print("  GMM gives: scan 1 → scan 0")
print("  GT gives:  scan 0 → scan 1")
print("  These are INVERSES of each other!")

print("\n" + "=" * 60)
print("Ground Truth Transformation (scan 0 to scan 1)")
print("=" * 60)

# Compute first ground truth transformation
Tw_c0 = poses_cam[0]
Tw_c1 = poses_cam[1]

print("\nWorld to Camera 0:")
print(Tw_c0)

print("\nWorld to Camera 1:")
print(Tw_c1)

# Convert to velodyne frame
Tw_v0 = pose_compose(Tw_c0, Tcv)
Tw_v1 = pose_compose(Tw_c1, Tcv)

print("\nWorld to Velodyne 0:")
print(Tw_v0)

print("\nWorld to Velodyne 1:")
print(Tw_v1)

# Compute relative transformation
Tv0_v1 = pose_compose(pose_inverse(Tw_v0), Tw_v1)

print("\nRelative transformation (Velodyne 0 to Velodyne 1):")
print(Tv0_v1)

rotation_gt = Tv0_v1[0:3, 0:3]
translation_gt = Tv0_v1[0:3, 3]
print(f"\nTranslation: {translation_gt}")
print(f"Rotation (ZYX): {RToZYX(rotation_gt)}")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("Either:")
print("1. Invert the GMM registration output before using it")
print("2. OR swap the source and target files")
print("3. OR invert the ground truth transformation")
