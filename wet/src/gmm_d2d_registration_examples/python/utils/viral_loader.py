#!/usr/bin/env python
"""
VIRAL dataset loading utilities.

GT loaded from NTU VIRAL GT CSV files: https://github.com/ntu-aris/ntuviral_gt
"""
import os
import numpy as np
import yaml
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

#TODO: fix GT angles ????

# Offset from body center to Leica prism (from ntuviral_evaluate.py)
T_BODY_PRISM = np.array([-0.293656, -0.012288, -0.273095])

# Default path to ntuviral_gt repository
DEFAULT_GT_PATH = '/home/sonieth2/thesis/VIRAL/ntuviral_gt'


def _opencv_matrix_constructor(loader, node):
    """Custom YAML constructor for opencv-matrix format."""
    mapping = loader.construct_mapping(node, deep=True)
    rows = mapping.get('rows', 4)
    cols = mapping.get('cols', 4)
    data = mapping.get('data', [])
    return np.array(data, dtype=np.float64).reshape(rows, cols)


def load_viral_lidar_config(bag_path, config_name='lidar_horz.yaml'):
    """
    Load lidar config from YAML file in same directory as bag.

    Args:
        bag_path: Path to .bag file
        config_name: YAML config filename (default: lidar_horz.yaml)

    Returns:
        topic: ROS topic string (or None)
        T_body_lidar: 4x4 numpy transform matrix (or None)
    """
    bag_dir = os.path.dirname(os.path.abspath(bag_path))
    config_path = os.path.join(bag_dir, config_name)

    if not os.path.exists(config_path):
        print(f'WARNING: Lidar config not found: {config_path}')
        return None, None

    # Register opencv-matrix constructor
    yaml.add_constructor('tag:yaml.org,2002:opencv-matrix', _opencv_matrix_constructor,
                         Loader=yaml.FullLoader)

    with open(config_path, 'r') as f:
        content = f.read()
        # Skip %YAML:1.0 header if present
        if content.startswith('%YAML'):
            content = '\n'.join(content.split('\n')[1:])

    config = yaml.full_load(content)

    topic = config.get('pointcloud_topic', None)
    T_body_lidar = config.get('T_Body_Lidar', None)  # Already numpy array from constructor

    if T_body_lidar is not None:
        t = T_body_lidar[:3, 3]
        print(f'  Lidar transform: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]')

    return topic, T_body_lidar


VIRAL_OUSTER_DTYPE = np.dtype([
    ('x', np.float32), ('y', np.float32), ('z', np.float32),
    ('_pad1', np.float32), ('intensity', np.float32), ('t', np.uint32),
    ('reflectivity', np.uint16), ('ring', np.uint8), ('_pad2', np.uint8),
    ('ambient', np.uint16), ('_pad3', np.uint16), ('range', np.uint32),
    ('_pad4', np.float32), ('_pad5', np.float32), ('_pad6', np.float32),
])


# def pointcloud2_to_numpy(msg):
#     """Convert ROS PointCloud2 to Nx4 numpy array (x, y, z, intensity)."""
#     points = np.frombuffer(msg.data, dtype=VIRAL_OUSTER_DTYPE)
#     return np.column_stack([points['x'], points['y'], points['z'], points['intensity']])

#TODO: viral datasets lidar is flipped in z: need to somehow generalize it 
def pointcloud2_to_numpy(msg):
    """Convert ROS PointCloud2 to Nx4 numpy array (x, y, z, intensity)."""
    points = np.frombuffer(msg.data, dtype=VIRAL_OUSTER_DTYPE)
    return np.column_stack([points['x'], points['y'], -points['z'], points['intensity']])


def get_viral_scan_count(bag_path, lidar_topic='/os1_cloud_node1/points'):
    """Return total number of scans in bag file."""
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == lidar_topic]
        if not connections:
            return 0
        return sum(1 for _ in reader.messages(connections=connections))


def load_viral_pointcloud(bag_path, scan_idx, lidar_topic='/os1_cloud_node1/points'):
    """Load single pointcloud from VIRAL rosbag by index. Returns Nx4 array."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == lidar_topic]
        if not connections:
            raise ValueError(f'Topic {lidar_topic} not found in bag')
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            if i == scan_idx:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                return pointcloud2_to_numpy(msg)
    raise IndexError(f'Scan index {scan_idx} out of range')


def apply_transform(points_4d, T):
    """
    Apply 4x4 homogeneous transform to points (xyz only, preserve intensity).

    Args:
        points_4d: Nx4 array (x, y, z, intensity)
        T: 4x4 transformation matrix (or None)

    Returns:
        Transformed Nx4 array
    """
    if T is None:
        return points_4d
    if len(points_4d) == 0:
        return points_4d

    xyz = points_4d[:, :3]
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])
    xyz_transformed = (T @ xyz_h.T).T[:, :3]

    return np.column_stack([xyz_transformed, points_4d[:, 3]])


def quat_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def load_viral_gt_csv(sequence_name, gt_base_path=None):
    """
    Load ground truth from NTU VIRAL CSV file.

    CSV format: %time, seq, stamp, x, y, z, qx, qy, qz, qw

    Args:
        sequence_name: e.g., 'eee_03'
        gt_base_path: Path to ntuviral_gt folder

    Returns:
        timestamps: Nx1 array of timestamps in seconds
        poses: List of 4x4 pose matrices
    """
    if gt_base_path is None:
        gt_base_path = DEFAULT_GT_PATH

    csv_path = os.path.join(gt_base_path, sequence_name, 'ground_truth.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'GT CSV not found: {csv_path}')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    # Use header.stamp (column 2) as timestamp
    timestamps = data[:, 2] / 1e9
    xyz = data[:, 3:6]
    quat = data[:, 6:10]  # qx, qy, qz, qw

    poses = []
    for i in range(len(timestamps)):
        T = np.eye(4)
        T[:3, :3] = quat_to_rotation_matrix(quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3])
        T[:3, 3] = xyz[i]
        poses.append(T)

    print(f'Loaded {len(poses)} GT poses from CSV')
    print(f'  Time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s | delta= {timestamps[-1]-timestamps[0]:.3f}s')

    return timestamps, poses


def load_viral_ground_truth(bag_path, lidar_topic='/os1_cloud_node1/points',
                            skip_scans=1, first_scan=0, max_scans=None,
                            gt_base_path=None, max_time_diff=0.05):
    """
    Load ground truth poses from NTU VIRAL CSV, aligned with lidar scans from bag.

    Args:
        bag_path: Path to .bag file (for lidar timestamps)
        lidar_topic: LiDAR topic to get scan timestamps
        skip_scans: Process every Nth scan (must match GMM generation)
        first_scan: First scan index to start from (must match GMM generation)
        max_scans: Maximum number of scans (must match GMM generation)
        gt_base_path: Path to ntuviral_gt folder
        max_time_diff: Maximum time difference for matching (seconds)

    Returns:
        poses: List of 4x4 numpy arrays (world poses for each scan)
        timestamps: List of timestamps for each scan (seconds)
    """
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f'Bag file not found: {bag_path}')

    # Extract sequence name from bag path (e.g., eee_03)
    bag_name = os.path.basename(bag_path).replace('.bag', '')
    sequence_name = bag_name

    # Load GT from CSV
    gt_timestamps, gt_poses = load_viral_gt_csv(sequence_name, gt_base_path)

    # Get lidar timestamps from bag
    typestore = get_typestore(Stores.ROS1_NOETIC)
    lidar_timestamps = []
    scan_idx = 0
    saved_count = 0

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == lidar_topic]
        if not connections:
            raise ValueError(f'Topic {lidar_topic} not found in bag')

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if scan_idx < first_scan:
                scan_idx += 1
                continue
            if (scan_idx - first_scan) % skip_scans != 0:
                scan_idx += 1
                continue
            if max_scans is not None and saved_count >= max_scans:
                break

            lidar_timestamps.append(timestamp / 1e9)  # Convert to seconds
            saved_count += 1
            scan_idx += 1

    print(f'Found {len(lidar_timestamps)} lidar scans')
    print(f'  Lidar time range: {lidar_timestamps[0]:.3f}s to {lidar_timestamps[-1]:.3f}s')

    # Match lidar timestamps to GT (with time validity check)
    gt_timestamps = np.array(gt_timestamps)
    aligned_poses = []
    aligned_timestamps = []
    skipped = 0

    for lidar_ts in lidar_timestamps:
        time_diffs = np.abs(gt_timestamps - lidar_ts)
        min_idx = np.argmin(time_diffs)
        min_diff = time_diffs[min_idx]

        if min_diff > max_time_diff:
            skipped += 1
            # Use identity for scans outside GT coverage
            aligned_poses.append(np.eye(4))
        else:
            aligned_poses.append(gt_poses[min_idx])

        aligned_timestamps.append(lidar_ts)

    if skipped > 0:
        print(f'  WARNING: {skipped} scans outside GT coverage (>{max_time_diff}s)')

    print(f'Aligned {len(aligned_poses)} poses to lidar scans')

    return aligned_poses, aligned_timestamps
