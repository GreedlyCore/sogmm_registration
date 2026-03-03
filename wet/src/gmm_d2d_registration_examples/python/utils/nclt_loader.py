#!/usr/bin/env python
"""
NCLT dataset loading utilities.

Reads velodyne_sync binary files from the NCLT dataset.

Binary format per point (8 bytes total):
  x: uint16 (little-endian)
  y: uint16 (little-endian)
  z: uint16 (little-endian)
  intensity: uint8
  laser_id:  uint8

Coordinate conversion: coord_m = raw * 0.005 - 100.0  (5 mm resolution, -100 m offset)

Data layout on disk:
  {base_path}/{scene}_vel/{scene}/velodyne_sync/*.bin
  Each .bin file is named by its microsecond timestamp.
"""
import os
import numpy as np

_SCALING = 0.005   # metres per raw unit
_OFFSET  = -100.0  # metres

_POINT_DTYPE = np.dtype([
    ('x',         '<u2'),
    ('y',         '<u2'),
    ('z',         '<u2'),
    ('intensity',  'u1'),
    ('laser_id',   'u1'),
])


def get_nclt_velodyne_dir(base_path, scene):
    """Return path to the velodyne_sync directory for *scene*.

    Args:
        base_path: NCLT dataset root, e.g. '/home/user/data'
        scene:     Date string, e.g. '2013-01-10'

    Returns:
        Absolute path to velodyne_sync directory.
    """
    return os.path.join(base_path, f'{scene}_vel', scene, 'velodyne_sync')


def get_nclt_sync_files(base_path, scene):
    """Return a time-sorted list of .bin file paths for *scene*.

    Args:
        base_path: NCLT dataset root.
        scene:     Scene date string.

    Returns:
        List[str]: Absolute paths sorted by timestamp (filename).

    Raises:
        FileNotFoundError: If velodyne_sync directory does not exist.
    """
    vel_dir = get_nclt_velodyne_dir(base_path, scene)
    if not os.path.isdir(vel_dir):
        raise FileNotFoundError(f'velodyne_sync directory not found: {vel_dir}')
    files = sorted(
        os.path.join(vel_dir, f)
        for f in os.listdir(vel_dir)
        if f.endswith('.bin')
    )
    return files


def get_nclt_scan_count(base_path, scene):
    """Return total number of velodyne scans available for *scene*."""
    return len(get_nclt_sync_files(base_path, scene))


def load_nclt_pointcloud(file_path):
    """Load a single NCLT velodyne_sync .bin file.

    Args:
        file_path: Absolute path to a .bin scan file.

    Returns:
        numpy.ndarray: Shape (N, 4), dtype float32, columns [x, y, z, intensity].
                       Coordinates are in metres.
    """
    with open(file_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=_POINT_DTYPE)

    x = raw['x'].astype(np.float32) * _SCALING + _OFFSET
    y = raw['y'].astype(np.float32) * _SCALING + _OFFSET
    # TODO: because main frame is Z-axis inverted: (proposed paper)
    z = -(raw['z'].astype(np.float32) * _SCALING + _OFFSET)  
    intensity = raw['intensity'].astype(np.float32)

    return np.column_stack([x, y, z, intensity])
