#!/usr/bin/env python
"""
TUM RGB-D dataset loading utilities.

Reads pre-converted point cloud .txt files produced by convert_tum_to_sogmm.py.

Format per file: N rows of  x,y,z,intensity  (comma-separated)

Data layout on disk:
  {base_path}/{dataset}/pointclouds/{scan_idx}.txt
"""
import os
import numpy as np


def get_tum_pointclouds_dir(base_path, dataset):
    """Return path to the pointclouds directory for *dataset*.

    Args:
        base_path: TUM data root, e.g. '~/thesis/sogmm_registration/data'
        dataset:   Dataset folder name, e.g. 'rgbd_dataset_freiburg3_long_office_household'

    Returns:
        Absolute path to pointclouds directory.
    """
    return os.path.join(base_path, dataset, 'pointclouds')


def get_tum_scan_files(base_path, dataset):
    """Return a sorted list of .txt file paths for *dataset*.

    Args:
        base_path: TUM data root.
        dataset:   Dataset folder name.

    Returns:
        List[str]: Absolute paths sorted by scan index (filename).

    Raises:
        FileNotFoundError: If pointclouds directory does not exist.
    """
    pcld_dir = get_tum_pointclouds_dir(base_path, dataset)
    if not os.path.isdir(pcld_dir):
        raise FileNotFoundError(f'pointclouds directory not found: {pcld_dir}')
    files = sorted(
        [os.path.join(pcld_dir, f) for f in os.listdir(pcld_dir) if f.endswith('.txt')],
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )
    return files


def get_tum_scan_count(base_path, dataset):
    """Return total number of point cloud scans available for *dataset*."""
    return len(get_tum_scan_files(base_path, dataset))


def load_tum_pointcloud(file_path):
    """Load a single TUM point cloud .txt file.

    Args:
        file_path: Absolute path to a .txt scan file.

    Returns:
        numpy.ndarray: Shape (N, 4), dtype float32, columns [x, y, z, intensity].
    """
    points = np.loadtxt(file_path, delimiter=',').astype(np.float32)
    points[:, 2] *= -1  # invert Z for better visualization (depth camera points down)
    return points
