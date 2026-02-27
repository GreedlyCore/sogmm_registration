#!/usr/bin/env python
"""
Utilities for working with GMM folders.
"""
import os
import re
import glob
import yaml

def parse_meta_file(gmm_folder):
    """
    Parse meta.yaml from GMM folder to extract dataset configuration.

    Returns:
        dict with keys: dataset, sequence (kitti), bag_file (viral), skip_scans (viral)
    """
    meta_path = os.path.join(gmm_folder, 'meta.yaml')
    config = {}

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Infer dataset from parent folder if not in meta.yaml
    if 'dataset' not in config:
        parent_folder = os.path.basename(os.path.dirname(gmm_folder))
        if parent_folder.startswith('kitti_'):
            config['dataset'] = 'KITTI'
            parts = parent_folder.split('_')
            if len(parts) >= 3:
                config['sequence'] = parts[-1]
        elif parent_folder.startswith('viral_'):
            config['dataset'] = 'VIRAL'
            config['bag_file'] = '_'.join(parent_folder.split('_')[1:])

    return config


def detect_gmm_range(gmm_folder):
    """
    Scan folder for .gmm files and detect first/last scan numbers and step.

    Args:
        gmm_folder: Path to folder containing .gmm files

    Returns:
        tuple: (first_scan, last_scan, skip_scans) or (None, None, None) if no files
    """
    gmm_files = glob.glob(os.path.join(gmm_folder, '*.gmm'))

    if not gmm_files:
        return None, None, None

    # Extract numeric part from filenames (e.g., "200.gmm" -> 200)
    scan_numbers = []
    for f in gmm_files:
        basename = os.path.basename(f)
        match = re.match(r'^(\d+)\.gmm$', basename)
        if match:
            scan_numbers.append(int(match.group(1)))

    if not scan_numbers:
        return None, None, None

    scan_numbers.sort()

    first_scan = scan_numbers[0]
    last_scan = scan_numbers[-1]

    # Detect skip_scans (step between consecutive files)
    if len(scan_numbers) >= 2:
        # Use the most common difference as skip_scans
        diffs = [scan_numbers[i+1] - scan_numbers[i] for i in range(len(scan_numbers)-1)]
        skip_scans = max(set(diffs), key=diffs.count)  # most frequent diff
    else:
        skip_scans = 1

    return first_scan, last_scan, skip_scans
