#!/usr/bin/env python
import numpy as np


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
