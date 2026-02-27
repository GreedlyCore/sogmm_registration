#!/usr/bin/env python
import copy
import numpy as np
import os
from os.path import expanduser
import pickle
from tqdm import tqdm

import gmm_d2d_registration_py
from utils.RToZYX import RToZYX
from utils.QuatToR import QuatToR
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse


def _gmm_center_of_mass(gmm_file):
    data = np.loadtxt(gmm_file, delimiter=',')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    means   = data[:, 0:3]   # (N, 3)
    weights = data[:, 12]    # (N,)
    weights /= weights.sum()
    return (weights[:, np.newaxis] * means).sum(axis=0)


def _gmm_pca_init(source_file, target_file):
    def _load(f):
        data = np.loadtxt(f, delimiter=',')
        if data.ndim == 1:
            data = data[np.newaxis, :]
        means   = data[:, 0:3]
        weights = data[:, 12]
        weights /= weights.sum()
        return means, weights

    def _weighted_pca(means, w):
        com = (w[:, np.newaxis] * means).sum(axis=0)
        centered = means - com
        cov = (centered * w[:, np.newaxis]).T @ centered
        _, _, Vt = np.linalg.svd(cov)
        return com, Vt.T  # columns = principal axes

    src_means, src_w = _load(source_file)
    tgt_means, tgt_w = _load(target_file)

    src_com, src_axes = _weighted_pca(src_means, src_w)
    tgt_com, tgt_axes = _weighted_pca(tgt_means, tgt_w)

    R = tgt_axes @ src_axes.T
    if np.linalg.det(R) < 0:  # fix reflections
        tgt_axes[:, -1] *= -1
        R = tgt_axes @ src_axes.T

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3]   = tgt_com - R @ src_com
    return T


def run_dataset(DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS, GMM_DIR=None, init_method='com'):

    cwd = os.getcwd()
    SANDBOX_NAME = 'gira3d-registration'
    matches = cwd.split(SANDBOX_NAME)
    GIRA3D_REGISTRATION_SANDBOX = matches[0] + SANDBOX_NAME
    DATA_DIR = GIRA3D_REGISTRATION_SANDBOX + '/data/' + DATASET + '/'
    RESULTS_DIR = DATA_DIR + 'results/'
    errors = np.zeros(((LAST_SCAN-FIRST_SCAN), 6))
    transforms = np.zeros(((LAST_SCAN-FIRST_SCAN), 6))
    ground_truths = np.zeros(((LAST_SCAN-FIRST_SCAN), 6))
    times = []

    # Check if results directory exists. Create if it does not
    if not os.path.isdir(RESULTS_DIR):
        print('Creating directory at ' + RESULTS_DIR)
        os.mkdir(RESULTS_DIR)

    K, Tbc, odometry, pointclouds = pickle.load(
        open(DATA_DIR + 'odometry.pkl', 'rb'))

    # GMM directory: custom path or default convention
    if GMM_DIR is None:
        GMM_DIR = os.path.join(DATA_DIR, str(NUM_COMPONENTS) + '_components')

    for i in tqdm(range(FIRST_SCAN, LAST_SCAN)):

        # Files are named according to matlab convention
        # Therefore index 0 in python corresponds to file name 1
        source_file = os.path.join(GMM_DIR, str(i+2) + '.gmm')
        target_file = os.path.join(GMM_DIR, str(i+1) + '.gmm')

        # Initial transform: COM translation or PCA-based alignment
        if init_method == 'pca':
            Tinit = _gmm_pca_init(source_file, target_file)
        else:  # 'com' — default
            src_com = _gmm_center_of_mass(source_file)
            tgt_com = _gmm_center_of_mass(target_file)
            Tinit = np.eye(4)
            Tinit[0:3, 3] = tgt_com - src_com
        output = gmm_d2d_registration_py.isoplanar_registration(                                                                                   
            Tinit, source_file, target_file, radius=0.05, min_delta=1e-5)                                                                       
        ret = gmm_d2d_registration_py.anisotropic_registration(   
            output[0], source_file, target_file, radius=0.1, min_delta=1e-7)
        Tout = ret[0]
        
        # print(ret[0])

        Rotation = Tout[0:3, 0:3]
        translation = np.transpose(Tout[0:3, 3])

        dpose = np.concatenate([translation, RToZYX(Rotation)])

        _, idx = odometry.closest_time(pointclouds.times[i])
        Twb1 = np.eye(4)
        Twb1[0:3, 0:3] = QuatToR(odometry.orientations[:, idx])
        Twb1[0:3, 3] = odometry.positions[:, idx]

        Twb2 = np.eye(4)
        _, idx = odometry.closest_time(pointclouds.times[i+1])
        Twb2[0:3, 0:3] = QuatToR(odometry.orientations[:, idx])
        Twb2[0:3, 3] = odometry.positions[:, idx]

        Twc1 = pose_compose(Twb1, Tbc)
        Twc2 = pose_compose(Twb2, Tbc)

        Tc1c2 = pose_compose(pose_inverse(Twc1), Twc2)

        dground_truth = np.array([np.array(Tc1c2[0:3, 3]), np.array(RToZYX(
            Tc1c2[0:3, 0:3]))]).flatten()

        transforms[i-FIRST_SCAN, :] = dpose
        ground_truths[i-FIRST_SCAN, :] = dground_truth
        errors[i-FIRST_SCAN, :] = dpose - dground_truth

    gmm_label = os.path.basename(os.path.normpath(GMM_DIR))
    with open(RESULTS_DIR + PREFIX + gmm_label + '_results.pkl', 'wb') as handle:
        pickle.dump([transforms, ground_truths, errors], handle)

    return transforms, ground_truths
