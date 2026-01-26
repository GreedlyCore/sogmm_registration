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


def run_dataset(DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS):

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

    for i in tqdm(range(FIRST_SCAN, LAST_SCAN)):

        # Files are named according to matlab convention
        # Therefore index 0 in python corresponds to file name 1
        source_file = DATA_DIR + \
            str(NUM_COMPONENTS) + '_components/' + str(i+2) + '.gmm'
        target_file = DATA_DIR + \
            str(NUM_COMPONENTS) + '_components/' + str(i+1) + '.gmm'

        Tinit = np.eye(4)
        output = gmm_d2d_registration_py.isoplanar_registration(
            Tinit, source_file, target_file)
        ret = gmm_d2d_registration_py.anisotropic_registration(
            output[0], source_file, target_file)
        Tout = ret[0]

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

    with open(RESULTS_DIR + PREFIX + str(NUM_COMPONENTS) + '_results.pkl', 'wb') as handle:
        pickle.dump([transforms, ground_truths, errors], handle)

    return transforms, ground_truths
