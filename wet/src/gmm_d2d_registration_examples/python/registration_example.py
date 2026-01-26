#!/usr/bin/env python
import copy
import numpy as np
import os
from os.path import expanduser
import open3d
#from sogmm_py.utils import np_to_open3d

import gmm_d2d_registration_py
from utils.RToZYX import RToZYX
from utils.open3d_visualizer import Open3DVisualizer

def load_np_from_txt(txt_file):
    data = np.loadtxt(txt_file, delimiter=',')
    return data

def main():

    cwd = os.getcwd()
    SANDBOX_NAME = 'gira3d-registration'
    matches = cwd.split(SANDBOX_NAME)
    GIRA3D_REGISTRATION_SANDBOX = matches[0] + SANDBOX_NAME
    DATA_DIR = GIRA3D_REGISTRATION_SANDBOX + '/data/rgbd_dataset_freiburg3_long_office_household/100_components/'
    PCLD_DIR = GIRA3D_REGISTRATION_SANDBOX + '/data/rgbd_dataset_freiburg3_long_office_household/pointclouds/'
    RESULTS_DIR = DATA_DIR + 'results/'

    source_idx = 1260
    target_idx = 1200

    source_file = DATA_DIR + str(source_idx) + '.gmm'
    target_file = DATA_DIR + str(target_idx) + '.gmm'

    Tinit = np.eye(4)
    output = gmm_d2d_registration_py.isoplanar_registration(Tinit, source_file, target_file)
    ret = gmm_d2d_registration_py.anisotropic_registration(output[0], source_file, target_file)
    Tout = ret[0]
    translation = (Tout[0:3,3]).flatten()
    correct_translation = np.array([-0.4217, -0.2275, 0.4124])
    assert(np.max(np.abs(translation-correct_translation)) < 1e-3)

    correct_zyx = np.array([0.1211, 0.6879, 0.4607])
    assert(np.max(np.abs(correct_zyx - RToZYX(Tout[0:3,0:3]))) < 1e-3)
    print('Tests Passed!')

    source_pcld_file = PCLD_DIR + str(source_idx) + '.txt'
    target_pcld_file = PCLD_DIR + str(target_idx) + '.txt'

    viz = Open3DVisualizer()
    source_np = load_np_from_txt(source_pcld_file)
    target_np = load_np_from_txt(target_pcld_file)
    np_array = np.concatenate([source_np, target_np])
    viz.plot3d(np_array)

    Rotation = Tout[0:3, 0:3]
    t = np.reshape(translation, (3,1))
    ts = np.tile(t, (1, np.shape(source_np)[0]))
    rotated_source = np.matmul(Rotation, source_np.transpose())
    transformed_source = rotated_source + ts
    np_array = np.concatenate([transformed_source.transpose(), target_np])
    viz.plot3d(np_array)

if __name__ == '__main__':
  main()
