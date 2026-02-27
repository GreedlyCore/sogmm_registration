import os
import numpy as np
from sklearn.mixture import GaussianMixture

from utils.save_gmm import save

def create_and_save_gmm_example():
    
    cwd = os.getcwd()
    SANDBOX_NAME = 'gira3d-registration'
    matches = cwd.split(SANDBOX_NAME)
    GIRA3D_REGISTRATION_SANDBOX = matches[0] + SANDBOX_NAME
    PCLD_DIR = GIRA3D_REGISTRATION_SANDBOX + '/data/rgbd_dataset_freiburg3_long_office_household/pointclouds/'

    idx = 1260
    n_components = 100

    path_to_save = './' + str(idx) + '.gmm'
    file_to_load = PCLD_DIR + str(idx) + '.txt'
    g = GaussianMixture(n_components)
    data = np.loadtxt(file_to_load, delimiter=',')
    g.fit(data)
    save(path_to_save, g)

if __name__ == "__main__":
    create_and_save_gmm_example()
