#!/usr/bin/env python
import numpy as np

def save(filepath, gmm):

    n_components = gmm.n_components
    data = np.zeros([n_components, 13])
    for i in range(0, n_components):
        data[i, 0:3] = gmm.means_[i,:]
        data[i, 3:12] = gmm.covariances_[i].flatten()
        data[i, 12] = gmm.weights_[i]
    np.savetxt(filepath, X=data, fmt='%.30f', delimiter=',')

def save_sogmm(filepath, gmm_4d):
    n_components = gmm_4d.n_components_
    data = np.zeros([n_components, 13])

    means_3d = gmm_4d.means_[:, :3]
    covs_4d = gmm_4d.covariances_.reshape(n_components, 4, 4)
    covs_3d = covs_4d[:, :3, :3]

    for i in range(n_components):
        data[i, 0:3] = means_3d[i, :]
        data[i, 3:12] = covs_3d[i].flatten()
        data[i, 12] = gmm_4d.weights_[i]
    np.savetxt(filepath, X=data, fmt='%.30f', delimiter=',')

