"""
GMM transformation utilities for world frame mapping.
"""
import numpy as np


def load_gmm_file(gmm_file):
    """
    Load GMM from .gmm file.

    Args:
        gmm_file: Path to .gmm file

    Returns:
        dict with 'means', 'covariances', 'weights' (each as numpy arrays)
    """
    data = np.loadtxt(gmm_file, delimiter=',')
    n_components = data.shape[0]

    means = data[:, 0:3]
    covariances = data[:, 3:12].reshape(n_components, 3, 3)
    weights = data[:, 12]

    return {
        'means': means,
        'covariances': covariances,
        'weights': weights,
        'n_components': n_components
    }


def transform_gmm(means, covariances, T):
    """
    Transform GMM components (means and covariances) to a new coordinate frame.

    For a Gaussian N(μ, Σ) transformed by T = [R|t]:
        μ' = R @ μ + t
        Σ' = R @ Σ @ R.T

    Args:
        means: (N, 3) array of component means
        covariances: (N, 3, 3) array of covariance matrices
        T: (4, 4) SE(3) transformation matrix

    Returns:
        means_transformed: (N, 3) transformed means
        covs_transformed: (N, 3, 3) transformed covariances
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Transform means: μ' = R @ μ + t
    means_transformed = (R @ means.T).T + t

    # Transform covariances: Σ' = R @ Σ @ R.T
    n_components = covariances.shape[0]
    covs_transformed = np.zeros_like(covariances)
    for i in range(n_components):
        covs_transformed[i] = R @ covariances[i] @ R.T

    return means_transformed, covs_transformed


def transform_gmm_dict(gmm_dict, T):
    """
    Transform a GMM dictionary to a new coordinate frame.

    Args:
        gmm_dict: dict with 'means', 'covariances', 'weights'
        T: (4, 4) SE(3) transformation matrix

    Returns:
        transformed GMM dict
    """
    means_t, covs_t = transform_gmm(
        gmm_dict['means'],
        gmm_dict['covariances'],
        T
    )
    return {
        'means': means_t,
        'covariances': covs_t,
        'weights': gmm_dict['weights'],
        'n_components': gmm_dict['n_components']
    }


def merge_gmms(gmm_list):
    """
    Merge multiple GMMs into a single GMM by concatenation.

    Note: This is a simple concatenation, not a proper GMM merge/reduction.
    For visualization purposes, this is sufficient.

    Args:
        gmm_list: list of GMM dicts with 'means', 'covariances', 'weights'

    Returns:
        merged GMM dict
    """
    all_means = []
    all_covs = []
    all_weights = []

    for gmm in gmm_list:
        all_means.append(gmm['means'])
        all_covs.append(gmm['covariances'])
        all_weights.append(gmm['weights'])

    return {
        'means': np.vstack(all_means),
        'covariances': np.vstack(all_covs),
        'weights': np.concatenate(all_weights),
        'n_components': sum(g['n_components'] for g in gmm_list)
    }
