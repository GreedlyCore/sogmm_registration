#!/usr/bin/env python
"""
Compare two saved GMM folders visually.

Usage:
    python compare_gmm_results.py <gmm_folder> <fsogmm_folder> [scan_idx]

    blue   = gmm_folder
    orange = fsogmm_folder
    grey   = original point cloud
"""
# python compare_gmm_results.py ./100_components_24021641_gmm ./100_components_24021641_fsogmm

import os
import sys
import numpy as np
import open3d as o3d
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

GIRA3D_SANDBOX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..'))
PCLD_DIR = os.path.join(
    GIRA3D_SANDBOX, 'data',
    'rgbd_dataset_freiburg3_long_office_household', 'pointclouds')

sys.path.insert(0, os.path.dirname(__file__))
from utils.open3d_visualizer import load_gmm_file, create_ellipsoid_mesh


def _valid(gmm):
    """Indices of components with finite means and covariances."""
    return [k for k in range(gmm['n_components'])
            if np.isfinite(gmm['means'][k]).all()
            and np.isfinite(gmm['covariances'][k]).all()]


def _log_density(gmm, points, valid):
    weights = gmm['weights'][valid]
    weights = weights / weights.sum()
    log_N = np.stack([
        multivariate_normal.logpdf(
            points,
            mean=gmm['means'][k].astype(np.float64),
            cov=gmm['covariances'][k].astype(np.float64),
            allow_singular=True)
        for k in valid
    ], axis=1)
    return logsumexp(log_N + np.log(weights + 1e-300)[None, :], axis=1)


def symmetric_kl(gmm_a, gmm_b, n_samples=50_000, seed=42):
    rng = np.random.default_rng(seed)
    va = _valid(gmm_a)
    vb = _valid(gmm_b)

    def sample(gmm, valid):
        w = gmm['weights'][valid]
        w = w / w.sum()
        idx = rng.choice(len(valid), size=n_samples, p=w)
        return np.stack([
            rng.multivariate_normal(
                gmm['means'][valid[i]].astype(np.float64),
                gmm['covariances'][valid[i]].astype(np.float64))
            for i in idx
        ])

    sa = sample(gmm_a, va)
    sb = sample(gmm_b, vb)

    kl_ab = float(np.mean(_log_density(gmm_a, sa, va) - _log_density(gmm_b, sa, vb)))
    kl_ba = float(np.mean(_log_density(gmm_b, sb, vb) - _log_density(gmm_a, sb, va)))
    return kl_ab, kl_ba


def add_ellipsoids(gmm, color, geometries):
    skipped = 0
    for k in range(gmm['n_components']):
        mesh, _ = create_ellipsoid_mesh(
            gmm['means'][k], gmm['covariances'][k], gmm['weights'][k],
            n_sigma=2, resolution=15)
        if mesh is None:
            skipped += 1
            continue
        mesh.paint_uniform_color(color)
        geometries.append(mesh)
    if skipped:
        print(f'  skipped {skipped} degenerate components')


def main():
    if len(sys.argv) < 3:
        print('Usage: python compare_gmm_results.py <gmm_folder> <fsogmm_folder> [scan_idx]')
        sys.exit(1)

    folder_a = sys.argv[1]
    folder_b = sys.argv[2]
    idx = int(sys.argv[3]) if len(sys.argv) > 3 else 1260

    gmm_a = load_gmm_file(os.path.join(folder_a, f'{idx}.gmm'))
    gmm_b = load_gmm_file(os.path.join(folder_b, f'{idx}.gmm'))

    geometries = []

    print(f'blue   = {folder_a}')
    add_ellipsoids(gmm_a, [0.2, 0.5, 1.0], geometries)

    print(f'orange = {folder_b}')
    add_ellipsoids(gmm_b, [1.0, 0.5, 0.1], geometries)

    pcld_path = os.path.join(PCLD_DIR, f'{idx}.txt')
    if os.path.exists(pcld_path):
        points = np.loadtxt(pcld_path, delimiter=',')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(pcd)
        print(f'grey   = scan {idx} ({points.shape[0]} pts)')

    #     KL divergence is in nats (natural log). Scale reference:
    #   - 0 = identical distributions
    #   - ln(2) ≈ 0.693 = roughly "half the mass is in different places"                                                                           
    #   - → ∞ = one distribution assigns zero probability where the other doesn't
    
    #   TODO: check sanity of this creation
    #   - KL(A||B) = 0.43 > KL(B||A) = 0.32 — asymmetry is the key insight: 
    #   sampling from sklearn-GMM (A), fsogmm (B) assigns noticeably lower
    #   probability to those points than vice versa. This means sklearn GMM has sharper/more concentrated components that fsogmm doesn't fully
    #   cover. fsogmm tends to be more "spread out".


    print('\nSymmetric KL (MC 50k samples)...')
    kl_ab, kl_ba = symmetric_kl(gmm_a, gmm_b)
    print(f'  KL(A||B) = {kl_ab:.4f}')
    print(f'  KL(B||A) = {kl_ba:.4f}')
    print(f'  symmetric = {(kl_ab + kl_ba) / 2:.4f}')

    o3d.visualization.draw_geometries(geometries,
                                      window_name=f'{folder_a} vs {folder_b}',
                                      width=1280, height=720)


if __name__ == '__main__':
    main()
