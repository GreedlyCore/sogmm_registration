#!/usr/bin/env python
"""
Debug script for GMM D2D registration.

Investigates different initialization techniques (identity, COM, PCA)
with per-stage cost logging and Open3D visualization of alignment.

Usage:
    python debug_registration.py <source.gmm> <target.gmm> [--init identity|com|pca]

Example:
    python debug_registration.py \
        ~/runs/nclt_2013-01-10/150_components_04030142/3880.gmm \
        ~/runs/nclt_2013-01-10/150_components_04030142/3881.gmm \
        --init com

Note: For per-iteration cost values, uncomment line 142 in
      wet/src/gmm_d2d_registration/include/gmm_d2d_registration/GMMD2DRegistration.h
      and recompile.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import open3d

import gmm_d2d_registration_py
from utils.open3d_visualizer import load_gmm_file, create_ellipsoid_mesh
from utils.RToZYX import RToZYX


"""
python debug_registration.py ~/runs/tum_rgbd_dataset_freiburg3_long_office_household/500_components_04030454/104.gmm  \
    ~/runs/tum_rgbd_dataset_freiburg3_long_office_household/500_components_04030454/105.gmm

python debug_registration.py ~/runs/tum_rgbd_dataset_freiburg3_long_office_household/50_components_04030502/1.gmm  \
~/runs/tum_rgbd_dataset_freiburg3_long_office_household/50_components_04030502/110.gmm

python debug_registration.py ~/runs/tum_rgbd_dataset_freiburg3_long_office_household/150_components_04030510/1.gmm  \
~/runs/tum_rgbd_dataset_freiburg3_long_office_household/150_components_04030510/751.gmm

"""






# ─── Initialization strategies ────────────────────────────────────────────────

def _weighted_centroid(gmm):
    w = gmm['weights']
    return (w[:, None] * gmm['means']).sum(axis=0) / w.sum()


def init_identity():
    return np.eye(4, dtype=np.float32)


def init_com(src_gmm, tgt_gmm):
    """Translate source center-of-mass to target center-of-mass."""
    c_src = _weighted_centroid(src_gmm)
    c_tgt = _weighted_centroid(tgt_gmm)
    T = np.eye(4, dtype=np.float32)
    T[0:3, 3] = (c_tgt - c_src).astype(np.float32)
    return T


def init_fgr(src_gmm, tgt_gmm, n_samples=5000, voxel_size=0.5):
    """Fast Global Registration on points sampled from both GMMs.

    Samples *n_samples* points proportional to component weights, builds Open3D
    point clouds, computes FPFH features, then runs FGR to get an initial pose.
    """
    rng = np.random.default_rng(42)

    def _sample(gmm):
        w = gmm['weights'] / gmm['weights'].sum()
        idx = rng.choice(len(w), size=n_samples, p=w)
        pts = np.array([
            rng.multivariate_normal(gmm['means'][k], gmm['covariances'][k])
            for k in idx
        ], dtype=np.float64)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts)
        return pcd

    def _prepare(pcd):
        down = pcd.voxel_down_sample(voxel_size)
        down.estimate_normals(
            open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = open3d.pipelines.registration.compute_fpfh_feature(
            down,
            open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return down, fpfh

    src_down, src_fpfh = _prepare(_sample(src_gmm))
    tgt_down, tgt_fpfh = _prepare(_sample(tgt_gmm))

    result = open3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        open3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size * 0.5))

    return result.transformation.astype(np.float32)


def init_pca(src_gmm, tgt_gmm):
    """Align principal axes of weighted mean distributions, then match centroids."""
    c_src = _weighted_centroid(src_gmm)
    c_tgt = _weighted_centroid(tgt_gmm)

    def _pca(gmm, centroid):
        means = gmm['means'] - centroid          # (N, 3)
        w = gmm['weights'] / gmm['weights'].sum()
        C = (w[:, None] * means).T @ means       # (3, 3) weighted covariance of means
        _, eigvecs = np.linalg.eigh(C)           # ascending eigenvalues
        return eigvecs[:, ::-1]                  # descending: principal axis first

    V_src = _pca(src_gmm, c_src)
    V_tgt = _pca(tgt_gmm, c_tgt)

    R = V_tgt @ V_src.T
    if np.linalg.det(R) < 0:          # fix reflection
        V_tgt[:, -1] *= -1
        R = V_tgt @ V_src.T

    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] = R.astype(np.float32)
    T[0:3, 3] = (c_tgt - R @ c_src).astype(np.float32)
    return T


# ─── GMM transform ─────────────────────────────────────────────────────────────

def transform_gmm(gmm, T):
    """Apply rigid T (4x4) to all GMM components."""
    R = T[0:3, 0:3].astype(np.float64)
    t = T[0:3, 3].astype(np.float64)
    new_means = (R @ gmm['means'].astype(np.float64).T).T + t
    new_covs = np.array([R @ gmm['covariances'][i].astype(np.float64) @ R.T
                         for i in range(gmm['n_components'])])
    return {
        'means': new_means,
        'covariances': new_covs,
        'weights': gmm['weights'],
        'n_components': gmm['n_components'],
    }


# ─── Printing ──────────────────────────────────────────────────────────────────

def print_transform(T, label):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    euler = RToZYX(R)  # [phi, theta, psi] in radians
    euler_deg = np.degrees(euler)
    # print(f"\n  [{label}]")
    print(f"    t = [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]  (meters)")
    print(f"    R (ZYX euler) = [{euler_deg[0]:+.3f}°, {euler_deg[1]:+.3f}°, {euler_deg[2]:+.3f}°]")


def print_stage(label, T, score):
    print(f"\n{'─'*50}")
    print(f"  Stage : {label}")
    print(f"  Score : {score:.6f}  (higher = better alignment)")
    print_transform(T, "T")


# ─── Visualization ─────────────────────────────────────────────────────────────

def _build_meshes(gmm, color, n_sigma=2, resolution=10):
    meshes = []
    for i in range(gmm['n_components']):
        mesh, _ = create_ellipsoid_mesh(
            gmm['means'][i], gmm['covariances'][i], gmm['weights'][i],
            n_sigma=n_sigma, resolution=resolution)
        if mesh is not None:
            mesh.paint_uniform_color(color)
            meshes.append(mesh)
    return meshes


def visualize(src_gmm, tgt_gmm, T_init, T_final, n_sigma=2):
    tgt_meshes  = _build_meshes(tgt_gmm, color=[0.2, 0.4, 1.0], n_sigma=n_sigma)  # blue
    src_before  = _build_meshes(transform_gmm(src_gmm, T_init),  color=[1.0, 0.3, 0.3], n_sigma=n_sigma)  # red
    src_after   = _build_meshes(transform_gmm(src_gmm, T_final), color=[0.2, 0.9, 0.2], n_sigma=n_sigma)  # green

    print(f"\n  Ellipsoids — target: {len(tgt_meshes)}, source: {len(src_before)}")
    print("  Close each window to proceed.\n")
    print("  Window 1: BEFORE  (target=blue, source@Tinit=red)")
    open3d.visualization.draw_geometries(
        tgt_meshes + src_before,
        window_name="BEFORE registration (source=red, target=blue)",
        width=1280, height=720)

    print("  Window 2: AFTER   (target=blue, source@T_final=green)")
    open3d.visualization.draw_geometries(
        tgt_meshes + src_after,
        window_name="AFTER registration (source=green, target=blue)",
        width=1280, height=720)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Debug GMM D2D registration')
    parser.add_argument('source', help='Source .gmm file (moving scan)')
    parser.add_argument('target', help='Target .gmm file (fixed reference)')
    parser.add_argument('--init', choices=['identity', 'com', 'pca', 'fgr'], default='identity',
                        help='Initialization method (default: identity)')
    parser.add_argument('--n_sigma', type=float, default=2.0,
                        help='Ellipsoid sigma scale for visualization (default: 2)')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Points sampled per GMM for FGR init (default: 5000)')
    parser.add_argument('--fgr_voxel', type=float, default=0.5,
                        help='Voxel size (metres) for FGR downsampling (default: 0.5)')
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    src_path = os.path.expanduser(args.source)
    tgt_path = os.path.expanduser(args.target)

    print(f"\nSource : {src_path}")
    print(f"Target : {tgt_path}")

    src_gmm = load_gmm_file(src_path)
    tgt_gmm = load_gmm_file(tgt_path)

    print(f"Source : {src_gmm['n_components']} components")
    print(f"Target : {tgt_gmm['n_components']} components")

    # ── Init ──────────────────────────────────────────────────────────────────
    print(f"\nInit method : {args.init.upper()}")

    if args.init == 'identity':
        Tinit = init_identity()
    elif args.init == 'com':
        Tinit = init_com(src_gmm, tgt_gmm)
    elif args.init == 'pca':
        Tinit = init_pca(src_gmm, tgt_gmm)
    elif args.init == 'fgr':
        Tinit = init_fgr(src_gmm, tgt_gmm,
                         n_samples=args.n_samples, voxel_size=args.fgr_voxel)

    print_transform(Tinit, "Tinit")

    # ── Registration ──────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print("  Running isoplanar registration ...")
    T_iso, score_iso = gmm_d2d_registration_py.isoplanar_registration(
        Tinit, src_path, tgt_path)
    print_stage("isoplanar", T_iso, score_iso)

    print(f"\n{'═'*50}")
    print("  Running anisotropic registration ...")
    T_final, score_final = gmm_d2d_registration_py.anisotropic_registration(
        T_iso, src_path, tgt_path)
    print_stage("anisotropic (final)", T_final, score_final)

    print(f"\n{'═'*50}")
    print(f"  Score delta: {score_final - score_iso:+.6f}  (aniso vs iso)")

    # ── Visualize ─────────────────────────────────────────────────────────────
    visualize(src_gmm, tgt_gmm, Tinit, T_final, n_sigma=args.n_sigma)


if __name__ == '__main__':
    main()
