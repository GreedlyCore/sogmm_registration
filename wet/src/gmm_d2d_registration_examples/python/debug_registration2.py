#!/usr/bin/env python
"""
Point-cloud-level FGR debug script.

Samples points from two GMMs, shows them before and after Fast Global
Registration — no GMM-based registration involved.

Usage:
    python debug_registration2.py <source.gmm> <target.gmm> [options]
"""

"""
python debug_registration2.py ~/runs/tum_rgbd_dataset_freiburg3_long_office_household/500_components_04030454/104.gmm  \
    ~/runs/tum_rgbd_dataset_freiburg3_long_office_household/500_components_04030454/105.gmm
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import open3d

from utils.open3d_visualizer import load_gmm_file


def _sample_gmm(gmm, n_samples, rng):
    w = gmm['weights'] / gmm['weights'].sum()
    idx = rng.choice(len(w), size=n_samples, p=w)
    pts = np.array([
        rng.multivariate_normal(gmm['means'][k], gmm['covariances'][k])
        for k in idx
    ], dtype=np.float64)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    return pcd


def _prepare_fgr(pcd, voxel_size, min_points=50):
    """Downsample and compute FPFH; auto-halve voxel_size if too few points remain."""
    down = pcd.voxel_down_sample(voxel_size)
    while len(down.points) < min_points and voxel_size > 1e-4:
        voxel_size /= 2.0
        down = pcd.voxel_down_sample(voxel_size)
    if len(down.points) < 2:
        raise RuntimeError(
            f"Only {len(down.points)} point(s) after downsampling — "
            "point cloud may be degenerate or voxel size is still too large.")
    down.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        down,
        open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return down, fpfh, voxel_size


def run_fgr(src_pcd, tgt_pcd, voxel_size):
    src_down, src_fpfh, v_src = _prepare_fgr(src_pcd, voxel_size)
    tgt_down, tgt_fpfh, v_tgt = _prepare_fgr(tgt_pcd, voxel_size)
    v_eff = min(v_src, v_tgt)
    print(f"  Effective voxel size: {v_eff:.4f} m  "
          f"({len(src_down.points)} src pts, {len(tgt_down.points)} tgt pts)")
    result = open3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        open3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=v_eff * 0.5))
    return result.transformation


def show(src_pcd, tgt_pcd, title):
    src_pcd.paint_uniform_color([1.0, 0.3, 0.3])  # red
    tgt_pcd.paint_uniform_color([0.2, 0.4, 1.0])  # blue
    print(f"  {title}  (source=red, target=blue) — close window to continue")
    open3d.visualization.draw_geometries(
        [src_pcd, tgt_pcd],
        window_name=title,
        width=1280, height=720)


def main():
    parser = argparse.ArgumentParser(description='Point-cloud FGR debug')
    parser.add_argument('source', help='Source .gmm file')
    parser.add_argument('target', help='Target .gmm file')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Points sampled per GMM (default: 5000)')
    parser.add_argument('--fgr_voxel', type=float, default=0.05,
                        help='Voxel size in metres for FGR (default: 0.05)')
    args = parser.parse_args()

    src_path = os.path.expanduser(args.source)
    tgt_path = os.path.expanduser(args.target)
    print(f"\nSource : {src_path}")
    print(f"Target : {tgt_path}")

    src_gmm = load_gmm_file(src_path)
    tgt_gmm = load_gmm_file(tgt_path)
    print(f"Source : {src_gmm['n_components']} components")
    print(f"Target : {tgt_gmm['n_components']} components")

    rng = np.random.default_rng(42)
    src_pcd = _sample_gmm(src_gmm, args.n_samples, rng)
    tgt_pcd = _sample_gmm(tgt_gmm, args.n_samples, rng)
    print(f"\nSampled {args.n_samples} points from each GMM")

    show(src_pcd, tgt_pcd, "BEFORE FGR")

    print(f"\nRunning FGR (voxel={args.fgr_voxel} m) ...")
    T = run_fgr(src_pcd, tgt_pcd, args.fgr_voxel)
    print("FGR transformation:")
    print(T)

    src_aligned = open3d.geometry.PointCloud(src_pcd)
    src_aligned.transform(T)
    show(src_aligned, tgt_pcd, "AFTER FGR")


if __name__ == '__main__':
    main()
