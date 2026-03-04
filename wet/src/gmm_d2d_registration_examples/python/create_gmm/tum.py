#!/usr/bin/env python
"""
Script to fit GMMs from TUM RGB-D 4D point cloud .txt files.

Expects files produced by convert_tum_to_sogmm.py:
  {pcld_dir}/{scan_idx}.txt  ->  N rows of  x,y,z,intensity  (comma-separated)

Usage:
    python create_and_save_gmm_tum.py --dataset rgbd_dataset_freiburg3_long_office_household --bw 0.05
    python create_and_save_gmm_tum.py --dataset rgbd_dataset_freiburg3_long_office_household --n_components 100 --every-n 3
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import numpy as np
import yaml
from tqdm import tqdm

from utils.save_gmm import save_sogmm
from utils.pcl_filters import voxel_filter, radius_filter, every_n_filter
from utils.gmm_fitter import fit_gmm, make_output_dir, detect_implementation
from utils.tum_loader import load_tum_pointcloud


def convert_tum_to_gmm(pcld_dir, output_dir, n_components=None,
                        start_idx=0, end_idx=None,
                        every_n=None, voxel=None, radius=None,
                        skip_scans=1,
                        implementation='fixed', bandwidth=None,
                        mahal_distance=None, redux_kmeans=None):
    """
    Fit GMMs from TUM 4D point cloud .txt files.

    Args:
        pcld_dir: Directory with {scan_idx}.txt files (x,y,z,intensity)
        output_dir: Directory to save GMM files
        n_components: Number of Gaussian components (fixed)
        start_idx: First scan index to process
        end_idx: Last scan index (exclusive, None = all)
        every_n: Keep every Nth point
        voxel: Voxel size (m), None to skip
        radius: Max point distance from origin (m)
        skip_scans: Process every Nth scan (default 1 = all scans)
        implementation: 'fixed' or 'sogmm'
        bandwidth: Bandwidth for SOGMM
        mahal_distance: Mahalanobis distance bound for EM
        redux_kmeans: Use every Nth point for KMeans++ init only
    """
    if implementation == 'fixed' and n_components is None:
        raise ValueError('fixed requires n_components')
    if implementation == 'sogmm' and bandwidth is None:
        raise ValueError('sogmm requires bandwidth')

    if not os.path.exists(pcld_dir):
        print(f'ERROR: Pointclouds directory not found: {pcld_dir}')
        sys.exit(1)

    txt_files = sorted([f for f in os.listdir(pcld_dir) if f.endswith('.txt')],
                       key=lambda x: int(x.split('.')[0]))
    if end_idx is None:
        end_idx = len(txt_files)
    txt_files = txt_files[start_idx:end_idx]

    print(f'Converting {len(txt_files)} scans to GMM format...')
    if implementation == 'sogmm':
        print(f'Implementation: {implementation}, bandwidth={bandwidth}')
    else:
        print(f'Implementation: {implementation}, components={n_components}')
    print(f'Every-N: {every_n}, Voxel: {voxel}m, Radius: {radius}m, Skip-scans: {skip_scans}')
    print(f'Mahalanobis bound: {mahal_distance}, Redux KMeans++: {redux_kmeans}')

    gmm_output_dir = make_output_dir(output_dir, implementation, n_components, bandwidth)

    meta = {
        'dataset': 'TUM',
        'implementation': implementation,
        'n_components': n_components if implementation == 'fixed' else None,
        'bandwidth':    bandwidth    if implementation == 'sogmm' else None,
        'skip_scans': skip_scans,
        'every_n': every_n, 'voxel': voxel, 'radius': radius,
        'mahal_distance': mahal_distance, 'redux_kmeans': redux_kmeans,
        'start_idx': start_idx,
        'end_idx': end_idx,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    with open(os.path.join(gmm_output_dir, 'meta.yaml'), 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    component_counts = []

    def _flush_meta():
        if implementation == 'sogmm' and component_counts:
            arr = np.array(component_counts, dtype=float)
            meta['components_avg'] = float(np.mean(arr))
            meta['components_std'] = float(np.std(arr))
            meta['components_n_scans'] = len(arr)
        meta_path = os.path.join(gmm_output_dir, 'meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    import time as _time
    try:
        for scan_i, txt_file in enumerate(tqdm(txt_files, desc='Processing scans')):
            if scan_i % skip_scans != 0:
                continue
            t0 = _time.time()
            filepath = os.path.join(pcld_dir, txt_file)

            points_4d = load_tum_pointcloud(filepath)
            print(f'\n  [{txt_file}] loadtxt: {_time.time()-t0:.2f}s, shape={points_4d.shape}')

            original = len(points_4d)

            if every_n is not None:
                points_4d = every_n_filter(points_4d, n=every_n, verbose=False)
            if radius is not None:
                points_4d = radius_filter(points_4d, 0.0, radius, verbose=False)
            if voxel is not None:
                points_4d = voxel_filter(points_4d, voxel, verbose=False)

            print(f'  [{txt_file}] after filters: {original} --> {len(points_4d)} pts')

            if len(points_4d) < 10:
                print(f'  [{txt_file}] too sparse, skipping')
                continue

            scan_name = txt_file.split('.')[0]
            stats_dir = os.path.join(gmm_output_dir, 'profiling')
            gmm_4d = fit_gmm(
                points_4d, implementation,
                n_components=n_components, bandwidth=bandwidth,
                redux_kmeans=redux_kmeans, mahal_distance=mahal_distance,
                stats_dir=stats_dir, scan_name=scan_name,
            )
            if gmm_4d is None:
                print(f'  WARNING: EM failed for {txt_file}')
                continue
            if implementation == 'sogmm':
                component_counts.append(gmm_4d.n_components_)
                print(f'  SOGMM: {gmm_4d.n_components_} components')

            output_path = os.path.join(gmm_output_dir, f'{scan_name}.gmm')
            save_sogmm(output_path, gmm_4d)

    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        _flush_meta()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    display_path = '~/' + os.path.relpath(gmm_output_dir, script_dir)
    print(f'\nSaved GMM files to: {display_path}')


def main():
    parser = argparse.ArgumentParser(description='Fit GMMs from TUM 4D point cloud .txt files')

    parser.add_argument('--dataset', type=str,
                        default='rgbd_dataset_freiburg3_long_office_household',
                        help='TUM dataset folder name under sogmm_registration/data/')
    parser.add_argument('--n_components', type=int, default=None)
    parser.add_argument('--bw',          type=float, default=None,
                        help='Bandwidth for SOGMM')
    parser.add_argument('--every-n',     type=int,   default=None, dest='every_n')
    parser.add_argument('--voxel',       type=float, default=None)
    parser.add_argument('--radius',      type=float, default=None)
    parser.add_argument('--skip-scan',   type=int,   default=1,    dest='skip_scans',
                        help='Process every Nth scan (default: 1 = all scans)')
    parser.add_argument('--start-id',    type=int,   default=0,    dest='start_idx')
    parser.add_argument('--end-id',      type=int,   default=None, dest='end_idx')
    parser.add_argument('--mahal',       type=float, default=None, dest='mahal_distance')
    parser.add_argument('--redux_kmeans',   type=int,   default=None)

    args = parser.parse_args()

    implementation, bandwidth, n_components = detect_implementation(args.bw, args.n_components)

    cwd = os.getcwd()
    SANDBOX_NAME = 'sogmm_registration'
    gira3d_root  = cwd.split(SANDBOX_NAME)[0] + SANDBOX_NAME

    pcld_dir   = os.path.join(gira3d_root, 'data', args.dataset, 'pointclouds')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'runs', f'tum_{args.dataset}')

    print(f'Dataset:         {args.dataset}')
    print(f'Pointclouds dir: {pcld_dir}')
    print(f'Output dir:      {output_dir}\n')

    convert_tum_to_gmm(
        pcld_dir=pcld_dir,
        output_dir=output_dir,
        n_components=n_components,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        every_n=args.every_n,
        voxel=args.voxel,
        radius=args.radius,
        skip_scans=args.skip_scans,
        implementation=implementation,
        bandwidth=bandwidth,
        mahal_distance=args.mahal_distance,
        redux_kmeans=args.redux_kmeans,
    )


if __name__ == '__main__':
    main()
