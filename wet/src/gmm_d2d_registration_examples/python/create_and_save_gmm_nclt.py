#!/usr/bin/env python
"""
Script to convert NCLT velodyne_sync point clouds to GMM format.
Reads binary .bin files directly (no ROS environment needed).

Data layout on disk:
    {base_path}/{scene}_vel/{scene}/velodyne_sync/*.bin

Usage example:
    python3 create_and_save_gmm_nclt.py \
        --scene 2013-01-10 \
        --every-n 5 --voxel 0.05 \
        --start-id 3880 --final-id 4070 \
        --n_components 150 --mahal_distance 2.0
"""
import os
import sys
import argparse

import numpy as np
import yaml
from tqdm import tqdm

from utils.nclt_loader import get_nclt_sync_files, load_nclt_pointcloud
from utils.pcl_filters import voxel_filter, radius_filter, every_n_filter
from utils.save_gmm import save_sogmm
from utils.gmm_fitter import fit_gmm, make_output_dir, detect_implementation

NCLT_BASE_PATH = os.path.expanduser('~/thesis/data')


def convert_nclt_to_gmm(scene, output_dir, n_components=None,
                         voxel=None, radius=None, every_n=None,
                         skip_scans=1, start_id=None, final_id=None,
                         implementation='fixed', bandwidth=None,
                         mahal_distance=None, redux_kmeans=None):
    """
    Convert NCLT velodyne_sync .bin files to GMM format.

    Args:
        scene:          Scene date string, e.g. '2013-01-10'
        output_dir:     Directory to save GMM files
        n_components:   Number of Gaussian components (fixed implementation)
        voxel:          Voxel size (m), None to skip
        radius:         Maximum distance from origin (m), None to skip
        every_n:        Take every Nth point, None to skip
        skip_scans:     Process every Nth scan
        start_id:       First scan index to process (inclusive)
        final_id:       Last scan index to process (inclusive)
        implementation: 'fixed' or 'sogmm'
        bandwidth:      Bandwidth for SOGMM
        mahal_distance: Mahalanobis distance bound for EM (passed to fit_gmm)
        redux_kmeans:   Use every Nth point for KMeans++ init only
    """
    if implementation == 'fixed' and n_components is None:
        raise ValueError('fixed implementation requires --n_components')
    if implementation == 'sogmm' and bandwidth is None:
        raise ValueError('sogmm implementation requires --bw')

    sync_files = get_nclt_sync_files(NCLT_BASE_PATH, scene)
    total = len(sync_files)
    print(f'NCLT / {scene}: {total} scans found')

    if implementation == 'sogmm':
        print(f'Implementation: {implementation}, bandwidth={bandwidth}')
    else:
        print(f'Implementation: {implementation}, components={n_components}')
    print(f'Every-N filter:  {every_n is not None} (every {every_n}th point)')
    print(f'Voxel filter:    {voxel is not None} (size={voxel}m)')
    print(f'Radius filter:   {radius is not None} (r={radius}m)')
    print(f'Skip scans:      {skip_scans}, start_id={start_id}, final_id={final_id}')
    print(f'Mahal bound:     {mahal_distance is not None} (λ={mahal_distance})')
    print(f'Redux KMeans++:  {redux_kmeans is not None} (every {redux_kmeans}th for init)')

    gmm_output_dir = make_output_dir(output_dir, implementation, n_components, bandwidth)

    meta = {
        'dataset': 'NCLT',
        'scene': scene,
        'implementation': implementation,
        'n_components': n_components if implementation == 'fixed' else None,
        'bandwidth': bandwidth if implementation == 'sogmm' else None,
        'every_n': every_n,
        'voxel': voxel,
        'radius': radius,
        'skip_scans': skip_scans,
        'start_id': start_id,
        'final_id': final_id,
        'mahal_distance': mahal_distance,
        'redux_kmeans': redux_kmeans,
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    meta_path = os.path.join(gmm_output_dir, 'meta.yaml')
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    lo = start_id if start_id is not None else 0
    hi = (final_id + 1) if final_id is not None else len(sync_files)
    files_to_process = sync_files[lo:hi]

    print(f'\nProcessing scans {lo} – {hi - 1} ({len(files_to_process)} total)...')
    scan_idx = lo
    saved_count = 0

    for file_path in tqdm(files_to_process, desc=f'scans {lo}-{hi-1}'):
        if (scan_idx - lo) % skip_scans != 0:
            scan_idx += 1
            continue

        points_4d = load_nclt_pointcloud(file_path)
        raw_count = len(points_4d)

        if every_n is not None:
            points_4d = every_n_filter(points_4d, n=every_n, verbose=False)
        if radius is not None:
            points_4d = radius_filter(points_4d, 0.5, radius, verbose=False)
        if voxel is not None:
            points_4d = voxel_filter(points_4d, voxel, verbose=False)

        print(f'  scan {scan_idx}: {raw_count} -> {len(points_4d)} pts after preprocessing')

        if len(points_4d) < 100:
            print(f'\nToo sparse: scan {scan_idx} has only {len(points_4d)} points, skipping')
            scan_idx += 1
            continue

        if implementation == 'fixed' and len(points_4d) < n_components * 3:
            print(f'\nWARNING: scan {scan_idx} has only {len(points_4d)} points '
                  f'for {n_components} components')

        scan_name = f'scan_{saved_count:06d}'
        stats_dir = os.path.join(gmm_output_dir, 'profiling')
        gmm_4d = fit_gmm(
            points_4d, implementation,
            n_components=n_components, bandwidth=bandwidth,
            redux_kmeans=redux_kmeans, mahal_distance=mahal_distance,
            stats_dir=stats_dir, scan_name=scan_name,
        )
        if gmm_4d is None:
            print(f'\nWARNING: EM fitting failed for scan {scan_idx}')
            scan_idx += 1
            continue

        if implementation == 'sogmm':
            print(f'SOGMM fitted with {gmm_4d.n_components_} components')

        output_path = os.path.join(gmm_output_dir, f'{scan_idx}.gmm')
        save_sogmm(output_path, gmm_4d)

        saved_count += 1
        scan_idx += 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    display_path = '~/' + os.path.relpath(gmm_output_dir, script_dir)
    print(f'\nSaved {saved_count} GMM files to: {display_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert NCLT velodyne_sync point clouds to GMM format')

    parser.add_argument('--scene', type=str, required=True,
                        help='NCLT scene date, e.g. 2013-01-10')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: ./runs/nclt_{scene})')
    parser.add_argument('--n_components', type=int, default=None,
                        help='Number of GMM components (fixed implementation)')
    parser.add_argument('--bw', type=float, default=None,
                        help='Bandwidth for sogmm')
    parser.add_argument('--every-n', type=int, default=None, dest='every_n')
    parser.add_argument('--voxel',   type=float, default=None)
    parser.add_argument('--radius',  type=float, default=None)
    parser.add_argument('--skip_scans', type=int, default=1,
                        help='Process every Nth scan (default: 1)')
    parser.add_argument('--start-id', type=int, default=None, dest='start_id',
                        help='First scan index to process (inclusive)')
    parser.add_argument('--final-id', type=int, default=None, dest='final_id',
                        help='Last scan index to process (inclusive)')
    parser.add_argument('--mahal_distance', type=float, default=None,
                        help='Mahalanobis distance bound for EM fitting')
    parser.add_argument('--redux_kmeans', type=int, default=None,
                        help='Use every Nth point for KMeans++ init only')

    args = parser.parse_args()

    implementation, bandwidth, n_components = detect_implementation(args.bw, args.n_components)

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, 'runs', f'nclt_{args.scene}')

    print(f'Scene:      {args.scene}')
    print(f'Output dir: {args.output_dir}\n')

    convert_nclt_to_gmm(
        scene=args.scene,
        output_dir=args.output_dir,
        n_components=n_components,
        every_n=args.every_n,
        voxel=args.voxel,
        radius=args.radius,
        skip_scans=args.skip_scans,
        start_id=args.start_id,
        final_id=args.final_id,
        implementation=implementation,
        bandwidth=bandwidth,
        mahal_distance=args.mahal_distance,
        redux_kmeans=args.redux_kmeans,
    )


if __name__ == '__main__':
    main()
