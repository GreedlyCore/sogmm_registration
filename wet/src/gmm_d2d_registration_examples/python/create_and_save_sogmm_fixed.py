#!/usr/bin/env python
"""
Create fixed-K GMM from XYZ point clouds using SOGMMLearner.fit_em
Supports cave / mine / tum datasets (contains inside no intensity)
"""

# As paper proposed:

# create:
# python3 create_and_save_gmm_fixed.py --dataset mine_001_part3 --n_components 100 --every-n 5 --radius 15
# python3 create_and_save_gmm_fixed.py --dataset mine_001_part3 --n_components 100 --every-n 5 --radius 15 --voxel 0.5
# python3 create_and_save_gmm_fixed.py --dataset cave --n_components 100 --every-n 5 --radius 15 --voxel 0.5

# eval odometry:
# python3 run_mine_dataset.py --gmm_dir runs/mine_001_part3/100_components_24020218
# python3 run_mine_dataset.py --gmm_dir runs/mine_001_part3/100_components_24020231


import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import yaml
from tqdm import tqdm

from utils.save_gmm import save_sogmm
from utils.pcl_filters import voxel_filter, radius_filter, every_n_filter

import sogmm_cpu

"""

python3 create_and_save_sogmm_fixed.py --dataset cave --n_components 100 --every-n 5 --radius 15 --start_idx 499 --end_idx 890


python3 run_cave_dataset.py /home/sonieth2/thesis/sogmm_registration/wet/src/gmm_d2d_registration_examples/python/runs/cave/100_components_24022157
"""

def main():
    parser = argparse.ArgumentParser(description='Create fixed-K GMM from XYZ point clouds')
    parser.add_argument('--dataset',      required=True, help='Dataset folder name (e.g. cave, mine_001_part3)')
    parser.add_argument('--n_components', required=True, type=int, help='Number of GMM components')
    parser.add_argument('--every-n',      type=int,   default=None, dest='every_n', help='Keep every Nth point')
    parser.add_argument('--voxel',        type=float, default=None, help='Voxel grid size (m)')
    parser.add_argument('--radius',       type=float, default=None, help='Max point distance from origin (m)')
    parser.add_argument('--start_idx',    type=int,   default=0)
    parser.add_argument('--end_idx',      type=int,   default=None)
    args = parser.parse_args()

    # Resolve paths
    sandbox = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', '..', '..')
    sandbox = os.path.normpath(sandbox)
    pcld_dir = os.path.join(sandbox, 'data', args.dataset, 'pointclouds')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'runs', args.dataset)

    if not os.path.exists(pcld_dir):
        print(f'ERROR: {pcld_dir} not found')
        sys.exit(1)

    txt_files = sorted([f for f in os.listdir(pcld_dir) if f.endswith('.txt')],
                       key=lambda x: int(x.split('.')[0]))
    end_idx = args.end_idx if args.end_idx is not None else len(txt_files)
    txt_files = txt_files[args.start_idx:end_idx]

    timestamp = datetime.now().strftime('%d%m%H%M')
    gmm_dir = os.path.join(output_dir, f'{args.n_components}_components_{timestamp}')
    os.makedirs(gmm_dir, exist_ok=True)

    meta = {
        'dataset': args.dataset,
        'n_components': args.n_components,
        'every_n': args.every_n,
        'voxel': args.voxel,
        'radius': args.radius,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    with open(os.path.join(gmm_dir, 'meta.yaml'), 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f'Dataset:    {args.dataset}  ({len(txt_files)} scans)')
    print(f'Components: {args.n_components}')
    print(f'Filters:    every_n={args.every_n}  voxel={args.voxel}  radius={args.radius}')
    print(f'Output:     {gmm_dir}\n')

    learner = sogmm_cpu.SOGMMLearner()

    for txt_file in tqdm(txt_files, desc='Fitting GMMs'):
        filepath = os.path.join(pcld_dir, txt_file)
        data_3d = np.loadtxt(filepath, delimiter=',')
        points = np.c_[data_3d, np.zeros(len(data_3d))].astype(np.float32)

        if args.every_n is not None:
            points = every_n_filter(points, n=args.every_n, verbose=False)
        if args.radius is not None:
            points = radius_filter(points, min_radius=0.0, max_radius=args.radius, verbose=False)
        if args.voxel is not None:
            points = voxel_filter(points, voxel_size=args.voxel, verbose=False)

        K = min(args.n_components, len(points) - 1)
        if K < 1:
            print(f'  SKIP {txt_file}: too few points ({len(points)})')
            continue

        sogmm = sogmm_cpu.SOGMMf4Host()
        t0 = time.time()
        learner.fit_em(points, K, sogmm)
        elapsed = time.time() - t0

        scan_name = txt_file.split('.')[0]
        out_path = os.path.join(gmm_dir, f'{scan_name}.gmm')
        save_sogmm(out_path, sogmm)
        tqdm.write(f'  {txt_file}: {len(points)} pts -> K={sogmm.n_components_}  ({elapsed:.2f}s)')

    print(f'\nDone. Saved to: {gmm_dir}')

if __name__ == '__main__':
    main()