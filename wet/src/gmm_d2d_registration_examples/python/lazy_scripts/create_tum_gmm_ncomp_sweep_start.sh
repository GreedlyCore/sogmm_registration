#!/usr/bin/env bash
cd "$(dirname "$0")/.."

# idx 0-100
python3 create_and_save_gmm_tum.py --n_components 50  --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
python3 create_and_save_gmm_tum.py --n_components 100 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
python3 create_and_save_gmm_tum.py --n_components 150 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
python3 create_and_save_gmm_tum.py --n_components 200 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
python3 create_and_save_gmm_tum.py --n_components 250 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
python3 create_and_save_gmm_tum.py --n_components 500 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
