#!/usr/bin/env bash
cd "$(dirname "$0")/.."

# idx 1500-1600
python3 create_and_save_gmm_tum.py --bw 0.09 --every-n 5 --voxel 0.05 --start_idx 1500 --end_idx 1600
python3 create_and_save_gmm_tum.py --bw 0.08 --every-n 5 --voxel 0.05 --start_idx 1500 --end_idx 1600
python3 create_and_save_gmm_tum.py --bw 0.07 --every-n 5 --voxel 0.05 --start_idx 1500 --end_idx 1600
python3 create_and_save_gmm_tum.py --bw 0.06 --every-n 5 --voxel 0.05 --start_idx 1500 --end_idx 1600
python3 create_and_save_gmm_tum.py --bw 0.05 --every-n 5 --voxel 0.05 --start_idx 1500 --end_idx 1600
python3 create_and_save_gmm_tum.py --bw 0.04 --every-n 5 --voxel 0.05 --start_idx 1500 --end_idx 1600
python3 create_and_save_gmm_tum.py --bw 0.03 --every-n 5 --voxel 0.05 --start_idx 1500 --end_idx 1600
