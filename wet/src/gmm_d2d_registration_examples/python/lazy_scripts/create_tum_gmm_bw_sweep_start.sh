#!/usr/bin/env bash
cd "$(dirname "$0")/.."

# idx 0-100
python3 create_and_save_gmm_tum.py --bw 0.06 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
python3 create_and_save_gmm_tum.py --bw 0.07 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
python3 create_and_save_gmm_tum.py --bw 0.08 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
python3 create_and_save_gmm_tum.py --bw 0.09 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
# ---
# python3 create_and_save_gmm_tum.py --bw 0.09 --every-n 5 --voxel 0.05 --start_idx 0 --end_idx 100
# add intermedaite --> [0.01, 0.0157, 0.0214, 0.0271, 0.0329, 0.0386, 0.0443, 0.05]
