#!/usr/bin/env bash
cd "$(dirname "$0")/.."

### 10^-7 GMM registration param
# python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw9_components_25022123 --no_plot --start_idx 0 --end_idx 90
# python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw8_components_25022118 --no_plot --start_idx 0 --end_idx 90
# python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw7_components_25022112 --no_plot --start_idx 0 --end_idx 90
# python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw6_components_25022058 --no_plot --start_idx 0 --end_idx 90
# python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw5_components_25021338 --no_plot --start_idx 0 --end_idx 90
# python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw4_components_25021452 --no_plot --start_idx 0 --end_idx 90
# python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw3_components_25021631 --no_plot --start_idx 0 --end_idx 90

### 10^-4 GMM registration param | idx 1500-1600

python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw9_components_25022200 --no_plot --start_idx 1500 --end_idx 1599
python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw8_components_25022206 --no_plot --start_idx 1500 --end_idx 1599
python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw7_components_25022213 --no_plot --start_idx 1500 --end_idx 1599
python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw6_components_25022222 --no_plot --start_idx 1500 --end_idx 1599
python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw5_components_25022233 --no_plot --start_idx 1500 --end_idx 1599
python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw4_components_25022247 --no_plot --start_idx 1500 --end_idx 1599
python3 run_tum_dataset.py --gmm_dir runs/tum_rgbd_dataset_freiburg3_long_office_household/bw3_components_25022307 --no_plot --start_idx 1500 --end_idx 1599
