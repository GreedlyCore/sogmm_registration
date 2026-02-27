#!/usr/bin/env python
"""
Run odometry estimation on VIRAL dataset GMMs and evaluate against ground truth.
Mirrors the structure of run_mine_dataset.py.

Usage:
    python run_viral_dataset.py \
        --bag /path/to/eee_03.bag \
        --gmm_dir ./runs/viral_eee_03/100_components_28010325
"""
import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from tqdm import tqdm

import gmm_d2d_registration_py
from utils.RToZYX import RToZYX
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse
from utils.plot_results import plot_results
from utils.viral_loader import load_viral_ground_truth, load_viral_lidar_config
from utils.quaternion import R_to_quat


def _save_tum(path, timestamps, poses):
    """Save trajectory in TUM format: timestamp tx ty tz qx qy qz qw"""
    with open(path, 'w') as f:
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = R_to_quat(T[:3, :3])
            f.write(f'{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} '
                    f'{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')


# python run_viral_dataset.py --bag ~/thesis/VIRAL/eee_03/eee_03.bag  --gmm_dir runs/viral_eee_03/100_components_28010325
# python run_viral_dataset.py --bag ~/thesis/VIRAL/nya_01/nya_01.bag  --gmm_dir runs/viral_nya_01/150_components_28010412
# python run_viral_dataset.py --bag ~/thesis/VIRAL/eee_03/eee_03.bag  --gmm_dir runs/viral_eee_03/150_components_19021747



# ● Now check first_scan is derived from GMM filenames (which it still is via int(gmm_files[0].stem)), so load_viral_ground_truth stays        
#   correct. No further changes needed there.
                                                                                                                                             
#   Usage now:                             
#   python3 create_and_save_gmm_viral.py \                                                                                                     
#     --every-n 5 --radius 30 --skip_scans 1 \                                                                                                 
#     --bag ~/thesis/VIRAL/eee_03/eee_03.bag \
#     --n_components 100 \
#     --start-id 1250 --final-id 2000

#   --start-id is now the single control for where bag iteration begins and saving starts.


def run_viral_dataset(bag_path, gmm_dir, gt_base_path=None, start_id=None, final_id=None):
    # Load meta to get skip_scans
    meta_path = os.path.join(gmm_dir, 'meta.yaml')
    skip_scans = 1
    if os.path.exists(meta_path):
        import yaml
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        skip_scans = meta.get('skip_scans', 1)
        print(f'Meta: skip_scans={skip_scans}, start_id={meta.get("start_id")}, n_components={meta.get("n_components")}')

    # Sorted GMM files by scan index
    gmm_files = sorted(
        [p for p in Path(gmm_dir).glob('*.gmm')],
        key=lambda p: int(p.stem)
    )

    # Filter by scan index range if specified
    if start_id is not None or final_id is not None:
        lo = start_id if start_id is not None else 0
        hi = final_id if final_id is not None else float('inf')
        gmm_files = [p for p in gmm_files if lo <= int(p.stem) <= hi]
        print(f'Filtered to scan range [{lo}, {hi}]: {len(gmm_files)} GMMs')

    n_scans = len(gmm_files)
    print(f'Found {n_scans} GMM files in {gmm_dir}')

    if n_scans < 2:
        raise ValueError('Need at least 2 GMM files to run odometry')

    # Derive range from GMM filenames (they encode bag scan indices)
    first_scan = int(gmm_files[0].stem)
    config_topic, _ = load_viral_lidar_config(bag_path)
    lidar_topic = config_topic or '/os1_cloud_node1/points'

    gt_poses, timestamps = load_viral_ground_truth(
        bag_path,
        lidar_topic=lidar_topic,
        skip_scans=skip_scans,
        first_scan=first_scan,
        max_scans=n_scans,
        gt_base_path=gt_base_path,
    )

    if len(gt_poses) < n_scans:
        print(f'WARNING: only {len(gt_poses)} GT poses for {n_scans} GMMs — truncating')
        n_scans = len(gt_poses)
        gmm_files = gmm_files[:n_scans]
        timestamps = timestamps[:n_scans]

    n_pairs = n_scans - 1
    transforms = np.zeros((n_pairs, 6))
    ground_truths = np.zeros((n_pairs, 6))

    for i in tqdm(range(n_pairs)):
        target_file = str(gmm_files[i])
        source_file = str(gmm_files[i + 1])

        Tinit = np.eye(4)
        output = gmm_d2d_registration_py.isoplanar_registration(Tinit, source_file, target_file)
        ret = gmm_d2d_registration_py.anisotropic_registration(output[0], source_file, target_file)
        Tout = ret[0]

        translation = Tout[0:3, 3]
        rotation = RToZYX(Tout[0:3, 0:3])
        transforms[i] = np.concatenate([translation, rotation])

        Tc1c2 = pose_compose(pose_inverse(gt_poses[i]), gt_poses[i + 1])
        dgt = np.concatenate([Tc1c2[0:3, 3], RToZYX(Tc1c2[0:3, 0:3])])
        ground_truths[i] = dgt

    return transforms, ground_truths, gt_poses, timestamps


def main():
    parser = argparse.ArgumentParser(description='Run VIRAL odometry evaluation')
    parser.add_argument('--bag', type=str, required=True, help='Path to .bag file')
    parser.add_argument('--gmm_dir', type=str, required=True, help='Path to GMM folder')
    parser.add_argument('--gt_base_path', type=str, default=None,
                        help='Path to ntuviral_gt folder (default: uses viral_loader default)')
    parser.add_argument('--start-id', type=int, default=None, dest='start_id',
                        help='First GMM scan index to use (inclusive, by file stem)')
    parser.add_argument('--final-id', type=int, default=None, dest='final_id',
                        help='Last GMM scan index to use (inclusive, by file stem)')
    args = parser.parse_args()

    args.bag = os.path.expanduser(args.bag)
    args.gmm_dir = os.path.expanduser(args.gmm_dir)

    transforms, ground_truths, gt_poses, timestamps = run_viral_dataset(
        args.bag, args.gmm_dir, args.gt_base_path, args.start_id, args.final_id)

    # Stats
    errors = transforms - ground_truths
    translation_errors = np.linalg.norm(errors[:, :3], axis=1)
    rotation_errors = np.linalg.norm(errors[:, 3:], axis=1)

    translation_rmse = np.sqrt(np.mean(translation_errors ** 2))
    rotation_rmse = np.sqrt(np.mean(rotation_errors ** 2))

    print(f'\nTranslation error: {np.mean(translation_errors):.4f} ± '
          f'{np.std(translation_errors):.4f} m  (RMSE: {translation_rmse:.4f})')
    print(f'Rotation error:    {np.mean(rotation_errors):.4f} ± '
          f'{np.std(rotation_errors):.4f} rad (RMSE: {rotation_rmse:.4f})')

    gmm_label = Path(args.gmm_dir).name
    bag_name = Path(args.bag).stem
    results_dir = os.path.join(os.path.dirname(args.gmm_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save pkl
    results_pkl = os.path.join(results_dir, f'{bag_name}_{gmm_label}_results.pkl')
    with open(results_pkl, 'wb') as f:
        pickle.dump([transforms, ground_truths, errors], f)
    print(f'Results saved to: {results_pkl}')

    # Build absolute trajectories for evo (starting from gt_poses[0])
    from utils.ZYXToR import ZYXToR
    odom_poses = [gt_poses[0]]
    for i in range(len(transforms)):
        T = np.eye(4)
        T[:3, :3] = ZYXToR(transforms[i, 3:6])
        T[:3, 3] = transforms[i, :3]
        odom_poses.append(pose_compose(odom_poses[-1], T))

    # Save TUM trajectories (N+1 poses, timestamps for scans 0..N)
    # odom_tum = os.path.join(results_dir, f'odom_{bag_name}_{gmm_label}.txt')
    # gt_tum = os.path.join(results_dir, f'gt_{bag_name}_{gmm_label}.txt')
    # _save_tum(odom_tum, timestamps, odom_poses)
    # _save_tum(gt_tum, timestamps, gt_poses)
    # print(f'TUM files saved:\n  {odom_tum}\n  {gt_tum}')
    # print(f'Run: evo_ape tum {gt_tum} {odom_tum} --align --correct_scale -p')

    save_path = os.path.join(results_dir, f'{bag_name}_{gmm_label}_trajectory.png')
    plot_results(transforms, ground_truths, 0, len(transforms), save_path=save_path)


if __name__ == '__main__':
    main()
