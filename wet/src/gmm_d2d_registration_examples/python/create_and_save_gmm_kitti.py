#!/usr/bin/env python
"""
Script to convert KITTI Velodyne point clouds to GMM format.
Supports voxel filtering/radius filtering to reduce point count.
"""
import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

from utils.save_gmm import save, save_sogmm
from sklearn.mixture import GaussianMixture
from gmm_d2d_registration_py import GMM3f
    
from sogmm_py import SOGMM
# Create GMM with profiling enabled
from gmm_py import GMMf4CPU
# K-Means++ initialization (same as in sogmm.py gmm_fit)
from kinit_py import KInitf4CPU

# GPU support (optional - will be None if not available)
try:
    from mean_shift_py import KernelType, MeanShift as MSf2
    from sogmm_gpu import SOGMMf4Device, SOGMMLearner
    from sogmm_cpu import SOGMMf4Host
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: GPU modules not available. Use --device cpu or install GPU support.")

def load_kitti_velodyne(filepath):
    """
    Load KITTI velodyne point cloud from .bin file.

    Args:
        filepath: Path to .bin file

    Returns:
        points: Nx4 array (x, y, z, reflectance)
    """
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return points


def voxel_filter(points, voxel_size=0.1):
    """
    Apply voxel grid to downsample point cloud.
    You probably may need this for faster CPU GMM fitting, just because typical KITTI scans have 100k+ points.
    

    Args:
        points: Nx3 or Nx4 array of points
        voxel_size: Size of voxel grid (in meters)

    Returns:
        filtered_points: Downsampled point cloud
    """
    xyz = points[:, :3]

    # Compute voxel indices
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    # Get unique voxels and their first occurrence
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    filtered_points = points[unique_indices]

    print(f'Voxel filtering: {len(points)} -> {len(filtered_points)} points '
          f'({100.0 * len(filtered_points) / len(points):.1f}%)')

    return filtered_points


def radius_filter(points, radius=5.0):
    """
    Remove points farther than specified radius from origin.

    Args:
        points: Nx3 or Nx4 array of points
        radius: Maximum allowed distance (in meters)

    Returns:
        filtered_points: Points within radius
    """
    xyz = points[:, :3]
    distances = np.linalg.norm(xyz, axis=1)
    mask = distances < radius
    
    filtered_points = points[mask]
    
    print(f'Radius filtering (<{radius}m): {len(points)} -> {len(filtered_points)} points '
          f'({100.0 * len(filtered_points) / len(points):.1f}%)')
    
    return filtered_points

def remove_plane_ransac(points, distance_threshold=0.3, ransac_n=3, num_iterations=1000):
    """
    Remove ground plane using RANSAC (simple numpy implementation).

    Args:
        points: Nx3 or Nx4 array of points
        distance_threshold: Max distance from plane to be considered inlier
        ransac_n: Number of points to sample for plane estimation
        num_iterations: Number of RANSAC iterations

    Returns:
        filtered_points: Points without the ground plane
    """
    if len(points) < ransac_n:
        return points

    xyz = points[:, :3]
    best_inliers = None
    best_count = 0

    for _ in range(num_iterations):
        idx = np.random.choice(len(xyz), ransac_n, replace=False)
        p0, p1, p2 = xyz[idx]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -np.dot(normal, p0)

        distances = np.abs(np.dot(xyz, normal) + d)
        inliers = distances < distance_threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        return points

    outliers = points[~best_inliers]
    print(f'RANSAC plane removal: {len(points)} -> {len(outliers)} points '
          f'(removed {best_count} plane points, {100.0 * len(outliers) / len(points):.1f}% remain)')
    return outliers


# TODO: add statistical outlier filter too ???
# when will be ported to pure cpp --> use pcl library for that ??? 

def convert_kitti_to_gmm(sequence_dir, output_dir, n_components=100,
                         use_voxel_filter=False, voxel_size=0.1,
                         use_radius_filter=False, radius=5.0,
                         use_plane_removal=False, ransac_distance=0.3, ransac_iters=1000,
                         start_idx=0, end_idx=None, implementation='cpp',
                         bandwidth=0.05, device='cpu'):
    """
    Convert KITTI velodyne scans to GMM format.

    Args:
        sequence_dir: Path to KITTI sequence (e.g., .../sequences/04/)
        output_dir: Directory to save GMM files
        n_components: Number of Gaussian components
        use_voxel_filter: Whether to apply voxel filtering
        voxel_size: Voxel size for filtering (in meters)
        use_radius_filter: Whether to remove points beyond radius
        radius: Maximum distance from origin (in meters)
        start_idx: First scan index to process
        end_idx: Last scan index to process (None = all)
        implementation: GMM implementation to use ('cpp', 'sklearn', 'sogmm', etc.)
        bandwidth: Bandwidth for SOGMM (only used with --implementation sogmm)
        device: Device to use for SOGMM ('cpu' or 'gpu', only applies to sogmm implementation)
    """
    velodyne_dir = os.path.join(sequence_dir, 'velodyne')

    if not os.path.exists(velodyne_dir):
        print(f'ERROR: Velodyne directory not found: {velodyne_dir}')
        print('\nYou need to download the KITTI velodyne laser data (80 GB):')
        print('https://www.cvlibs.net/datasets/kitti/eval_odometry.php')
        print('\nExtract it so that velodyne/ folders appear in each sequence directory.')
        sys.exit(1)

    bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])

    if end_idx is None:
        end_idx = len(bin_files)

    bin_files = bin_files[start_idx:end_idx]

    print(f'Converting {len(bin_files)} scans to GMM format...')
    if implementation == 'sogmm':
        print(f'Implementation: {implementation} (adaptive)')
        print(f'Device: {device.upper()}')
        print(f'Bandwidth: {bandwidth}')
        if device == 'gpu' and not HAS_GPU:
            print('ERROR: GPU support not available. Please build sogmm_open3d or use --device cpu')
            sys.exit(1)
    else:
        print(f'Implementation: {implementation}')
        print(f'Components: {n_components}')
    print(f'Voxel filtering: {use_voxel_filter} (voxel_size={voxel_size}m)')
    print(f'Radius filtering: {use_radius_filter} (radius={radius}m)')
    print(f'Plane removal: {use_plane_removal} (distance={ransac_distance}m, iters={ransac_iters})')

    # Create timestamp in format DDMMHHMM
    timestamp = datetime.now().strftime("%d%m%H%M")

    if implementation == 'sogmm':
        gmm_output_dir = os.path.join(output_dir, f"adaptive_bw{int(bandwidth*100)}_components_{timestamp}")
    else:
        gmm_output_dir = os.path.join(output_dir, f'{n_components}_components_{timestamp}')
    os.makedirs(gmm_output_dir, exist_ok=True)

    # Write metadata file with filter parameters
    meta_path = os.path.join(gmm_output_dir, 'meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"# GMM Generation Metadata\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"implementation: {implementation}\n")
        if implementation == 'sogmm':
            f.write(f"bandwidth: {bandwidth}\n")
            f.write(f"device: {device}\n")
        else:
            f.write(f"n_components: {n_components}\n")
        f.write(f"\n# Filtering parameters\n")
        f.write(f"voxel_filter: {use_voxel_filter}\n")
        f.write(f"voxel_size: {voxel_size}\n")
        f.write(f"radius_filter: {use_radius_filter}\n")
        f.write(f"radius: {radius}\n")
        f.write(f"plane_removal: {use_plane_removal}\n")
        f.write(f"ransac_distance: {ransac_distance}\n")
        f.write(f"ransac_iters: {ransac_iters}\n")
        f.write(f"\n# Scan range\n")
        f.write(f"start_idx: {start_idx}\n")
        f.write(f"end_idx: {end_idx}\n")
        f.write(f"n_scans: {len(bin_files)}\n")

    for bin_file in tqdm(bin_files, desc='Processing scans'):
        filepath = os.path.join(velodyne_dir, bin_file)
        points_4d = load_kitti_velodyne(filepath)  # Nx4 (x, y, z, intensity)

        # Apply filtering to full 4D points to keep xyz and intensity aligned
        if use_radius_filter:
            points_4d = radius_filter(points_4d, radius)
        if use_voxel_filter:
            points_4d = voxel_filter(points_4d, voxel_size)
        if use_plane_removal:
            points_4d = remove_plane_ransac(points_4d, ransac_distance, 3, ransac_iters)

        # Remove points that are too close (likely vehicle itself)
        distances = np.linalg.norm(points_4d[:, :3], axis=1)
        points_4d = points_4d[distances > 2.0]  # Keep points > 2m away

        # Extract xyz for fixed GMM, full 4D for SOGMM
        xyz = points_4d[:, :3]
        xyz_with_intensity = points_4d

        if len(xyz) < n_components * 3:
            print(f'\nWARNING: {bin_file} has only {len(xyz)} points, '
                  f'might be too few for {n_components} components')
        
        if implementation == 'cpp':
            gmm = GMM3f()
            gmm.fit(xyz, n_components)
        elif implementation == 'sklearn':
            gmm = GaussianMixture(n_components=n_components, covariance_type='full')
            gmm.fit(xyz)
        elif implementation == 'fsogmm':
            # Fixed component SOGMM --- need to bypass SOGMM wrapper to enable profiling
            n_samples = xyz_with_intensity.shape[0]
            kinit = KInitf4CPU()
            _, indices = kinit.resp_calc(xyz_with_intensity, n_components)
            resp = np.zeros((n_samples, n_components), dtype=np.float32)
            resp[indices, np.arange(n_components)] = 1

            stats_dir = os.path.join(gmm_output_dir, 'profiling')
            os.makedirs(stats_dir, exist_ok=True)
            stats_file = f'gmm_stats_{bin_file.split(".")[0]}.csv'

            local_model = GMMf4CPU(n_components, True, stats_dir, stats_file)
            success = local_model.fit(xyz_with_intensity, resp)

            if not success:
                print(f'WARNING: EM fitting failed for {bin_file}')
                continue
            gmm_4d = local_model
        
        elif implementation == 'sogmm':
            stats_dir = os.path.join(gmm_output_dir, 'profiling')
            os.makedirs(stats_dir, exist_ok=True)
            stats_file_prefix = bin_file.split(".")[0]

            if device == 'gpu':
                # GPU SOGMM implementation
                # Extract mean-shift data (distance + intensity)
                d = np.array([np.linalg.norm(x) for x in xyz_with_intensity[:, 0:3]])[:, np.newaxis]
                g = xyz_with_intensity[:, 3][:, np.newaxis]
                ms_data = np.concatenate((d, g), axis=1)

                # Create GPU learner with profiling support
                # After rebuilding sogmm_open3d, this will create profiling CSV files:
                #   - {stats_file_prefix}_ms.csv  (Mean Shift timings)
                #   - {stats_file_prefix}_em.csv  (GPU EM timings: e_step, m_step)
                # Note: KInit doesn't support profiling in current version
                learner_gpu = SOGMMLearner(
                    bandwidth=bandwidth,
                    kernel='flat',
                    save_stats=True,
                    stats_dir=stats_dir,
                    stats_file_prefix=f"gpu_{stats_file_prefix}"
                )
                model_gpu = SOGMMf4Device()

                # Fit on GPU
                learner_gpu.fit(ms_data, xyz_with_intensity, model_gpu)
                n_components_gpu = model_gpu.n_components_

                # Convert to CPU for saving
                model_cpu = SOGMMf4Host(n_components_gpu)
                model_gpu.to_host(model_cpu)
                local_model = model_cpu
                print(f"SOGMM (GPU) fitted with {n_components_gpu} components...")
            else:
                # CPU SOGMM implementation (original)
                sg = SOGMM(bandwidth, save_stats=True, stats_dir=stats_dir, stats_file_prefix=stats_file_prefix)
                local_model = sg.fit(xyz_with_intensity)
                print(f"SOGMM (CPU) fitted with {local_model.n_components_} components...")

            gmm_4d = local_model


        # Save GMM (use 1-based indexing to match MATLAB convention --> first scan is named 1.gmm)
        idx = int(bin_file.split('.')[0]) + 1
        output_path = os.path.join(gmm_output_dir, f'{idx}.gmm')

        if implementation in ['fsogmm', 'sogmm']:
            save_sogmm(output_path, gmm_4d)
        else:
            save(output_path, gmm)

def main():
    parser = argparse.ArgumentParser(
        description='Convert KITTI velodyne scans to GMM format')

    parser.add_argument('--sequence', type=str, required=True,
                        help='KITTI sequence number (e.g., 00, 04, 07)')
    parser.add_argument('--n_scans', type=int, required=True,
                        help='Number of scans to convert')
    parser.add_argument('--kitti_dir', type=str,
                        default='./dataset/kitti',
                        help='Path to KITTI dataset root')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: ./kitti_sequence_{ID})')
    parser.add_argument('--n_components', type=int, default=100,
                        help='Number of GMM components (default: 100)')
    parser.add_argument('--voxel_filter', action='store_true',
                        help='Enable voxel filtering to reduce point count')
    parser.add_argument('--voxel_size', type=float, default=0.1,
                        help='Voxel size in meters (default: 0.1)')
    parser.add_argument('--radius_filter', action='store_true',
                        help='Enable radius filtering to remove distant points')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='Maximum distance from origin in meters (default: 5.0)')
    parser.add_argument('--remove_plane', action='store_true',
                        help='Enable RANSAC ground plane removal')
    parser.add_argument('--ransac_distance', type=float, default=0.3,
                        help='RANSAC distance threshold in meters (default: 0.3)')
    parser.add_argument('--ransac_iters', type=int, default=1000,
                        help='RANSAC iterations (default: 1000)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='First scan index (default: 0)')
    parser.add_argument('--implementation', type=str,
                        choices=['sklearn', 'cpp', 'fsogmm', 'sogmm'], default='cpp',
                        help='GMM implementation(CPU): sklearn  or handmade cpp or fixed sogmm model or adaptive sogmm')
    parser.add_argument('--bandwidth', type=float, default=0.05,
                        help='Bandwidth for SOGMM (only used with --implementation sogmm, default: 0.05)')
    parser.add_argument('--device', type=str,
                        choices=['cpu', 'gpu'], default='cpu',
                        help='Device to use for SOGMM (only applies to --implementation sogmm, default: cpu)')

    args = parser.parse_args()

    # Expand ~ to home directory
    args.kitti_dir = os.path.expanduser(args.kitti_dir)

    # Construct paths for downloaded KITTI dataset
    sequence_dir = os.path.join(args.kitti_dir, 'data_odometry_velodyne',
                                'dataset', 'sequences', args.sequence)

    # Set output directory to kitti_sequence_{ID} in current directory
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, f'kitti_sequence_{args.sequence}')

    if not os.path.exists(sequence_dir):
        print(f'ERROR: Sequence directory not found: {sequence_dir}')
        print(f'\nMake sure you have downloaded the KITTI velodyne data to:')
        print(f'{args.kitti_dir}/data_odometry_velodyne/')
        sys.exit(1)

    # Calculate end_idx from n_scans
    end_idx = args.start_idx + args.n_scans

    print(f'KITTI Sequence: {args.sequence}')
    print(f'Sequence dir: {sequence_dir}')
    print(f'Output dir: {args.output_dir}')
    print(f'Processing scans {args.start_idx} to {end_idx-1} ({args.n_scans} total)')
    print()

    convert_kitti_to_gmm(
        sequence_dir=sequence_dir,
        output_dir=args.output_dir,
        n_components=args.n_components,
        use_voxel_filter=args.voxel_filter,
        voxel_size=args.voxel_size,
        use_radius_filter=args.radius_filter,
        radius=args.radius,
        use_plane_removal=args.remove_plane,
        ransac_distance=args.ransac_distance,
        ransac_iters=args.ransac_iters,
        start_idx=args.start_idx,
        end_idx=end_idx,
        implementation=args.implementation,
        bandwidth=args.bandwidth,
        device=args.device
    )


if __name__ == '__main__':
    main()