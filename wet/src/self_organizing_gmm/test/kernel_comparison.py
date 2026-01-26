#!/usr/bin/env python
"""
Kernel Comparison Script for MeanShift in SOGMM Pipeline
Compares Flat, Gaussian, and Cauchy kernels
"""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors
import time
from dataclasses import dataclass, asdict
from typing import List, Dict
import json
import os

from cprint import cprint
from sogmm_py.utils import matrix_to_tensor, o3d_to_np, calculate_depth_metrics, np_to_o3d
from sogmm_py.utils import read_log_trajectory, ImageUtils
from sogmm_py.vis_open3d import VisOpen3D
import open3d as o3d

from sogmm_cpu import SOGMMf4Host, SOGMMLearner, SOGMMInference
from kinit_py import KInitf2CPU
from gmm_py import GMMf2CPU


@dataclass
class KernelResults:
    """Results for a single kernel"""
    kernel_name: str
    num_modes: int
    fit_time: float
    f_score: float
    precision: float
    recall: float
    recon_mean: float
    recon_std: float


class CustomMeanShift:
    """Custom MeanShift implementation with different kernel functions"""

    def __init__(self, bandwidth: float, kernel: str = 'flat'):
        """
        Args:
            bandwidth: Bandwidth parameter
            kernel: Kernel type - 'flat', 'gaussian', or 'cauchy'
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.mode_centers_ = None
        self.labels_ = None

    def _kernel_weight(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute kernel weights based on distance

        Args:
            distance: Array of distances

        Returns:
            Array of weights
        """
        if self.kernel == 'flat':
            # Uniform kernel - all points have equal weight
            return np.ones_like(distance)
        elif self.kernel == 'gaussian':
            # Gaussian kernel
            return np.exp(-0.5 * (distance / self.bandwidth) ** 2)
        elif self.kernel == 'cauchy':
            # Cauchy kernel (heavy-tailed)
            return 1.0 / (1.0 + (distance / self.bandwidth) ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _get_bin_seeds(self, X: np.ndarray) -> np.ndarray:
        """
        Generate bin seeds similar to sklearn implementation

        Args:
            X: Input data of shape (N, 2)

        Returns:
            Seeds array of shape (n_seeds, 2)
        """
        min_pt = X.min(axis=0)
        max_pt = X.max(axis=0)

        # Calculate grid dimensions
        width = int(np.floor((max_pt[0] - min_pt[0]) / self.bandwidth + 0.5)) + 1
        height = int(np.floor((max_pt[1] - min_pt[1]) / self.bandwidth + 0.5)) + 1

        # Hash points into bins
        bin_sizes = {}
        for point in X:
            r = int(np.floor((point[1] - min_pt[1]) / self.bandwidth + 0.5))
            c = int(np.floor((point[0] - min_pt[0]) / self.bandwidth + 0.5))
            idx = r * width + c
            bin_sizes[idx] = bin_sizes.get(idx, 0) + 1

        # Get valid bins (with at least 1 point)
        valid_bins = [idx for idx, count in bin_sizes.items() if count >= 1]

        if len(valid_bins) == 0:
            raise RuntimeError("No valid bins found")

        # Convert bin indices back to coordinates
        seeds = []
        for idx in valid_bins:
            r = idx // width
            c = idx % width
            x = c * self.bandwidth + min_pt[0]
            y = r * self.bandwidth + min_pt[1]
            seeds.append([x, y])

        return np.array(seeds)

    def fit(self, X: np.ndarray):
        """
        Fit MeanShift with custom kernel

        Args:
            X: Input data of shape (N, 2)
        """
        # Get initial seeds
        seeds = self._get_bin_seeds(X)
        n_seeds = len(seeds)

        stop_thresh = 1e-3 * self.bandwidth
        max_iter = 300

        mode_centers = []
        points_within_counts = []

        # Iterate over each seed
        for i in range(n_seeds):
            current = seeds[i].copy()
            completed_iter = 0

            while True:
                # Find points within bandwidth
                distances = np.linalg.norm(X - current, axis=1)
                in_bandwidth_mask = distances < self.bandwidth

                if not np.any(in_bandwidth_mask):
                    points_within_counts.append(0)
                    break

                points_within = X[in_bandwidth_mask]
                distances_within = distances[in_bandwidth_mask]

                # Store old position
                old_center = current.copy()

                # Apply kernel weighting
                weights = self._kernel_weight(distances_within)
                weights = weights / weights.sum()

                # Compute weighted mean (mean shift)
                current = np.sum(points_within * weights[:, np.newaxis], axis=0)

                # Check convergence
                shift = np.linalg.norm(current - old_center)
                if shift < stop_thresh or completed_iter >= max_iter:
                    points_within_counts.append(len(points_within))
                    mode_centers.append(current)
                    break

                completed_iter += 1

        if len(mode_centers) == 0:
            self.mode_centers_ = np.array([]).reshape(0, 2)
            self.labels_ = np.array([])
            return self

        mode_centers = np.array(mode_centers)
        points_within_counts = np.array(points_within_counts)

        # Post-process: remove duplicate modes
        # Sort by points within (descending)
        sorted_indices = np.argsort(points_within_counts)[::-1]
        sorted_indices = sorted_indices[points_within_counts[sorted_indices] > 0]

        if len(sorted_indices) == 0:
            self.mode_centers_ = np.array([]).reshape(0, 2)
            self.labels_ = np.array([])
            return self

        sorted_centers = mode_centers[sorted_indices]

        # Mark unique modes
        unique_mask = np.ones(len(sorted_centers), dtype=bool)

        for i in range(len(sorted_centers)):
            if unique_mask[i]:
                # Find all modes within bandwidth of this mode
                distances = np.linalg.norm(sorted_centers - sorted_centers[i], axis=1)
                close_modes = distances < self.bandwidth
                # Mark duplicates as non-unique (except current)
                close_modes[i] = False
                unique_mask[close_modes] = False

        self.mode_centers_ = sorted_centers[unique_mask]

        # Assign labels
        if len(self.mode_centers_) > 0:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.mode_centers_)
            _, indices = nbrs.kneighbors(X)
            self.labels_ = indices.flatten()
        else:
            self.labels_ = np.zeros(len(X), dtype=int)

        return self

    def get_num_modes(self) -> int:
        """Get number of modes found"""
        return len(self.mode_centers_) if self.mode_centers_ is not None else 0


class KernelComparison:
    """Main class for running kernel comparison experiments"""

    def __init__(self, bandwidth: float = 0.02, output_dir: str = "./kernel_comparison_results"):
        """
        Args:
            bandwidth: Bandwidth parameter for MeanShift
            output_dir: Directory to save results
        """
        self.bandwidth = bandwidth
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fit_sogmm_with_kernel(self, kernel: str, Y: np.ndarray, pcld: np.ndarray,
                               num_samples: int = 300000, max_depth: float = 2.2) -> Dict:
        """
        Fit SOGMM using specified kernel for MeanShift

        Args:
            kernel: Kernel type ('flat', 'gaussian', 'cauchy')
            Y: 2D data for MeanShift (depth, grayscale) of shape (N, 2)
            pcld: 4D point cloud data of shape (N, 4)
            num_samples: Number of samples for reconstruction
            max_depth: Maximum depth for reconstruction

        Returns:
            Dictionary with results
        """
        cprint.info(f"\n{'='*70}")
        cprint.info(f"Testing kernel: {kernel}")
        cprint.info(f"{'='*70}")

        # Step 1: Run MeanShift with specified kernel
        cprint.info("Step 1: Running MeanShift...")
        ms_start = time.time()

        if kernel in ['flat', 'gaussian'] and kernel == 'flat':
            # Use sklearn for flat kernel (fastest baseline)
            ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True)
            ms.fit(Y)
            num_modes = len(np.unique(ms.labels_))
        elif kernel == 'gaussian':
            # sklearn doesn't support gaussian kernel in MeanShift, use custom
            ms = CustomMeanShift(bandwidth=self.bandwidth, kernel=kernel)
            ms.fit(Y)
            num_modes = ms.get_num_modes()
        else:
            # Use custom implementation
            ms = CustomMeanShift(bandwidth=self.bandwidth, kernel=kernel)
            ms.fit(Y)
            num_modes = ms.get_num_modes()

        ms_time = time.time() - ms_start
        cprint.ok(f"MeanShift completed in {ms_time:.2f}s - Found {num_modes} modes")

        # Step 2: Initialize SOGMM with K components
        cprint.info(f"Step 2: Fitting SOGMM with K={num_modes} components...")
        sogmm_start = time.time()

        sogmm = SOGMMf4Host()
        learner = SOGMMLearner(self.bandwidth)

        # Use fit_em to directly specify K instead of running MeanShift again
        learner.fit_em(pcld, num_modes, sogmm)

        sogmm_time = time.time() - sogmm_start
        cprint.ok(f"SOGMM fitting completed in {sogmm_time:.2f}s")

        # Step 3: Reconstruction
        cprint.info(f"Step 3: Reconstructing point cloud...")
        recon_start = time.time()

        inference = SOGMMInference()
        reconstructed_points = inference.reconstruct(sogmm, num_samples, max_depth)

        recon_time = time.time() - recon_start
        cprint.ok(f"Reconstruction completed in {recon_time:.2f}s")

        # Step 4: Calculate metrics
        cprint.info("Step 4: Calculating metrics...")
        f_score, precision, recall, recon_mean, recon_std = calculate_depth_metrics(
            np_to_o3d(pcld[:, 0:3]),
            np_to_o3d(reconstructed_points)
        )

        cprint.ok(f"F-score: {f_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        cprint.ok(f"Recon Mean: {recon_mean:.4f}, Recon Std: {recon_std:.4f}")

        total_time = ms_time + sogmm_time + recon_time

        return {
            'kernel': kernel,
            'num_modes': num_modes,
            'ms_time': ms_time,
            'sogmm_time': sogmm_time,
            'recon_time': recon_time,
            'total_time': total_time,
            'f_score': f_score,
            'precision': precision,
            'recall': recall,
            'recon_mean': recon_mean,
            'recon_std': recon_std,
            'reconstructed_points': reconstructed_points,
            'sogmm': sogmm
        }

    def run_comparison(self, Y: np.ndarray, pcld: np.ndarray,
                      kernels: List[str] = ['flat', 'gaussian', 'cauchy'],
                      visualize: bool = False) -> List[KernelResults]:
        """
        Run comparison for multiple kernels

        Args:
            Y: 2D data for MeanShift (depth, grayscale)
            pcld: 4D point cloud data
            kernels: List of kernel names to compare
            visualize: Whether to visualize results

        Returns:
            List of KernelResults
        """
        results = []
        reconstructions = {}

        for kernel in kernels:
            try:
                result = self.fit_sogmm_with_kernel(kernel, Y, pcld)

                results.append(KernelResults(
                    kernel_name=kernel,
                    num_modes=result['num_modes'],
                    fit_time=result['total_time'],
                    f_score=result['f_score'],
                    precision=result['precision'],
                    recall=result['recall'],
                    recon_mean=result['recon_mean'],
                    recon_std=result['recon_std']
                ))

                reconstructions[kernel] = result['reconstructed_points']

            except Exception as e:
                cprint.err(f"Error with kernel {kernel}: {str(e)}")
                continue

        # Print comparison table
        self._print_results_table(results)

        # Save results to JSON
        self._save_results(results)

        # Visualize if requested
        if visualize:
            self._visualize_results(pcld, reconstructions)

        return results

    def _print_results_table(self, results: List[KernelResults]):
        """Print formatted results table"""
        print("\n" + "="*100)
        print(f"{'Kernel':<12} {'Modes':<8} {'Time (s)':<12} {'F-score':<12} "
              f"{'Precision':<12} {'Recall':<12} {'Mean':<10} {'Std':<10}")
        print("="*100)

        for r in results:
            print(f"{r.kernel_name:<12} {r.num_modes:<8} {r.fit_time:<12.2f} "
                  f"{r.f_score:<12.4f} {r.precision:<12.4f} {r.recall:<12.4f} "
                  f"{r.recon_mean:<10.4f} {r.recon_std:<10.4f}")

        print("="*100)

        # Find best kernel by F-score
        if results:
            best = max(results, key=lambda x: x.f_score)
            cprint.ok(f"\nBest kernel by F-score: {best.kernel_name} (F-score: {best.f_score:.4f})")

    def _save_results(self, results: List[KernelResults]):
        """Save results to JSON file"""
        output_file = os.path.join(self.output_dir, "kernel_comparison_results.json")

        results_dict = [asdict(r) for r in results]

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        cprint.ok(f"\nResults saved to: {output_file}")

    def _visualize_results(self, original_pcld: np.ndarray, reconstructions: Dict[str, np.ndarray]):
        """Visualize reconstruction results"""
        cprint.info("\nVisualizing results...")

        # Create visualization for each kernel
        for kernel, recon_points in reconstructions.items():
            cprint.info(f"Showing reconstruction for kernel: {kernel}")

            vis = VisOpen3D(visible=True)

            # Original point cloud in gray
            original_pcd = np_to_o3d(original_pcld[:, 0:3])
            original_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            vis.add_geometry(original_pcd)

            # Reconstructed point cloud in color
            recon_pcd = np_to_o3d(recon_points)
            if kernel == 'flat':
                recon_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            elif kernel == 'gaussian':
                recon_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green
            elif kernel == 'cauchy':
                recon_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

            vis.add_geometry(recon_pcd)
            vis.render()


def main():
    """Main function for standalone execution"""
    from config_parser import ConfigParser

    # Load config
    parser = ConfigParser()
    config = parser.get_config()

    # Load dataset
    cprint.info("Loading dataset...")
    
    pcld_path = o3d.data.LivingRoomPointClouds().paths[0]
    
    pcld_o3d = o3d.io.read_point_cloud(pcld_path)
    
    pcld = o3d_to_np(pcld_o3d)

    # Prepare 2D data (depth, grayscale)
    d = np.array([np.linalg.norm(x) for x in pcld[:, 0:3]])[:, np.newaxis]
    g = pcld[:, 3][:, np.newaxis]
    Y = np.concatenate((d, g), axis=1)

    cprint.ok(f"Loaded point cloud with {len(pcld)} points")

    # Run comparison
    comparison = KernelComparison(bandwidth=config.bandwidth)
    results = comparison.run_comparison(
        Y, pcld,
        kernels=['flat', 'gaussian', 'cauchy'],
        visualize=config.show_plots
    )

    cprint.ok("\nKernel comparison completed!")


if __name__ == "__main__":
    main()
