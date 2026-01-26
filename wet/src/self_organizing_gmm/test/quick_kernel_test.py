#!/usr/bin/env python
"""
Quick test for kernel comparison on synthetic 2D data
Use this to verify the implementation works before running on real datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from kernel_comparison import CustomMeanShift
from sklearn.cluster import MeanShift


def test_kernels_on_2d_data():
    """Quick test on 2D synthetic data"""
    print("="*70)
    print("Quick Kernel Test on Synthetic 2D Data")
    print("="*70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, _ = datasets.make_blobs(n_samples=1000, centers=4, random_state=42)
    print(f"   Generated {len(X)} points with 4 clusters")

    # Test each kernel
    bandwidth = 1.0
    kernels = ['flat', 'gaussian', 'cauchy']
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, kernel in enumerate(kernels):
        print(f"\n2. Testing {kernel} kernel...")

        if kernel == 'flat':
            # Use sklearn for baseline
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
        else:
            # Use custom implementation
            ms = CustomMeanShift(bandwidth=bandwidth, kernel=kernel)
            ms.fit(X)
            labels = ms.labels_
            cluster_centers = ms.mode_centers_

        n_modes = len(cluster_centers)
        print(f"   Found {n_modes} modes")

        results[kernel] = {
            'n_modes': n_modes,
            'centers': cluster_centers,
            'labels': labels
        }

        # Plot
        ax = axes[idx]
        colors = plt.cm.Spectral(np.linspace(0, 1, n_modes))

        for k in range(n_modes):
            cluster_mask = labels == k
            ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1],
                      c=[colors[k]], s=20, alpha=0.6)

        # Plot cluster centers
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                  c='red', s=200, marker='X', edgecolors='black', linewidths=2,
                  label='Cluster Centers')

        ax.set_title(f'{kernel.capitalize()} Kernel\n({n_modes} modes)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kernel_comparison_2d_test.png', dpi=150)
    print(f"\n3. Plot saved to: kernel_comparison_2d_test.png")

    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"{'Kernel':<15} {'Number of Modes':<20}")
    print("-"*70)
    for kernel in kernels:
        print(f"{kernel:<15} {results[kernel]['n_modes']:<20}")
    print("="*70)

    plt.show()


def test_kernel_properties():
    """Test kernel weight functions"""
    print("\n" + "="*70)
    print("Kernel Weight Function Test")
    print("="*70)

    bandwidth = 1.0
    distances = np.linspace(0, 3*bandwidth, 100)

    # Compute weights for each kernel
    flat_weights = np.ones_like(distances)
    flat_weights[distances > bandwidth] = 0

    gaussian_weights = np.exp(-0.5 * (distances / bandwidth) ** 2)

    cauchy_weights = 1.0 / (1.0 + (distances / bandwidth) ** 2)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, flat_weights, 'r-', linewidth=2, label='Flat (Uniform)')
    plt.plot(distances, gaussian_weights, 'g-', linewidth=2, label='Gaussian')
    plt.plot(distances, cauchy_weights, 'b-', linewidth=2, label='Cauchy')

    plt.axvline(bandwidth, color='black', linestyle='--', alpha=0.5, label=f'Bandwidth = {bandwidth}')
    plt.xlabel('Distance from center', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Kernel Weight Functions Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kernel_weights.png', dpi=150)
    print("Plot saved to: kernel_weights.png")
    plt.show()

    print("\nKernel properties:")
    print("-"*70)
    print("Flat kernel:")
    print("  - Hard cutoff at bandwidth")
    print("  - All points within bandwidth have equal weight")
    print("  - Fast to compute")
    print()
    print("Gaussian kernel:")
    print("  - Smooth decay with distance")
    print("  - Points closer to center have more influence")
    print("  - Most commonly used in theory")
    print()
    print("Cauchy kernel:")
    print("  - Heavy-tailed distribution")
    print("  - More robust to outliers")
    print("  - Slower decay than Gaussian")
    print("="*70)


if __name__ == "__main__":
    print("\nThis script runs two quick tests:\n")
    print("1. Test kernels on 2D synthetic data")
    print("2. Visualize kernel weight functions")
    print()

    # Test on 2D data
    test_kernels_on_2d_data()

    # Test kernel properties
    test_kernel_properties()

    print("\n" + "="*70)
    print("Quick test completed!")
    print("If everything works correctly, you can now run the full comparison:")
    print("  python3 kernel_comparison.py")
    print("or")
    print("  ./run_kernel_comparison.sh")
    print("="*70 + "\n")
