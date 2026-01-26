#!/usr/bin/env python
"""
Diagnostic script to inspect GMM files and identify issues.
"""
import numpy as np
import os

def check_gmm_file(filepath):
    """
    Load and validate a GMM file.

    GMM file format (CSV):
    Each row: mean_x, mean_y, mean_z, cov_xx, cov_xy, cov_xz, cov_yx, cov_yy, cov_yz, cov_zx, cov_zy, cov_zz, weight
    Total: 13 columns per component
    """
    data = np.loadtxt(filepath, delimiter=',')
    n_components = data.shape[0]

    # Extract components
    means = data[:, 0:3]
    covariances = data[:, 3:12].reshape(n_components, 3, 3)
    weights = data[:, 12]

    # Check weights
    weight_sum = weights.sum()
    weight_issues = []
    if abs(weight_sum - 1.0) > 0.01:
        weight_issues.append(f"Weights don't sum to 1: {weight_sum:.6f}")
    if np.any(weights < 0):
        weight_issues.append(f"Negative weights found: {weights[weights < 0]}")
    if np.any(weights == 0):
        weight_issues.append(f"{np.sum(weights == 0)} zero-weight components")

    # Check covariances
    cov_issues = []
    singular_count = 0
    for i in range(n_components):
        cov = covariances[i]

        # Check symmetry
        if not np.allclose(cov, cov.T):
            cov_issues.append(f"Component {i}: Non-symmetric covariance")

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < 1e-8):
            singular_count += 1
            if singular_count <= 3:  # Only show first 3
                cov_issues.append(f"Component {i}: Near-singular (min eig={eigvals.min():.2e})")

    if singular_count > 3:
        cov_issues.append(f"... and {singular_count-3} more singular components")

    # Mean position check
    mean_range = (means.min(), means.max())
    mean_center = means.mean(axis=0)

    return {
        'n_components': n_components,
        'weight_sum': weight_sum,
        'weight_range': (weights.min(), weights.max()),
        'weight_issues': weight_issues,
        'cov_issues': cov_issues,
        'singular_count': singular_count,
        'mean_range': mean_range,
        'mean_center': mean_center
    }


def compare_gmm_directories(adaptive_dir, fixed_dir, n_samples=10):
    """Compare adaptive vs fixed GMMs."""
    print("=" * 80)
    print("GMM Diagnostic Report")
    print("=" * 80)
    print()

    # Get common files
    adaptive_files = set(os.listdir(adaptive_dir))
    fixed_files = set(os.listdir(fixed_dir))
    common_files = sorted(adaptive_files & fixed_files)

    # Sample files
    if len(common_files) > n_samples:
        indices = np.linspace(0, len(common_files)-1, n_samples, dtype=int)
        sampled_files = [common_files[i] for i in indices]
    else:
        sampled_files = common_files

    print(f"Comparing {len(sampled_files)} GMM files...\n")

    adaptive_stats = []
    fixed_stats = []

    for filename in sampled_files:
        adaptive_path = os.path.join(adaptive_dir, filename)
        fixed_path = os.path.join(fixed_dir, filename)

        print(f"File: {filename}")
        print("-" * 40)

        # Adaptive GMM
        print("ADAPTIVE:")
        try:
            adaptive_info = check_gmm_file(adaptive_path)
            adaptive_stats.append(adaptive_info)

            print(f"  Components: {adaptive_info['n_components']}")
            print(f"  Weights: sum={adaptive_info['weight_sum']:.6f}, "
                  f"range=({adaptive_info['weight_range'][0]:.6f}, {adaptive_info['weight_range'][1]:.6f})")
            print(f"  Singular components: {adaptive_info['singular_count']}")

            if adaptive_info['weight_issues']:
                print(f"  ⚠️  WEIGHT ISSUES:")
                for issue in adaptive_info['weight_issues']:
                    print(f"     - {issue}")

            if adaptive_info['cov_issues']:
                print(f"  ⚠️  COVARIANCE ISSUES:")
                for issue in adaptive_info['cov_issues'][:5]:  # Show max 5
                    print(f"     - {issue}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

        # Fixed GMM
        print("\nFIXED (100 components):")
        try:
            fixed_info = check_gmm_file(fixed_path)
            fixed_stats.append(fixed_info)

            print(f"  Components: {fixed_info['n_components']}")
            print(f"  Weights: sum={fixed_info['weight_sum']:.6f}, "
                  f"range=({fixed_info['weight_range'][0]:.6f}, {fixed_info['weight_range'][1]:.6f})")
            print(f"  Singular components: {fixed_info['singular_count']}")

            if fixed_info['weight_issues']:
                print(f"  ⚠️  WEIGHT ISSUES:")
                for issue in fixed_info['weight_issues']:
                    print(f"     - {issue}")

            if fixed_info['cov_issues']:
                print(f"  ⚠️  COVARIANCE ISSUES:")
                for issue in fixed_info['cov_issues'][:5]:
                    print(f"     - {issue}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if adaptive_stats:
        avg_adaptive_components = np.mean([s['n_components'] for s in adaptive_stats])
        avg_adaptive_singular = np.mean([s['singular_count'] for s in adaptive_stats])
        print(f"Adaptive GMMs:")
        print(f"  Average components: {avg_adaptive_components:.1f}")
        print(f"  Average singular components: {avg_adaptive_singular:.1f} "
              f"({100*avg_adaptive_singular/avg_adaptive_components:.1f}%)")

    if fixed_stats:
        avg_fixed_singular = np.mean([s['singular_count'] for s in fixed_stats])
        print(f"\nFixed GMMs:")
        print(f"  Components: 100")
        print(f"  Average singular components: {avg_fixed_singular:.1f} ({avg_fixed_singular:.1f}%)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose GMM file issues')
    parser.add_argument('--sequence', type=str, default='00',
                       help='KITTI sequence number')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of files to sample')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sequence_dir = os.path.join(script_dir, f'kitti_sequence_{args.sequence}')

    adaptive_dir = os.path.join(sequence_dir, 'adaptive_components')
    fixed_dir = os.path.join(sequence_dir, '100_components')

    if not os.path.exists(adaptive_dir):
        print(f"ERROR: Adaptive directory not found: {adaptive_dir}")
        exit(1)

    if not os.path.exists(fixed_dir):
        print(f"ERROR: Fixed directory not found: {fixed_dir}")
        exit(1)

    compare_gmm_directories(adaptive_dir, fixed_dir, args.n_samples)
