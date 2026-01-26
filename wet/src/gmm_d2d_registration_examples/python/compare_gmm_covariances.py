#!/usr/bin/env python
"""
Compare covariance statistics between adaptive and fixed GMMs.
"""
import numpy as np
import os

def analyze_gmm_covariances(filepath):
    """Load GMM and compute covariance statistics."""
    data = np.loadtxt(filepath, delimiter=',')
    n_components = data.shape[0]

    means = data[:, 0:3]
    covariances = data[:, 3:12].reshape(n_components, 3, 3)
    weights = data[:, 12]

    # Compute eigenvalues for all components
    all_eigvals = []
    for i in range(n_components):
        eigvals = np.linalg.eigvalsh(covariances[i])
        all_eigvals.append(eigvals)

    all_eigvals = np.array(all_eigvals)

    # Compute determinants (volume of ellipsoids)
    determinants = np.array([np.linalg.det(covariances[i]) for i in range(n_components)])

    # Compute Frobenius norms (overall scale)
    frobenius_norms = np.array([np.linalg.norm(covariances[i], 'fro') for i in range(n_components)])

    # Check symmetry
    max_asymmetry = 0
    for i in range(n_components):
        asymmetry = np.max(np.abs(covariances[i] - covariances[i].T))
        max_asymmetry = max(max_asymmetry, asymmetry)

    return {
        'n_components': n_components,
        'means': means,
        'covariances': covariances,
        'weights': weights,
        'eigenvalues': all_eigvals,
        'determinants': determinants,
        'frobenius_norms': frobenius_norms,
        'max_asymmetry': max_asymmetry
    }


def compare_single_pair(adaptive_file, fixed_file, file_id):
    """Compare one adaptive vs one fixed GMM."""
    print(f"\n{'='*80}")
    print(f"File: {file_id}.gmm")
    print('='*80)

    adaptive = analyze_gmm_covariances(adaptive_file)
    fixed = analyze_gmm_covariances(fixed_file)

    print(f"\nADAPTIVE ({adaptive['n_components']} components):")
    print(f"  Mean position range: [{adaptive['means'].min():.2f}, {adaptive['means'].max():.2f}]")
    print(f"  Mean center: {adaptive['means'].mean(axis=0)}")
    print(f"  Eigenvalues (min/median/max): {adaptive['eigenvalues'].min():.6f} / "
          f"{np.median(adaptive['eigenvalues']):.6f} / {adaptive['eigenvalues'].max():.6f}")
    print(f"  Determinants (min/median/max): {adaptive['determinants'].min():.6e} / "
          f"{np.median(adaptive['determinants']):.6e} / {adaptive['determinants'].max():.6e}")
    print(f"  Frobenius norms (min/median/max): {adaptive['frobenius_norms'].min():.6f} / "
          f"{np.median(adaptive['frobenius_norms']):.6f} / {adaptive['frobenius_norms'].max():.6f}")
    print(f"  Max asymmetry: {adaptive['max_asymmetry']:.6e}")

    print(f"\nFIXED (100 components):")
    print(f"  Mean position range: [{fixed['means'].min():.2f}, {fixed['means'].max():.2f}]")
    print(f"  Mean center: {fixed['means'].mean(axis=0)}")
    print(f"  Eigenvalues (min/median/max): {fixed['eigenvalues'].min():.6f} / "
          f"{np.median(fixed['eigenvalues']):.6f} / {fixed['eigenvalues'].max():.6f}")
    print(f"  Determinants (min/median/max): {fixed['determinants'].min():.6e} / "
          f"{np.median(fixed['determinants']):.6e} / {fixed['determinants'].max():.6e}")
    print(f"  Frobenius norms (min/median/max): {fixed['frobenius_norms'].min():.6f} / "
          f"{np.median(fixed['frobenius_norms']):.6f} / {fixed['frobenius_norms'].max():.6f}")
    print(f"  Max asymmetry: {fixed['max_asymmetry']:.6e}")

    # Compare scales
    print(f"\nCOMPARISON:")
    eig_ratio = np.median(adaptive['eigenvalues']) / np.median(fixed['eigenvalues'])
    det_ratio = np.median(adaptive['determinants']) / np.median(fixed['determinants'])
    frob_ratio = np.median(adaptive['frobenius_norms']) / np.median(fixed['frobenius_norms'])

    print(f"  Eigenvalue ratio (adaptive/fixed): {eig_ratio:.4f}")
    print(f"  Determinant ratio (adaptive/fixed): {det_ratio:.4f}")
    print(f"  Frobenius norm ratio (adaptive/fixed): {frob_ratio:.4f}")

    if eig_ratio > 2.0 or eig_ratio < 0.5:
        print(f"  ⚠️  SIGNIFICANT SCALE DIFFERENCE!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare adaptive vs fixed GMM covariances')
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--n_samples', type=int, default=5)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sequence_dir = os.path.join(script_dir, f'kitti_sequence_{args.sequence}')

    adaptive_dir = os.path.join(sequence_dir, 'adaptive_components')
    fixed_dir = os.path.join(sequence_dir, '100_components')

    # Get common files
    adaptive_files = sorted([f for f in os.listdir(adaptive_dir) if f.endswith('.gmm')])

    # Sample files evenly
    indices = np.linspace(0, len(adaptive_files)-1, args.n_samples, dtype=int)
    sampled_files = [adaptive_files[i] for i in indices]

    for filename in sampled_files:
        file_id = filename.replace('.gmm', '')
        adaptive_file = os.path.join(adaptive_dir, filename)
        fixed_file = os.path.join(fixed_dir, filename)

        if os.path.exists(adaptive_file) and os.path.exists(fixed_file):
            compare_single_pair(adaptive_file, fixed_file, file_id)
