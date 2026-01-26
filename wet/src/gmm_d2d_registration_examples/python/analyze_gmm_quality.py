import numpy as np
import sys
from pathlib import Path

def analyze_gmm(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    means = data[:, 0:3]
    covs_flat = data[:, 3:12]
    weights = data[:, 12]

    n_components = len(data)
    covs = covs_flat.reshape(n_components, 3, 3)

    dets = np.array([np.linalg.det(cov) for cov in covs])
    eigenvalues = np.array([np.linalg.eigvals(cov) for cov in covs])

    print(f"File: {filepath.name}")
    print(f"Components: {n_components}")
    print(f"Weights - min: {weights.min():.6f}, max: {weights.max():.6f}, sum: {weights.sum():.6f}")
    print(f"Determinants - min: {dets.min():.6e}, max: {dets.max():.6e}, mean: {dets.mean():.6e}")
    print(f"Eigenvalues - min: {eigenvalues.min():.6e}, max: {eigenvalues.max():.6e}")
    print(f"Degenerate covs (det < 1e-10): {(dets < 1e-10).sum()}")
    print(f"Small weight components (< 0.001): {(weights < 0.001).sum()}")
    print()

    return {
        'n_components': n_components,
        'weight_min': weights.min(),
        'weight_max': weights.max(),
        'det_min': dets.min(),
        'det_max': dets.max(),
        'det_mean': dets.mean(),
        'degenerate_count': (dets < 1e-10).sum()
    }

if len(sys.argv) > 1:
    analyze_gmm(Path(sys.argv[1]))
else:
    print("=== Fixed 100 components ===")
    analyze_gmm(Path("kitti_sequence_00/100_components/1.gmm"))
    analyze_gmm(Path("kitti_sequence_00/100_components/100.gmm"))

    print("=== Adaptive components ===")
    analyze_gmm(Path("kitti_sequence_00/adaptive_components/1.gmm"))
    analyze_gmm(Path("kitti_sequence_00/adaptive_components/100.gmm"))
