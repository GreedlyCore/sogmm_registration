#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from utils.ZYXToR import ZYXToR
from utils.pose_compose import pose_compose


def plot_results(transforms, ground_truths, FIRST_SCAN, LAST_SCAN, save_path=None):

    # plot the results
    Tgmm = [np.eye(4)]
    Tgt = [np.eye(4)]

    # Use actual array size (handles skip_scans for VIRAL dataset)
    n_pairs = transforms.shape[0]
    for i in range(n_pairs):
        T = np.eye(4)
        T[0:3, 0:3] = ZYXToR(transforms[i, 3:6])
        T[0:3, 3] = np.transpose(transforms[i, 0:3])
        T2 = pose_compose(Tgmm[-1], T)
        Tgmm.append(T2)

        T = np.eye(4)
        T[0:3, 0:3] = ZYXToR(ground_truths[i, 3:6])
        T[0:3, 3] = np.transpose(ground_truths[i, 0:3])
        T2 = pose_compose(Tgt[-1], T)
        Tgt.append(T2)

    xyz_gmm = np.zeros((len(Tgmm), 3))
    xyz_gt = np.zeros((len(Tgt), 3))
    for i in range(0, len(Tgmm)):
        xyz_gmm[i, :] = np.transpose(Tgmm[i][0:3, 3])
        xyz_gt[i, :] = np.transpose(Tgt[i][0:3, 3])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # XY top-down view
    axes[0].plot(xyz_gmm[:, 0], xyz_gmm[:, 1], label='GMM D2D Registration')
    axes[0].plot(xyz_gt[:, 0],  xyz_gt[:, 1],  label='Ground Truth')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Top-down (XY)')
    axes[0].set_aspect('equal')
    axes[0].legend()

    # 3D view — no forced equal aspect so both trajectories are always visible
    ax3 = fig.add_subplot(1, 2, 2, projection='3d')
    ax3.plot(xyz_gmm[:, 0], xyz_gmm[:, 1], xyz_gmm[:, 2], label='GMM D2D Registration')
    ax3.plot(xyz_gt[:, 0],  xyz_gt[:, 1],  xyz_gt[:, 2],  label='Ground Truth')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D view')
    ax3.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
