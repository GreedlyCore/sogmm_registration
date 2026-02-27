import numpy as np
from utils.ZYXToR import ZYXToR
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse


def _rel_to_SE3(vec):
    T = np.eye(4)
    T[0:3, 0:3] = ZYXToR(vec[3:6])
    T[0:3, 3] = vec[0:3]
    return T


def _rot_angle(R):
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_a)


def compute_metrics(transforms, ground_truths):
    n = len(transforms)

    t_sq, r_sq = [], []
    for i in range(n):
        P_rel = _rel_to_SE3(transforms[i])
        Q_rel = _rel_to_SE3(ground_truths[i])
        E = pose_compose(pose_inverse(Q_rel), P_rel)
        t_sq.append(np.linalg.norm(E[0:3, 3]) ** 2)
        r_sq.append(_rot_angle(E[0:3, 0:3]) ** 2)

    rmse_t = np.sqrt(np.mean(t_sq))
    rmse_r = np.sqrt(np.mean(r_sq))
    oe_t   = np.mean(np.sqrt(t_sq))
    oe_r   = np.mean(np.sqrt(r_sq))

    print(f'RMSE  translation : {rmse_t:.6f} m')
    print(f'RMSE  rotation    : {np.degrees(rmse_r):.6f} deg  ({rmse_r:.6f} rad)')
    print(f'OE    translation : {oe_t:.6f} m')
    print(f'OE    rotation    : {np.degrees(oe_r):.6f} deg  ({oe_r:.6f} rad)')
