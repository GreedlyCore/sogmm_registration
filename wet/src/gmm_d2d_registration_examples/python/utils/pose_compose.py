import numpy as np


def pose_compose(T1, T2):

    R1 = T1[0:3, 0:3]
    t1 = T1[0:3, 3]
    R2 = T2[0:3, 0:3]
    t2 = T2[0:3, 3]

    Tout = np.eye(4)
    Tout[0:3, 3] = t1 + np.matmul(R1, t2)
    Tout[0:3, 0:3] = np.matmul(R1, R2)
    return Tout
