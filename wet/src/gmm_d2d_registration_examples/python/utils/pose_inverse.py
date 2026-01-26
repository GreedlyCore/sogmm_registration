import numpy as np


def pose_inverse(Tin):

    R = Tin[0:3, 0:3]
    T = Tin[0:3, 3]
    T = -1 * np.matmul(np.linalg.inv(R), T)
    R = np.linalg.inv(R)

    Tout = np.eye(4)
    Tout[0:3, 0:3] = R
    Tout[0:3, 3] = T
    return Tout
