import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import arcsin as asin
from numpy import arctan2 as atan2


def RToZYX(R):
    eps = 2.2204e-16

    theta = -asin(R[2, 0])
    phi = 0
    psi = 0

    if cos(theta) > eps:
        phi = atan2(R[2, 1], R[2, 2])
        psi = atan2(R[1, 0], R[0, 0])

    return np.array([phi, theta, psi])
