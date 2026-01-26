import numpy as np
from numpy import cos as cos
from numpy import sin as sin


def ZYXToR(angles):
    # rot = ZYXToR(angles)
    # convert ZYX Euler angles to rotation matrix

    ang = angles.flatten()
    if len(ang) != 3:
        print('np.shape(angles) must equal 3')
        return np.eye(3)

    phi = ang[0]
    theta = ang[1]
    psi = ang[2]

    rot = np.eye(3)
    rot[0, 0] = cos(theta)*cos(psi)
    rot[0, 1] = cos(psi)*sin(theta)*sin(phi) - cos(phi)*sin(psi)
    rot[0, 2] = cos(phi)*cos(psi)*sin(theta) + sin(phi)*sin(psi)

    rot[1, 0] = cos(theta)*sin(psi)
    rot[1, 1] = cos(phi)*cos(psi) + sin(theta)*sin(phi)*sin(psi)
    rot[1, 2] = -cos(psi)*sin(phi) + cos(phi)*sin(theta)*sin(psi)

    rot[2, 0] = -sin(theta)
    rot[2, 1] = cos(theta)*sin(phi)
    rot[2, 2] = cos(theta)*cos(phi)

    return rot
