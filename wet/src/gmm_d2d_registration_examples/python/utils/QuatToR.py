import numpy as np


def QuatToR(quat):
    # rot = QuatToR(quat)
    # convert quaternion to rotation matrix
    # Assumes input is q = [qx qy qz qw]

    a = quat[3]  # w
    b = quat[0]  # x
    c = quat[1]  # y
    d = quat[2]  # z

    rot = np.zeros((3, 3))

    rot[0, 0] = a*a + b*b - c*c - d*d
    rot[0, 1] = 2*b*c - 2*a*d
    rot[0, 2] = 2*b*d + 2*a*c
    rot[1, 0] = 2*b*c + 2*a*d
    rot[1, 1] = a*a - b*b + c*c - d*d
    rot[1, 2] = 2*c*d - 2*a*b
    rot[2, 0] = 2*b*d - 2*a*c
    rot[2, 1] = 2*c*d + 2*a*b
    rot[2, 2] = a*a - b*b - c*c + d*d

    return rot
