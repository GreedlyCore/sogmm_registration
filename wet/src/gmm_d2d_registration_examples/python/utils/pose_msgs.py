import numpy as np


class PoseMsgs:
    positions = None  # 3 x N
    orientations = None  # 4 x N
    linear_velocity = None  # 3 x N
    angular_velocity = None  # 3 x N
    times = None  # 1 x N

    def __init__(self):
        pass

    def closest_time(self, time):
        idx = np.argmin(np.abs(time - self.times))
        diff = np.min(np.abs(time - self.times))
        return (diff, idx)
