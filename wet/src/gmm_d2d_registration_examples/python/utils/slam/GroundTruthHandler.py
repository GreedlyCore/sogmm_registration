import matplotlib.pyplot as plt
import numpy as np

from utils.RToZYX import RToZYX
from utils.ZYXToR import ZYXToR
from utils.QuatToR import QuatToR
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse

class GroundTruthHandler:
    gt_poses = None

    def __init__(self, ground_truths):
        self.gt_poses = np.zeros(np.shape(ground_truths))
        Tgt = np.eye(4)
        for i in range(0, np.shape(ground_truths)[0]):
            R = ZYXToR(ground_truths[i,3:6])
            t = np.transpose(ground_truths[i,0:3])
            T = np.eye(4)
            T[0:3,0:3] = R
            T[0:3, 3] = t
            Tgt = pose_compose(Tgt, T)
            row = np.append(np.transpose(Tgt[0:3,3]), RToZYX(Tgt[0:3, 0:3]))
            self.gt_poses[i, :] = row

    def plot(self, ax, idx=None):

        if idx == None:
            idx = np.shape(self.gt_poses)[0]-1

        xyz = self.gt_poses[1:idx,0:3] 
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], label='Ground Truth')
