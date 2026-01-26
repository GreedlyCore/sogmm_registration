import matplotlib.pyplot as plt
import numpy as np

import gtsam

class PoseGraph:
    pose_graph = None
    values = []
    covariance = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.01,
                                                   0.04, 0.04, 0.04])
    key = 1
    odometry = gtsam.Pose3()

    def __init__(self):
        isamParams = gtsam.ISAM2Params()
        isamParams.setRelinearizeThreshold(0.01)
        self.pose_graph = gtsam.ISAM2(isamParams)

        new_factor = gtsam.NonlinearFactorGraph()
        new_value = gtsam.Values()

        sigma_init_x = gtsam.noiseModel.Isotropic.Precisions([0.1, 0.1, 0.1, 0.02, 0.02, 0.02])

        new_value.insert(self.key, gtsam.Pose3());
        new_factor.add(gtsam.PriorFactorPose3(1, gtsam.Pose3(), sigma_init_x));
        
        self.update(new_factor, new_value);
        self.increment_key()

    def update(self, new_factor, new_value):
        self.pose_graph.update(new_factor, new_value)
        self.values = self.pose_graph.calculateEstimate()

    def increment_key(self):
      self.key = self.key + 1

    def to_gtsam(self, T):
      gtsamT = gtsam.Pose3(gtsam.Rot3(T[0:3, 0:3]),
                           gtsam.Point3(T[0:3,3]))
      return gtsamT

    def from_gtsam(self, gtsamT):
      T = gtsamT.matrix()
      return T

    def update_odometry_incrementally(self, T):
      self.odometry = self.odometry.compose(self.to_gtsam(T))

    def get_curr_pose(self):
      T = self.odometry.matrix()
      return T

    def add_loop_constraint(self, k, delta):
      new_factor = gtsam.NonlinearFactorGraph()
      new_factor.add(gtsam.BetweenFactorPose3(k, self.key, delta, self.covariance))
      self.pose_graph.update(new_factor, gtsam.Values())

    def create_between_factor(self):
      new_factor = gtsam.NonlinearFactorGraph()
      new_value = gtsam.Values()
      new_factor.add(gtsam.BetweenFactorPose3(self.key-1, self.key, 
					      self.odometry, self.covariance))
      self.odometry = gtsam.Pose3()
      last_pose = self.values.atPose3(self.key-1)
      new_value.insert(self.key, last_pose.compose(self.odometry))
      self.update(new_factor, new_value)

    def calculate_estimate(self):
      self.values = self.pose_graph.calculateEstimate();

    def plot(self, ax, color=None):
        xyz = np.zeros((self.key-1, 3))
        for i in range(1,self.key-1):
            T = self.values.atPose3(i).matrix()
            xyz[i,:] = np.transpose(T[0:3, 3])
        if color is None:
            ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], label='GMM')
        else:
            ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], label='GMM', color=color)

