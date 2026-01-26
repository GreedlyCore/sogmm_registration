import scipy.io
import numpy as np
import pickle

from utils.pointcloud2_msgs import PointCloud2Msgs
from utils.pose_msgs import PoseMsgs

dataset = 'mine_001_part3'

mat = scipy.io.loadmat('../../../../data/' + dataset + '/ground_truth.mat')
positions = mat['positions']
orientations = mat['orientations']
times = mat['times']
odometry = PoseMsgs()
odometry.positions = positions
odometry.orientations = orientations
odometry.times = times.flatten()

mat = scipy.io.loadmat('../../../../data/' + dataset + '/K.mat')
K = mat['K']

mat = scipy.io.loadmat('../../../../data/' + dataset + '/Tbc.mat')
Tbc = mat['Tbc']

mat = scipy.io.loadmat('../../../../data/' + dataset + '/pointcloud_times.mat')
pointclouds = PointCloud2Msgs()
pointclouds.times = mat['times'].flatten()

with open('mine_odometry.pkl', 'wb') as handle:
    pickle.dump([K, Tbc, odometry, pointclouds], handle)
