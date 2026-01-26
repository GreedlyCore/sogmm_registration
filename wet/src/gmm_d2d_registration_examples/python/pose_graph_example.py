#!/usr/bin/env python
import copy
import numpy as np
import os
from os.path import expanduser
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import gtsam

import gmm_d2d_registration_py
from utils.RToZYX import RToZYX
from utils.ZYXToR import ZYXToR
from utils.QuatToR import QuatToR
from utils.pose_compose import pose_compose
from utils.pose_inverse import pose_inverse
from utils.run_dataset import run_dataset
from utils.plot_results import plot_results

from utils.slam.PoseGraph import PoseGraph
from utils.slam.GroundTruthHandler import GroundTruthHandler

def find_potential_loop_closures(values, curr_pose, distance):
    keys = np.arange(1, len(values.keys()))
    idxs = []

    curr_translation = curr_pose[0:3, 3]
    for k in values.keys():
        T = values.atPose3(k).matrix()
        translation = T[0:3,3]
        d = np.linalg.norm(curr_translation-translation)
        if (d < distance):
            idxs.append(k)
    return idxs

def draw(groundtruth_handler, pose_graph, plt, ax, i, color=None):
    plt.cla()
    groundtruth_handler.plot(ax, i)
    pose_graph.plot(ax, color)
    ax.legend()
    ax.set_aspect('equal', 'box')
    plt.draw()
    plt.pause(0.05)

    if not os.path.isdir('./images'):
        os.mkdir('./images')

    plt.savefig('./images/' + f"{i:06d}" + '.png')

def main():
    cwd = os.getcwd()
    SANDBOX_NAME = 'gira-registration'
    matches = cwd.split(SANDBOX_NAME)
    GIRA3D_REGISTRATION_SANDBOX = matches[0] + SANDBOX_NAME
    REGISTRATION_RESULTS_DIR = GIRA3D_REGISTRATION_SANDBOX + '/data/slam/results/'
    DATA_DIR = GIRA3D_REGISTRATION_SANDBOX + '/data/slam/'
    NUM_COMPONENTS = 70
    DISTANCE_BETWEEN_FACTORS = 0.5
    MAX_DISTANCE_TO_SEARCH = 10.0
    LOOK_BACK_KEYS = 25
    MIN_SCORE_TO_CLOSE_LOOP = 0.65
    count = 0

    ax = plt.figure().add_subplot(projection='3d')

    ground_truths_tmp, transforms_tmp = pickle.load(
        open(REGISTRATION_RESULTS_DIR + 'isoplanarhybrid_' + str(NUM_COMPONENTS)
             + '_results.pkl', 'rb'))

    idxs = [1]

    # Prepend a zero to the ground truth and transforms
    ground_truths = np.zeros((np.shape(ground_truths_tmp)[0]+1,
                              np.shape(ground_truths_tmp)[1]))
    transforms = np.zeros((np.shape(transforms_tmp)[0]+1,
                           np.shape(transforms_tmp)[1]))

    ground_truths[1:,:] = ground_truths_tmp
    transforms[1:,:] = transforms_tmp

    # current pose
    Tm = np.eye(4)
    pose_graph = PoseGraph()
    groundtruth_handler = GroundTruthHandler(ground_truths)
    for i in range(1, np.shape(transforms)[0]):
        T = np.eye(4)
        T[0:3, 0:3] = ZYXToR(transforms[i,3:6])
        T[0:3, 3] = transforms[i, 0:3]

        Tm = pose_compose(Tm, T)

        pose_graph.update_odometry_incrementally(T)
        T = pose_graph.get_curr_pose()

        if np.linalg.norm(T[0:3, 3]) > DISTANCE_BETWEEN_FACTORS:
            pose_graph.create_between_factor()
            Tm = pose_graph.values.atPose3(pose_graph.key).matrix()
            idxs.append(i)

            # do some plotting
            idxs_potential = find_potential_loop_closures(pose_graph.values, Tm, MAX_DISTANCE_TO_SEARCH)
            idxs_potential_bools = np.array(idxs_potential) < (pose_graph.key - LOOK_BACK_KEYS)
            idxs_in_range = [y for y, x in enumerate(idxs_potential_bools) if x]

            draw(groundtruth_handler, pose_graph, plt, ax, i)

            if len(idxs_in_range) and count > 1:
                print('Found potential loop closure within a ' + str(MAX_DISTANCE_TO_SEARCH) + ' radius\n')

                closed_loop = False

                for j in range(1, len(idxs_in_range)):
                    k = idxs_potential[idxs_in_range[j]]

                    source_file = DATA_DIR + 'gmms/' + f"{i:06d}" + '.gmm'
                    target_file = DATA_DIR + 'gmms/' + f"{idxs[k]:06d}" + '.gmm'

                    key_target = k

                    T0t = pose_graph.values.atPose3(key_target).matrix()
                    T0s = pose_graph.values.atPose3(pose_graph.key).matrix()
                    Tts = pose_compose(pose_inverse(T0t), T0s)

                    # align source with target
                    output = gmm_d2d_registration_py.isoplanar_registration(Tts, source_file, target_file)
                    Tts = output[0]
                    score = output[1]

                    if score > MIN_SCORE_TO_CLOSE_LOOP:
                        print('Got loop closure between ' + str(i) + ' and ' +
                              str(idxs[k]) + ' with score ' + str(score))
                        delta = pose_graph.to_gtsam(Tts)
                        pose_graph.add_loop_constraint(k, delta)
                        closed_loop = True
                        count = 0
                    else:
                        print('Failed to get loop closure between ' + str(i) +
                              ' and ' + str(idxs[k]) + ' with score ' + str(score))

                    pose_graph.calculate_estimate()
                    Tm = pose_graph.values.atPose3(len(pose_graph.values.keys())).matrix()

                    if closed_loop:
                        draw(groundtruth_handler, pose_graph, plt, ax, i)

            else:
                count = count + 1

            pose_graph.increment_key()

    print('Done!')
    draw(groundtruth_handler, pose_graph, plt, ax, i)
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
