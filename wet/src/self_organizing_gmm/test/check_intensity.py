#!/usr/bin/env python
"""
Check intensity field in VIRAL bag lidar scans. Intensity required for properly
working of adaptive part of SOGMM algorithm.

python check_intensity.py --bag ~/thesis/VIRAL/eee_03/eee_03.bag
python check_intensity.py --bag ~/thesis/VIRAL/nya_01/nya_01.bag --n 10
"""
import argparse
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from utils.viral_loader import VIRAL_OUSTER_DTYPE


def check_intensity(bag_path, topic='/os1_cloud_node1/points', n_scans=5):
    typestore = get_typestore(Stores.ROS1_NOETIC)

    print(f'Bag : {bag_path}')
    print(f'Topic: {topic}')
    print(f'Checking first {n_scans} scans...\n')

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = [c.topic for c in reader.connections]
            print(f'ERROR: topic not found. Available:\n  ' + '\n  '.join(available))
            return

        for i, (conn, ts, raw) in enumerate(reader.messages(connections=connections)):
            if i >= n_scans:
                break

            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            pts = np.frombuffer(msg.data, dtype=VIRAL_OUSTER_DTYPE)

            intensity  = pts['intensity']
            reflectivity = pts['reflectivity']
            ambient    = pts['ambient']

            nonzero_i  = np.count_nonzero(intensity)
            nonzero_r  = np.count_nonzero(reflectivity)

            print(f'Scan {i:3d} | {len(pts):6d} pts')
            print(f'  intensity    min={intensity.min():.2f}  max={intensity.max():.2f}'
                  f'  mean={intensity.mean():.2f}  nonzero={nonzero_i}/{len(pts)}')
            print(f'  reflectivity min={reflectivity.min()}  max={reflectivity.max()}'
                  f'  nonzero={nonzero_r}/{len(pts)}')
            print(f'  ambient      min={ambient.min()}  max={ambient.max()}')
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--topic', default='/os1_cloud_node1/points')
    parser.add_argument('--n', type=int, default=5, dest='n_scans',
                        help='Number of scans to inspect (default: 5)')
    args = parser.parse_args()

    import os
    args.bag = os.path.expanduser(args.bag)
    check_intensity(args.bag, args.topic, args.n_scans)


if __name__ == '__main__':
    main()