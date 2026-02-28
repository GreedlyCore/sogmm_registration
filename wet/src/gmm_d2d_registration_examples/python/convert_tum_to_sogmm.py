#!/usr/bin/env python
"""
Convert TUM RGB-D rosbag to per-scan 4D point cloud .txt files.

Each output .txt: N rows of  x,y,z,intensity  (comma-separated, float32).
Intensity: luminance 0.299·R + 0.587·G + 0.114·B  in [0, 1].
Depth→3D back-projection uses camera intrinsics read from the bag.
RGB lookup uses TF chain: depth_optical → camera → rgb_optical.

Output path default:
  sogmm_registration/data/{bag_name}/pointclouds/{depth_idx}.txt

Usage:
    python convert_tum_to_sogmm.py --bag ~/thesis/mai_city/bags/rgbd_dataset_freiburg3_long_office_household.bag
    python convert_tum_to_sogmm.py --bag ~/thesis/mai_city/bags/rgbd_dataset_freiburg3_long_office_household.bag --start-id 100 --final-id 500
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


DEPTH_TOPIC      = '/camera/depth/image'
RGB_TOPIC        = '/camera/rgb/image_color'
DEPTH_INFO_TOPIC = '/camera/depth/camera_info'
RGB_INFO_TOPIC   = '/camera/rgb/camera_info'
TF_TOPIC         = '/tf'

RGB_SYNC_MAX_DT = 50e6   # 50ms in nanoseconds
MAX_RGB_BUFFER  = 60     # rolling buffer size (~2s at 30Hz)

"""
Bag file: rgbd_dataset_freiburg3_long_office_household.bag
============================================================
Total messages: 31121
Total topics: 6
============================================================
Topics Details
============================================================

Topic: /camera/depth/camera_info
  Type: sensor_msgs/msg/CameraInfo
  Messages: 2510
  Frequency: 28.78 Hz
  Time range: 19:33:00.723 - 19:34:27.895
  Duration: 87.17 seconds
  Connections: 1

Topic: /camera/depth/image
  Type: sensor_msgs/msg/Image
  Messages: 2509
  Frequency: 28.78 Hz
  Time range: 19:33:00.723 - 19:34:27.862
  Duration: 87.14 seconds
  Connections: 1

Topic: /camera/rgb/camera_info
  Type: sensor_msgs/msg/CameraInfo
  Messages: 2586
  Frequency: 29.65 Hz
  Time range: 19:33:00.722 - 19:34:27.895
  Duration: 87.17 seconds
  Connections: 1

Topic: /camera/rgb/image_color
  Type: sensor_msgs/msg/Image
  Messages: 2585
  Frequency: 29.65 Hz
  Time range: 19:33:00.722 - 19:34:27.862
  Duration: 87.14 seconds
  Connections: 1

Topic: /cortex_marker_array
  Type: visualization_msgs/msg/MarkerArray
  Messages: 8726
  Frequency: 100.00 Hz
  Time range: 19:33:00.630 - 19:34:27.879
  Duration: 87.25 seconds
  Connections: 1

Topic: /tf
  Type: tf/msg/tfMessage
  Messages: 12205
  Frequency: 139.77 Hz
  Time range: 19:33:00.630 - 19:34:27.947
  Duration: 87.32 seconds
  Connections: 1

============================================================
Topics List (Quick Reference)
============================================================
  /camera/depth/camera_info                sensor_msgs/msg/CameraInfo     (2510 msgs)
  /camera/depth/image                      sensor_msgs/msg/Image          (2509 msgs)
  /camera/rgb/camera_info                  sensor_msgs/msg/CameraInfo     (2586 msgs)
  /camera/rgb/image_color                  sensor_msgs/msg/Image          (2585 msgs)
  /cortex_marker_array                     visualization_msgs/msg/MarkerArray (8726 msgs)
  /tf                                      tf/msg/tfMessage               (12205 msgs)
"""
def _quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)  ],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)  ],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2)],
    ])


def _decode_image(msg):
    """Returns (array, depth_scale_or_None).
    depth_scale: divide raw depth values by this to get meters.
    """
    data = np.frombuffer(msg.data, dtype=np.uint8)
    h, w, enc = msg.height, msg.width, msg.encoding
    if '16U' in enc:
        return data.view(np.uint16).reshape(h, w), 5000.0  # TUM 16-bit: /5000 = m
    elif enc == '32FC1':
        return data.view(np.float32).reshape(h, w), 1.0    # already in meters
    elif enc == 'rgb8':
        return data.reshape(h, w, 3), None
    elif enc == 'bgr8':
        return data.reshape(h, w, 3)[:, :, ::-1], None
    else:
        raise ValueError(f'Unsupported image encoding: {enc}')


def _read_intrinsics(bag_path, typestore, topic):
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            return np.array(msg.K, dtype=np.float64).reshape(3, 3)
    raise RuntimeError(f'No CameraInfo on {topic}')


def _load_tf_tree(bag_path, typestore):
    """Returns dict (parent, child) -> T 4x4."""
    with Reader(bag_path) as reader:
        tf_conns = [c for c in reader.connections if c.topic == TF_TOPIC]
        if not tf_conns:
            return {}
        _, msgdef_str = tf_conns[0].msgdef
    typestore.register(get_types_from_msg(msgdef_str, 'tf/msg/tfMessage'))

    seen = {}
    with Reader(bag_path) as reader:
        tf_conns = [c for c in reader.connections if c.topic == TF_TOPIC]
        for conn, ts, raw in reader.messages(connections=tf_conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            for t in msg.transforms:
                key = (t.header.frame_id, t.child_frame_id)
                if key not in seen:
                    tr, ro = t.transform.translation, t.transform.rotation
                    T = np.eye(4)
                    T[:3, :3] = _quat_to_R(ro.x, ro.y, ro.z, ro.w)
                    T[:3, 3]  = [tr.x, tr.y, tr.z]
                    seen[key] = T
    return seen


def _compute_T_depth_to_rgb_optical(tf_tree):
    """
    T such that  P_rgb_optical = T @ P_depth_optical.

    TF convention: T(parent, child) maps child-frame pts to parent-frame.
    Chain: depth_optical → depth_frame → camera → rgb_frame → rgb_optical

        P_depth_frame  = T1 @ P_depth_optical
        P_camera       = T2 @ P_depth_frame
        P_rgb_frame    = inv(T3) @ P_camera
        P_rgb_optical  = inv(T4) @ P_rgb_frame
    """
    T1 = tf_tree.get(('/openni_depth_frame', '/openni_depth_optical_frame'))
    T2 = tf_tree.get(('/openni_camera',      '/openni_depth_frame'))
    T3 = tf_tree.get(('/openni_camera',      '/openni_rgb_frame'))
    T4 = tf_tree.get(('/openni_rgb_frame',   '/openni_rgb_optical_frame'))

    if any(t is None for t in [T1, T2, T3, T4]):
        print('WARNING: TF chain incomplete, using identity for depth→rgb')
        return np.eye(4)

    return np.linalg.inv(T4) @ np.linalg.inv(T3) @ T2 @ T1


def _backproject(depth_img, depth_scale, rgb_img, K_depth, K_rgb, T_depth_to_rgb):
    """
    Back-project depth image to Nx4 [x,y,z,intensity] in depth_optical_frame.
    """
    H, W = depth_img.shape
    z = depth_img.astype(np.float32) / float(depth_scale)

    u_g = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    v_g = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))

    fx_d, fy_d = float(K_depth[0, 0]), float(K_depth[1, 1])
    cx_d, cy_d = float(K_depth[0, 2]), float(K_depth[1, 2])

    x = (u_g - cx_d) * z / fx_d
    y = (v_g - cy_d) * z / fy_d

    valid = z > 0
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1)  # Nx3

    if len(pts) == 0:
        return np.empty((0, 4), dtype=np.float32)

    # Project to RGB frame to look up colors
    pts_h   = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    pts_rgb = (T_depth_to_rgb.astype(np.float32) @ pts_h.T).T[:, :3]

    fx_r, fy_r = float(K_rgb[0, 0]), float(K_rgb[1, 1])
    cx_r, cy_r = float(K_rgb[0, 2]), float(K_rgb[1, 2])
    H_r, W_r   = rgb_img.shape[:2]

    u_r = (fx_r * pts_rgb[:, 0] / pts_rgb[:, 2] + cx_r).astype(np.int32)
    v_r = (fy_r * pts_rgb[:, 1] / pts_rgb[:, 2] + cy_r).astype(np.int32)

    in_bounds = (u_r >= 0) & (u_r < W_r) & (v_r >= 0) & (v_r < H_r)

    intensity = np.full(len(pts), 0.5, dtype=np.float32)
    if in_bounds.any():
        rgb_vals = rgb_img[v_r[in_bounds], u_r[in_bounds]].astype(np.float32) / 255.0
        intensity[in_bounds] = (0.299 * rgb_vals[:, 0]
                                + 0.587 * rgb_vals[:, 1]
                                + 0.114 * rgb_vals[:, 2])

    return np.column_stack([pts, intensity]).astype(np.float32)


# ---------------------------------------------------------------------------

def convert(bag_path, output_dir, start_id=None, final_id=None, skip_scans=1):
    if not os.path.exists(bag_path):
        print(f'ERROR: Bag not found: {bag_path}')
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    typestore = get_typestore(Stores.ROS1_NOETIC)

    print('Reading camera intrinsics...')
    K_depth = _read_intrinsics(bag_path, typestore, DEPTH_INFO_TOPIC)
    K_rgb   = _read_intrinsics(bag_path, typestore, RGB_INFO_TOPIC)
    print(f'  K_depth  fx={K_depth[0,0]:.2f}  fy={K_depth[1,1]:.2f}  cx={K_depth[0,2]:.2f}  cy={K_depth[1,2]:.2f}')
    print(f'  K_rgb    fx={K_rgb[0,0]:.2f}  fy={K_rgb[1,1]:.2f}  cx={K_rgb[0,2]:.2f}  cy={K_rgb[1,2]:.2f}')

    print('Reading TF...')
    tf_tree = _load_tf_tree(bag_path, typestore)
    T_d2r = _compute_T_depth_to_rgb_optical(tf_tree)
    t = T_d2r[:3, 3]
    print(f'  T depth→rgb_optical: translation=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]')

    depth_idx   = 0
    saved_count = 0
    rgb_buffer  = {}  # ts_ns -> rgb_img

    with Reader(bag_path) as reader:
        depth_conns = [c for c in reader.connections if c.topic == DEPTH_TOPIC]
        rgb_conns   = [c for c in reader.connections if c.topic == RGB_TOPIC]
        all_conns   = depth_conns + rgb_conns

        for conn, ts, raw in tqdm(reader.messages(connections=all_conns), desc='Converting'):

            # Buffer RGB
            if conn.topic == RGB_TOPIC:
                msg = typestore.deserialize_ros1(raw, conn.msgtype)
                img, _ = _decode_image(msg)
                rgb_buffer[ts] = img
                if len(rgb_buffer) > MAX_RGB_BUFFER:
                    del rgb_buffer[min(rgb_buffer.keys())]
                continue

            # Range filter
            if start_id is not None and depth_idx < start_id:
                depth_idx += 1
                continue
            if final_id is not None and depth_idx > final_id:
                break
            if (depth_idx - (start_id or 0)) % skip_scans != 0:
                depth_idx += 1
                continue

            if not rgb_buffer:
                depth_idx += 1
                continue

            # Match closest RGB
            best_rgb_ts = min(rgb_buffer, key=lambda t: abs(t - ts))
            if abs(best_rgb_ts - ts) > RGB_SYNC_MAX_DT:
                print(f'\nWARNING: no RGB match for depth frame {depth_idx}, skipping')
                depth_idx += 1
                continue

            depth_msg = typestore.deserialize_ros1(raw, conn.msgtype)
            depth_img, depth_scale = _decode_image(depth_msg)
            rgb_img = rgb_buffer[best_rgb_ts]

            points_4d = _backproject(depth_img, depth_scale, rgb_img, K_depth, K_rgb, T_d2r)

            if len(points_4d) == 0:
                depth_idx += 1
                continue

            out_path = os.path.join(output_dir, f'{depth_idx}.txt')
            np.savetxt(out_path, points_4d, delimiter=',', fmt='%.6f')

            saved_count += 1
            if saved_count % 100 == 1:
                print(f'\n  [{depth_idx}] {len(points_4d)} pts  →  {out_path}')
            depth_idx += 1

    print(f'\nDone. Saved {saved_count} files to: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Convert TUM rosbag to 4D point cloud .txt files')
    parser.add_argument('--bag', type=str, required=True,
                        help='Path to .bag file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir (default: sogmm_registration/data/{bag_name}/pointclouds/)')
    parser.add_argument('--start-id',  type=int, default=None, dest='start_id')
    parser.add_argument('--final-id',  type=int, default=None, dest='final_id')
    parser.add_argument('--skip-scans', type=int, default=1,   dest='skip_scans')
    args = parser.parse_args()

    args.bag = os.path.expanduser(args.bag)

    if args.output_dir is None:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        gira3d_root = os.path.abspath(os.path.join(script_dir, '../../../../'))
        bag_name    = Path(args.bag).stem
        args.output_dir = os.path.join(gira3d_root, 'data', bag_name, 'pointclouds')

    print(f'Bag:        {args.bag}')
    print(f'Output dir: {args.output_dir}\n')

    convert(args.bag, args.output_dir,
            start_id=args.start_id,
            final_id=args.final_id,
            skip_scans=args.skip_scans)


if __name__ == '__main__':
    main()
