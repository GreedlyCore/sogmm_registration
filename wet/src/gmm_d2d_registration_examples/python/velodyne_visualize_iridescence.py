#!/usr/bin/env python
"""
Velodyne point cloud visualizer using iridescence
"""
import numpy as np
import argparse
import os
import glob
from pyridescence import guik, imgui

KITTI_BASE_PATH = os.path.expanduser("./dataset/kitti/data_odometry_velodyne/dataset")


def load_velodyne_bin(bin_file):
    """Load Velodyne .bin file (KITTI format: x, y, z, intensity)."""
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape((-1, 4))
    return points


def voxel_filter(points, voxel_size=0.1):
    """Apply voxel grid to downsample point cloud."""
    xyz = points[:, :3]
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    filtered_points = points[unique_indices]
    print(f'Voxel: {len(points)} -> {len(filtered_points)} ({100.0 * len(filtered_points) / len(points):.1f}%)')
    return filtered_points


def radius_filter(points, radius=50.0):
    """Remove points farther than specified radius from origin."""
    xyz = points[:, :3]
    distances = np.linalg.norm(xyz, axis=1)
    mask = distances < radius
    filtered_points = points[mask]
    print(f'Radius (<{radius}m): {len(points)} -> {len(filtered_points)} ({100.0 * len(filtered_points) / len(points):.1f}%)')
    return filtered_points


def remove_plane_ransac(points, distance_threshold=0.3, ransac_n=3, num_iterations=1000):
    """
    Remove ground plane using RANSAC (simple numpy implementation).
    """
    if len(points) < ransac_n:
        return points

    xyz = points[:, :3]
    best_inliers = None
    best_count = 0

    for _ in range(num_iterations):
        # Sample 3 random points
        idx = np.random.choice(len(xyz), ransac_n, replace=False)
        p0, p1, p2 = xyz[idx]

        # Compute plane normal
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -np.dot(normal, p0)

        # Calculate distances
        distances = np.abs(np.dot(xyz, normal) + d)
        inliers = distances < distance_threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        return points

    outliers = points[~best_inliers]
    print(f'RANSAC: removed {best_count} plane pts ({100.0 * best_count / len(points):.1f}%), {len(outliers)} remain')
    return outliers


class IridescenceVisualizer:
    """Interactive point cloud visualizer using pyridescence."""

    def __init__(self, file_list, start_index=0, voxel_size=None, radius=None,
                 remove_plane=False, ransac_distance=0.3, ransac_n=3, ransac_iters=1000,
                 point_size=0.05, colormap='rainbow'):
        self.file_list = file_list
        self.current_index = start_index
        self.voxel_size = voxel_size
        self.radius = radius
        self.remove_plane = remove_plane
        self.ransac_distance = ransac_distance
        self.ransac_n = ransac_n
        self.ransac_iters = ransac_iters
        self.point_size = point_size
        self.colormap = colormap

        # GUI state
        self.auto_play = False
        self.play_speed = 1
        self.frame_counter = 0

        # Filter enable states (for checkboxes)
        self.voxel_enabled = voxel_size is not None
        self.radius_enabled = radius is not None
        self.plane_enabled = remove_plane

        # Default values if not provided
        if self.voxel_size is None:
            self.voxel_size = 0.1
        if self.radius is None:
            self.radius = 50.0

        if self.voxel_enabled:
            print(f"Voxel filter enabled (size={self.voxel_size})")
        if self.radius_enabled:
            print(f"Radius filter enabled (radius={self.radius})")
        if self.plane_enabled:
            print(f"RANSAC plane removal enabled")

    def _load_pointcloud(self, index):
        """Load and filter point cloud at given index."""
        file_path = self.file_list[index]
        points = load_velodyne_bin(file_path)
        original_count = len(points)

        if self.radius_enabled:
            points = radius_filter(points, self.radius)

        if self.voxel_enabled:
            points = voxel_filter(points, self.voxel_size)

        if self.plane_enabled:
            points = remove_plane_ransac(points, self.ransac_distance,
                                         self.ransac_n, self.ransac_iters)

        ratio = len(points) / original_count if original_count > 0 else 0
        print(f"[{index + 1}/{len(self.file_list)}] {os.path.basename(file_path)} "
              f"({len(points)} pts, {100.0 * ratio:.1f}%)")

        return points[:, :3].astype(np.float32)

    def _get_shader_setting(self):
        """Get shader setting based on colormap selection."""
        if self.colormap == 'rainbow':
            return guik.Rainbow()
        elif self.colormap == 'flat_red':
            return guik.FlatRed()
        elif self.colormap == 'flat_green':
            return guik.FlatGreen()
        elif self.colormap == 'flat_blue':
            return guik.FlatBlue()
        elif self.colormap == 'flat_orange':
            return guik.FlatOrange()
        else:
            return guik.Rainbow()

    def run(self):
        """Run the interactive visualizer."""
        viewer = guik.LightViewer.instance()
        viewer.set_title("Velodyne Point Cloud Viewer")

        # Set point rendering style
        viewer.set_point_shape(self.point_size, metric=True, circle=True)

        # Load initial point cloud
        points = self._load_pointcloud(self.current_index)
        viewer.update_points("pointcloud", points, self._get_shader_setting())

        # Add coordinate axes (X=red, Y=green, Z=blue)
        viewer.update_coord("coords", guik.VertexColor().scale(2.0))

        # Register UI callback
        def ui_callback():
            imgui.begin("Controls", None)

            # Navigation info
            imgui.text(f"Frame: {self.current_index + 1} / {len(self.file_list)}")
            imgui.text(f"File: {os.path.basename(self.file_list[self.current_index])}")
            imgui.separator()

            # Navigation buttons
            if imgui.button("< Prev (E)"):
                self._prev()
                self._update_view(viewer)
            imgui.same_line()
            if imgui.button("Next (Q) >"):
                self._next()
                self._update_view(viewer)

            imgui.separator()

            # Auto-play controls
            changed, self.auto_play = imgui.checkbox("Auto-play", self.auto_play)
            if self.auto_play:
                changed, self.play_speed = imgui.slider_int("Speed", self.play_speed, 1, 10)

            imgui.separator()

            # Point size control
            changed, new_size = imgui.slider_float("Point size", self.point_size, 0.05, 2.0)
            if changed:
                self.point_size = new_size
                viewer.set_point_shape(self.point_size, metric=True, circle=True)

            imgui.separator()

            # Filter controls
            imgui.text_colored(np.array([1.0, 0.8, 0.3, 1.0], dtype=np.float32), "Filters:")

            filters_changed = False

            # Radius filter
            changed, self.radius_enabled = imgui.checkbox("Radius filter", self.radius_enabled)
            if changed:
                filters_changed = True
            if self.radius_enabled:
                changed, new_radius = imgui.slider_float("  Radius (m)", self.radius, 5.0, 100.0)
                if changed:
                    self.radius = new_radius
                    filters_changed = True

            # Voxel filter
            changed, self.voxel_enabled = imgui.checkbox("Voxel filter", self.voxel_enabled)
            if changed:
                filters_changed = True
            if self.voxel_enabled:
                changed, new_voxel = imgui.slider_float("  Voxel size (m)", self.voxel_size, 0.01, 1.0)
                if changed:
                    self.voxel_size = new_voxel
                    filters_changed = True

            # RANSAC plane removal
            changed, self.plane_enabled = imgui.checkbox("Remove plane (RANSAC)", self.plane_enabled)
            if changed:
                filters_changed = True
            if self.plane_enabled:
                changed, new_dist = imgui.slider_float("  Distance thresh", self.ransac_distance, 0.1, 1.0)
                if changed:
                    self.ransac_distance = new_dist
                    filters_changed = True
                changed, new_iters = imgui.slider_int("  Iterations", self.ransac_iters, 100, 5000)
                if changed:
                    self.ransac_iters = new_iters
                    filters_changed = True

            # Reload if filters changed
            if filters_changed:
                self._update_view(viewer)

            imgui.separator()

            # Jump to frame
            changed, new_idx = imgui.slider_int("Go to", self.current_index, 0, len(self.file_list) - 1)
            if changed and new_idx != self.current_index:
                self.current_index = new_idx
                self._update_view(viewer)

            imgui.separator()

            # Help
            imgui.text_colored(np.array([0.5, 0.8, 1.0, 1.0], dtype=np.float32), "Keyboard:")
            imgui.text("  Q - Next frame")
            imgui.text("  E - Previous frame")
            imgui.text("  Space - Toggle auto-play")

            imgui.end()

        viewer.register_ui_callback("controls", ui_callback)

        # Print controls
        print("\nControls:")
        print("  Q - Next frame")
        print("  E - Previous frame")
        print("  Mouse - Rotate/Pan/Zoom")
        print("  Close window to exit\n")

        # Main loop
        while viewer.spin_once():
            # Keyboard navigation (Q=next, E=prev)
            if imgui.is_key_pressed(ord('Q')):
                self._next()
                self._update_view(viewer)
            if imgui.is_key_pressed(ord('E')):
                self._prev()
                self._update_view(viewer)

            # Auto-play logic
            if self.auto_play:
                self.frame_counter += 1
                if self.frame_counter >= (60 // self.play_speed):
                    self.frame_counter = 0
                    if self.current_index < len(self.file_list) - 1:
                        self.current_index += 1
                        self._update_view(viewer)
                    else:
                        self.auto_play = False

    def _next(self):
        if self.current_index < len(self.file_list) - 1:
            self.current_index += 1
        else:
            print("Already at last frame")

    def _prev(self):
        if self.current_index > 0:
            self.current_index -= 1
        else:
            print("Already at first frame")

    def _update_view(self, viewer):
        points = self._load_pointcloud(self.current_index)
        viewer.update_points("pointcloud", points, self._get_shader_setting())


def get_file_list(input_path):
    """Get list of .bin files from input (sequence number or file path)."""
    # Check if input is a 2-digit sequence number (e.g., "00", "01")
    if len(input_path) == 2 and input_path.isdigit():
        sequence_path = os.path.join(KITTI_BASE_PATH, "sequences", input_path, "velodyne")
        sequence_path = os.path.expanduser(sequence_path)
        if os.path.exists(sequence_path):
            bin_files = sorted(glob.glob(os.path.join(sequence_path, "*.bin")))
            if bin_files:
                print(f"Loading KITTI sequence {input_path} ({len(bin_files)} files)")
                return bin_files, 0
            else:
                raise ValueError(f"No .bin files found in {sequence_path}")
        else:
            raise ValueError(f"KITTI sequence path does not exist: {sequence_path}")

    # Otherwise treat as file path
    bin_dir = os.path.dirname(input_path)
    bin_files = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))

    if not bin_files:
        raise ValueError(f"No .bin files found in {bin_dir}")

    try:
        start_index = bin_files.index(input_path)
    except ValueError:
        raise ValueError(f"File {input_path} not found in directory")

    print(f"Found {len(bin_files)} files in {bin_dir}")
    return bin_files, start_index


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Velodyne point clouds using pyridescence (iridescence)'
    )
    parser.add_argument('input', type=str,
                       help='Path to .bin file or KITTI sequence number (e.g., "00")')
    parser.add_argument('--voxel-size', type=float, default=None,
                       help='Voxel size for downsampling (meters)')
    parser.add_argument('--radius', type=float, default=None,
                       help='Maximum radius from origin (meters)')
    parser.add_argument('--remove-plane', action='store_true',
                       help='Remove ground plane using RANSAC')
    parser.add_argument('--ransac-distance', type=float, default=0.3,
                       help='RANSAC distance threshold (default: 0.3m)')
    parser.add_argument('--ransac-n', type=int, default=3,
                       help='RANSAC sample points (default: 3)')
    parser.add_argument('--ransac-iters', type=int, default=1000,
                       help='RANSAC iterations (default: 1000)')
    parser.add_argument('--point-size', type=float, default=0.05,
                       help='Point rendering size (default: 0.05)')
    parser.add_argument('--colormap', type=str, default='rainbow',
                       choices=['rainbow', 'flat_red', 'flat_green', 'flat_blue', 'flat_orange'],
                       help='Color scheme (default: rainbow)')

    args = parser.parse_args()

    file_list, start_index = get_file_list(args.input)

    visualizer = IridescenceVisualizer(
        file_list=file_list,
        start_index=start_index,
        voxel_size=args.voxel_size,
        radius=args.radius,
        remove_plane=args.remove_plane,
        ransac_distance=args.ransac_distance,
        ransac_n=args.ransac_n,
        ransac_iters=args.ransac_iters,
        point_size=args.point_size,
        colormap=args.colormap
    )

    visualizer.run()


if __name__ == '__main__':
    main()
