#!/usr/bin/env python
"""
Visualize GMM map with pyridescence (iridescence).

Shows:
- GMM ellipsoids colored by scan index
- Trajectory as colored dots connected by lines
"""
import numpy as np
import argparse
import pickle
import os
from pyridescence import guik, imgui


def load_map_data(map_file):
    """Load map data from pickle file."""
    with open(map_file, 'rb') as f:
        return pickle.load(f)


def create_trajectory_lines(trajectory):
    """
    Create line vertices for trajectory visualization.

    Args:
        trajectory: (N, 3) array of positions

    Returns:
        vertices: (2*(N-1), 3) array for line segments
    """
    n = len(trajectory)
    if n < 2:
        return np.zeros((0, 3), dtype=np.float32)

    # Create pairs of points for line segments
    vertices = np.zeros((2 * (n - 1), 3), dtype=np.float32)
    for i in range(n - 1):
        vertices[2 * i] = trajectory[i]
        vertices[2 * i + 1] = trajectory[i + 1]

    return vertices


def trajectory_colors(n_poses):
    """
    Generate colors for trajectory (rainbow gradient from start to end).

    Args:
        n_poses: Number of poses

    Returns:
        colors: (n_poses, 4) RGBA colors
    """
    colors = np.zeros((n_poses, 4), dtype=np.float32)
    for i in range(n_poses):
        t = i / max(n_poses - 1, 1)
        # Rainbow: red -> yellow -> green -> cyan -> blue
        if t < 0.25:
            colors[i] = [1.0, t * 4, 0.0, 1.0]
        elif t < 0.5:
            colors[i] = [1.0 - (t - 0.25) * 4, 1.0, 0.0, 1.0]
        elif t < 0.75:
            colors[i] = [0.0, 1.0, (t - 0.5) * 4, 1.0]
        else:
            colors[i] = [0.0, 1.0 - (t - 0.75) * 4, 1.0, 1.0]
    return colors


class GMMMapVisualizer:
    """Interactive GMM map visualizer using pyridescence."""

    def __init__(self, map_data):
        self.map_data = map_data
        self.gmm = map_data['gmm']
        self.trajectory = map_data['trajectory']
        self.world_poses = map_data['world_poses']

        # Visualization settings
        self.ellipsoid_scale = 2.0  # n_sigma for ellipsoid size
        self.point_size = 0.3
        self.show_ellipsoids = True
        self.show_trajectory = True
        self.show_trajectory_points = True
        self.ellipsoid_alpha = 0.5
        self.max_ellipsoids = 5000  # Limit for performance

    def run(self):
        """Run the interactive visualizer."""
        viewer = guik.LightViewer.instance()
        viewer.set_title("GMM Map Visualization")

        # Add coordinate axes
        viewer.update_coord("coords", guik.VertexColor().scale(5.0))

        # Draw trajectory lines
        if len(self.trajectory) > 1:
            self._draw_trajectory(viewer)

        # Draw trajectory points
        self._draw_trajectory_points(viewer)

        # Draw GMM ellipsoids
        if self.gmm is not None:
            self._draw_ellipsoids(viewer)

        # Register UI callback
        def ui_callback():
            imgui.begin("Map Controls", None)

            # Info
            imgui.text(f"Trajectory poses: {len(self.trajectory)}")
            if self.gmm is not None:
                imgui.text(f"GMM components: {self.gmm['n_components']}")

            imgui.separator()

            # Visibility toggles
            changed, self.show_trajectory = imgui.checkbox("Show trajectory lines", self.show_trajectory)
            if changed:
                if self.show_trajectory:
                    self._draw_trajectory(viewer)
                else:
                    viewer.remove_drawable("trajectory_lines")

            changed, self.show_trajectory_points = imgui.checkbox("Show trajectory points", self.show_trajectory_points)
            if changed:
                if self.show_trajectory_points:
                    self._draw_trajectory_points(viewer)
                else:
                    viewer.remove_drawable("trajectory_points")

            changed, self.show_ellipsoids = imgui.checkbox("Show ellipsoids", self.show_ellipsoids)
            if changed:
                if self.show_ellipsoids and self.gmm is not None:
                    self._draw_ellipsoids(viewer)
                else:
                    viewer.remove_drawable("ellipsoids")

            imgui.separator()

            # Ellipsoid settings
            if self.gmm is not None:
                imgui.text_colored(np.array([1.0, 0.8, 0.3, 1.0], dtype=np.float32), "Ellipsoid settings:")
                changed, new_scale = imgui.slider_float("Scale (sigma)", self.ellipsoid_scale, 0.5, 5.0)
                if changed:
                    self.ellipsoid_scale = new_scale
                    if self.show_ellipsoids:
                        self._draw_ellipsoids(viewer)

                changed, new_alpha = imgui.slider_float("Alpha", self.ellipsoid_alpha, 0.1, 1.0)
                if changed:
                    self.ellipsoid_alpha = new_alpha
                    if self.show_ellipsoids:
                        self._draw_ellipsoids(viewer)

            imgui.separator()

            # Trajectory point size
            changed, new_size = imgui.slider_float("Point size", self.point_size, 0.1, 2.0)
            if changed:
                self.point_size = new_size
                if self.show_trajectory_points:
                    self._draw_trajectory_points(viewer)

            imgui.separator()

            # Camera controls
            if imgui.button("Reset view"):
                viewer.reset_center()

            if imgui.button("Top-down view"):
                # Look from above
                viewer.lookat(np.array([0, 0, 50], dtype=np.float32))

            imgui.end()

        viewer.register_ui_callback("controls", ui_callback)

        print("\nGMM Map Visualization")
        print("=" * 40)
        print(f"Trajectory poses: {len(self.trajectory)}")
        if self.gmm is not None:
            print(f"GMM components: {self.gmm['n_components']}")
        print("\nControls:")
        print("  Mouse - Rotate/Pan/Zoom")
        print("  Close window to exit\n")

        # Main loop
        viewer.spin()

    def _draw_trajectory(self, viewer):
        """Draw trajectory as connected lines."""
        if len(self.trajectory) < 2:
            return

        line_vertices = create_trajectory_lines(self.trajectory)
        colors = trajectory_colors(len(self.trajectory))

        # Create line colors (each segment needs start and end color)
        line_colors = np.zeros((len(line_vertices), 4), dtype=np.float32)
        for i in range(len(self.trajectory) - 1):
            line_colors[2 * i] = colors[i]
            line_colors[2 * i + 1] = colors[i + 1]

        viewer.update_thin_lines(
            "trajectory_lines",
            line_vertices.astype(np.float32),
            line_colors,
            [],  # indices
            False,  # line_strip
            guik.VertexColor()
        )

    def _draw_trajectory_points(self, viewer):
        """Draw trajectory poses as colored spheres."""
        if len(self.trajectory) == 0:
            return

        points = self.trajectory.astype(np.float32)
        colors = trajectory_colors(len(points))

        viewer.update_points(
            "trajectory_points",
            points,
            guik.VertexColor().set_point_shape(self.point_size, True, True)
        )

        # Draw individual spheres for better visibility
        for i, (pos, color) in enumerate(zip(self.trajectory, colors)):
            viewer.update_sphere(
                f"pose_{i}",
                guik.FlatColor(color[0], color[1], color[2], 1.0)
                    .translate(pos[0], pos[1], pos[2])
                    .scale(self.point_size)
            )

    def _draw_ellipsoids(self, viewer):
        """Draw GMM ellipsoids."""
        if self.gmm is None:
            return

        means = self.gmm['means']
        covs = self.gmm['covariances']
        n_components = min(self.gmm['n_components'], self.max_ellipsoids)

        if n_components < self.gmm['n_components']:
            print(f"Warning: Showing only {n_components}/{self.gmm['n_components']} ellipsoids for performance")

        # Prepare data for update_normal_dists
        means_list = [means[i].astype(np.float32).reshape(3, 1) for i in range(n_components)]
        covs_list = [covs[i].astype(np.float32) for i in range(n_components)]

        # Use rainbow coloring with transparency
        viewer.update_normal_dists(
            "ellipsoids",
            means_list,
            covs_list,
            self.ellipsoid_scale,
            guik.Rainbow().set_alpha(self.ellipsoid_alpha)
        )


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GMM map with pyridescence'
    )

    parser.add_argument('--map_file', type=str, required=True,
                       help='Path to map .pkl file from gmm_mapping.py')

    args = parser.parse_args()

    if not os.path.exists(args.map_file):
        print(f"Map file not found: {args.map_file}")
        return

    print(f"Loading map from: {args.map_file}")
    map_data = load_map_data(args.map_file)

    visualizer = GMMMapVisualizer(map_data)
    visualizer.run()


if __name__ == '__main__':
    main()
