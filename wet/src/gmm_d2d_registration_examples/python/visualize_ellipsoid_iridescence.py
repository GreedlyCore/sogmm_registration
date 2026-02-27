#!/usr/bin/env python
"""
GMM ellipsoid visualizer using iridescence
"""
import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt
from pyridescence import guik, glk, imgui


def load_gmm_file(gmm_file):
    """
    Load GMM from .gmm file.

    Returns:
        dict with 'means', 'covariances', 'weights' (each as numpy arrays)
    """
    data = np.loadtxt(gmm_file, delimiter=',')
    n_components = data.shape[0]

    means = data[:, 0:3]
    covariances = data[:, 3:12].reshape(n_components, 3, 3)
    weights = data[:, 12]

    return {
        'means': means,
        'covariances': covariances,
        'weights': weights,
        'n_components': n_components
    }


def weights_to_viridis(weights):
    """Convert weights to Viridis RGB colors (N x 4 RGBA)."""
    weights = np.array(weights)
    w_min, w_max = weights.min(), weights.max()

    if w_max - w_min < 1e-10:
        normalized = np.full_like(weights, 0.5)
    else:
        normalized = (weights - w_min) / (w_max - w_min)

    return plt.cm.viridis(normalized)  # Returns RGBA


def create_ellipsoid_vertices(mean, covariance, n_sigma=2, resolution=16):
    """
    Create ellipsoid vertices from GMM component parameters.

    Args:
        mean: 3D mean vector
        covariance: 3x3 covariance matrix
        n_sigma: Number of standard deviations (1, 2, or 3)
        resolution: Sphere resolution

    Returns:
        vertices: Nx3 array of ellipsoid surface points
        indices: Triangle indices for mesh
    """
    # Eigendecomposition for principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-6)

    # Generate unit sphere vertices
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution * 2)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Stack and reshape to Nx3
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Scale by n_sigma * sqrt(eigenvalues)
    radii = n_sigma * np.sqrt(eigenvalues)
    vertices = vertices * radii

    # Rotate by eigenvectors
    vertices = vertices @ eigenvectors.T

    # Translate to mean
    vertices = vertices + mean

    return vertices.astype(np.float32)


class IridescenceEllipsoidVisualizer:
    """Interactive GMM ellipsoid visualizer using pyridescence."""

    def __init__(self, gmm_dir, start_index=0):
        self.gmm_dir = gmm_dir
        self.gmm_files = self._get_gmm_files()
        self.current_index = start_index

        # Visualization params
        self.n_sigma = 2  # 1, 2, or 3
        self.sigma_options = [1, 2, 3]
        self.resolution = 16
        self.point_size = 0.03
        self.show_axes = True

        # Auto-play state
        self.auto_play = False
        self.play_speed = 1
        self.frame_counter = 0

        # Color mode
        self.use_viridis = True

        # Current component count (for UI display)
        self.n_components = 0

        print(f"Found {len(self.gmm_files)} GMM files in {gmm_dir}")

    def _get_gmm_files(self):
        """Get sorted list of .gmm files."""
        gmm_files = sorted(
            glob.glob(os.path.join(self.gmm_dir, "*.gmm")),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        if not gmm_files:
            raise ValueError(f"No .gmm files found in {self.gmm_dir}")
        return gmm_files

    def _load_gmm_as_points(self, index):
        """Load GMM and create ellipsoid point cloud."""
        gmm_file = self.gmm_files[index]
        gmm_data = load_gmm_file(gmm_file)
        self.n_components = gmm_data['n_components']

        print(f"[{index + 1}/{len(self.gmm_files)}] {os.path.basename(gmm_file)} "
              f"({self.n_components} components)")

        all_points = []
        all_colors = []

        # Get colors for all components
        colors = weights_to_viridis(gmm_data['weights'])

        for i in range(gmm_data['n_components']):
            vertices = create_ellipsoid_vertices(
                gmm_data['means'][i],
                gmm_data['covariances'][i],
                n_sigma=self.n_sigma,
                resolution=self.resolution
            )
            all_points.append(vertices)

            # Assign color to all vertices of this ellipsoid
            if self.use_viridis:
                color = colors[i, :3]  # RGB only
            else:
                color = np.array([0.2, 0.6, 1.0])  # Default blue

            vertex_colors = np.tile(color, (len(vertices), 1))
            all_colors.append(vertex_colors)

        points = np.vstack(all_points).astype(np.float32)
        colors = np.vstack(all_colors).astype(np.float32)

        return points, colors

    def run(self):
        """Run the interactive visualizer."""
        viewer = guik.LightViewer.instance()
        viewer.set_title("GMM Ellipsoid Visualizer (iridescence)")

        viewer.set_point_shape(self.point_size, metric=True, circle=True)

        # Load initial GMM
        points, colors = self._load_gmm_as_points(self.current_index)
        cloud_buffer = glk.PointCloudBuffer(points)
        cloud_buffer.add_color(colors)
        viewer.update_drawable("ellipsoids", cloud_buffer, guik.VertexColor())

        if self.show_axes:
            viewer.update_coord("coords", guik.VertexColor().scale(2.0))

        # UI callback
        def ui_callback():
            imgui.begin("GMM Controls", None)

            # Frame info
            imgui.text(f"Frame: {self.current_index + 1} / {len(self.gmm_files)}")
            imgui.text(f"File: {os.path.basename(self.gmm_files[self.current_index])}")
            imgui.text(f"Components: {self.n_components}")
            imgui.separator()

            # Navigation
            if imgui.button("< Prev (E)"):
                self._prev()
                self._update_view(viewer)
            imgui.same_line()
            if imgui.button("Next (Q) >"):
                self._next()
                self._update_view(viewer)

            imgui.separator()

            # Sigma selection (radio buttons)
            imgui.text_colored(np.array([1.0, 0.8, 0.3, 1.0], dtype=np.float32), "Sigma Level:")
            sigma_changed = False
            for sigma_val in self.sigma_options:
                selected = (self.n_sigma == sigma_val)
                if imgui.radio_button(f"{sigma_val}σ", selected):
                    if self.n_sigma != sigma_val:
                        self.n_sigma = sigma_val
                        sigma_changed = True
                if sigma_val < 3:
                    imgui.same_line()

            if sigma_changed:
                self._update_view(viewer)

            imgui.separator()

            # Point size
            changed, new_size = imgui.slider_float("Point size", self.point_size, 0.01, 0.2)
            if changed:
                self.point_size = new_size
                viewer.set_point_shape(self.point_size, metric=True, circle=True)

            # Resolution
            changed, new_res = imgui.slider_int("Resolution", self.resolution, 8, 32)
            if changed:
                self.resolution = new_res
                self._update_view(viewer)

            imgui.separator()

            # Color mode
            changed, self.use_viridis = imgui.checkbox("Viridis colormap", self.use_viridis)
            if changed:
                self._update_view(viewer)

            # Show axes
            changed, self.show_axes = imgui.checkbox("Show axes", self.show_axes)
            if changed:
                if self.show_axes:
                    viewer.update_coord("coords", guik.VertexColor().scale(2.0))
                else:
                    viewer.remove("coords")

            imgui.separator()

            # Auto-play
            changed, self.auto_play = imgui.checkbox("Auto-play", self.auto_play)
            if self.auto_play:
                changed, self.play_speed = imgui.slider_int("Speed", self.play_speed, 1, 10)

            imgui.separator()

            # Jump to frame
            changed, new_idx = imgui.slider_int("Go to", self.current_index, 0, len(self.gmm_files) - 1)
            if changed and new_idx != self.current_index:
                self.current_index = new_idx
                self._update_view(viewer)

            imgui.separator()

            # Help
            imgui.text_colored(np.array([0.5, 0.8, 1.0, 1.0], dtype=np.float32), "Keyboard:")
            imgui.text("  Q - Next frame")
            imgui.text("  E - Previous frame")
            imgui.text("  1/2/3 - Set sigma")

            imgui.end()

        viewer.register_ui_callback("controls", ui_callback)

        print("\nControls:")
        print("  Q - Next frame")
        print("  E - Previous frame")
        print("  1/2/3 - Set sigma level")
        print("  Mouse - Rotate/Pan/Zoom\n")

        # Main loop
        while viewer.spin_once():
            # Keyboard: Q=next, E=prev
            if imgui.is_key_pressed(ord('Q')):
                self._next()
                self._update_view(viewer)
            if imgui.is_key_pressed(ord('E')):
                self._prev()
                self._update_view(viewer)

            # Keyboard: 1/2/3 for sigma
            for sigma_val in [1, 2, 3]:
                if imgui.is_key_pressed(ord(str(sigma_val))):
                    if self.n_sigma != sigma_val:
                        self.n_sigma = sigma_val
                        self._update_view(viewer)

            # Auto-play
            if self.auto_play:
                self.frame_counter += 1
                if self.frame_counter >= (60 // self.play_speed):
                    self.frame_counter = 0
                    if self.current_index < len(self.gmm_files) - 1:
                        self.current_index += 1
                        self._update_view(viewer)
                    else:
                        self.auto_play = False

    def _next(self):
        if self.current_index < len(self.gmm_files) - 1:
            self.current_index += 1
        else:
            print("Already at last frame")

    def _prev(self):
        if self.current_index > 0:
            self.current_index -= 1
        else:
            print("Already at first frame")

    def _update_view(self, viewer):
        points, colors = self._load_gmm_as_points(self.current_index)
        cloud_buffer = glk.PointCloudBuffer(points)
        cloud_buffer.add_color(colors)
        viewer.update_drawable("ellipsoids", cloud_buffer, guik.VertexColor())


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GMM as N-sigma ellipsoids using iridescence')
    parser.add_argument('gmm_dir', type=str,
                        help='Path to directory containing .gmm files')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting scan index (0-based)')
    parser.add_argument('--sigma', type=int, default=2, choices=[1, 2, 3],
                        help='Initial sigma level (default: 2)')
    parser.add_argument('--point-size', type=float, default=0.03,
                        help='Point rendering size (default: 0.03)')
    parser.add_argument('--resolution', type=int, default=16,
                        help='Ellipsoid resolution (default: 16)')

    args = parser.parse_args()

    if not os.path.isdir(args.gmm_dir):
        print(f"ERROR: Directory not found: {args.gmm_dir}")
        return

    visualizer = IridescenceEllipsoidVisualizer(
        gmm_dir=args.gmm_dir,
        start_index=args.start_index
    )
    visualizer.n_sigma = args.sigma
    visualizer.point_size = args.point_size
    visualizer.resolution = args.resolution

    visualizer.run()


if __name__ == '__main__':
    main()
