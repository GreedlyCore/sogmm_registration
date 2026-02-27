import open3d
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

KITTI_BASE_PATH = "./dataset/kitti/data_odometry_velodyne/dataset"


def load_gmm_file(gmm_file):
    """
    Load GMM from .gmm file.

    Args:
        gmm_file: Path to .gmm file

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


def create_ellipsoid_mesh(mean, covariance, weight, n_sigma=3, resolution=20):
    """
    Create an ellipsoid mesh from GMM component parameters using 3-sigma rule.

    TODO: Add Box-Muller sampling method as an alternative visualization option

    Args:
        mean: 3D mean vector [x, y, z]
        covariance: 3x3 covariance matrix
        weight: Component weight (visualized via Viridis colormap)
        n_sigma: Number of standard deviations for ellipsoid size (default: 3)
        resolution: Mesh resolution (default: 20)

    Returns:
        Open3D TriangleMesh of the ellipsoid
    """
    # Cast to float64 for numerical stability (float32 EM can leave near-singular covs)
    covariance = covariance.astype(np.float64)

    if not np.all(np.isfinite(covariance)):
        return None, weight

    # Eigenvalue decomposition to get principal axes and scales
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None, weight

    # Ensure eigenvalues are positive (numerical stability)
    eigenvalues = np.maximum(eigenvalues, 1e-6)

    # Create unit sphere
    mesh = open3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)

    # Scale by n_sigma * sqrt(eigenvalues) for each principal axis
    radii = n_sigma * np.sqrt(eigenvalues)

    # Transform: scale, rotate, translate
    vertices = np.asarray(mesh.vertices)

    # Scale by radii
    vertices = vertices * radii

    # Rotate by eigenvectors (principal axes)
    vertices = vertices @ eigenvectors.T

    # Translate to mean
    vertices = vertices + mean

    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()

    # Color based on weight (you can customize this)
    # Using a blue color scheme, but feel free to change
    color = np.array([0.2, 0.6, 1.0])  # Light blue
    mesh.paint_uniform_color(color)

    return mesh, weight  # Return weight for opacity setting

class Open3DVisualizer():
    _viz = None

    def __init__(self):
        self._viz = open3d.visualization.Visualizer()
        self._viz.create_window()

    def destroy_window(self):
        self._viz.destroy_window()

    def plot3d(self, np_array):
        pcd = self._numpy_to_pcd(np_array)
        open3d.visualization.draw_geometries([pcd])

    def _numpy_to_pcd(self, np_array):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np_array[:, 0:3])
        return pcd


class InteractiveVisualizer():

    def __init__(self, load_callback, file_list, start_index=0, 
                voxel_size=None, radius=None, remove_plane=False,
                ransac_distance=0.3, ransac_n=3, ransac_iters=1000):
        self.load_callback = load_callback
        self.file_list = self._process_file_list(file_list)
        self.current_index = start_index
        self.voxel_size = voxel_size
        self.radius = radius
        self.remove_plane = remove_plane
        self.ransac_distance = ransac_distance
        self.ransac_n = ransac_n
        self.ransac_iters = ransac_iters
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.pcd = None
        
        # Status display
        if voxel_size is not None:
            print(f"Voxel filter enabled (size={voxel_size})")
        if radius is not None:
            print(f"Radius filter enabled (radius={radius})")
        if remove_plane:
            print(f"RANSAC plane removal enabled (distance={ransac_distance}, n={ransac_n}, iters={ransac_iters})")

    def _remove_plane_ransac(self, points, distance_threshold=0.3, ransac_n=3, num_iterations=1000):
        """
        Remove ground plane using RANSAC algorithm.
        
        Args:
            points: Nx3 or Nx4 array of points
            distance_threshold: Max distance from plane to be considered inlier
            ransac_n: Number of points to sample for plane estimation
            num_iterations: Number of RANSAC iterations
        
        Returns:
            filtered_points: Points without the ground plane
        """
        if len(points) < ransac_n:
            return points
        
        # Use only XYZ coordinates for plane fitting
        xyz = points[:, :3]
        
        # Convert to Open3D point cloud for RANSAC
        pcd_o3d = open3d.geometry.PointCloud()
        pcd_o3d.points = open3d.utility.Vector3dVector(xyz)
        
        # Find plane using RANSAC
        plane_model, inliers = pcd_o3d.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        # Get outliers (non-plane points)
        outliers = np.delete(points, inliers, axis=0)
        
        # Print statistics
        plane_points = len(inliers)
        remaining_points = len(outliers)
        total_points = len(points)
        
        print(f'RANSAC plane removal: removed {plane_points} plane points '
            f'({100.0 * plane_points / total_points:.1f}%), '
            f'{remaining_points} points remain ({100.0 * remaining_points / total_points:.1f}%)')
        
        return outliers
    
    def _voxel_filter(self, points, voxel_size=0.1):
        """Apply voxel grid to downsample point cloud."""
        xyz = points[:, :3]
        voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        filtered_points = points[unique_indices]
        print(f'Voxel filtering: {len(points)} -> {len(filtered_points)} points '
            f'({100.0 * len(filtered_points) / len(points):.1f}%)')
        return filtered_points

    def _radius_filter(self, points, radius=5.0):
        """Remove points farther than specified radius from origin."""
        xyz = points[:, :3]
        distances = np.linalg.norm(xyz, axis=1)
        mask = distances < radius
        filtered_points = points[mask]
        print(f'Radius filtering (<{radius}m): {len(points)} -> {len(filtered_points)} points '
            f'({100.0 * len(filtered_points) / len(points):.1f}%)')
        return filtered_points
    
    def _process_file_list(self, file_list):
        if isinstance(file_list, str) and len(file_list) == 2 and file_list.isdigit():
            sequence_path = os.path.join(KITTI_BASE_PATH, "sequences", file_list, "velodyne")
            if os.path.exists(sequence_path):
                bin_files = sorted(glob.glob(os.path.join(sequence_path, "*.bin")))
                if bin_files:
                    print(f"Loading KITTI sequence {file_list} from {sequence_path} ({len(bin_files)} files)")
                    return bin_files
                else:
                    raise ValueError(f"No .bin files found in {sequence_path}")
            else:
                raise ValueError(f"KITTI sequence path does not exist: {sequence_path}")
        return file_list

    def _numpy_to_pcd(self, np_array):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np_array[:, 0:3])
        return pcd

    def _load_current_pointcloud(self):
        file_path = self.file_list[self.current_index]
        points = self.load_callback(file_path)
        original_count = len(points)
        
        # Apply filters in sequence
        if self.radius is not None:
            points = self._radius_filter(points, self.radius)
        
        if self.voxel_size is not None:
            points = self._voxel_filter(points, self.voxel_size)
        
        if self.remove_plane:
            points = self._remove_plane_ransac(points, 
                                            self.ransac_distance,
                                            self.ransac_n,
                                            self.ransac_iters)
        
        # Calculate overall reduction
        remaining_ratio = len(points) / original_count if original_count > 0 else 0
        print(f"[{self.current_index + 1}/{len(self.file_list)}] Loading: {os.path.basename(file_path)} "
            f"({len(points)} points, {100.0 * remaining_ratio:.1f}% of original)")
        
        return self._numpy_to_pcd(points)

    def _next_pointcloud(self, vis):
        if self.current_index < len(self.file_list) - 1:
            self.current_index += 1
            self._update_pointcloud()
        else:
            print("Already at the last point cloud")
        return False

    def _prev_pointcloud(self, vis):
        if self.current_index > 0:
            self.current_index -= 1
            self._update_pointcloud()
        else:
            print("Already at the first point cloud")
        return False

    def _update_pointcloud(self):
        new_pcd = self._load_current_pointcloud()
        self.pcd.points = new_pcd.points
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        self.vis.create_window()
        self.pcd = self._load_current_pointcloud()
        self.vis.add_geometry(self.pcd)

        self.vis.register_key_callback(ord('Q'), self._next_pointcloud)
        self.vis.register_key_callback(ord('E'), self._prev_pointcloud)

        print("\nControls:")
        print("  Q - Next point cloud")
        print("  E - Previous point cloud")
        print("  ESC - Exit\n")

        self.vis.run()
        self.vis.destroy_window()


class InteractiveEllipsoidVisualizer():
    """Interactive visualizer for GMM ellipsoids with keyboard navigation and Viridis colormap for weights."""

    def __init__(self, gmm_dir, start_index=0):
        """
        Initialize ellipsoid visualizer.

        Args:
            gmm_dir: Directory containing .gmm files (1.gmm, 2.gmm, etc.)
            start_index: Starting file index (0-based, will be converted to 1-based for file loading)
        """
        self.gmm_dir = gmm_dir
        self.gmm_files = self._get_gmm_files()
        self.current_index = start_index

        # Use the new GUI-based visualizer for proper transparency support
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering

        self.gui = gui
        self.rendering = rendering
        self.app = gui.Application.instance
        self.app.initialize()

        self.window = None
        self.scene_widget = None
        self.ellipsoid_names = []

        print(f"Found {len(self.gmm_files)} GMM files in {gmm_dir}")

    def _get_gmm_files(self):
        """Get sorted list of .gmm files (1-based naming: 1.gmm, 2.gmm, ...)"""
        gmm_files = sorted(glob.glob(os.path.join(self.gmm_dir, "*.gmm")),
                          key=lambda x: int(os.path.basename(x).split('.')[0]))
        if not gmm_files:
            raise ValueError(f"No .gmm files found in {self.gmm_dir}")
        return gmm_files

    def _load_current_gmm(self):
        """Load current GMM and create ellipsoid meshes."""
        gmm_file = self.gmm_files[self.current_index]
        gmm_data = load_gmm_file(gmm_file)

        print(f"[{self.current_index + 1}/{len(self.gmm_files)}] Loading: {os.path.basename(gmm_file)} "
              f"({gmm_data['n_components']} components)")

        # Create ellipsoid meshes for all components
        meshes_with_weights = []
        skipped = 0
        for i in range(gmm_data['n_components']):
            mesh, weight = create_ellipsoid_mesh(
                gmm_data['means'][i],
                gmm_data['covariances'][i],
                gmm_data['weights'][i],
                n_sigma=2,
                resolution=20
            )
            if mesh is None:
                skipped += 1
                continue
            meshes_with_weights.append((mesh, weight))
        if skipped:
            print(f"  ({skipped} degenerate components skipped)")

        return meshes_with_weights

    def _weights_to_viridis_colors(self, weights):
        """
        Convert weights to Viridis colormap RGB values.
        Uses perceptually uniform Viridis colormap:
        - Low weight (0) -> Dark blue/purple
        - High weight (1) -> Bright yellow

        Returns:
            Array of RGB colors (N x 3), each value in [0, 1]
        """
        weights = np.array(weights)
        w_min = weights.min()
        w_max = weights.max()

        # Normalize weights to [0, 1]
        if w_max - w_min < 1e-10:  # All weights are the same
            normalized = np.full_like(weights, 0.5)
        else:
            normalized = (weights - w_min) / (w_max - w_min)

        # Get Viridis colors (returns RGBA, we take only RGB)
        viridis = plt.cm.viridis(normalized)[:, :3]  # Shape: (N, 3)

        return viridis

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == self.gui.KeyName.Q:
            self._next_gmm()
            return self.gui.Widget.EventCallbackResult.HANDLED
        elif event.key == self.gui.KeyName.E:
            self._prev_gmm()
            return self.gui.Widget.EventCallbackResult.HANDLED
        return self.gui.Widget.EventCallbackResult.IGNORED

    def _next_gmm(self):
        """Load next GMM."""
        if self.current_index < len(self.gmm_files) - 1:
            self.current_index += 1
            self._update_visualization()
        else:
            print("Already at the last GMM")

    def _prev_gmm(self):
        """Load previous GMM."""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_visualization()
        else:
            print("Already at the first GMM")

    def _update_visualization(self):
        """Update visualization with new GMM using Viridis colormap."""
        # Remove old ellipsoids
        for name in self.ellipsoid_names:
            self.scene_widget.scene.remove_geometry(name)
        self.ellipsoid_names = []

        # Load new ellipsoids
        meshes_with_weights = self._load_current_gmm()
        weights = [w for _, w in meshes_with_weights]
        colors = self._weights_to_viridis_colors(weights)

        # Add new ellipsoids with Viridis colors and full opacity
        for idx, ((mesh, weight), color) in enumerate(zip(meshes_with_weights, colors)):
            mesh.compute_vertex_normals()

            # Create material with Viridis color and full opacity
            mat = self.rendering.MaterialRecord()
            mat.shader = "defaultLit"  # Use standard shader (no transparency)
            # base_color: [R, G, B, Alpha=1.0] - full opacity
            mat.base_color = np.array([color[0], color[1], color[2], 1.0])

            # Add geometry with material
            name = f"ellipsoid_{idx}"
            self.scene_widget.scene.add_geometry(name, mesh, mat)
            self.ellipsoid_names.append(name)

        # Update view
        self.scene_widget.scene.show_axes(False)

    def run(self):
        """Run the interactive visualizer with GUI."""
        # Create window
        self.window = self.app.create_window("GMM Ellipsoid Visualizer", 1280, 720)

        # Create 3D scene widget
        self.scene_widget = self.gui.SceneWidget()
        self.scene_widget.scene = self.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1, 1, 1, 1])  # White background
        self.scene_widget.scene.show_skybox(False)  # Remove skybox

        # Add scene to window
        self.window.add_child(self.scene_widget)

        # Set up keyboard callback
        self.scene_widget.set_on_key(self._on_key)

        # Load initial GMM
        meshes_with_weights = self._load_current_gmm()
        weights = [w for _, w in meshes_with_weights]
        colors = self._weights_to_viridis_colors(weights)

        # Add all ellipsoids with Viridis colors and full opacity
        for idx, ((mesh, weight), color) in enumerate(zip(meshes_with_weights, colors)):
            mesh.compute_vertex_normals()

            # Create material with Viridis color and full opacity
            mat = self.rendering.MaterialRecord()
            mat.shader = "defaultLit"  # Use standard shader (no transparency)
            mat.base_color = np.array([color[0], color[1], color[2], 1.0])  # [R, G, B, Alpha=1.0]

            # Add geometry
            name = f"ellipsoid_{idx}"
            self.scene_widget.scene.add_geometry(name, mesh, mat)
            self.ellipsoid_names.append(name)

        # Set up camera
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())
        self.scene_widget.scene.show_axes(False)

        print("\nControls:")
        print("  Q - Next GMM scan")
        print("  E - Previous GMM scan")
        print("  Mouse - Rotate/Pan/Zoom")
        print("  Close window to exit\n")

        # Run application
        self.app.run()
