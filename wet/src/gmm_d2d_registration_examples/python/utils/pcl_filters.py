#!/usr/bin/env python
"""
Point cloud filtering utilities
"""
import numpy as np


def voxel_filter(points, voxel_size=0.1, verbose=True):
    """
    Apply voxel grid downsampling to point cloud.

    Args:
        points: Nx3 or Nx4 array of points
        voxel_size: Size of voxel grid (meters)        

    Returns:
        Downsampled point cloud
    """
    if len(points) == 0:
        return points

    xyz = points[:, :3]
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    filtered = points[unique_indices]

    if verbose:
        print(f'Voxel filter: {len(points)} -> {len(filtered)} pts '
              f'({100.0 * len(filtered) / len(points):.1f}%)')

    return filtered


def radius_filter(points, radius=30.0, verbose=True):
    """
    Remove points beyond specified radius from origin.

    Args:
        points: Nx3 or Nx4 array of points
        radius: Maximum distance (meters)
        
    Returns:
        Points within radius
    """
    if len(points) == 0:
        return points

    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = distances < radius
    filtered = points[mask]

    if verbose:
        print(f'Radius filter (<{radius}m): {len(points)} -> {len(filtered)} pts '
              f'({100.0 * len(filtered) / len(points):.1f}%)')

    return filtered


def min_distance_filter(points, min_dist=1.0, verbose=False):
    """
    Remove points closer than min_dist from origin (sensor noise/vehicle).

    Args:
        points: Nx3 or Nx4 array of points
        min_dist: Minimum distance (meters)

    Returns:
        Points beyond min_dist
    """
    if len(points) == 0:
        return points

    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = distances > min_dist
    filtered = points[mask]

    if verbose:
        print(f'Min dist filter (>{min_dist}m): {len(points)} -> {len(filtered)} pts')

    return filtered


def every_n_filter(points, n=5, min_range=0.0, max_range=np.inf, verbose=True):
    """
    Downsample by taking every Nth point with optional range filtering.

    Args:
        points: Nx3 or Nx4 array of points
        n: Take every Nth point (default: 5)
        min_range: Minimum distance from origin (meters)
        max_range: Maximum distance from origin (meters)

    Returns:
        Downsampled point cloud (every Nth point within range)
    """
    if len(points) == 0:
        return points

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if n == 1 and min_range == 0.0 and max_range == np.inf:
        return points

    xyz = points[:, :3]
    distances_sq = np.sum(xyz**2, axis=1)
    min_sq = min_range * min_range
    max_sq = max_range * max_range

    # Apply range filter and every_n downsampling
    filtered_indices = []
    cnt = 0

    for i in range(len(points)):
        d2 = distances_sq[i]
        if d2 < min_sq or d2 > max_sq:
            continue
        if cnt % n != 0:
            cnt += 1
            continue
        filtered_indices.append(i)
        cnt += 1

    filtered = points[filtered_indices]

    if verbose:
        print(f'Every-{n} filter: {len(points)} -> {len(filtered)} pts '
              f'({100.0 * len(filtered) / len(points):.1f}%)')

    return filtered


def remove_plane_ransac(points, distance_threshold=0.3, ransac_n=3,
                        num_iterations=1000, verbose=True):
    """
    Remove ground plane using RANSAC.

    Args:
        points: Nx3 or Nx4 array of points
        distance_threshold: Max distance from plane to be inlier
        ransac_n: Points to sample for plane estimation
        num_iterations: RANSAC iterations

    Returns:
        Points without ground plane
    """
    if len(points) < ransac_n:
        return points

    xyz = points[:, :3]
    best_inliers = None
    best_count = 0

    for _ in range(num_iterations):
        idx = np.random.choice(len(xyz), ransac_n, replace=False)
        p0, p1, p2 = xyz[idx]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -np.dot(normal, p0)

        distances = np.abs(np.dot(xyz, normal) + d)
        inliers = distances < distance_threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        return points

    outliers = points[~best_inliers]

    if verbose:
        print(f'RANSAC plane: {len(points)} -> {len(outliers)} pts '
              f'(removed {best_count}, {100.0 * len(outliers) / len(points):.1f}% remain)')

    return outliers
