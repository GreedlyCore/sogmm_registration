#!/usr/bin/env python
"""
Interactive GMM ellipsoid visualizer.

Visualizes Gaussian Mixture Models as N-sigma ellipsoids with colour based on component weights.

TODO: check it again, is colours mapped correctly?
Viridis Color Mapping Function (lines 336-359)
- Converts weights to Viridis RGB colors
- Low weight → Dark blue/purple
- High weight → Bright yellow
- Perceptually uniform mapping


TODO: Add Box-Muller sampling method as an alternative visualization option
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.open3d_visualizer import InteractiveEllipsoidVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GMM as N-sigma ellipsoids from .gmm files')
    parser.add_argument('gmm_dir', type=str,
                       help='Path to directory containing .gmm files (e.g., kitti_sequence_00/100_components)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Starting scan index (0-based, default: 0)')

    args = parser.parse_args()

    # Validate directory
    if not os.path.exists(args.gmm_dir):
        print(f"ERROR: Directory not found: {args.gmm_dir}")
        return

    if not os.path.isdir(args.gmm_dir):
        print(f"ERROR: Path is not a directory: {args.gmm_dir}")
        return

    print(f"GMM directory: {args.gmm_dir}")
    print(f"Starting at index: {args.start_index}\n")

    visualizer = InteractiveEllipsoidVisualizer(
        gmm_dir=args.gmm_dir,
        start_index=args.start_index
    )
    visualizer.run()


if __name__ == '__main__':
    main()
