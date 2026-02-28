import os
import sys
import numpy as np
from datetime import datetime

from gmm_py import GMMf4CPU
from kinit_py import KInitf4CPU
from sogmm_py import SOGMM


def detect_implementation(bw, n_components):
    """Returns (implementation, bandwidth, n_components). Exits on invalid combination."""
    if bw is not None and n_components is not None:
        print('ERROR: Specify either --bw OR --n_components, not both')
        sys.exit(1)
    elif bw is not None:
        return 'sogmm', bw, None
    elif n_components is not None:
        return 'fixed', None, n_components
    else:
        print('ERROR: Must specify either --bw or --n_components')
        sys.exit(1)


def make_output_dir(base_dir, implementation, n_components, bandwidth):
    """Create and return timestamped output directory."""
    timestamp = datetime.now().strftime('%d%m%H%M')
    if implementation == 'sogmm':
        name = f'bw{int(bandwidth * 100)}_components_{timestamp}'
    else:
        name = f'{n_components}_components_{timestamp}'
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def fit_gmm(points_4d, implementation, n_components=None, bandwidth=None,
            redux_kmeans=None, mahal_distance=None, stats_dir=None, scan_name='scan'):
    """
    Fit a GMM to 4D points.
    Returns fitted model or None on failure (fixed EM divergence).
    """
    if stats_dir is not None:
        os.makedirs(stats_dir, exist_ok=True)

    if implementation == 'fixed':
        if points_4d.shape[1] == 3:
            points_4d = np.hstack([points_4d, np.zeros((points_4d.shape[0], 1), dtype=np.float32)])
        n_samples = points_4d.shape[0]
        kinit = KInitf4CPU()

        if redux_kmeans is not None:
            points_sub = points_4d[::redux_kmeans]
            centers, _ = kinit.resp_calc(points_sub, n_components)
            dists = kinit.euclidean_dists(points_4d, centers)
            assignments = np.argmin(dists, axis=1)
            resp = np.zeros((n_samples, n_components), dtype=np.float32)
            resp[np.arange(n_samples), assignments] = 1
        else:
            _, indices = kinit.resp_calc(points_4d, n_components)
            resp = np.zeros((n_samples, n_components), dtype=np.float32)
            resp[indices, np.arange(n_components)] = 1

        local_model = GMMf4CPU(n_components, True, stats_dir or '.', f'gmm_stats_{scan_name}.csv')
        if mahal_distance is not None:
            success = local_model.fit_mahal(points_4d, resp, mahal_distance)
        else:
            success = local_model.fit(points_4d, resp)

        return local_model if success else None

    elif implementation == 'sogmm':
        sg = SOGMM(bandwidth, save_stats=True, stats_dir=stats_dir, stats_file_prefix=scan_name)
        return sg.fit(points_4d, mahal_distance=mahal_distance)
