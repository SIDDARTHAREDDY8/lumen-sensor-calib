"""LiDAR-side board plane via RANSAC plane fitting."""

import numpy as np


def fit_plane_ransac(points, thresh=0.03, iters=300, seed=0):
    """Fit plane n.x = d (||n|| = 1) to 3D points, robust to outliers.

    Returns (n, d, inlier_fraction, inlier_points).
    """
    rng = np.random.default_rng(seed)
    best_inliers = None
    n_pts = len(points)
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = n @ p0
        dist = np.abs(points @ n - d)
        inl = dist < thresh
        if best_inliers is None or inl.sum() > best_inliers.sum():
            best_inliers = inl
    inl_pts = points[best_inliers]
    centroid = inl_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl_pts - centroid)
    n = Vt[-1]
    # orient normal toward the sensor origin (consistent with camera side)
    if n @ centroid > 0:
        n = -n
    d = float(n @ centroid)
    return n, d, float(best_inliers.mean()), inl_pts
