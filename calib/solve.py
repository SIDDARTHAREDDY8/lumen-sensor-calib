"""Extrinsic solve from plane correspondences.

For each capture i we have the same physical board plane expressed in
two frames:
    camera: n_c . x = d_c        LiDAR: n_l . x = d_l

Rotation:  n_c = R n_l  ->  Kabsch on stacked normals.
Translation: d_c = d_l + n_c . t  ->  linear least squares for t.

This is the classic plane-correspondence formulation used by
target-based LiDAR-camera calibrators. Needs >= 3 captures with
non-parallel board orientations.
"""

import numpy as np


def _skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def _expm_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = _skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def solve_extrinsic(planes_cam, planes_lidar):
    """planes_* : lists of (n, d). Returns (R, t): LiDAR -> camera."""
    n_c = np.stack([p[0] for p in planes_cam])
    n_l = np.stack([p[0] for p in planes_lidar])

    H = n_l.T @ n_c                      # Kabsch covariance
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:             # proper rotation only
        Vt[-1] *= -1
        R = Vt.T @ U.T

    d_c = np.array([p[1] for p in planes_cam])
    d_l = np.array([p[1] for p in planes_lidar])
    t, *_ = np.linalg.lstsq(n_c, d_c - d_l, rcond=None)
    return R, t


def bundle_adjust(R_ext_init, t_ext_init, board_poses_init,
                 corners_uv_list, lidar_inliers, K, iters=30):
    """Joint bundle adjustment: extrinsic + all board poses, one cost.

    Residuals:
      * camera: reprojection error of board corners (2 per corner)
      * LiDAR: z-distance of each inlier point from the board plane,
        evaluated in the board frame (1 per point)

    This is the maximum-likelihood estimator under the noise model and
    lets the accurate LiDAR planes constrain the weakly-observed
    camera-side board tilt -- the same joint-optimization structure as
    production target-based calibrators (e.g. Kalibr-style).
    """
    n_cap = len(corners_uv_list)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    from .simulate import board_corners
    corners = board_corners()

    Rs = [R.copy() for R, _ in board_poses_init]
    ts = [t.copy() for _, t in board_poses_init]
    R_e, t_e = R_ext_init.copy(), t_ext_init.copy()
    e3 = np.array([0.0, 0.0, 1.0])

    def unpack_grad():
        # returns residual vector and Jacobian (n_res, 6 + 6*n_cap)
        r_all, rows = [], []
        P = 6 + 6 * n_cap
        # we build J as list of (start_row, block) and assemble at end
        for i in range(n_cap):
            R_i, t_i = Rs[i], ts[i]
            uv = corners_uv_list[i]
            # --- camera residuals ---
            p = (R_i @ corners.T).T + t_i
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            q = np.stack([fx * x / z + cx, fy * y / z + cy], axis=1)
            r_cam = (q - uv).reshape(-1)
            J_cam = np.zeros((2 * len(corners), P))
            o = 6 + 6 * i
            for k in range(len(corners)):
                Jp = np.array([[fx / z[k], 0, -fx * x[k] / z[k] ** 2],
                               [0, fy / z[k], -fy * y[k] / z[k] ** 2]])
                J_cam[2 * k:2 * k + 2, o:o + 3] = -Jp @ R_i @ _skew(corners[k])
                J_cam[2 * k:2 * k + 2, o + 3:o + 6] = Jp
            r_all.append(r_cam)
            rows.append(J_cam)
            # --- LiDAR residuals: z of point in board frame ---
            pts = lidar_inliers[i]
            p_c = (R_e @ pts.T).T + t_e
            p_b = (R_i.T @ (p_c - t_i).T).T
            r_lid = p_b[:, 2]                      # should be 0
            J_lid = np.zeros((len(pts), P))
            n_c = R_i @ e3                        # board normal, cam frame
            # w.r.t. extrinsic
            for k in range(len(pts)):
                J_lid[k, 0:3] = -e3 @ R_i.T @ R_e @ _skew(pts[k])
                J_lid[k, 3:6] = e3 @ R_i.T
            # w.r.t. board pose i
            #   dz/d(delta_r_i) = (e3 x p_b)^T ,  dz/d(delta_t_i) = -n_c^T
            cross = np.cross(np.tile(e3, (len(pts), 1)), p_b)
            J_lid[:, o:o + 3] = cross
            J_lid[:, o + 3:o + 6] = -np.tile(n_c, (len(pts), 1))
            r_all.append(r_lid)
            rows.append(J_lid)
        return np.concatenate(r_all), np.vstack(rows)

    r, _ = unpack_grad()
    prev = float(r @ r)
    for _ in range(iters):
        r, J = unpack_grad()
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        alpha, improved = 1.0, False
        for _ in range(12):
            R_ec = R_e @ _expm_so3(alpha * step[0:3])
            t_ec = t_e + alpha * step[3:6]
            Rs_c = [R @ _expm_so3(alpha * step[6 + 6 * i:6 + 6 * i + 3])
                    for i, R in enumerate(Rs)]
            ts_c = [t + alpha * step[6 + 6 * i + 3:6 + 6 * i + 6]
                    for i, t in enumerate(ts)]
            Rs_s, ts_s, R_e_s, t_e_s = Rs, ts, R_e, t_e
            Rs, ts, R_e, t_e = Rs_c, ts_c, R_ec, t_ec
            rc, _ = unpack_grad()
            if float(rc @ rc) < prev:
                prev, improved = float(rc @ rc), True
                break
            Rs, ts, R_e, t_e = Rs_s, ts_s, R_e_s, t_e_s
            alpha *= 0.5
        if not improved or np.linalg.norm(step) < 1e-9:
            break
    return R_e, t_e
