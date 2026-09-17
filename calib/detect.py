"""Camera-side board pose from corner detections.

Standard planar-target pipeline (Zhang-style, first step):
  1. work in normalized image coords (K^-1), so the homography H maps
     board-plane coords (X, Y) -> normalized coords (x, y)
  2. estimate H with the DLT
  3. decompose H = [r1 r2 t] into rotation + translation, then
     re-orthogonalize R

No OpenCV dependency; the math is the same one used in real
target-based calibrators.
"""

import numpy as np

from .simulate import board_corners


def _normalize_pts(pts):
    """Hartley normalization: centroid -> origin, mean dist -> sqrt(2)."""
    c = pts.mean(axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1]])
    return T, (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :2]


def _dlt_homography(src, dst):
    """DLT homography mapping src (N,2) -> dst (N,2), with Hartley
    normalization for noise robustness."""
    T1, src_n = _normalize_pts(src)
    T2, dst_n = _normalize_pts(dst)
    n = src.shape[0]
    A = np.zeros((2 * n, 9))
    for i, ((X, Y), (x, y)) in enumerate(zip(src_n, dst_n)):
        A[2 * i] = [-X, -Y, -1, 0, 0, 0, x * X, x * Y, x]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, y * X, y * Y, y]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ H @ T1
    return H / H[2, 2]


def _skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def _expm_so3(w):
    """Rodrigues exponential map: axis-angle vector -> rotation matrix."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = _skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def refine_pose(R_init, t_init, corners_uv, K, iters=50):
    """Gauss-Newton refinement of (R, t) minimizing reprojection error.

    Same role as the iterative stage of solvePnP: the DLT homography
    decomposition is only an algebraic initializer and can sit in a
    poor basin for near-fronto-parallel views.
    """
    corners = board_corners()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    R, t = R_init.copy(), t_init.copy()

    def residuals(R, t):
        p = (R @ corners.T).T + t
        q = np.stack([fx * p[:, 0] / p[:, 2] + cx,
                      fy * p[:, 1] / p[:, 2] + cy], axis=1)
        return (q - corners_uv).reshape(-1), p

    r, _ = residuals(R, t)
    cost = float(r @ r)
    for _ in range(iters):
        r, p = residuals(R, t)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        J = np.zeros((2 * len(corners), 6))
        for i in range(len(corners)):
            Jp = np.array([[fx / z[i], 0, -fx * x[i] / z[i] ** 2],
                           [0, fy / z[i], -fy * y[i] / z[i] ** 2]])
            J[2 * i:2 * i + 2, :3] = -Jp @ R @ _skew(corners[i])
            J[2 * i:2 * i + 2, 3:] = Jp
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        # backtracking line search
        alpha, improved = 1.0, False
        for _ in range(10):
            Rc = R @ _expm_so3(alpha * step[:3])
            tc = t + alpha * step[3:]
            rc, _ = residuals(Rc, tc)
            if float(rc @ rc) < cost:
                R, t, r, cost, improved = Rc, tc, rc, float(rc @ rc), True
                break
            alpha *= 0.5
        if not improved or np.linalg.norm(step) < 1e-10:
            break
    return R, t


def board_pose_from_corners(corners_uv, K):
    """Estimate (R, t): board frame -> camera frame, from 2D detections.

    DLT homography for initialization, Gauss-Newton refinement on the
    geometric (reprojection) error -- the same two-stage structure as
    iterative solvePnP.
    """
    corners = board_corners()
    Kinv = np.linalg.inv(K)
    uv_h = np.hstack([corners_uv, np.ones((len(corners_uv), 1))])
    norm = (Kinv @ uv_h.T).T[:, :2]          # normalized image coords
    H = _dlt_homography(corners[:, :2], norm)  # (X,Y) -> (x,y)

    lam = 1.0 / np.linalg.norm(H[:, 0])
    r1, r2 = lam * H[:, 0], lam * H[:, 1]
    t = lam * H[:, 2]
    if t[2] < 0:                             # board must be in front
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R_approx = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R_approx)        # nearest proper rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return refine_pose(R, t, corners_uv, K)


def plane_from_board_pose(R, t):
    """Board plane in camera frame as (n, d) with n.x = d, ||n|| = 1,
    normal oriented toward the camera origin."""
    n = R[:, 2].copy()
    c = t + R @ np.array([0.0, 0.0, 0.0])  # board origin in camera frame
    if n @ c > 0:                            # flip to face the camera
        n = -n
    d = float(n @ c)
    return n, d
