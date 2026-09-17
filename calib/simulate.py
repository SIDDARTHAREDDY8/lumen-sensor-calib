"""Synthetic multi-sensor data with KNOWN ground-truth extrinsics.

Generates paired observations of a checkerboard target:
  * camera: noisy 2D corner detections (known intrinsics K)
  * LiDAR: noisy 3D points sampled on the board plane (+ outliers)

All noise parameters are explicit and disclosed. Nothing here touches
real hardware: this is a correctness harness for the calibration math.
"""

import numpy as np

# --- target geometry -------------------------------------------------------
# Deliberately large, frame-filling target: real calibration guides insist
# on this, because tilt observability (hence extrinsic accuracy) scales
# with the target's apparent size.
NX, NY = 7, 5          # inner corner grid
SQUARE = 0.08          # checker square size, meters
BOARD_W = (NX + 1) * SQUARE
BOARD_H = (NY + 1) * SQUARE

# --- default camera intrinsics (synthetic) ---------------------------------
K_DEFAULT = np.array([[1000.0, 0.0, 640.0],
                      [0.0, 1000.0, 360.0],
                      [0.0, 0.0, 1.0]])
IMG_W, IMG_H = 1280, 720


def board_corners():
    """(NX*NY, 3) corner coordinates in the board frame (z = 0)."""
    xs = np.arange(NX) * SQUARE
    ys = np.arange(NY) * SQUARE
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(NX * NY)], axis=1)


def random_rotation(rng, max_deg, min_deg=0.0):
    """Small random rotation (axis-angle, uniform axis)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(min_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _default_gt_extrinsic(rng):
    """Ground-truth LiDAR->camera transform (synthetic, but fixed per seed)."""
    yaw, pitch, roll = np.deg2rad([25.0, -8.0, 12.0])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), \
        np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    t = np.array([0.40, -0.15, 0.10])
    return R, t


def _board_pose_in_camera(rng):
    """Random board pose, guaranteed visible in the synthetic camera."""
    while True:
        center = np.array([rng.uniform(-0.8, 0.8),
                           rng.uniform(-0.6, 0.6),
                           rng.uniform(1.2, 2.2)])
        # board tilted 20-45 deg off fronto-parallel, as real
        # calibration guides instruct (fronto-parallel views are degenerate)
        R = random_rotation(rng, 45.0, min_deg=20.0)
        # shift corners so the grid is centered on `center`
        offset = np.array([NX * SQUARE / 2, NY * SQUARE / 2, 0.0])
        t = center - R @ offset
        pts_cam = (R @ board_corners().T).T + t
        if np.any(pts_cam[:, 2] < 0.5):
            continue
        uv = (K_DEFAULT @ pts_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        if (np.all(uv[:, 0] > 20) and np.all(uv[:, 0] < IMG_W - 20)
                and np.all(uv[:, 1] > 20) and np.all(uv[:, 1] < IMG_H - 20)):
            return R, t


def generate(n_captures=16, pixel_noise=0.75, lidar_noise=0.02,
             outlier_frac=0.05, lidar_points=600, seed=0):
    """Generate one synthetic calibration dataset.

    Returns dict with:
      gt_R, gt_t   : ground-truth LiDAR->camera extrinsic
      K            : camera intrinsics
      captures     : list of dicts {corners_uv, lidar_points}
      params       : the noise knobs used (for disclosure)
    """
    rng = np.random.default_rng(seed)
    gt_R, gt_t = _default_gt_extrinsic(rng)
    # inverse: camera->LiDAR, to move board poses into the LiDAR frame
    R_cl = gt_R.T
    t_cl = -gt_R.T @ gt_t

    corners = board_corners()
    captures = []
    true_poses = []
    for _ in range(n_captures):
        R_cb, t_cb = _board_pose_in_camera(rng)
        true_poses.append((R_cb, t_cb))
        # camera observation: project + pixel noise
        pts_cam = (R_cb @ corners.T).T + t_cb
        uv = (K_DEFAULT @ pts_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        uv += rng.normal(0, pixel_noise, size=uv.shape)
        # LiDAR observation: sample board surface, move to LiDAR frame
        u = rng.uniform(-0.03, BOARD_W + 0.03, size=lidar_points)
        v = rng.uniform(-0.03, BOARD_H + 0.03, size=lidar_points)
        pts_board = np.stack([u, v, np.zeros_like(u)], axis=1)
        pts_cam_s = (R_cb @ pts_board.T).T + t_cb
        pts_lidar = (R_cl @ pts_cam_s.T).T + t_cl
        pts_lidar += rng.normal(0, lidar_noise, size=pts_lidar.shape)
        # outliers: uniform scatter in a 4 m box around the board
        n_out = int(lidar_points * outlier_frac)
        if n_out:
            c = pts_lidar.mean(axis=0)
            pts_lidar[:n_out] = rng.uniform(c - 2, c + 2, size=(n_out, 3))
        captures.append({"corners_uv": uv, "lidar_points": pts_lidar,
                         "board_corners": corners})

    return {"gt_R": gt_R, "gt_t": gt_t, "K": K_DEFAULT,
            "captures": captures, "true_poses": true_poses,
            "params": dict(n_captures=n_captures, pixel_noise=pixel_noise,
                           lidar_noise=lidar_noise, outlier_frac=outlier_frac,
                           lidar_points=lidar_points, seed=seed)}
