"""Error metrics against ground truth."""

import numpy as np


def rotation_error_deg(R_est, R_true):
    cosang = (np.trace(R_est.T @ R_true) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cosang, -1, 1))))


def translation_error_m(t_est, t_true):
    return float(np.linalg.norm(t_est - t_true))


def plane_residual_m(planes_cam, planes_lidar, R_est, t_est):
    """|d_c - d_l - n_c.t| per capture: how well the solved extrinsic
    explains every observed plane pair."""
    res = []
    for (n_c, d_c), (n_l, d_l) in zip(planes_cam, planes_lidar):
        n_c_pred = R_est @ n_l
        res.append(abs(d_c - d_l - float(n_c_pred @ t_est)))
    return np.array(res)
