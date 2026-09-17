"""Deterministic self-tests for the calibration pipeline."""

import numpy as np

from calib import simulate, detect, fit, solve, metrics


def _run_pipeline(seed):
    data = simulate.generate(seed=seed)
    gt_R, gt_t, K = data["gt_R"], data["gt_t"], data["K"]
    pc, pl, inl, bp, cuv = [], [], [], [], []
    for cap in data["captures"]:
        R_cb, t_cb = detect.board_pose_from_corners(cap["corners_uv"], K)
        bp.append((R_cb, t_cb))
        cuv.append(cap["corners_uv"])
        pc.append(detect.plane_from_board_pose(R_cb, t_cb))
        n_l, d_l, _, ip = fit.fit_plane_ransac(cap["lidar_points"])
        pl.append((n_l, d_l))
        inl.append(ip)
    R0, t0 = solve.solve_extrinsic(pc, pl)
    R1, t1 = solve.bundle_adjust(R0, t0, bp, cuv, inl, K)
    return (metrics.rotation_error_deg(R1, gt_R),
            metrics.translation_error_m(t1, gt_t))


def test_end_to_end_seed0():
    rot, trans = _run_pipeline(0)
    assert rot < 0.5, f"rotation gate: {rot}"
    assert trans < 0.02, f"translation gate: {trans}"


def test_end_to_end_seed1():
    rot, trans = _run_pipeline(1)
    assert rot < 0.5, f"rotation gate: {rot}"
    assert trans < 0.02, f"translation gate: {trans}"


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(7)
    pts = rng.uniform(-1, 1, size=(300, 3))
    pts[:, 2] = 2.0 + rng.normal(0, 0.01, size=300)   # plane z = 2
    pts[:30] = rng.uniform(-3, 3, size=(30, 3))       # 10% outliers
    n_est, d_est, frac, _ = fit.fit_plane_ransac(pts, seed=1)
    assert abs(abs(n_est @ np.array([0, 0, 1])) - 1.0) < 1e-3
    assert abs(abs(d_est) - 2.0) < 0.02   # normal may face either way
    assert frac > 0.85
