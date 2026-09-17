#!/usr/bin/env python3
"""End-to-end LiDAR-camera extrinsic calibration demo.

Pipeline (all on SYNTHETIC data, see README):
  simulate -> camera board pose (DLT homography) -> LiDAR plane (RANSAC)
           -> extrinsic solve (plane correspondences) -> report + plots

Usage:
    python3 run_calibration.py [--captures 12] [--seed 0] [--out outputs]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calib import simulate, detect, fit, solve, metrics

# acceptance gates for the default noise profile
GATE_ROT_DEG = 0.5
GATE_TRANS_M = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=int, default=16)
    ap.add_argument("--pixel-noise", type=float, default=0.75,
                    help="camera corner noise, pixels (1-sigma)")
    ap.add_argument("--lidar-noise", type=float, default=0.02,
                    help="LiDAR point noise, meters (1-sigma)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = simulate.generate(n_captures=args.captures,
                             pixel_noise=args.pixel_noise,
                             lidar_noise=args.lidar_noise,
                             seed=args.seed)
    gt_R, gt_t, K = data["gt_R"], data["gt_t"], data["K"]

    planes_cam, planes_lidar, inliers = [], [], []
    lidar_inlier_pts, board_poses, corners_uv_list = [], [], []
    for cap in data["captures"]:
        R_cb, t_cb = detect.board_pose_from_corners(cap["corners_uv"], K)
        board_poses.append((R_cb, t_cb))
        corners_uv_list.append(cap["corners_uv"])
        planes_cam.append(detect.plane_from_board_pose(R_cb, t_cb))
        n_l, d_l, frac, inl_pts = fit.fit_plane_ransac(cap["lidar_points"])
        planes_lidar.append((n_l, d_l))
        inliers.append(frac)
        lidar_inlier_pts.append(inl_pts)

    # stage 1: closed-form init from plane correspondences
    R_init, t_init = solve.solve_extrinsic(planes_cam, planes_lidar)
    # stage 2: joint bundle adjustment over extrinsic + all board poses
    R_est, t_est = solve.bundle_adjust(R_init, t_init, board_poses,
                                       corners_uv_list, lidar_inlier_pts, K)

    rot_err = metrics.rotation_error_deg(R_est, gt_R)
    tr_err = metrics.translation_error_m(t_est, gt_t)
    resid = metrics.plane_residual_m(planes_cam, planes_lidar, R_est, t_est)

    ok = rot_err < GATE_ROT_DEG and tr_err < GATE_TRANS_M

    print("=" * 60)
    print("LiDAR-camera extrinsic calibration  (SYNTHETIC DATA)")
    print("=" * 60)
    print(f"captures: {args.captures}   pixel noise: {args.pixel_noise}px"
          f"   lidar noise: {args.lidar_noise * 100:.1f}cm   seed: {args.seed}")
    print(f"mean RANSAC inlier fraction: {np.mean(inliers):.3f}")
    print()
    print("ground truth t (LiDAR->cam):", np.round(gt_t, 4))
    print("estimated    t (LiDAR->cam):", np.round(t_est, 4))
    print()
    print(f"rotation error:    {rot_err:.3f} deg   (gate < {GATE_ROT_DEG})")
    print(f"translation error: {tr_err * 100:.2f} cm    (gate < {GATE_TRANS_M * 100:.0f} cm)")
    print(f"plane residual:    mean {resid.mean() * 100:.2f} cm,"
          f" max {resid.max() * 100:.2f} cm over {len(resid)} captures")
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 60)

    # --- plots -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["tx err", "ty err", "tz err"],
              np.abs(t_est - gt_t) * 100, color="#4C78A8")
    ax[0].set_ylabel("cm")
    ax[0].set_title("per-axis translation error")
    ax[1].hist(resid * 100, bins=10, color="#72B66B", edgecolor="black")
    ax[1].set_xlabel("plane residual (cm)")
    ax[1].set_title("residuals across captures")
    fig.suptitle(f"extrinsic calibration report "
                 f"(rot err {rot_err:.3f} deg, trans err {tr_err * 100:.2f} cm)")
    fig.tight_layout()
    plot_path = os.path.join(args.out, "calibration_report.png")
    fig.savefig(plot_path, dpi=120)
    print("plot:", plot_path)

    with open(os.path.join(args.out, "results.txt"), "w") as f:
        f.write(f"rotation_error_deg={rot_err:.4f}\n")
        f.write(f"translation_error_m={tr_err:.6f}\n")
        f.write(f"plane_residual_mean_m={resid.mean():.6f}\n")
        f.write(f"plane_residual_max_m={resid.max():.6f}\n")
        f.write(f"result={'PASS' if ok else 'FAIL'}\n")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
