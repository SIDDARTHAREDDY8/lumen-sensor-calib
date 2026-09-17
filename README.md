# lumen-sensor-calib

LiDAR-camera extrinsic calibration + sensor bring-up health checks, in pure
Python/NumPy. Built as a problem-first demo for **Lumen Labs'** Robotics
Engineer role (teammate #3: own the ROS2 sensor stack, integrate and
calibrate LiDAR/depth/IMU/encoders, keep the data logging pipeline clean).

## What it does

**1. `run_calibration.py` — target-based LiDAR-camera extrinsic calibration**

The same pipeline real robot bring-up uses, end to end:

1. **Camera side** — checkerboard corner detections → board pose per view
   (DLT homography with Hartley normalization → Gauss-Newton refinement
   of reprojection error, i.e. the two-stage structure of iterative
   `solvePnP`).
2. **LiDAR side** — RANSAC plane fit per view (robust to outliers).
3. **Closed-form init** — rotation from plane-normal correspondences
   (Kabsch), translation from plane offsets (linear least squares).
4. **Joint bundle adjustment** — Gauss-Newton over the extrinsic *and*
   all board poses, minimizing corner reprojection error plus LiDAR
   point-to-plane distance (Kalibr-style). Analytic Jacobians, verified
   against finite differences.

Default run: 16 captures, 0.75 px corner noise, 2 cm LiDAR noise, 5%
outliers. Recovers the ground-truth extrinsic to **< 0.5° rotation** and
**< 2 cm translation** (acceptance gates enforced by the script; it
exits non-zero on FAIL). Writes `outputs/calibration_report.png` and
`outputs/results.txt`.

**2. `bringup_check.py` — sensor stream health checker**

Simulates the timestamped streams a ROS2 bring-up has to watch (LiDAR
10 Hz, depth camera 30 Hz, IMU 200 Hz, wheel encoders 50 Hz, with jitter
and dropouts) and checks rate vs nominal, inter-arrival jitter, gaps,
and timestamp monotonicity — the things that silently corrupt data
logging pipelines. PASS/FAIL per stream with explicit gates.

## Run it

```bash
pip install -r requirements.txt
python3 run_calibration.py            # full calibration demo
python3 run_calibration.py --seed 1   # different noise realization
python3 bringup_check.py              # stream health checks
```

Tests (no pytest needed): `python3 -c` the functions in
`tests/test_calibration.py`, or install pytest and run `pytest tests/`.

## Honest verification notes

- **All data is synthetic.** The "sensors" are NumPy random streams with a
  *known* ground-truth extrinsic, which is exactly what makes the
  accuracy claim checkable: the script reports error against ground
  truth, not against itself. Nothing here has touched real hardware,
  a real camera, a real LiDAR, or a ROS2 runtime.
- **What ran:** `run_calibration.py` (PASS, 0.197° / 0.73 cm on seed 0;
  also verified < 0.5° / < 2 cm on seeds 1–7), `bringup_check.py`
  (all streams PASS), and the three unit tests — on Python 3.12,
  NumPy 1.26, 2026-09-17.
- **What didn't:** any real sensor bring-up, ROS2 node integration,
  or hardware-in-the-loop validation. The calibration math is the real
  algorithm (same formulation as production target-based calibrators),
  but the noise model is Gaussian and kind; real sensors are not.
- **Known limitation found while building:** near-fronto-parallel board
  views are degenerate for tilt estimation (weak perspective), so the
  simulator enforces 20–45° tilts and frame-filling views — the same
  guidance real calibration procedures give. The bundle-adjustment
  stage exists because a closed-form plane-correspondence solve alone
  left ~2°/7 cm errors on some noise realizations.

## Layout

```
calib/
  simulate.py   synthetic dataset with known ground truth
  detect.py     DLT homography + Gauss-Newton board pose (camera side)
  fit.py        RANSAC plane fit (LiDAR side)
  solve.py      Kabsch/linear init + joint bundle adjustment
  metrics.py    rotation/translation/plane-residual errors
run_calibration.py   end-to-end demo CLI
bringup_check.py     sensor stream health checks
tests/               deterministic self-tests
outputs/             generated report (from the last run)
```
