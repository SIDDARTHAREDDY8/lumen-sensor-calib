#!/usr/bin/env python3
"""Sensor bring-up health checker (ROS2-style, synthetic streams).

Simulates timestamped message streams for the sensors a mobile robot
bring-up needs (LiDAR, depth camera, IMU, wheel encoders) and checks
what actually breaks data logging pipelines in practice:
  * stream rate vs nominal (dropped frames)
  * inter-arrival jitter (scheduling latency)
  * gaps (sensor stalls / USB resets)
  * non-monotonic timestamps (clock jumps)

Usage:
    python3 bringup_check.py [--seed 0]
"""
import argparse
import numpy as np

SENSORS = {
    # name:            (nominal_hz, duration_s, jitter_ms_1sigma, drop_prob)
    "lidar_3d":        (10.0,  20.0, 2.0,  0.005),
    "depth_camera":    (30.0,  20.0, 1.5,  0.010),
    "imu":             (200.0, 20.0, 0.4,  0.002),
    "wheel_encoders":  (50.0,  20.0, 1.0,  0.010),
}

GATE_RATE_TOL = 0.05      # measured rate within 5% of nominal
GATE_JITTER_FRAC = 0.30   # jitter std <= 30% of nominal period
GATE_MAX_GAP_S = 0.5      # no single gap longer than 0.5 s


def simulate_stream(nominal_hz, duration, jitter_ms, drop_prob, seed):
    rng = np.random.default_rng(seed)
    period = 1.0 / nominal_hz
    n = int(nominal_hz * duration)
    stamps = np.arange(n) * period
    stamps += rng.normal(0, jitter_ms / 1000.0, size=n)   # scheduling jitter
    keep = rng.random(n) > drop_prob                      # dropped frames
    return np.sort(stamps[keep])


def check_stream(name, nominal_hz, stamps):
    period = 1.0 / nominal_hz
    dt = np.diff(stamps)
    measured_hz = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    rate_ok = abs(measured_hz - nominal_hz) / nominal_hz <= GATE_RATE_TOL
    jitter = float(np.std(dt))
    jitter_ok = jitter / period <= GATE_JITTER_FRAC
    max_gap = float(dt.max())
    gap_ok = max_gap <= GATE_MAX_GAP_S
    mono_ok = bool(np.all(dt > 0))
    ok = rate_ok and jitter_ok and gap_ok and mono_ok
    return dict(name=name, nominal_hz=nominal_hz, measured_hz=measured_hz,
                jitter_ms=jitter * 1000, max_gap_ms=max_gap * 1000,
                monotonic=mono_ok, ok=ok,
                checks=dict(rate=rate_ok, jitter=jitter_ok,
                            gaps=gap_ok, monotonic=mono_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 72)
    print("sensor bring-up health check  (SYNTHETIC STREAMS)")
    print("=" * 72)
    print(f"{'sensor':<16}{'nom Hz':>8}{'meas Hz':>10}"
          f"{'jitter ms':>11}{'max gap ms':>12}  verdict")
    print("-" * 72)
    all_ok = True
    for i, (name, (hz, dur, jit, drop)) in enumerate(SENSORS.items()):
        stamps = simulate_stream(hz, dur, jit, drop, seed=args.seed + i)
        r = check_stream(name, hz, stamps)
        all_ok &= r["ok"]
        print(f"{name:<16}{hz:>8.1f}{r['measured_hz']:>10.2f}"
              f"{r['jitter_ms']:>11.2f}{r['max_gap_ms']:>12.1f}"
              f"  {'PASS' if r['ok'] else 'FAIL'}")
    print("-" * 72)
    print("OVERALL:", "ALL STREAMS HEALTHY" if all_ok else "ISSUES DETECTED")
    print("gates: rate within 5%, jitter<=30% of period, no gap>500ms,"
          " monotonic stamps")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
