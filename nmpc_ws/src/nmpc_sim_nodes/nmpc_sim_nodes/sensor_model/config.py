"""Numeric preset from SENSOR_NOISE_MODEL.md section 4 ("Light").
All angles are stored internally in radians (the doc tabulates them in
degrees), all rates in rad/s. LSB fields are 0.0 (quantization disabled) since
section 4's table leaves them unspecified -- section 3.5 explicitly allows
this ("can be set to an effectively-zero step size if quantization is not of
interest yet").
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

import _pkg_paths

_DEG = np.deg2rad(1.0)


@dataclass(frozen=True)
class SensorNoiseConfig:
    # position (x, y) -- GPS-like, section 3.1
    pos_white_sigma: float       # [m]
    pos_bias_tau: float          # [s]
    pos_bias_sigma: float        # [m]
    pos_outlier_p: float         # [-], 0 disables
    pos_outlier_sigma: float     # [m]

    # heading (psi) -- compass/AHRS, section 3.2
    psi_white_sigma: float       # [rad]
    psi_bias_tau: float          # [s]
    psi_bias_sigma: float        # [rad]

    # yaw rate (r) -- gyroscope, section 3.3
    r_white_sigma: float         # [rad/s]
    r_bias_tau: float            # [s]
    r_bias_sigma: float          # [rad/s]

    # body-frame accelerometer (ax, ay) -- IMU, replaces the old direct u,v
    # velocity channel entirely: this sensor model no longer measures u,v at
    # all (that reconstruction is the UKF's job now). White + Gauss-Markov
    # bias, same two-component shape as the psi/r groups above (real MEMS
    # accelerometers exhibit both a noise-density term and a slowly-drifting
    # bias-instability term per their Allan-variance spec, not white noise
    # alone).
    accel_white_sigma: float      # [m/s^2]
    accel_bias_tau: float         # [s]
    accel_bias_sigma: float       # [m/s^2]

    # rudder angle (delta) / propeller speed (n) -- actuator encoders, 3.5
    delta_white_sigma: float      # [rad]
    delta_bias_sigma: float       # [rad]
    delta_lsb: float              # [rad], 0 disables quantization
    n_white_sigma: float          # [rps]
    n_bias_sigma: float           # [rps]
    n_lsb: float                  # [rps], 0 disables quantization


LIGHT = SensorNoiseConfig(
    pos_white_sigma=1.5, pos_bias_tau=120.0, pos_bias_sigma=1.0,
    pos_outlier_p=0.0, pos_outlier_sigma=0.0,

    psi_white_sigma=1.0 * _DEG, psi_bias_tau=300.0, psi_bias_sigma=0.5 * _DEG,

    r_white_sigma=0.1 * _DEG, r_bias_tau=300.0, r_bias_sigma=0.05 * _DEG,

    # Starting values -- not yet empirically tuned; see test_ukf.py.
    accel_white_sigma=0.02, accel_bias_tau=300.0, accel_bias_sigma=0.01,

    # 0.1 deg / 0.05 deg per section 4's delta row
    delta_white_sigma=0.1 * _DEG, delta_bias_sigma=0.05 * _DEG, delta_lsb=0.0,
    # section 4 tabulates n's noise in rpm (0.5 / 0.2); the state's n is in rps
    # (config.py's RPS_MIN/MAX, section 1's own units column) so both values
    # are converted here by dividing by 60.
    n_white_sigma=0.5 / 60.0, n_bias_sigma=0.2 / 60.0, n_lsb=0.0,
)

# SBG_ELLIPSE_N: derived from SBG Systems' real Ellipse-N datasheet (a
# semi-expensive, marine-relevant GNSS-aided INS -- IMU + GPS + rate gyro in
# one integrated module: https://www.sbg-systems.com/ins/ellipse-n/ and its
# performance-specifications page), not hand-picked placeholder numbers.
# Source figures (as published, 1-sigma/RMS over full temperature range):
#   Gyroscope:     Angular Random Walk (ARW) = 0.18 deg/sqrt(hr)
#                  In-run bias instability    = 7 deg/hr
#   Accelerometer: Velocity Random Walk (VRW) = 57 ug/sqrt(Hz)
#                  In-run bias instability    = 14 ug
#   GNSS:          Single-point horizontal position accuracy = 1.5 m RMS
#   Heading:       Marine, single-antenna magnetic mode       = 0.8 deg RMS
#
# Conversion to this project's per-sample sigma_white / sigma_bias split:
#   Gyro/accel white noise: ARW/VRW are continuous-time noise DENSITIES
#   (deg/s/sqrt(Hz) once ARW is divided by 60; ug/sqrt(Hz) for VRW as
#   published). Discrete sample sigma at this project's dt=0.1s
#   (fs=10Hz) = density * sqrt(fs) -- the standard random-walk-to-sample-noise
#   conversion (e.g. Groves, "Principles of GNSS, Inertial, and Multisensor
#   Integrated Navigation Systems").
#   Gyro/accel bias: "in-run bias instability" IS the Allan-variance
#   stationary-wander amplitude -- maps directly onto this project's
#   Gauss-Markov bias_sigma, no conversion needed beyond units.
#   GPS position / heading: the datasheet gives one COMBINED RMS figure, not
#   separately split into white vs. slowly-correlated (bias) components.
#   This project splits each combined figure using a 2/3-bias : 1/3-white
#   VARIANCE ratio (i.e. bias_sigma = total*sqrt(2/3), white_sigma =
#   total*sqrt(1/3)) -- a documented approximation, not a datasheet number:
#   single-point GPS error is dominated by slowly-varying atmospheric/
#   multipath effects (correlating over tens of seconds as geometry/
#   atmosphere drift) rather than fast receiver thermal noise, and magnetic
#   heading error is likewise dominated by slowly-varying calibration/local-
#   disturbance residuals rather than fast noise -- both are the same
#   underlying assumption applied consistently across the two channels.
#   pos_bias_tau/psi_bias_tau/r_bias_tau/accel_bias_tau: the datasheet gives
#   no correlation-time figures at all, so these four are carried over
#   unchanged from LIGHT's own values (also not datasheet-derived there).
_SBG_G0 = 9.80665  # [m/s^2] standard gravity, for converting ug -> m/s^2
SBG_ELLIPSE_N = SensorNoiseConfig(
    pos_white_sigma=0.8660254, pos_bias_tau=120.0, pos_bias_sigma=1.2247449,
    pos_outlier_p=0.0, pos_outlier_sigma=0.0,  # not covered by the datasheet's aggregate accuracy figure

    psi_white_sigma=0.4618802 * _DEG, psi_bias_tau=300.0, psi_bias_sigma=0.6531973 * _DEG,

    r_white_sigma=0.00948683 * _DEG, r_bias_tau=300.0, r_bias_sigma=(7.0 / 3600.0) * _DEG,

    accel_white_sigma=57e-6 * np.sqrt(10.0) * _SBG_G0, accel_bias_tau=300.0, accel_bias_sigma=14e-6 * _SBG_G0,

    # Actuator encoders (delta, n) aren't part of an IMU/GPS/gyro nav module
    # -- out of scope for this datasheet-derived preset, carried over from
    # LIGHT unchanged.
    delta_white_sigma=0.1 * _DEG, delta_bias_sigma=0.05 * _DEG, delta_lsb=0.0,
    n_white_sigma=0.5 / 60.0, n_bias_sigma=0.2 / 60.0, n_lsb=0.0,
)

PRESETS = {"light": LIGHT, "sbg_ellipse_n": SBG_ELLIPSE_N}


def load_preset_and_seed(seed: Optional[int] = None, path: Optional[str] = None) -> tuple:
    """Reads sim_params.yaml's mmg_node.sensor_preset/sensor_seed fields,
    returning (preset_name, SensorNoiseConfig, seed). `seed` overrides the
    yaml's own sensor_seed (mainly so a script's --seed flag can pick a
    different reproducible noise realization per run); leave unset to use
    the yaml's value. The one entry point standalone test_*.py harnesses
    should use instead of hardcoding a preset name/seed of their own."""
    p = _pkg_paths.load_sim_params(path)["mmg_node"]["ros__parameters"]
    preset_name = str(p["sensor_preset"])
    resolved_seed = int(seed) if seed is not None else int(p["sensor_seed"])
    return preset_name, PRESETS[preset_name], resolved_seed
