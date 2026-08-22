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

    # surge/sway velocity (u, v) -- section 3.4. "direct" mode uses these
    # white/bias sigmas straight on u and v; "derived" mode instead
    # differentiates the (already-noisy) position measurement per section 2.h.
    velocity_mode: str            # "direct" | "derived"
    uv_white_sigma: float         # [m/s], used only when velocity_mode == "direct"
    uv_bias_sigma: float          # [m/s], used only when velocity_mode == "direct"

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

    velocity_mode="direct", uv_white_sigma=0.1, uv_bias_sigma=0.02,

    # 0.1 deg / 0.05 deg per section 4's delta row
    delta_white_sigma=0.1 * _DEG, delta_bias_sigma=0.05 * _DEG, delta_lsb=0.0,
    # section 4 tabulates n's noise in rpm (0.5 / 0.2); the state's n is in rps
    # (config.py's RPS_MIN/MAX, section 1's own units column) so both values
    # are converted here by dividing by 60.
    n_white_sigma=0.5 / 60.0, n_bias_sigma=0.2 / 60.0, n_lsb=0.0,
)

PRESETS = {"light": LIGHT}


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
