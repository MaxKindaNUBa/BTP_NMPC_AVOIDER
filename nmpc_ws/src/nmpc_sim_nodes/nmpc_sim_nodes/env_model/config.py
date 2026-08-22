"""Config dataclasses for env_model (WAVE_CURRENT_DISTURBANCE_PLAN.md section
3). Mirrors sensor_model/config.py's "frozen dataclass, plain numeric fields"
convention -- mmg_node builds one of each from its own declared ROS params
(current_enabled/current_mean_speed/... in sim_params.yaml's mmg_node section).

load_current_config()/load_wave_config() below are the equivalent path for
non-ROS callers (standalone test_*.py harnesses): they read the exact same
sim_params.yaml section directly, so a script calling these and a live
mmg_node launched off the same file can never drift apart -- no separate
hardcoded copy of hs/tp/gamma/Tc/sigma/etc. living inside a test script."""
import math
from dataclasses import dataclass
from typing import Optional

import _pkg_paths


@dataclass(frozen=True)
class CurrentConfig:
    mean_vx: float          # [m/s], earth-frame
    mean_vy: float          # [m/s], earth-frame
    time_constant_s: float  # [s], OU mean-reversion time constant (Tc)
    sigma: float            # [m/s per sqrt(s)], driving noise intensity
    seed: int


@dataclass(frozen=True)
class WaveConfig:
    hs: float               # [m] significant wave height, scaled-model units
    tp: float                # [s] peak period
    gamma: float             # JONSWAP peak-enhancement factor (3.3 = standard North Sea fit)
    mean_heading: float      # [rad], earth-fixed mean wave direction
    num_components: int      # spectral discretization for Newman's approximation
    seed: int


def load_current_config(seed: Optional[int] = None, path: Optional[str] = None) -> CurrentConfig:
    """Reads sim_params.yaml's mmg_node.current_* fields. `seed` overrides
    the yaml's own current_seed (mainly so a script's --seed flag can still
    pick a different reproducible realization per run without needing its
    own copy of mean_vx/Tc/sigma); leave unset to use the yaml's value."""
    p = _pkg_paths.load_sim_params(path)["mmg_node"]["ros__parameters"]
    speed = float(p["current_mean_speed"])
    heading = float(p["current_mean_heading"])
    return CurrentConfig(
        mean_vx=speed * math.cos(heading),
        mean_vy=speed * math.sin(heading),
        time_constant_s=float(p["current_time_constant"]),
        sigma=float(p["current_sigma"]),
        seed=int(seed) if seed is not None else int(p["current_seed"]),
    )


def load_wave_config(seed: Optional[int] = None, path: Optional[str] = None) -> WaveConfig:
    """Reads sim_params.yaml's mmg_node.wave_* fields. `seed` overrides the
    yaml's own wave_seed, same reasoning as load_current_config()."""
    p = _pkg_paths.load_sim_params(path)["mmg_node"]["ros__parameters"]
    return WaveConfig(
        hs=float(p["wave_hs"]), tp=float(p["wave_tp"]), gamma=float(p["wave_gamma"]),
        mean_heading=float(p["wave_mean_heading"]), num_components=int(p["wave_num_components"]),
        seed=int(seed) if seed is not None else int(p["wave_seed"]),
    )
