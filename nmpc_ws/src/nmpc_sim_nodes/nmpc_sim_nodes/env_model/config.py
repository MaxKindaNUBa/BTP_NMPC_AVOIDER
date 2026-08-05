"""Config dataclasses for env_model (WAVE_CURRENT_DISTURBANCE_PLAN.md section
3). Mirrors sensor_model/config.py's "frozen dataclass, plain numeric fields"
convention -- env_node builds one of each from its own declared params."""
from dataclasses import dataclass


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
