"""CurrentModel: a mean-reverting Ornstein-Uhlenbeck process on the earth-frame
current velocity (vx, vy) -- WAVE_CURRENT_DISTURBANCE_PLAN.md section 3.

Cartesian, not polar (speed+heading), specifically to avoid heading-wraparound
math: two independent OU components mean-revert toward (mean_vx, mean_vy)
directly, no atan2/angle-wrap bookkeeping needed anywhere in this class.

Own RNG instance (np.random.default_rng(seed)) -- never touches global numpy
RNG state, matching sensor_model/noise_primitives.py's convention.
"""
import numpy as np

from env_model.config import CurrentConfig


class CurrentModel:
    def __init__(self, config: CurrentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.vx = float(config.mean_vx)
        self.vy = float(config.mean_vy)

    def step(self, dt: float):
        """One Euler-Maruyama update of the OU process on each Cartesian
        component independently:
            v[k+1] = v[k] + dt * (-(v[k]-mean)/Tc) + sqrt(dt) * sigma * N(0,1)
        Returns the new (vx, vy)."""
        c = self.config
        dt = float(dt)
        dw = self.rng.normal(0.0, 1.0, size=2) * np.sqrt(dt)

        self.vx += dt * (-(self.vx - c.mean_vx) / c.time_constant_s) + c.sigma * dw[0]
        self.vy += dt * (-(self.vy - c.mean_vy) / c.time_constant_s) + c.sigma * dw[1]

        return self.vx, self.vy
