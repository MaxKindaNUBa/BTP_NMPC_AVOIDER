"""SensorModel: composes the section-2 primitives into the section-3
per-signal specs and section-5 cross-signal rules. Stateful across calls
(OU bias values, static actuator offsets, previous measured position for the
"heavy" derived-velocity mode) -- one instance per run, stepped once per
solver tick (dt = config.dt, matching the rest of the sim).

Independence (section 5): every signal below draws from its own RNG stream
(one child generator per signal, spawned from a single run seed), so no two
signals share so much as a random draw unless explicitly modeled that way
(there is no such coupling in this first implementation).

Plant-model mismatch (section 5, mandatory): this class only ever transforms
a copy of the state handed to it. It never mutates the caller's true state,
so whoever integrates the plant dynamics keeps doing so on the clean value.
"""
import math

import numpy as np

from sensor_model.config import SensorNoiseConfig
from sensor_model.noise_primitives import GaussMarkov, OutlierWhiteGaussian, StaticOffset, WhiteGaussian, quantize


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SensorModel:
    def __init__(self, config: SensorNoiseConfig, dt: float, seed: int):
        self.config = config
        self.dt = float(dt)

        # 9 independent signal streams: x, y, psi, r, u, v (direct mode only),
        # delta, n, plus one spare for outlier draws on x/y.
        seeds = np.random.SeedSequence(seed).spawn(8)
        (self._rng_x, self._rng_y, self._rng_psi, self._rng_r,
         self._rng_uv, self._rng_delta, self._rng_n, self._rng_outlier) = (
            np.random.default_rng(s) for s in seeds)

        c = config
        self._bias_x = GaussMarkov(self._rng_x, dt, c.pos_bias_tau, c.pos_bias_sigma)
        self._bias_y = GaussMarkov(self._rng_y, dt, c.pos_bias_tau, c.pos_bias_sigma)
        self._white_x = OutlierWhiteGaussian(self._rng_x, c.pos_white_sigma, c.pos_outlier_p, c.pos_outlier_sigma)
        self._white_y = OutlierWhiteGaussian(self._rng_outlier, c.pos_white_sigma, c.pos_outlier_p, c.pos_outlier_sigma)

        self._bias_psi = GaussMarkov(self._rng_psi, dt, c.psi_bias_tau, c.psi_bias_sigma)
        self._white_psi = WhiteGaussian(self._rng_psi, c.psi_white_sigma)

        self._bias_r = GaussMarkov(self._rng_r, dt, c.r_bias_tau, c.r_bias_sigma)
        self._white_r = WhiteGaussian(self._rng_r, c.r_white_sigma)

        # "direct" mode (light preset): small white + small static bias on u, v.
        self._bias_u = StaticOffset(self._rng_uv, c.uv_bias_sigma)
        self._bias_v = StaticOffset(self._rng_uv, c.uv_bias_sigma)
        self._white_u = WhiteGaussian(self._rng_uv, c.uv_white_sigma)
        self._white_v = WhiteGaussian(self._rng_uv, c.uv_white_sigma)

        self._offset_delta = StaticOffset(self._rng_delta, c.delta_bias_sigma)
        self._white_delta = WhiteGaussian(self._rng_delta, c.delta_white_sigma)
        self._offset_n = StaticOffset(self._rng_n, c.n_bias_sigma)
        self._white_n = WhiteGaussian(self._rng_n, c.n_white_sigma)

        # "derived" mode (heavy preset) state: previous step's measured x, y,
        # used to finite-difference a velocity per section 2.h/3.4. None until
        # the first call, at which point there is nothing to differentiate
        # against yet.
        self._prev_meas_xy = None

    def measure(self, mmg_state, delta: float, n: float):
        """mmg_state = [u, v, r, x, y, psi] (true). Returns a new
        (mmg_state_meas, delta_meas, n_meas) -- never mutates the input."""
        u, v, r, x, y, psi = (float(s) for s in mmg_state)
        c = self.config

        x_meas = x + self._bias_x.step() + self._white_x.sample()
        y_meas = y + self._bias_y.step() + self._white_y.sample()

        psi_meas = _wrap(psi + self._bias_psi.step() + self._white_psi.sample())

        r_meas = r + self._bias_r.step() + self._white_r.sample()

        if c.velocity_mode == "direct":
            u_meas = u + self._bias_u.step() + self._white_u.sample()
            v_meas = v + self._bias_v.step() + self._white_v.sample()
        else:
            u_meas, v_meas = self._derive_velocity(x_meas, y_meas, psi_meas, u, v)

        delta_meas = quantize(delta + self._offset_delta.step() + self._white_delta.sample(), c.delta_lsb)
        n_meas = quantize(n + self._offset_n.step() + self._white_n.sample(), c.n_lsb)

        return np.array([u_meas, v_meas, r_meas, x_meas, y_meas, psi_meas], dtype=float), delta_meas, n_meas

    def _derive_velocity(self, x_meas, y_meas, psi_meas, u_true, v_true):
        """Section 3.4 'GPS-derived' + 2.h: finite-difference the noisy
        position, then rotate the inertial-frame rate into body-frame u, v
        using the (also noisy) heading, since u/v are surge/sway, not
        inertial x_dot/y_dot."""
        if self._prev_meas_xy is None:
            self._prev_meas_xy = (x_meas, y_meas)
            return u_true, v_true  # first sample: nothing to differentiate against yet

        x_prev, y_prev = self._prev_meas_xy
        vx = (x_meas - x_prev) / self.dt
        vy = (y_meas - y_prev) / self.dt
        self._prev_meas_xy = (x_meas, y_meas)

        cos_psi, sin_psi = math.cos(psi_meas), math.sin(psi_meas)
        u_meas = vx * cos_psi + vy * sin_psi
        v_meas = -vx * sin_psi + vy * cos_psi
        return u_meas, v_meas
