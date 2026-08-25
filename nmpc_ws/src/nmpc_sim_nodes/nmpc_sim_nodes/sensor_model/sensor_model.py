"""SensorModel: composes the section-2 primitives into the section-3
per-signal specs and section-5 cross-signal rules. Stateful across calls
(OU bias values, static actuator offsets) -- one instance per run, stepped
once per solver tick (dt = config.dt, matching the rest of the sim).

This model measures GPS position (x,y), compass/AHRS heading (psi), rate
gyro (r), IMU accelerometer (ax,ay), and actuator encoders (delta,n) --
explicitly not surge/sway velocity (u,v) directly; reconstructing u,v (and
estimating ocean current) from this sensor set is ukf_node's job.

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

        # 8 independent signal streams: x, y, psi, r, accel (ax+ay share one,
        # same convention the old u,v channel used), delta, n, plus one spare
        # for outlier draws on x/y.
        seeds = np.random.SeedSequence(seed).spawn(8)
        (self._rng_x, self._rng_y, self._rng_psi, self._rng_r,
         self._rng_accel, self._rng_delta, self._rng_n, self._rng_outlier) = (
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

        # IMU accelerometer (ax, ay): white + Gauss-Markov bias-instability
        # drift, same two-component shape as psi/r above.
        self._bias_ax = GaussMarkov(self._rng_accel, dt, c.accel_bias_tau, c.accel_bias_sigma)
        self._bias_ay = GaussMarkov(self._rng_accel, dt, c.accel_bias_tau, c.accel_bias_sigma)
        self._white_ax = WhiteGaussian(self._rng_accel, c.accel_white_sigma)
        self._white_ay = WhiteGaussian(self._rng_accel, c.accel_white_sigma)

        self._offset_delta = StaticOffset(self._rng_delta, c.delta_bias_sigma)
        self._white_delta = WhiteGaussian(self._rng_delta, c.delta_white_sigma)
        self._offset_n = StaticOffset(self._rng_n, c.n_bias_sigma)
        self._white_n = WhiteGaussian(self._rng_n, c.n_white_sigma)

    def measure(self, mmg_state, delta: float, n: float, accel_true=(0.0, 0.0)):
        """mmg_state = [u, v, r, x, y, psi] (true; u,v are used only to
        satisfy this signature's existing callers -- they are NOT measured).
        accel_true = (ax, ay) body-frame true acceleration, since this class
        has no dynamics knowledge of its own and can't compute it itself.
        Returns (r_meas, x_meas, y_meas, psi_meas, ax_meas, ay_meas,
        delta_meas, n_meas) -- never mutates the input."""
        _, _, r, x, y, psi = (float(s) for s in mmg_state)
        ax, ay = (float(s) for s in accel_true)
        c = self.config

        x_meas = x + self._bias_x.step() + self._white_x.sample()
        y_meas = y + self._bias_y.step() + self._white_y.sample()

        psi_meas = _wrap(psi + self._bias_psi.step() + self._white_psi.sample())

        r_meas = r + self._bias_r.step() + self._white_r.sample()

        ax_meas = ax + self._bias_ax.step() + self._white_ax.sample()
        ay_meas = ay + self._bias_ay.step() + self._white_ay.sample()

        delta_meas = quantize(delta + self._offset_delta.step() + self._white_delta.sample(), c.delta_lsb)
        n_meas = quantize(n + self._offset_n.step() + self._white_n.sample(), c.n_lsb)

        return r_meas, x_meas, y_meas, psi_meas, ax_meas, ay_meas, delta_meas, n_meas
