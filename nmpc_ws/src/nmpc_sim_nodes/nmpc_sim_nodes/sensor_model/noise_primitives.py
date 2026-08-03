"""Generic noise-generator primitives from SENSOR_NOISE_MODEL.md section 2.
Each class holds whatever state it needs (RNG, current bias value) and exposes
a single `step()`/`sample()` call per solver tick -- callers never touch the
underlying recursion.
"""
import numpy as np


class WhiteGaussian:
    """2.a: y_k = x_k + w_k, w_k ~ N(0, sigma^2) i.i.d. Zero sigma -> no-op."""

    def __init__(self, rng: np.random.Generator, sigma: float):
        self.rng = rng
        self.sigma = float(sigma)

    def sample(self) -> float:
        if self.sigma == 0.0:
            return 0.0
        return float(self.rng.normal(0.0, self.sigma))


class OutlierWhiteGaussian(WhiteGaussian):
    """2.f: contaminated-Gaussian mixture on top of the white-noise draw --
    rare heavy-tailed spikes (multipath etc.) instead of the plain N(0, sigma^2)
    used by WhiteGaussian. p == 0 makes this identical to WhiteGaussian."""

    def __init__(self, rng: np.random.Generator, sigma: float, p: float, sigma_outlier: float):
        super().__init__(rng, sigma)
        self.p = float(p)
        self.sigma_outlier = float(sigma_outlier)

    def sample(self) -> float:
        if self.p > 0.0 and self.rng.random() < self.p:
            return float(self.rng.normal(0.0, self.sigma_outlier))
        return super().sample()


class GaussMarkov:
    """2.b: discretized Ornstein-Uhlenbeck colored drift, exact (not
    Euler-approximated) discretization -- correct for any dt.

        a = exp(-dt / tau), q = sigma_b * sqrt(1 - a^2)
        b_k = a * b_{k-1} + q * eps_k,   eps_k ~ N(0, 1)
        b_0 ~ N(0, sigma_b^2)

    sigma_b == 0 short-circuits to an always-zero bias (no drift at all).
    """

    def __init__(self, rng: np.random.Generator, dt: float, tau: float, sigma_b: float):
        self.rng = rng
        self.sigma_b = float(sigma_b)
        if self.sigma_b == 0.0:
            self._a = 0.0
            self._q = 0.0
            self.value = 0.0
        else:
            self._a = float(np.exp(-dt / tau))
            self._q = self.sigma_b * float(np.sqrt(1.0 - self._a ** 2))
            self.value = float(rng.normal(0.0, self.sigma_b))

    def step(self) -> float:
        if self.sigma_b == 0.0:
            return 0.0
        self.value = self._a * self.value + self._q * float(self.rng.normal(0.0, 1.0))
        return self.value


class StaticOffset:
    """2.d (the 'held constant' variant): b_k = b_0, drawn once at construction
    and never updated again -- the encoder-mounting-offset model used for the
    actuator signals (delta, n), where a full OU process would be overkill."""

    def __init__(self, rng: np.random.Generator, sigma_b: float):
        self.value = float(rng.normal(0.0, sigma_b)) if sigma_b != 0.0 else 0.0

    def step(self) -> float:
        return self.value


def quantize(x: float, lsb: float) -> float:
    """2.e: y_k = round(x_k / LSB) * LSB. lsb <= 0 disables quantization."""
    if lsb <= 0.0:
        return x
    return round(x / lsb) * lsb
