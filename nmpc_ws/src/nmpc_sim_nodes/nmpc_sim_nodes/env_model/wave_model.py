"""WaveModel: JONSWAP-spectrum wave-drift force, upgrading Preliminary_func.py's
single fixed-frequency regular-wave drift term (WAVE_CURRENT_DISTURBANCE_PLAN.md
section 3) to a real Hs/Tp/gamma sea state -- same Wave_Data/*.mat tables, same
(heading, frequency) RegularGridInterpolators, same Lambda/FS_force/FS_moment
non-dimensionalization Preliminary_func.py already uses for this hull, just
summed over many spectral components instead of evaluated at one.

Mean drift force:  2 * sum_i D_i(heading, omega_i) * S(omega_i) * domega
Slow drift force (Newman's approximation), on top of the mean:
  2 * sum_i sum_j sqrt(S_i S_j) * domega * sqrt(D_i D_j) * cos((omega_i-omega_j) t
      + phase_i - phase_j)
t is an elapsed-time counter incremented by `dt` (the constructor arg, matching
env_node's own per-tick dt) once per force() call.
"""
import math
import os

import numpy as np
import scipy.io
from scipy.interpolate import RegularGridInterpolator

from env_model.config import WaveConfig

# Same hull as casadi_mmg.py / Preliminary_func.py (Lpp = 2.902 m); these
# lookup tables are specific to this hull's geometry (see Wave_Data/README.md).
_LPP = 2.902
_LAMBDA = 320.0 / _LPP
_FS_FORCE = _LAMBDA ** 3
_FS_MOMENT = _LAMBDA ** 4

# Wave_Data/ stays at the repo root -- same env-var-overridable path convention
# Preliminary_func.py already uses (see that file's own comment for why).
_WAVE_DATA_DIR = os.environ.get(
    "NMPC_WAVE_DATA_DIR", "/mnt/0BF1C240574D9C37/BTP_NMPC_AVOIDER/Wave_Data") + "/"


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _signed_sqrt(a: np.ndarray) -> np.ndarray:
    """sqrt(|a|) with a's sign preserved -- these drift coefficients can be
    negative, and sqrt(D_i * D_j) in the Newman cross-term needs a real,
    sign-consistent stand-in for sqrt(D_i)*sqrt(D_j) when D_i, D_j have
    differing signs (a plain complex sqrt would poison the whole sum)."""
    return np.sign(a) * np.sqrt(np.abs(a))


def _load_drift_interpolators():
    """Loads Wave_Data/*.mat and builds the same (heading, frequency)
    RegularGridInterpolators Preliminary_func.py builds, over the *raw*
    non-dimensional drift coefficients (no Amplitude**2/2 baked in -- that
    factor was specific to Preliminary_func.py's single-regular-wave case;
    here the spectral density S(omega) plays that role per-component)."""
    X_wave = np.array(scipy.io.loadmat(_WAVE_DATA_DIR + "DX_SALV_REV1.mat")["DX"])
    Y_wave = np.array(scipy.io.loadmat(_WAVE_DATA_DIR + "DY_SALV_REV1.mat")["DY"])
    N_wave = np.array(scipy.io.loadmat(_WAVE_DATA_DIR + "DN_SALV_REV1.mat")["DN"])

    # dimensional (model-scale) angular frequency axis: Froude-scales the
    # table's non-dimensional frequency column by sqrt(Lambda), same as
    # Preliminary_func.py's `wave_frequency = 0.439*(Lamda)**0.5` convention.
    freq_axis = np.real(2.0 * np.pi * X_wave[:, 0]) * math.sqrt(_LAMBDA)
    heading_axis = -np.arange(-np.pi, np.pi + 0.01, np.deg2rad(15.0))

    D_X = np.real(X_wave[:, 1:])
    D_Y = -np.real(Y_wave[:, 1:])
    D_N = -np.real(N_wave[:, 1:])

    f_dx = RegularGridInterpolator((heading_axis, freq_axis), D_X.T, bounds_error=False, fill_value=None)
    f_dy = RegularGridInterpolator((heading_axis, freq_axis), D_Y.T, bounds_error=False, fill_value=None)
    f_dn = RegularGridInterpolator((heading_axis, freq_axis), D_N.T, bounds_error=False, fill_value=None)
    return f_dx, f_dy, f_dn


def _jonswap_spectrum(omega: np.ndarray, hs: float, tp: float, gamma: float) -> np.ndarray:
    """Standard JONSWAP one-sided spectral density S(omega) [m^2 s]: a
    Pierson-Moskowitz base spectrum shaped by JONSWAP's peak-enhancement
    factor gamma (Hasselmann et al. 1973 form, A_gamma normalization so total
    variance stays ~ (Hs/4)^2 regardless of gamma)."""
    omega = np.asarray(omega, dtype=float)
    omega_p = 2.0 * np.pi / tp
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    peak_shape = np.exp(-((omega - omega_p) ** 2) / (2.0 * sigma ** 2 * omega_p ** 2))
    a_gamma = 1.0 - 0.287 * np.log(gamma)
    omega_safe = np.maximum(omega, 1e-6)
    s_pm = ((5.0 / 16.0) * (hs ** 2) * (omega_p ** 4) * (omega_safe ** -5)
            * np.exp(-1.25 * (omega_p / omega_safe) ** 4))
    return a_gamma * s_pm * (gamma ** peak_shape)


class WaveModel:
    def __init__(self, config: WaveConfig, dt: float):
        self.config = config
        self._dt = float(dt)
        self._t = 0.0

        rng = np.random.default_rng(config.seed)
        self._f_dx, self._f_dy, self._f_dn = _load_drift_interpolators()

        omega_p = 2.0 * np.pi / config.tp
        n = int(config.num_components)
        omega_grid = np.linspace(0.4 * omega_p, 3.0 * omega_p, n)
        self.domega = float(omega_grid[1] - omega_grid[0]) if n > 1 else 0.0
        # Equally-spaced omega_i (the raw linspace grid) would make every
        # (omega_i - omega_j) an exact multiple of domega, so the Newman
        # slow-drift double sum's cos((omega_i-omega_j)*t + ...) term becomes
        # an exactly periodic function of t with period 2*pi/domega -- a
        # well-known artifact of equal-frequency discretization in irregular
        # wave synthesis (the whole force trace repeats itself every ~13s at
        # this hull's default Tp, regardless of the randomized phases below).
        # Jittering each component within its own bin breaks that
        # commensurability while leaving the Riemann-sum integration
        # (S(omega_i)*domega) just as valid -- S is still sampled once per
        # equal-width bin, only the sample point inside the bin moves.
        jitter = rng.uniform(-0.5 * self.domega, 0.5 * self.domega, size=n) if n > 1 else np.zeros(n)
        self.omega = omega_grid + jitter
        self.S = _jonswap_spectrum(self.omega, config.hs, config.tp, config.gamma)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=n)

    def force(self, psi: float):
        """Returns (fx, fy, fn): body-frame surge/sway drift force [N] and
        yaw drift moment [N*m] at the vessel's current heading psi, for this
        tick. Advances the internal elapsed-time counter by `dt` each call."""
        rel_heading = _wrap(self.config.mean_heading - float(psi))

        pts = np.column_stack([np.full(self.omega.shape, rel_heading), self.omega])
        d_x = self._f_dx(pts)
        d_y = self._f_dy(pts)
        d_n = self._f_dn(pts)

        mean_x = 2.0 * np.sum(d_x * self.S * self.domega) / _FS_FORCE
        mean_y = 2.0 * np.sum(d_y * self.S * self.domega) / _FS_FORCE
        mean_n = 2.0 * np.sum(d_n * self.S * self.domega) / _FS_MOMENT

        sqrt_s_ij = np.outer(np.sqrt(self.S), np.sqrt(self.S))
        cos_term = np.cos((self.omega[:, None] - self.omega[None, :]) * self._t
                           + (self.phase[:, None] - self.phase[None, :]))

        slow_x = 2.0 * np.sum(sqrt_s_ij * self.domega * np.outer(_signed_sqrt(d_x), _signed_sqrt(d_x)) * cos_term) / _FS_FORCE
        slow_y = 2.0 * np.sum(sqrt_s_ij * self.domega * np.outer(_signed_sqrt(d_y), _signed_sqrt(d_y)) * cos_term) / _FS_FORCE
        slow_n = 2.0 * np.sum(sqrt_s_ij * self.domega * np.outer(_signed_sqrt(d_n), _signed_sqrt(d_n)) * cos_term) / _FS_MOMENT

        self._t += self._dt

        return float(mean_x + slow_x), float(mean_y + slow_y), float(mean_n + slow_n)
