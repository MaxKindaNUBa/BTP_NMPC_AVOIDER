"""UKFConfig: tunable sigma-point shaping and noise-covariance parameters for
ukf_node's UnscentedKalmanFilter. Structural constants (state dim, ordering)
live in ukf_core.py next to the code that depends on them, same split
nmpc/config.py vs nmpc/params.py uses.
"""
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

import _pkg_paths

_pkg_paths.ensure_on_path()

from sensor_model.config import SensorNoiseConfig  # noqa: E402

STATE_DIM = 12  # [u, v, r, x, y, psi, vcx, vcy, ax_bias, ay_bias, pos_bias_x, pos_bias_y]
MEAS_DIM = 6    # [x, y, psi, r, ax, ay]


@dataclass(frozen=True)
class UKFConfig:
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    # [u,v,r,x,y,psi,vcx,vcy,ax_bias,ay_bias,pos_bias_x,pos_bias_y].
    # vcx/vcy = current_sigma**2*dt (sim_params.yaml current_sigma=0.000019:
    # 0.000019**2*0.1 = 3.61e-11) -- the exact per-step variance CurrentModel's
    # own Euler-Maruyama OU step (env_model/current_model.py: v += ... +
    # sigma*sqrt(dt)*N(0,1)) adds, dropping the negligible mean-reversion term
    # since current_time_constant=21600s >> dt (re-derive this if either yaml
    # value changes materially -- see project memory on this file's history).
    # CORRECTED 2026-08-23: this literal previously hardcoded 0.00019 (10x
    # sim_params.yaml's real current_sigma), giving 3.61e-9 -- 100x too large
    # -- which was then the "physically-derived starting point" tune_ukf.py's
    # NEES search scaled ~71x further, leaving the checked-in
    # sim_params.yaml q_diag ~7147x too large for vcx/vcy and causing the
    # UKF's current estimate to track noise rather than the true, near-
    # constant current (see sim_params.yaml's ukf_node.q_diag comment for the
    # full account). This class default is only the load_ukf_config()
    # fallback when sim_params.yaml omits q_diag -- fixed here too so that
    # fallback path can't reintroduce the same 100x error.
    # ax_bias/ay_bias = accel_bias_sigma**2*(1-exp(-dt/accel_bias_tau)**2)
    # (LIGHT preset defaults: 0.01**2*(1-exp(-0.1/300)**2) = 6.664e-8).
    # pos_bias_x/pos_bias_y = pos_bias_sigma**2*(1-exp(-dt/pos_bias_tau)**2)
    # (LIGHT preset defaults: 1.0**2*(1-exp(-0.1/120)**2) = 1.665e-3). Both
    # bias pairs use the exact per-step driving-noise variance of the SAME
    # Gauss-Markov process sensor_model.noise_primitives.GaussMarkov itself
    # uses, so the filter's internal model of each bias matches its real
    # generator -- see accel_bias_decay()/pos_bias_decay() below.
    q_diag: Sequence[float] = field(default_factory=lambda: (
        1.0e-4, 1.0e-4, np.deg2rad(0.05) ** 2, 1.0e-4, 1.0e-4, np.deg2rad(0.02) ** 2,
        3.61e-11, 3.61e-11, 6.664e-8, 6.664e-8, 1.665e-3, 1.665e-3,
    ))
    # ax_bias/ay_bias initial variance = accel_bias_sigma**2 (0.01**2);
    # pos_bias_x/pos_bias_y initial variance = pos_bias_sigma**2 (1.0**2) --
    # both matching GaussMarkov's own b_0 ~ N(0, sigma_b**2) initial draw.
    # vcx/vcy initial variance = current OU process's own STATIONARY variance
    # sigma**2*Tc/2 (sim_params.yaml defaults: 0.000019**2*21600/2 = 3.8988e-6
    # -- re-derive from whatever current_sigma/current_time_constant a given
    # deployment actually uses, don't copy this literal blindly) -- the
    # mathematically correct prior uncertainty for "what is the current right
    # now" given only the process's long-run statistics and no observation
    # yet, NOT "totally unknown"/arbitrarily wide. Only a net win paired with seeding
    # vcx/vcy at CurrentConfig's own mean (see ukf_node.py's __init__ /
    # test_ukf.py's _current_mean()) rather than always 0 -- a tight prior
    # centered on the wrong value is worse than a wide one, not better
    # (measured: same q_diag/r_diag, tightening p0 alone made vcx RMSE ~3x
    # WORSE; tightening it WITH the seed fix made it ~2x better). See
    # sim_params.yaml's own p0_diag comment for the full derivation/numbers.
    p0_diag: Sequence[float] = field(default_factory=lambda: (
        0.01, 0.01, np.deg2rad(1.0) ** 2, 0.25, 0.25, np.deg2rad(5.0) ** 2,
        3.8988e-6, 3.8988e-6, 1.0e-4, 1.0e-4, 1.0, 1.0,
    ))
    r_diag: Sequence[float] = None  # if None, derived from a SensorNoiseConfig via default_r_diag()

    # Sage-Husa adaptive noise estimation -- see ukf/ukf_core.py's module
    # docstring for the mechanism. DEFAULT OFF: it does fix NEES calibration
    # (held-out avg NEES 22->11.8 in this project's own validation), but NEES
    # only measures whether reported uncertainty matches actual error, not
    # whether the point estimate is accurate -- here it achieved calibration
    # by increasing Q (less smoothing), which made vcx RMSE ~3x WORSE on
    # every tested seed versus the same q_diag/r_diag with adaptation off.
    # Left available for further experimentation, not a default
    # recommendation. forgetting_factor in (0,1): higher = slower to adapt /
    # more smoothing (effective memory ~= 1/(1-forgetting_factor) samples --
    # 0.98 => ~50 samples => ~5s at dt=0.1s).
    adapt_r: bool = False
    adapt_q: bool = False
    forgetting_factor: float = 0.98


def _ou_discretize(stationary_var: float, tau: float, dt: float) -> tuple:
    """(q, p0) for ANY Ornstein-Uhlenbeck / Gauss-Markov state given its
    stationary variance and mean-reversion time constant -- the one formula
    shared by current_noise_diag() and bias_noise_diag() below, since both
    vcx/vcy and the sensor biases are the same underlying process, just
    parameterized differently by their respective generators (CurrentModel
    vs sensor_model.noise_primitives.GaussMarkov).

    q (process noise added each predict step) is the EXACT discrete-time OU
    step variance, not the small-dt/tau shortcut, so this stays correct
    across regimes: q = stationary_var * (1 - exp(-2*dt/tau)). p0 (initial
    covariance) is always the stationary variance itself, independent of dt
    -- the right prior for "what is this state right now" given only the
    process's long-run statistics and no observation yet.
    """
    stationary_var, tau, dt = float(stationary_var), float(tau), float(dt)
    if tau <= 0.0:
        return stationary_var, stationary_var
    q = stationary_var * (1.0 - np.exp(-2.0 * dt / tau))
    return q, stationary_var


def current_noise_diag(sigma: float, time_constant_s: float, dt: float) -> tuple:
    """(q, p0) for the vcx/vcy rows (indices 6,7), derived LIVE from
    CurrentModel's own OU parameters instead of a hand-copied numeric literal.

    q_diag/p0_diag used to hardcode these for one specific (sigma, Tc) pair
    (sim_params.yaml's current_sigma=0.000019, current_time_constant=21600.0
    at the time): q=sigma**2*dt=3.61e-11, p0=sigma**2*Tc/2=3.8988e-6. Whoever
    changes either yaml value for a test -- e.g. a faster-varying current,
    sigma and Tc each 1000x off from those defaults -- silently left the
    filter's noise model sized for the OLD, much calmer current, so it
    couldn't keep up with the new one (looked like the UKF "can't track a
    varying current" when the actual bug was a stale Q). Deriving both here
    from whatever CurrentConfig actually holds closes that off permanently.

    CurrentModel's own sigma is a driving-noise INTENSITY (units m/s per
    sqrt(s)), not the process's stationary std directly, so its stationary
    variance is sigma**2*Tc/2 (standard OU result) -- unlike GaussMarkov's
    bias_sigma below, which already IS the stationary std.
    """
    sigma, tc = float(sigma), float(time_constant_s)
    if tc <= 0.0:
        return sigma ** 2 * dt, sigma ** 2 * dt
    return _ou_discretize(sigma ** 2 * tc / 2.0, tc, dt)


def bias_noise_diag(bias_sigma: float, tau: float, dt: float) -> tuple:
    """(q, p0) for a sensor-bias row (ax_bias/ay_bias or pos_bias_x/pos_bias_y),
    derived LIVE from the sensor preset's own GaussMarkov parameters instead
    of a hand-copied numeric literal -- same bug class as current_noise_diag()
    above: q_diag[8:12]/p0_diag[8:12] used to hardcode these for one specific
    sensor_preset's accel_bias_sigma/accel_bias_tau/pos_bias_sigma/
    pos_bias_tau (whichever preset was active when someone last hand-derived
    the numbers), so switching sensor_preset -- or retuning a preset's own
    bias_sigma/bias_tau -- silently left the filter's noise model sized for
    the OLD preset's bias statistics.

    bias_sigma here already IS the process's stationary std (matching
    sensor_model.noise_primitives.GaussMarkov's own b_0 ~ N(0, sigma_b**2)
    and step noise q = sigma_b*sqrt(1-a**2), a=exp(-dt/tau) -- see that
    class's docstring), so its stationary variance is simply bias_sigma**2,
    unlike CurrentModel's sigma above.
    """
    return _ou_discretize(float(bias_sigma) ** 2, tau, dt)


def accel_bias_decay(sensor_cfg: SensorNoiseConfig, dt: float) -> float:
    """The Gauss-Markov AR(1) decay coefficient a = exp(-dt/accel_bias_tau) --
    the SAME formula sensor_model.noise_primitives.GaussMarkov uses internally
    -- for the ax_bias/ay_bias state rows' predict-step propagation
    (bias_next = a * bias_prev). Sourced from the sensor preset actually
    driving the run (not a separately hand-tuned UKF parameter), so the
    filter's internal bias model can never silently drift out of sync with
    the sensor model generating the real bias it's trying to estimate."""
    return float(np.exp(-dt / sensor_cfg.accel_bias_tau))


def pos_bias_decay(sensor_cfg: SensorNoiseConfig, dt: float) -> float:
    """Same idea as accel_bias_decay(), for the pos_bias_x/pos_bias_y state
    rows -- a = exp(-dt/pos_bias_tau), matching GPS's own Gauss-Markov bias
    process (pos_bias_tau, typically much shorter than accel_bias_tau -- 120s
    vs 300s in the LIGHT preset -- so this is genuinely a different decay
    coefficient, not reusable from accel_bias_decay())."""
    return float(np.exp(-dt / sensor_cfg.pos_bias_tau))


def load_ukf_config(path=None):
    """Reads sim_params.yaml's ukf_node.* fields, returning (UKFConfig,
    r_diag, sensor_preset_name). r_diag falls back to
    default_r_diag(PRESETS[sensor_preset]) when the yaml doesn't set
    ukf_node.r_diag explicitly (the normal case -- see sim_params.yaml's own
    comment: ukf_node.py itself does the same fallback at parameter-declare
    time). The one entry point standalone test_*.py harnesses should use
    instead of hardcoding a copy of alpha/beta/kappa/q_diag/p0_diag, mirroring
    env_model/config.py's load_current_config()/load_wave_config() and
    sensor_model/config.py's load_preset_and_seed()."""
    from sensor_model.config import PRESETS  # local import: avoid a hard
    p = _pkg_paths.load_sim_params(path)["ukf_node"]["ros__parameters"]  # dependency for callers that don't need it
    preset_name = str(p["sensor_preset"])
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown sensor_model preset {preset_name!r}; expected one of {list(PRESETS)}")

    q_diag = list(float(v) for v in p["q_diag"])
    p0_diag = list(float(v) for v in p["p0_diag"])

    # Override the vcx/vcy (indices 6,7) entries with a LIVE derivation from
    # mmg_node's actual current_sigma/current_time_constant, rather than
    # trusting whatever numeric literal happens to be sitting in this yaml's
    # q_diag/p0_diag list for those two slots -- see current_noise_diag()'s
    # docstring for why a stale hand-copied literal here is exactly the bug
    # that made the UKF look unable to track a faster-varying current.
    from env_model.config import load_current_config  # local import: avoid a hard
    current_cfg = load_current_config(path=path)       # dependency for callers that don't need it
    dt = float(p["dt"])
    q_cur, p0_cur = current_noise_diag(current_cfg.sigma, current_cfg.time_constant_s, dt)
    q_diag[6] = q_diag[7] = q_cur
    p0_diag[6] = p0_diag[7] = p0_cur

    # Same override for the ax_bias/ay_bias (8,9) and pos_bias_x/pos_bias_y
    # (10,11) entries, from the ACTIVE sensor preset's own bias_sigma/tau --
    # see bias_noise_diag()'s docstring for why a stale hand-copied literal
    # here is the same bug class, triggered by switching sensor_preset (or
    # retuning a preset's bias_sigma/tau) instead of by changing current's
    # params.
    #
    # q_diag[8:12] is the live physical value TIMES an explicit scale factor
    # (accel_bias_q_scale/pos_bias_q_scale below, default 1.0) rather than
    # the pure physical value alone (unlike current's q_diag[6:7], which has
    # no such knob) -- because tune_ukf.py's own NEES search treats these two
    # groups as tunable (see _GROUPS/_expand_q there) and its result has to
    # persist as something that survives this function re-deriving the
    # physical baseline on every load. Before 2026-08-25 the search's
    # group_scale was baked directly into the yaml's absolute q_diag[8:12]
    # literal; now that those slots are always live-overridden, the scale
    # itself is the only thing left for tune_ukf.py to report/persist. Set to
    # 1.0 for "trust the physical model as-is" (this project's own measured
    # result: doing so left ax_bias/ay_bias RMSE completely unchanged from
    # the previous NEES-tuned absolute values, i.e. Q sizing was never their
    # bottleneck) or to whatever a fresh tune_ukf.py run converges to.
    sensor_cfg = PRESETS[preset_name]
    q_ax, p0_ax = bias_noise_diag(sensor_cfg.accel_bias_sigma, sensor_cfg.accel_bias_tau, dt)
    q_pos, p0_pos = bias_noise_diag(sensor_cfg.pos_bias_sigma, sensor_cfg.pos_bias_tau, dt)
    q_diag[8] = q_diag[9] = q_ax * float(p.get("accel_bias_q_scale", 1.0))
    p0_diag[8] = p0_diag[9] = p0_ax
    q_diag[10] = q_diag[11] = q_pos * float(p.get("pos_bias_q_scale", 1.0))
    p0_diag[10] = p0_diag[11] = p0_pos

    config = UKFConfig(
        alpha=float(p["alpha"]), beta=float(p["beta"]), kappa=float(p["kappa"]),
        q_diag=tuple(q_diag),
        p0_diag=tuple(p0_diag),
        adapt_r=bool(p.get("adapt_r", False)),
        adapt_q=bool(p.get("adapt_q", False)),
        forgetting_factor=float(p.get("forgetting_factor", 0.98)),
    )
    r_diag = (np.array([float(v) for v in p["r_diag"]], dtype=float) if "r_diag" in p
              else default_r_diag(PRESETS[preset_name]))
    return config, r_diag, preset_name


def default_r_diag(sensor_cfg: SensorNoiseConfig) -> np.ndarray:
    """[x, y, psi, r, ax, ay] measurement-noise diagonal, sized from a
    SensorNoiseConfig's white-noise sigmas only.

    Documented limitation: psi/r still ignore their own Gauss-Markov
    bias-instability terms (psi_bias_sigma, r_bias_sigma) -- the 12-dim UKF
    state augments accel bias and position (GPS) bias, not psi/r bias, so
    residual sensor bias in those two channels is still absorbed into the
    filter's state estimate rather than tracked explicitly. Empirically this
    residual is small: isolating each untracked bias source in turn showed
    pos_bias_sigma alone accounted for ~4.6x more of the residual
    current-estimate error than psi_bias_sigma or r_bias_sigma combined
    (disabling pos_bias alone dropped vcx bias from 0.071 to 0.0155 m/s;
    disabling psi/r bias individually changed nothing measurable) -- which is
    why pos bias, not psi/r bias, was the one worth augmenting as a state.

    Both accel and position are like this for the same underlying reason:
    each is the *only* channel a specific hidden state is observable
    through -- accel for current (a mismatch between predicted and measured
    acceleration can only be "explained" by wrong current or wrong bias), and
    position for velocity/current together (a drifting GPS bias looks
    identical, through the kinematics, to the vessel's actual velocity being
    slightly wrong). Widening R to account for either bias (rather than
    tracking it as a state) was tried first for accel and barely helped -- R
    only discounts trust in each individual sample, it can't represent that
    the error is correlated across time, so a persistent bias still gets
    absorbed at a slightly slower rate, not stopped. Both are explicit UKF
    states instead (see accel_bias_decay()/pos_bias_decay() and STATE_DIM
    above), so R's x/y and ax/ay rows stay white-noise-only -- both biases
    are subtracted out via the state estimate, not discounted via R.
    """
    c = sensor_cfg
    return np.array([
        c.pos_white_sigma ** 2,
        c.pos_white_sigma ** 2,
        c.psi_white_sigma ** 2,
        c.r_white_sigma ** 2,
        c.accel_white_sigma ** 2,
        c.accel_white_sigma ** 2,
    ], dtype=float)
