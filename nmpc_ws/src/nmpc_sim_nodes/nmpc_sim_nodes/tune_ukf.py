"""tune_ukf.py: AUTOMATED Q/R tuning for ukf.ukf_core.UnscentedKalmanFilter via
NEES (Normalized Estimation Error Squared) consistency -- not the manual
"disable a noise source, eyeball the plot, repeat" process used earlier in
this project's development. This script drives a numerical optimizer; no
human looks at a plot inside the tuning loop.

--- The idea ---
NEES = (x_true - x_hat)^T @ inv(P) @ (x_true - x_hat)

If Q/R/P0 correctly describe the filter's actual uncertainty, NEES follows a
chi-squared distribution with STATE_DIM degrees of freedom (12 here) -- its
expected value is exactly STATE_DIM, REGARDLESS of how large or small the
estimation error itself is. That's what makes it useful for tuning:
  - mean(NEES) >> STATE_DIM: the filter is OVERCONFIDENT -- P claims more
    certainty than it has earned. This is the "locks onto a wrong value and
    can't correct" failure mode chased manually earlier in this project.
    Fix: increase Q and/or R for the offending states.
  - mean(NEES) << STATE_DIM: the filter is UNDERCONFIDENT -- sluggish, slow
    to converge even with good data. Fix: decrease Q and/or R.
Since this is simulation, ground truth is available every step, so NEES (not
the weaker NIS, which only uses the innovation) is the right tool.

--- The search ---
Rather than search all 12 q_diag + 6 r_diag entries independently (18
dimensions, expensive and poorly conditioned), this searches ONE scale
factor per physically-meaningful GROUP of states (_GROUPS below) plus one
overall scale for R -- 7 free parameters total, each multiplying the
existing sim_params.yaml values (which are already physically-derived
starting points for the current/bias states, not guesses -- see
ukf/config.py). scipy.optimize.minimize (Nelder-Mead, gradient-free) drives
the search, minimizing (mean_NEES - STATE_DIM)**2 averaged across several
TRAINING seeds (never just one -- seed-to-seed variance is exactly what made
manual tuning unreliable earlier in this project's history). The result is
then checked against separate VALIDATION seeds never seen by the optimizer,
to catch overfitting to a specific noise realization.

Uses the exact same turning-circle maneuver and yaml-driven current/wave models as
test_ukf.py (imported from there, not redefined) -- same sim_time/dt/sensor
config resolution, so a tuned config found here is valid for that harness
(and for ukf_node.py/mmg_node.py, which read the same sim_params.yaml).

Run: ros2 run nmpc_sim_nodes tune_ukf
     ros2 run nmpc_sim_nodes tune_ukf --train-seeds 1,2,3 --val-seeds 233,42 --sim-time 90
"""
import argparse
import dataclasses
import os

import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from casadi_mmg_solver.casadi_mmg import make_casadi_accel_function, make_casadi_integrator  # noqa: E402
from sensor_model.config import load_preset_and_seed  # noqa: E402
from sensor_model.sensor_model import SensorModel  # noqa: E402
from ukf.config import accel_bias_decay, load_ukf_config, pos_bias_decay  # noqa: E402
from ukf.ukf_core import IDX_PSI_STATE, STATE_DIM, UnscentedKalmanFilter  # noqa: E402

from . import test_ukf as tu  # noqa: E402 -- reuse the turning-circle schedule + env-model builder, not a re-derived copy

RESULTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "tune_ukf_results")

# Group q_diag's 12 indices by physical meaning (matches ukf/ukf_core.py's
# state ordering: [u,v,r,x,y,psi,vcx,vcy,ax_bias,ay_bias,pos_bias_x,pos_bias_y]).
#
# "current" (6,7) is DELIBERATELY NOT included -- this project's own priority
# is accurate vcx/vcy specifically (other states may trade off worse), and
# NEES-consistency is the wrong objective for that: it only measures whether
# P matches actual error, which the optimizer can satisfy just as well by
# inflating Q (P grows to match a bad estimate) as by improving the estimate
# itself. That's exactly what happened here before: leaving "current" in this
# search previously converged it to ~7147x its physically-correct value
# (q_diag[vcx]=current_sigma**2*dt, see ukf/config.py) -- NEES-legal, but it
# made vcx/vcy track noise instead of the true, near-constant current. With
# "current" excluded, q_diag[6]/[7] simply pass through from base_q
# unscaled -- i.e. stay pinned at whatever load_ukf_config() resolves (the
# physically-derived value, per sim_params.yaml/ukf/config.py's own
# comments) -- regardless of what this optimizer finds for everything else.
_GROUPS = {
    "uv": (0, 1),
    "psi_r": (2, 5),
    "xy": (3, 4),
    "accel_bias": (8, 9),
    "pos_bias": (10, 11),
}
_GROUP_NAMES = list(_GROUPS.keys())
N_PARAMS = len(_GROUP_NAMES) + 1  # + 1 overall R scale

POST_TRANSIENT_T = 30.0  # [s] discard the initial low-observability ramp phase, same convention as test_ukf.py


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _expand_q(base_q: np.ndarray, group_scales: np.ndarray) -> np.ndarray:
    q = np.array(base_q, dtype=float).copy()
    for name, scale in zip(_GROUP_NAMES, group_scales):
        for idx in _GROUPS[name]:
            q[idx] = base_q[idx] * scale
    return q


def run_rollout(sim_time: float, seed: int, q_diag: np.ndarray, r_diag: np.ndarray,
                 ukf_config_base, dt: float, preset) -> np.ndarray:
    """One closed-loop rollout (same turning-circle maneuver + yaml-driven current/
    wave models as test_ukf.py's run()), returning the per-step NEES history
    -- the true state (with bias components read from the sensor model's own
    internal Gauss-Markov state, same convention test_ukf.py uses) vs the
    filter's own (x_hat, P)."""
    sensor_model = SensorModel(preset, dt, seed)
    mmg_params = _pkg_paths.load_sim_params()["mmg_node"]["ros__parameters"]
    current_model, wave_model = tu._make_env_models(dt, mmg_params)

    plant_step = make_casadi_integrator(dt, method="rk4", sym_type=ca.SX, with_env=True)
    accel_fn = make_casadi_accel_function(sym_type=ca.SX)
    ax_decay = accel_bias_decay(preset, dt)
    gps_decay = pos_bias_decay(preset, dt)

    cfg = dataclasses.replace(ukf_config_base, q_diag=tuple(q_diag))
    ukf = UnscentedKalmanFilter(cfg, plant_step, accel_fn, r_diag, ax_decay, gps_decay)

    # Pre-settled to the true n=N_TRIM/delta=0 trim equilibrium -- see
    # test_ukf.py's _settle_to_trim()/run() for the full rationale (a
    # guessed/mismatched trim IC forces an avoidable transient into the
    # single most current-unobservable window of the run, which this
    # optimizer would otherwise train against every rollout).
    current_enabled = bool(mmg_params["current_enabled"])
    u0, v0, r0 = tu._settle_to_trim(plant_step, dt, tu.N_TRIM, current_enabled)
    mmg_state = np.array([u0, v0, r0, 0.0, 0.0, 0.0])

    vcx0, vcy0 = tu._current_mean(current_enabled)  # see its docstring -- required for p0_diag[vcx]/[vcy] to help, not hurt
    ukf.reset(np.array([*mmg_state, vcx0, vcy0, 0.0, 0.0, 0.0, 0.0]))

    n_steps = int(sim_time / dt)
    nees_hist = np.full(n_steps, np.nan)

    for step in range(n_steps):
        t = step * dt
        delta, n = tu._turning_circle_schedule(t)

        current = current_model.step(dt) if current_model is not None else (0.0, 0.0)
        wave_force = wave_model.force(float(mmg_state[5])) if wave_model is not None else (0.0, 0.0, 0.0)
        current_dm, wave_dm = ca.DM(list(current)), ca.DM(list(wave_force))

        true_accel = np.array(accel_fn(ca.DM(mmg_state), ca.DM([delta, n]), current_dm, wave_dm)).flatten()
        r_m, x_m, y_m, psi_m, ax_m, ay_m, delta_m, n_m = sensor_model.measure(
            mmg_state, delta, n, accel_true=(true_accel[0], true_accel[1]))

        x_hat, P, ok, status = ukf.step([delta_m, n_m], [x_m, y_m, psi_m, r_m, ax_m, ay_m])

        if ok:
            x_true = np.array([
                mmg_state[0], mmg_state[1], mmg_state[2], mmg_state[3], mmg_state[4], mmg_state[5],
                current[0], current[1],
                sensor_model._bias_ax.value, sensor_model._bias_ay.value,
                sensor_model._bias_x.value, sensor_model._bias_y.value,
            ])
            err = x_true - x_hat
            err[IDX_PSI_STATE] = _wrap(err[IDX_PSI_STATE])
            try:
                nees_hist[step] = err @ np.linalg.solve(P, err)
            except np.linalg.LinAlgError:
                pass  # leave as nan -- excluded from the mean below

        next_state, _ = plant_step(ca.DM(mmg_state), ca.DM([delta, n]), current_dm, wave_dm)
        mmg_state = np.array(next_state).flatten()

    return nees_hist


def _mean_post_transient_nees(nees_hist: np.ndarray, dt: float) -> float:
    post = nees_hist[int(POST_TRANSIENT_T / dt):]
    post = post[np.isfinite(post)]
    return float(post.mean()) if len(post) > 0 else float("nan")


def objective(log_params: np.ndarray, base_q: np.ndarray, base_r: np.ndarray,
              ukf_config_base, dt: float, preset, sim_time: float, seeds) -> float:
    scales = np.exp(log_params)
    q = _expand_q(base_q, scales[:-1])
    r = np.asarray(base_r, dtype=float) * scales[-1]

    nees_means = []
    for seed in seeds:
        nees_hist = run_rollout(sim_time, seed, q, r, ukf_config_base, dt, preset)
        m = _mean_post_transient_nees(nees_hist, dt)
        if not np.isfinite(m):
            return 1.0e6  # filter failed outright for this candidate -- reject it hard
        nees_means.append(m)

    mean_nees = float(np.mean(nees_means))
    loss = (mean_nees - STATE_DIM) ** 2
    scale_str = " ".join(f"{n}={s:.3f}" for n, s in zip(_GROUP_NAMES, scales[:-1]))
    print(f"  {scale_str} r={scales[-1]:.3f}  -> mean_NEES={mean_nees:.2f} (target {STATE_DIM})  loss={loss:.3f}")
    return loss


def _initial_simplex(x0: np.ndarray, step: float = 0.7) -> np.ndarray:
    """scipy's own default initial-simplex construction uses a FIXED tiny
    step (0.00025) for any x0 component that's exactly 0 -- since x0 here is
    all zeros (log-scale 0 == scale 1.0, the natural starting point), that
    default makes the initial simplex degenerately small, and Nelder-Mead
    reports "converged" after a single iteration having barely moved (this
    is exactly what happened on the first run of this script -- nit=1, loss
    barely changed from the unscaled starting point). Building the simplex
    explicitly with a real per-dimension step (0.7 in log-space =~ 2x scale)
    fixes it."""
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        v = x0.copy()
        v[i] += step
        simplex.append(v)
    return np.array(simplex)


def tune(train_seeds, sim_time: float, dt: float, preset, ukf_config_base, base_r, maxiter: int):
    base_q = np.array(ukf_config_base.q_diag, dtype=float)
    x0 = np.zeros(N_PARAMS)  # log-scales: start at scale=1.0 everywhere (today's sim_params.yaml values)

    print(f"--- Tuning against {len(train_seeds)} training seeds: {train_seeds}, sim_time={sim_time}s ---")
    result = minimize(
        objective, x0, args=(base_q, base_r, ukf_config_base, dt, preset, sim_time, train_seeds),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 0.05, "fatol": 0.5, "adaptive": True,
                 "initial_simplex": _initial_simplex(x0)},
    )
    scales = np.exp(result.x)
    tuned_q = _expand_q(base_q, scales[:-1])
    tuned_r = np.asarray(base_r, dtype=float) * scales[-1]
    print(f"\n--- Converged (nit={result.nit}, final loss={result.fun:.3f}) ---")
    for name, s in zip(_GROUP_NAMES, scales[:-1]):
        print(f"  {name:>12} scale = {s:.4f}")
    print(f"  {'R (overall)':>12} scale = {scales[-1]:.4f}")
    return tuned_q, tuned_r, scales


def validate(label, q_diag, r_diag, val_seeds, sim_time, dt, preset, ukf_config_base):
    print(f"\n--- Validating [{label}] on held-out seeds {val_seeds} (never seen by the optimizer) ---")
    nees_means = []
    for seed in val_seeds:
        nees_hist = run_rollout(sim_time, seed, q_diag, r_diag, ukf_config_base, dt, preset)
        m = _mean_post_transient_nees(nees_hist, dt)
        nees_means.append(m)
        print(f"  seed={seed:<6} mean_NEES={m:.2f}")
    print(f"  -> average mean_NEES across held-out seeds: {np.mean(nees_means):.2f}  (target {STATE_DIM})")
    return nees_means


def plot_nees_comparison(baseline_hist, tuned_hist, dt: float, seed: int, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(baseline_hist)) * dt
    ax.plot(t, baseline_hist, color="tab:red", alpha=0.7, linewidth=0.9, label="baseline (sim_params.yaml as-is)")
    ax.plot(t, tuned_hist, color="tab:blue", alpha=0.8, linewidth=0.9, label="tuned")
    ax.axhline(STATE_DIM, color="k", linestyle="--", linewidth=1.2, label=f"target (STATE_DIM={STATE_DIM})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.set_title(f"NEES consistency: baseline vs tuned (seed={seed}, held-out)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seeds", default="1,2,3", help="comma-separated seeds the optimizer sees (default: 1,2,3)")
    parser.add_argument("--val-seeds", default="233,42", help="comma-separated held-out seeds for validation (default: 233,42)")
    parser.add_argument("--sim-time", type=float, default=90.0,
                         help="rollout duration per seed in seconds (default: 90 -- shorter than test_ukf.py's "
                              "default since this runs many rollouts; override for a more thorough search)")
    parser.add_argument("--maxiter", type=int, default=80, help="Nelder-Mead iteration cap (default: 80)")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="where to write the NEES comparison plot")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)
    train_seeds = [int(s) for s in args.train_seeds.split(",")]
    val_seeds = [int(s) for s in args.val_seeds.split(",")]

    mmg_params = _pkg_paths.load_sim_params()["mmg_node"]["ros__parameters"]
    dt = float(mmg_params["dt"])
    preset_name, preset, _ = load_preset_and_seed(seed=train_seeds[0])
    ukf_config_base, base_r, ukf_preset_name = load_ukf_config()
    if ukf_preset_name != preset_name:
        print(f"WARNING: mmg_node.sensor_preset={preset_name!r} != ukf_node.sensor_preset={ukf_preset_name!r}")

    base_q = np.array(ukf_config_base.q_diag, dtype=float)

    validate("baseline (today's sim_params.yaml)", base_q, base_r, val_seeds, args.sim_time, dt, preset, ukf_config_base)

    tuned_q, tuned_r, scales = tune(train_seeds, args.sim_time, dt, preset, ukf_config_base, base_r, args.maxiter)

    validate("tuned", tuned_q, tuned_r, val_seeds, args.sim_time, dt, preset, ukf_config_base)

    # one before/after NEES-over-time plot, on the first held-out seed, for a visual sanity check
    baseline_hist = run_rollout(args.sim_time, val_seeds[0], base_q, base_r, ukf_config_base, dt, preset)
    tuned_hist = run_rollout(args.sim_time, val_seeds[0], tuned_q, tuned_r, ukf_config_base, dt, preset)
    plot_path = os.path.join(args.results_dir, "nees_comparison.png")
    plot_nees_comparison(baseline_hist, tuned_hist, dt, val_seeds[0], plot_path)
    print(f"\n  wrote {plot_path}")

    print("\n--- Suggested sim_params.yaml ukf_node.ros__parameters values ---")
    print("    # only indices 0-5 (uv/psi_r/xy) matter from this array -- 6:12 are always")
    print("    # LIVE-overridden by load_ukf_config()/ukf_node.py, see ukf/config.py")
    print(f"    q_diag: [{', '.join(f'{v:.4e}' for v in tuned_q)}]")
    print(f"    r_diag: [{', '.join(f'{v:.4e}' for v in tuned_r)}]")
    accel_bias_scale = float(scales[_GROUP_NAMES.index("accel_bias")])
    pos_bias_scale = float(scales[_GROUP_NAMES.index("pos_bias")])
    print("    # accel_bias/pos_bias q_diag[8:12] is ALWAYS the live physical value times")
    print("    # these two scales now -- set THESE, not a q_diag entry, to apply this tuning:")
    print(f"    accel_bias_q_scale: {accel_bias_scale:.4f}")
    print(f"    pos_bias_q_scale: {pos_bias_scale:.4f}")
    print("\n(Not written automatically -- review the validation NEES above, then apply by hand if it looks right.)")


if __name__ == "__main__":
    main()
