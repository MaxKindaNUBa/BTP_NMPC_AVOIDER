"""Standalone comparison harness for ukf.ukf_core.UnscentedKalmanFilter,
driven entirely by the project's own sim_params.yaml -- not a hand-picked
scenario. mmg_node's dt/sensor_preset/sensor_seed and current_enabled/
wave_enabled (+ their configs) are read the exact same way mmg_node itself
does (env_model.config.load_current_config/load_wave_config,
sensor_model.config.load_preset_and_seed), and ukf_node's alpha/beta/kappa/
q_diag/p0_diag/r_diag are read the same way ukf_node.py does
(ukf.config.load_ukf_config) -- so this harness exercises exactly what a live
`ros2 launch bringup.launch.py` run would, with no separate copy of any
tunable to drift out of sync.

Control input is a canned turning-circle maneuver defined in this file
(pre-settle open-loop to the true n=N_TRIM/delta=0 STRAIGHT trim equilibrium
-- see _settle_to_trim() -- then hold a constant +TURN_DELTA_DEG rudder for
the whole run, sweeping heading through repeated full 360-degree circles
instead of a zig-zag's narrow oscillating band -- see TURN_DELTA_DEG's own
comment for why: only the current-vector component that rotates into
alignment with the vessel's high-SNR surge axis is observed at any given
heading, so a maneuver that visits every heading gives much more even
observability of both current components than one confined to a band around
the nominal course) -- no scenario.json,
no NMPC, no ROS graph. Each tick: sample current/wave (if enabled) exactly
like mmg_node._tick does, synthesize the true IMU accelerometer reading, run
it through SensorModel.measure(), then through a bare UnscentedKalmanFilter
instance, and integrate the true plant one step forward.

The pre-settle exists because seeding the true plant at a hand-guessed "trim"
speed while commanding a DIFFERENT n (e.g. this file's old approach: start at
a hardcoded u=0.78 while ramping n from 0) forces a real, avoidable
acceleration transient into the plant during the single most
current-unobservable window of the whole run: straight-line, zero rudder,
where u and vcx are only separable through their DIFFERENCE (relative/water
velocity, what the accelerometer actually senses), not individually -- so
that transient's correction gets split ~arbitrarily between u and vcx and,
because nothing rotates the confusion axis while going straight, accumulates
into a slow-decaying vcx lock-in bias rather than averaging out (confirmed
empirically: RMSE of (u-vcx) came out ~5x tighter than either state alone
during such a window). Settling the true plant to its OWN exact equilibrium
first (rather than guessing it, or starting from rest -- tried, and worse:
0->trim from rest is a bigger, slower transient than even the original
mismatched guess) removes that self-inflicted worst case instead of just
relocating it.

Same harness pattern as test_sensor_model.py/test_env_model.py: a plain
in-memory log dict, a fixed (non-timestamped) results directory that gets
overwritten each run, a CSV via the stdlib csv module, and matplotlib PNGs --
no ExperimentLogger, no per-run timestamped subdirectory.

Writes one PNG per UKF state EXCEPT x/y (u, v, r, psi, vcx, vcy, ax_bias,
ay_bias, pos_bias_x, pos_bias_y) plus a single combined state_xy.png plan-view
trajectory plot (true/sensor/UKF position overlaid in the X-Y plane, not
separate x(t)/y(t) time series -- much more directly readable for a
turning-circle maneuver, see plot_xy_trajectory()), plus a signals CSV, under
<repo_root>/nmpc_sim_logs/test_ukf_results/ -- each per-state plot overlays
true state, sensor output (where the sensor actually measures that quantity:
x, y, psi, r -- u, v, vcx, vcy, ax_bias, ay_bias, pos_bias_x, pos_bias_y have
no direct sensor equivalent, so those plots show true vs UKF estimate only;
the bias states' "true" lines are read back from the sensor model's own
internal Gauss-Markov bias state, to verify the filter's bias estimates
actually track the real drifting sensor biases), and the UKF's estimate.
Angle states (psi) are unwrapped for plotting (see plot_state()) so a
multi-revolution turning-circle maneuver doesn't look like a broken estimate
just because the raw true/estimate traces wrap at different points.

Run: ros2 run nmpc_sim_nodes test_ukf
     ros2 run nmpc_sim_nodes test_ukf --seed 7
"""
import argparse
import csv
import math
import os

import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from casadi_mmg_solver.casadi_mmg import make_casadi_accel_function, make_casadi_integrator  # noqa: E402
from env_model.config import load_current_config, load_wave_config  # noqa: E402
from env_model.current_model import CurrentModel  # noqa: E402
from env_model.wave_model import WaveModel  # noqa: E402
from sensor_model.config import load_preset_and_seed  # noqa: E402
from sensor_model.sensor_model import SensorModel  # noqa: E402
from ukf.config import accel_bias_decay, load_ukf_config, pos_bias_decay  # noqa: E402
from ukf.ukf_core import UnscentedKalmanFilter  # noqa: E402

RESULTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "test_ukf_results")

# --- canned turning-circle maneuver, saved here (no scenario.json) ---
# REPLACED 2026-08-25 (was a zig-zag): a zig-zag confined to a narrow heading
# band around the nominal course only ever rotates the current vector through
# that same narrow band -- current (vcx,vcy) only enters the dynamics via
# uc=vcx*cos(psi)+vcy*sin(psi), vc=-vcx*sin(psi)+vcy*cos(psi) (see
# casadi_mmg.py), and only the component landing in ur (riding on the
# vessel's own large, high-SNR surge u) is well observed -- whatever's
# perpendicular to the zig-zag's heading band stays stuck in the much
# lower-SNR sway/vr channel the entire run. A constant-rudder turning circle
# sweeps psi through the full 360 degrees, so EVERY direction of the true
# current vector rotates into the high-SNR surge channel at some point during
# the turn, instead of one direction being permanently favored -- see project
# discussion for the full reasoning (this also matches an observed pattern:
# vcy RMSE was consistently worse than vcx RMSE under the old zig-zag, in a
# project where current_mean_heading sits closer to the zig-zag's own heading
# band than perpendicular to it).
TURN_DELTA_DEG = 30.0   # [deg] constant rudder angle for the whole run
N_TRIM = 15.0           # [rps]

# [s] a full 360-degree circle at TURN_DELTA_DEG/N_TRIM measured (directly
# simulating the plant) at ~103s -- SIM_TIME_DEFAULT below covers ~3 of them,
# so the maneuver isn't a lucky one-off partial sweep.
_MEASURED_CIRCLE_PERIOD_S = 103.0

# [s] open-loop pre-settle duration for _settle_to_trim() -- NOT part of the
# logged run (see run()). n=N_TRIM/delta=0's true equilibrium converges slowly
# (measured: still ~4% off at 100s, ~0.02% off by 300s), so this is generous
# on purpose -- it's cheap (a single trajectory, no sigma points) relative to
# the logged run's own cost.
TRIM_SETTLE_TIME = 400.0

SIM_TIME_DEFAULT = 3.0 * _MEASURED_CIRCLE_PERIOD_S

# (key, axis label, is_angle, has_sensor) -- has_sensor: whether
# SensorModel.measure() provides a direct measurement of this exact state
# (x, y, psi, r do; u, v, vcx, vcy, ax_bias, ay_bias, pos_bias_x, pos_bias_y
# don't -- see module docstring). ax_bias/ay_bias/pos_bias_x/pos_bias_y's
# "true" values are read back from the sensor model's own internal
# Gauss-Markov state (not something mmg_state carries), specifically to
# verify the bias state augmentation actually tracks the real drifting
# biases instead of misattributing them as current.
_STATES = [
    ("u", "Surge u (m/s)", False, False),
    ("v", "Sway v (m/s)", False, False),
    ("r", "Yaw rate (deg/s)", True, True),
    ("x", "X pos (m)", False, True),
    ("y", "Y pos (m)", False, True),
    ("psi", "Heading (deg)", True, True),
    ("vcx", "Current vcx (m/s)", False, False),
    ("vcy", "Current vcy (m/s)", False, False),
    ("ax_bias", "Accel bias ax (m/s^2)", False, False),
    ("ay_bias", "Accel bias ay (m/s^2)", False, False),
    ("pos_bias_x", "GPS bias x (m)", False, False),
    ("pos_bias_y", "GPS bias y (m)", False, False),
]


def _turning_circle_schedule(t: float):
    """n=N_TRIM from t=0 (the plant is already pre-settled to the STRAIGHT
    trim by _settle_to_trim() before t=0 -- see run()), constant
    +TURN_DELTA_DEG rudder held for the entire run -- a short turn-entry
    transient, then a steady circle repeating for the rest of sim_time.
    Returns (delta_rad, n_rps)."""
    return np.deg2rad(TURN_DELTA_DEG), N_TRIM


def _make_env_models(dt: float, mmg_params: dict):
    """Builds CurrentModel/WaveModel from sim_params.yaml's mmg_node.* fields
    exactly like mmg_node._init_env_models does -- None when disabled."""
    current_model = CurrentModel(load_current_config()) if bool(mmg_params["current_enabled"]) else None
    wave_model = WaveModel(load_wave_config(), dt) if bool(mmg_params["wave_enabled"]) else None
    return current_model, wave_model


def _current_mean(current_enabled: bool):
    """(mean_vx, mean_vy) from CurrentConfig, or (0, 0) when current is
    disabled -- matches what the main loop itself feeds the plant in that
    case. Shared by _settle_to_trim (the fixed current to settle the plant
    under) and run()'s/run_rollout()'s own UKF seed: seeding vcx/vcy at this
    mean, not 0, is what makes ukf/config.py's tightened p0_diag[vcx]/[vcy]
    (see its own comment) a net win instead of a regression -- a tight prior
    centered on the wrong value (0) just makes the filter slow to unlearn a
    bad guess; centered on this mean, it lets the filter trust that it's
    already close to right from t=0 and use its tiny process noise to hold
    that trust rather than spending the whole run re-discovering it."""
    if not current_enabled:
        return 0.0, 0.0
    cc = load_current_config()
    return cc.mean_vx, cc.mean_vy


def _settle_to_trim(plant_step, dt: float, n_trim: float, current_enabled: bool,
                     settle_time: float = TRIM_SETTLE_TIME):
    """Pre-converges the TRUE plant (open-loop, delta=0, n=n_trim, starting
    from rest) to its own exact equilibrium (u, v, r) before the logged run
    starts -- see this file's module docstring for why a guessed/mismatched
    trim IC corrupts the UKF's current estimate. Settles under a FIXED
    current at _current_mean() -- not the live, stochastic CurrentModel --
    since a short settle sees essentially no OU drift from its own
    start-at-mean anyway, and stepping the live model here would burn RNG
    draws off the run's own reproducible current trajectory before logging
    even begins. No wave force (this project's own MMG model has no
    closed-form trim under an oscillating wave force to settle to;
    wave_enabled defaults off -- see sim_params.yaml).

    Returns (u, v, r); the caller keeps x, y, psi at 0 -- this settle isn't
    logged, so the run itself starts at the origin, exactly straight."""
    current_dm = ca.DM(list(_current_mean(current_enabled)))
    zero3 = ca.DM([0.0, 0.0, 0.0])
    control_dm = ca.DM([0.0, n_trim])

    state = np.zeros(6)
    for _ in range(int(settle_time / dt)):
        next_state, _ = plant_step(ca.DM(state), control_dm, current_dm, zero3)
        state = np.array(next_state).flatten()
    return float(state[0]), float(state[1]), float(state[2])


def run(sim_time: float, seed) -> dict:
    mmg_params = _pkg_paths.load_sim_params()["mmg_node"]["ros__parameters"]
    dt = float(mmg_params["dt"])

    preset_name, preset, sensor_seed = load_preset_and_seed(seed=seed)
    sensor_model = SensorModel(preset, dt, sensor_seed)
    current_model, wave_model = _make_env_models(dt, mmg_params)

    ukf_config, r_diag, ukf_preset_name = load_ukf_config()
    if ukf_preset_name != preset_name:
        print(f"WARNING: mmg_node.sensor_preset={preset_name!r} != ukf_node.sensor_preset={ukf_preset_name!r} "
              f"in sim_params.yaml -- r_diag won't match the sensor model actually driving this run.")

    plant_step = make_casadi_integrator(dt, method="rk4", sym_type=ca.SX, with_env=True)
    accel_fn = make_casadi_accel_function(sym_type=ca.SX)
    ax_bias_decay = accel_bias_decay(preset, dt)
    gps_bias_decay = pos_bias_decay(preset, dt)
    ukf = UnscentedKalmanFilter(ukf_config, plant_step, accel_fn, r_diag, ax_bias_decay, gps_bias_decay)

    current_enabled = bool(mmg_params["current_enabled"])
    u0, v0, r0 = _settle_to_trim(plant_step, dt, N_TRIM, current_enabled)
    mmg_state = np.array([u0, v0, r0, 0.0, 0.0, 0.0])  # u, v, r, x, y, psi

    vcx0, vcy0 = _current_mean(current_enabled)  # see _current_mean()'s docstring: matches ukf_node.py's own seed
    ukf.reset(np.array([*mmg_state, vcx0, vcy0, 0.0, 0.0, 0.0, 0.0]))

    n_steps = int(sim_time / dt)
    log = {k: np.zeros(n_steps) for k in ("t", "success")}
    # Constant-at-seed "null" baseline for print_summary()'s current comparison --
    # see its own comment for why this matters more than it sounds.
    log["vcx_null"] = np.full(n_steps, vcx0)
    log["vcy_null"] = np.full(n_steps, vcy0)
    for key, _, _, has_sensor in _STATES:
        log[f"{key}_t"] = np.zeros(n_steps)
        log[f"{key}_e"] = np.zeros(n_steps)
        if has_sensor:
            log[f"{key}_s"] = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt
        delta, n = _turning_circle_schedule(t)

        current = current_model.step(dt) if current_model is not None else (0.0, 0.0)
        wave_force = wave_model.force(float(mmg_state[5])) if wave_model is not None else (0.0, 0.0, 0.0)
        current_dm, wave_dm = ca.DM(list(current)), ca.DM(list(wave_force))

        true_accel = np.array(accel_fn(ca.DM(mmg_state), ca.DM([delta, n]), current_dm, wave_dm)).flatten()
        r_m, x_m, y_m, psi_m, ax_m, ay_m, delta_m, n_m = sensor_model.measure(
            mmg_state, delta, n, accel_true=(true_accel[0], true_accel[1]))

        x_hat, _, success, _ = ukf.step([delta_m, n_m], [x_m, y_m, psi_m, r_m, ax_m, ay_m])

        true_full = {"u": mmg_state[0], "v": mmg_state[1], "r": mmg_state[2],
                     "x": mmg_state[3], "y": mmg_state[4], "psi": mmg_state[5],
                     "vcx": current[0], "vcy": current[1],
                     "ax_bias": sensor_model._bias_ax.value, "ay_bias": sensor_model._bias_ay.value,
                     "pos_bias_x": sensor_model._bias_x.value, "pos_bias_y": sensor_model._bias_y.value}
        est_full = {"u": x_hat[0], "v": x_hat[1], "r": x_hat[2],
                    "x": x_hat[3], "y": x_hat[4], "psi": x_hat[5],
                    "vcx": x_hat[6], "vcy": x_hat[7],
                    "ax_bias": x_hat[8], "ay_bias": x_hat[9],
                    "pos_bias_x": x_hat[10], "pos_bias_y": x_hat[11]}
        sensor_full = {"r": r_m, "x": x_m, "y": y_m, "psi": psi_m}

        log["t"][step] = t
        log["success"][step] = success
        for key, _, _, has_sensor in _STATES:
            log[f"{key}_t"][step] = true_full[key]
            log[f"{key}_e"][step] = est_full[key]
            if has_sensor:
                log[f"{key}_s"][step] = sensor_full[key]

        next_state, _ = plant_step(ca.DM(mmg_state), ca.DM([delta, n]), current_dm, wave_dm)
        mmg_state = np.array(next_state).flatten()

    return log


def save_csv(log: dict, out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"]
        for key, _, _, has_sensor in _STATES:
            header.append(f"{key}_true")
            if has_sensor:
                header.append(f"{key}_sensor")
            header.append(f"{key}_ukf")
        writer.writerow(header)
        for i in range(len(log["t"])):
            row = [log["t"][i]]
            for key, _, _, has_sensor in _STATES:
                row.append(log[f"{key}_t"][i])
                if has_sensor:
                    row.append(log[f"{key}_s"][i])
                row.append(log[f"{key}_e"][i])
            writer.writerow(row)


def _angle_error(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def print_summary(log: dict):
    print("\n--- UKF estimate - true (per-signal error stats) ---")
    print(f"{'signal':>8} {'rms_err':>14} {'bias':>14}")
    for key, _, is_angle, _ in _STATES:
        true_v, est_v = log[f"{key}_t"], log[f"{key}_e"]
        err = _angle_error(est_v, true_v) if is_angle else (est_v - true_v)
        print(f"{key:>8} {np.sqrt(np.mean(err ** 2)):14.5f} {np.mean(err):14.5f}")
    n_fail = int(np.sum(log["success"] == 0))
    print(f"filter update failures: {n_fail}/{len(log['success'])}")

    # A low current RMSE alone doesn't prove the filter is sensing anything --
    # a "null" estimator that just holds its seed forever, never correcting,
    # would score well too whenever the true current stays close to that seed
    # (e.g. a long current_time_constant relative to the run length). This
    # compares the real UKF against exactly that trivial baseline, so a
    # near-flat true current can't be mistaken for evidence of good tracking.
    print("\n--- current vs a \"never updates from its seed\" null baseline ---")
    for key in ("vcx", "vcy"):
        true_v, est_v, null_v = log[f"{key}_t"], log[f"{key}_e"], log[f"{key}_null"]
        ukf_rmse = float(np.sqrt(np.mean((est_v - true_v) ** 2)))
        null_rmse = float(np.sqrt(np.mean((null_v - true_v) ** 2)))
        pct = (null_rmse - ukf_rmse) / null_rmse * 100.0 if null_rmse > 0 else float("nan")
        verdict = "better" if pct >= 0 else "WORSE"
        print(f"  {key:>4}  ukf_rmse={ukf_rmse:.5f}  null_rmse={null_rmse:.5f}  "
              f"(ukf is {abs(pct):.1f}% {verdict} than just guessing the seed forever)")


def plot_state(log: dict, key: str, label: str, is_angle: bool, has_sensor: bool, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    scale = math.degrees(1.0) if is_angle else 1.0

    true_v_raw = log[f"{key}_t"]
    est_v_raw = log[f"{key}_e"]
    if is_angle:
        # print_summary()'s error stats already go through _angle_error (a
        # wrapped difference), so they were never affected by this -- but
        # plotting the raw values directly was not: psi accumulates
        # continuously in the true plant state (integrator never wraps it,
        # see casadi_mmg.py's kinematics), and the turning-circle maneuver
        # (unlike the old zig-zag, confined to +-20deg) now sweeps through
        # multiple full 360deg revolutions, while the UKF's OWN psi state is
        # wrapped to +-pi every step (ukf_core.py's _wrap()) -- so the raw
        # true trace climbs past 360/720/1080deg while the raw estimate
        # trace sawtooths between +-180, making a genuinely accurate
        # estimate look completely broken. Fix: unwrap the true trace to a
        # continuous reference, then reconstruct the estimate/sensor traces
        # as that same reference plus their small WRAPPED error from it (the
        # same _angle_error() print_summary() already trusts), so all three
        # curves stay continuous and directly comparable regardless of how
        # many revolutions the maneuver makes.
        true_unwrapped = np.unwrap(true_v_raw)
        true_v = true_unwrapped * scale
        est_v = (true_unwrapped + _angle_error(est_v_raw, true_v_raw)) * scale
    else:
        true_v = true_v_raw * scale
        est_v = est_v_raw * scale
    ax.plot(log["t"], true_v, "k-", linewidth=1.4, label="true state")
    if has_sensor:
        sensor_v_raw = log[f"{key}_s"]
        sensor_v = (true_unwrapped + _angle_error(sensor_v_raw, true_v_raw)) * scale if is_angle \
            else sensor_v_raw * scale
        ax.plot(log["t"], sensor_v, color="tab:red", alpha=0.6, linewidth=0.8, label="sensor output")
    ax.plot(log["t"], est_v, color="tab:blue", alpha=0.9, linewidth=1.1, label="UKF prediction")

    # Without this, matplotlib's default offset notation prints tick labels
    # relative to a hidden baseline (e.g. "0.0002" on an axis actually
    # spanning 0.9998-1.0009, with a tiny, easy-to-miss "+1" printed near the
    # top of the axis) whenever values cluster tightly around a round number
    # -- exactly what a large current_mean_speed does to vcx/vcy. Disabling
    # it prints the real absolute values instead.
    ax.ticklabel_format(useOffset=False, axis="y")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(label)
    ax.set_title(f"{label}: true vs {'sensor vs ' if has_sensor else ''}UKF prediction")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_xy_trajectory(log: dict, out_path: str):
    """Single X-Y plan-view plot (true/sensor/UKF position overlaid as 3
    curves in the horizontal plane) replacing the separate state_x.png/
    state_y.png time-series -- for a turning-circle maneuver, x(t) and y(t)
    each individually just oscillate/ramp with the turn and are hard to
    visually judge; the actual quantity of interest (does the UKF's
    estimated position trace the same circles as the true position) is much
    more directly readable as a spatial trajectory plot."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(log["x_t"], log["y_t"], "k-", linewidth=1.4, label="true position")
    ax.plot(log["x_s"], log["y_s"], color="tab:red", alpha=0.5, linewidth=0.6, label="sensor (GPS) position")
    ax.plot(log["x_e"], log["y_e"], color="tab:blue", alpha=0.9, linewidth=1.1, label="UKF prediction")
    ax.scatter([log["x_t"][0]], [log["y_t"][0]], color="green", marker="P", s=80, zorder=5, label="start")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Trajectory: true vs sensor vs UKF prediction")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-time", type=float, default=SIM_TIME_DEFAULT,
                         help=f"maneuver duration in seconds (default: {SIM_TIME_DEFAULT:.0f}, "
                              "covering ~3 full turning circles)")
    parser.add_argument("--seed", type=int, default=None,
                         help="override sensor_model RNG seed; default: use sim_params.yaml's own sensor_seed")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="where to write plots/CSV")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)

    mmg_params = _pkg_paths.load_sim_params()["mmg_node"]["ros__parameters"]
    print(f"--- Running UKF test: sim_time={args.sim_time:.0f}s, dt={mmg_params['dt']}s, "
          f"sensor_preset={mmg_params['sensor_preset']!r}, current_enabled={mmg_params['current_enabled']}, "
          f"wave_enabled={mmg_params['wave_enabled']} (all from sim_params.yaml) ---")
    log = run(args.sim_time, args.seed)

    csv_path = os.path.join(args.results_dir, "signals.csv")
    save_csv(log, csv_path)
    print(f"  wrote {csv_path}")

    xy_path = os.path.join(args.results_dir, "state_xy.png")
    plot_xy_trajectory(log, xy_path)
    print(f"  wrote {xy_path}")

    for key, label, is_angle, has_sensor in _STATES:
        if key in ("x", "y"):  # combined into the single xy trajectory plot above instead
            continue
        out_path = os.path.join(args.results_dir, f"state_{key}.png")
        plot_state(log, key, label, is_angle, has_sensor, out_path)
        print(f"  wrote {out_path}")

    print_summary(log)
    print(f"\nAll output under: {args.results_dir}")


if __name__ == "__main__":
    main()
