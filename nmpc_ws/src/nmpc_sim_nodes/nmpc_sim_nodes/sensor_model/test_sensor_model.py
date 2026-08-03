"""Standalone comparison harness for sensor_model (SENSOR_NOISE_MODEL.md).

Drives the real MMG plant integrator (casadi_mmg_solver, the same one
mmg_node uses) through a fixed maneuver to get a "true" trajectory, then
passes every sample through SensorModel.measure() to get the "measured"
(noisy) trajectory -- exactly what sensor_node would produce with
enabled=True. "Without noise" is just the true trajectory itself, since
that's bit-identical to what sensor_node returns with enabled=False
(sensor_node.py's pass-through path).

No rclpy, no other node needs to be running -- this only imports the plant
integrator and the sensor_model package directly.

Run: ros2 run nmpc_sim_nodes test_sensor_model
     ros2 run nmpc_sim_nodes test_sensor_model --sim-time 60 --seed 7
"""
import argparse
import csv
import os

import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from casadi_mmg_solver.casadi_mmg import make_casadi_integrator  # noqa: E402
from sensor_model.config import PRESETS  # noqa: E402
from sensor_model.sensor_model import SensorModel  # noqa: E402

RESULTS_DIR = os.path.expanduser("~/nmpc_sim_logs/test_sensor_model_results")

# (key, axis label, is_angle) -- angle signals are stored/plotted in degrees
# and get wrap-aware error computation.
_SIGNALS = [
    ("x", "X pos (m)", False),
    ("y", "Y pos (m)", False),
    ("psi", "Heading (deg)", True),
    ("r", "Yaw rate (deg/s)", True),
    ("u", "Surge u (m/s)", False),
    ("v", "Sway v (m/s)", False),
    ("delta", "Rudder delta (deg)", True),
    ("n", "Propeller n (rps)", False),
]


def _angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrap-aware a - b, result in (-pi, pi]."""
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _control_schedule(t: float, n_trim: float = 15.0):
    """Canned maneuver: ramp up to trim rps, turn one way, turn the other,
    settle straight. Gives every signal (u, v, r, psi, x, y) some genuine
    excitation instead of a static/constant trajectory."""
    if t < 10.0:
        return 0.0, n_trim * (t / 10.0)
    if t < 40.0:
        return np.deg2rad(20.0), n_trim
    if t < 70.0:
        return np.deg2rad(-20.0), n_trim
    return 0.0, n_trim


def run_true_trajectory(dt: float, sim_time: float) -> dict:
    """Integrates the real MMG plant (RK4, same integrator mmg_node uses)
    through _control_schedule. Returns a dict of numpy arrays, one per
    signal, keyed like _SIGNALS plus 't'."""
    plant_step = make_casadi_integrator(dt, method="rk4", sym_type=ca.SX)
    mmg_state = np.array([0.78, 0.0, 0.0, 0.0, 0.0, 0.0])  # u, v, r, x, y, psi

    n_steps = int(sim_time / dt)
    log = {k: np.zeros(n_steps) for k, _, _ in _SIGNALS}
    log["t"] = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt
        delta, n = _control_schedule(t)

        log["t"][step] = t
        log["u"][step], log["v"][step], log["r"][step] = mmg_state[0], mmg_state[1], mmg_state[2]
        log["x"][step], log["y"][step], log["psi"][step] = mmg_state[3], mmg_state[4], mmg_state[5]
        log["delta"][step], log["n"][step] = delta, n

        next_state, _ = plant_step(ca.DM(mmg_state), ca.DM([delta, n]))
        mmg_state = np.array(next_state).flatten()

    return log


def run_noisy(log_true: dict, preset_name: str, seed: int, dt: float) -> dict:
    """Feeds every true sample through SensorModel.measure() -- the exact
    same call sensor_node makes per /sensor/measure request when enabled."""
    model = SensorModel(PRESETS[preset_name], dt, seed)
    n_steps = len(log_true["t"])
    log = {k: np.zeros(n_steps) for k, _, _ in _SIGNALS}

    for i in range(n_steps):
        mmg_state = [log_true["u"][i], log_true["v"][i], log_true["r"][i],
                     log_true["x"][i], log_true["y"][i], log_true["psi"][i]]
        meas_state, delta_meas, n_meas = model.measure(mmg_state, log_true["delta"][i], log_true["n"][i])
        log["u"][i], log["v"][i], log["r"][i], log["x"][i], log["y"][i], log["psi"][i] = meas_state
        log["delta"][i], log["n"][i] = delta_meas, n_meas

    log["t"] = log_true["t"]
    return log


# ------------------------------------------------------------------ viewing
def print_summary(log_true: dict, log_noisy: dict, preset_name: str):
    print(f"\n--- {preset_name} preset: measured - true (per-signal error stats) ---")
    print(f"{'signal':>8} {'rms_err':>14} {'bias':>14} {'max_abs_err':>14}")
    for key, label, is_angle in _SIGNALS:
        true_v, noisy_v = log_true[key], log_noisy[key]
        err = _angle_error(noisy_v, true_v) if is_angle else (noisy_v - true_v)
        if is_angle:
            err = np.rad2deg(err)
        rms = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        max_abs = float(np.max(np.abs(err)))
        unit = label.split("(")[-1].rstrip(")") if "(" in label else ""
        print(f"{key:>8} {rms:14.5f} {bias:14.5f} {max_abs:14.5f}  [{unit}]")


def save_csv(log_true: dict, log_noisy: dict, out_path: str):
    keys = [k for k, _, _ in _SIGNALS]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + [f"{k}_true" for k in keys] + [f"{k}_meas" for k in keys])
        for i in range(len(log_true["t"])):
            row = [log_true["t"][i]] + [log_true[k][i] for k in keys] + [log_noisy[k][i] for k in keys]
            writer.writerow(row)


def plot_timeseries(log_true: dict, log_noisy: dict, preset_name: str, out_path: str):
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
    for ax, (key, label, is_angle) in zip(axes.flat, _SIGNALS):
        true_v, noisy_v = log_true[key], log_noisy[key]
        if is_angle:
            true_v, noisy_v = np.rad2deg(true_v), np.rad2deg(noisy_v)
        ax.plot(log_true["t"], true_v, "k-", linewidth=1.3, label="true")
        ax.plot(log_true["t"], noisy_v, color="tab:red", alpha=0.75, linewidth=0.8, label="measured")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"sensor_model: true vs measured ({preset_name} preset)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trajectory_xy(log_true: dict, log_noisy: dict, preset_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(log_true["y"], log_true["x"], "k-", linewidth=1.5, label="true trajectory")
    ax.plot(log_noisy["y"], log_noisy["x"], color="tab:red", alpha=0.6, linewidth=0.9, label="measured trajectory")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title(f"Trajectory: true vs measured ({preset_name} preset)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ entry point
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-time", type=float, default=90.0, help="maneuver duration in seconds (default: 90)")
    parser.add_argument("--dt", type=float, default=0.1, help="integration/measurement step (default: 0.1, matches sim_params.yaml)")
    parser.add_argument("--seed", type=int, default=42, help="sensor_model RNG seed (default: 42)")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="where to write plots/CSVs")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"--- Building true trajectory: sim_time={args.sim_time}s, dt={args.dt}s ---")
    log_true = run_true_trajectory(args.dt, args.sim_time)

    preset_name = "light"
    log_noisy = run_noisy(log_true, preset_name, args.seed, args.dt)

    ts_path = os.path.join(args.results_dir, f"{preset_name}_timeseries.png")
    traj_path = os.path.join(args.results_dir, f"{preset_name}_trajectory.png")
    csv_path = os.path.join(args.results_dir, f"{preset_name}_signals.csv")

    plot_timeseries(log_true, log_noisy, preset_name, ts_path)
    plot_trajectory_xy(log_true, log_noisy, preset_name, traj_path)
    save_csv(log_true, log_noisy, csv_path)

    print_summary(log_true, log_noisy, preset_name)
    print(f"  wrote {ts_path}")
    print(f"  wrote {traj_path}")
    print(f"  wrote {csv_path}")

    print(f"\nAll output under: {args.results_dir}")


if __name__ == "__main__":
    main()
