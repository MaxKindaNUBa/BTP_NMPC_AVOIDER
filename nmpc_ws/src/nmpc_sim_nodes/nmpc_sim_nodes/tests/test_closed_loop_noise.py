"""Standalone headless run of the full closed-loop pipeline (acados NMPC +
MMG plant integrator, driven exactly like mmg_node/nmpc_node do at runtime)
on the package's scenario.json, twice: once with sensor noise disabled (true
state straight into NMPC), once with sensor_model's noise (whichever preset
sim_params.yaml's mmg_node.sensor_preset actually configures) run through a
UnscentedKalmanFilter built from sim_params.yaml's own tuned ukf_node.*
config (ukf.config.load_ukf_config() -- the same entry point test_ukf.py/
tune_ukf.py/ukf_node.py itself all use, not a bare UKFConfig() default) --
since the sensor no longer measures u,v directly, reconstructing them from
the noisy GPS/gyro/IMU stream is exactly what ukf_node exists for; feeding
NMPC a raw noisy measurement directly, with no estimator in between, is the
"Known gap" this repo's UKF work fixed, so that path no longer exists here.
Saves one final plot comparing the two true (executed) paths against the
scenario's waypoints/obstacles.

No rclpy, no ROS graph, no rviz, no live matplotlib window -- this only
imports the plant integrator, the acados NMPC solver, and sensor_model
directly, and only ever writes a saved PNG. There is no real-time wall-clock
throttling anywhere in this loop (unlike a live run, where mmg_node's
create_timer(dt, ...) paces ticks to real time) -- it simply runs every tick
back-to-back as fast as the CPU allows, which is reported at the end as an
"effective speedup" versus real time (typically well above the "5x" a live
run would need rviz/matplotlib overhead to fall behind).

Run: ros2 run nmpc_sim_nodes test_closed_loop_noise
     ros2 run nmpc_sim_nodes test_closed_loop_noise --seed 7
"""
import argparse
import dataclasses
import json
import os
import time

import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from casadi_mmg_solver.casadi_mmg import make_casadi_accel_function, make_casadi_integrator  # noqa: E402
from nmpc.params import DEFAULT_CONFIG  # noqa: E402
from nmpc.nmpc_acados import AcadosNMPC  # noqa: E402
from nmpc.path_following import compute_path_angle, select_active_waypoint  # noqa: E402
from sensor_model.config import load_preset_and_seed  # noqa: E402
from sensor_model.sensor_model import SensorModel  # noqa: E402
from ukf.config import accel_bias_decay, load_ukf_config, pos_bias_decay  # noqa: E402
from ukf.ukf_core import UnscentedKalmanFilter  # noqa: E402

from . import test_ukf as tu  # noqa: E402 -- reuse its env-model builder, not a re-derived copy

RESULTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "test_closed_loop_noise_results")


def _load_scenario(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _build_solver() -> AcadosNMPC:
    """Same ACADOS_CODE_EXPORT_DIR/ACADOS_JSON_FILE override nmpc_node.py applies
    (section 2.6 packaging gotcha: these must be absolute, not cwd-relative, or
    first-solve code-gen fails to find/write its output)."""
    cfg = dataclasses.replace(
        DEFAULT_CONFIG,
        ACADOS_CODE_EXPORT_DIR=_pkg_paths.ACADOS_CODE_EXPORT_DIR_DEFAULT,
        ACADOS_JSON_FILE=_pkg_paths.ACADOS_JSON_FILE_DEFAULT,
    )
    return AcadosNMPC(cfg)


def _reset_solver(nmpc: AcadosNMPC):
    """Clears the SQP-RTI warm-start iterate so the second run doesn't inherit
    the first run's internal solver state -- reuses the one compiled solver
    instead of paying acados' code-gen/compile cost twice."""
    nmpc.solver.reset()
    nmpc._last_delta = nmpc.config.DELTA_TRIM
    nmpc._last_n = nmpc.config.N_TRIM


def _make_ukf(dt: float, preset):
    """Builds the UKF from sim_params.yaml's own ukf_node.* config (alpha/
    beta/kappa/q_diag/r_diag/p0_diag) via load_ukf_config() -- previously a
    bare UKFConfig() default, which silently diverged from the tuned config
    (e.g. q_diag[ax_bias]/[ay_bias] stuck at a much-larger placeholder value
    while r_diag/the bias-decay coefficients below already resolved the
    preset actually configured -- an internal inconsistency within this one
    function). Returns (ukf, ukf_preset_name) -- the caller checks
    ukf_preset_name against mmg_node's own preset_name, same as test_ukf.py."""
    predict_integrator = make_casadi_integrator(dt, method="rk4", sym_type=ca.SX, with_env=True)
    accel_fn = make_casadi_accel_function(sym_type=ca.SX)
    ukf_config, r_diag, ukf_preset_name = load_ukf_config()
    ax_bias_decay = accel_bias_decay(preset, dt)
    gps_bias_decay = pos_bias_decay(preset, dt)
    ukf = UnscentedKalmanFilter(ukf_config, predict_integrator, accel_fn, r_diag,
                                 ax_bias_decay, gps_bias_decay)
    return ukf, ukf_preset_name


def run_closed_loop(nmpc: AcadosNMPC, scenario: dict, sensor_model, ukf: UnscentedKalmanFilter = None) -> dict:
    """Runs one full closed-loop rollout: mmg_node's true-state integration +
    nmpc_node's solve. sensor_model=None (ukf ignored) is the no-noise run --
    true state straight into NMPC, matching sensor_enabled=False. Otherwise
    sensor_model's noisy [x,y,psi,r,ax,ay] measurement is filtered through
    `ukf` (required together with sensor_model -- the sensor no longer
    measures u,v directly, so there is no other way to get NMPC a state to
    solve against) and the UKF's estimate is what NMPC sees, matching
    use_ukf=True. Stops on GOAL_REACHED (map_node's 'endpoint' SIM_TIME_MODE)
    or after the scenario's own 'sim_time' seconds, whichever comes first.

    CORRECTED 2026-08-25: the true plant integration used to ALWAYS run with
    with_env defaulted off (no current, no wave), regardless of
    sim_params.yaml's mmg_node.current_enabled/wave_enabled -- silently
    untrue to a live bringup.launch.py run (where they're both on), and unable
    to reproduce/diagnose current-driven x/y drift reports at all. Now built
    with with_env=True and stepped through current_model/wave_model exactly
    like test_ukf.py's run() does (same _make_env_models() helper, reused --
    not a re-derived copy), for both the clean and noisy rollouts, so
    log_clean is now a true no-sensor-noise/no-estimator baseline under the
    SAME environmental disturbance the noisy run sees, not a disturbance-free
    one -- a fairer comparison, and the only way this harness can actually
    exercise the thing being investigated."""
    cfg = nmpc.config
    dt = cfg.dt
    plant_step = make_casadi_integrator(dt, method="rk4", sym_type=ca.SX, with_env=True)
    accel_fn = make_casadi_accel_function(sym_type=ca.SX) if sensor_model is not None else None
    mmg_params = _pkg_paths.load_sim_params()["mmg_node"]["ros__parameters"]
    current_model, wave_model = tu._make_env_models(dt, mmg_params)

    mmg_state = np.array(scenario["mmg_init"], dtype=float)
    delta, n = float(cfg.DELTA_TRIM), float(cfg.N_TRIM)

    if sensor_model is not None:
        assert ukf is not None, "sensor_model requires a ukf to reconstruct u,v from its noisy measurement"
        vcx0, vcy0 = tu._current_mean(current_model is not None)
        ukf.reset(np.array([*mmg_state, vcx0, vcy0, 0.0, 0.0, 0.0, 0.0], dtype=float))

    waypoints = [tuple(wp) for wp in scenario["waypoints"]]
    obstacles = [tuple(o) for o in scenario.get("obstacles", [])]
    last_idx = len(waypoints) - 1
    target_idx = 1
    wp_radius = cfg.WP_RADIUS
    max_steps = int(float(scenario.get("sim_time", cfg.SIM_TIME_FIXED)) / dt)

    xs, ys, ts = [], [], []
    x_hats, y_hats = [], []  # UKF-estimated position, for the noisy run only -- see plot_comparison()
    status = "MAX_STEPS_REACHED"
    t = 0.0

    for _ in range(max_steps):
        prev_wp, target_wp = waypoints[target_idx - 1], waypoints[target_idx]
        chi_p = compute_path_angle(prev_wp, target_wp)
        x_d, y_d = target_wp

        current = current_model.step(dt) if current_model is not None else (0.0, 0.0)
        wave_force = wave_model.force(float(mmg_state[5])) if wave_model is not None else (0.0, 0.0, 0.0)
        current_dm, wave_dm = ca.DM(list(current)), ca.DM(list(wave_force))

        if sensor_model is not None:
            true_accel = np.array(accel_fn(ca.DM(mmg_state), ca.DM([delta, n]), current_dm, wave_dm)).flatten()
            r_m, x_m, y_m, psi_m, ax_m, ay_m, delta_m, n_m = sensor_model.measure(
                list(mmg_state), delta, n, accel_true=(true_accel[0], true_accel[1]))
            x_hat, _, _, _ = ukf.step([delta_m, n_m], [x_m, y_m, psi_m, r_m, ax_m, ay_m])
            meas_state, meas_delta, meas_n = x_hat[:6], delta_m, n_m
            x_hats.append(float(x_hat[3]))
            y_hats.append(float(x_hat[4]))
            # matches mmg_node.py's request.current = ukf_response.estimated_current
            # (use_ukf=True path) -- the UKF's OWN current estimate, not the true
            # value, since that's exactly what the live system feeds the solver.
            nmpc_current = (float(x_hat[6]), float(x_hat[7]))
        else:
            meas_state, meas_delta, meas_n = list(mmg_state), delta, n
            # no-noise/no-UKF run: matches mmg_node.py's (not self.use_ukf) path,
            # which falls back to current_msg -- i.e. the actual /env/current_state
            # reading (the true current) when there's no estimator in the loop.
            nmpc_current = (float(current[0]), float(current[1]))

        result = nmpc.solve(meas_state, meas_delta, meas_n, chi_p, x_d, y_d,
                             obstacles=obstacles, current=nmpc_current)
        delta, n = result["delta"], result["n"]

        xs.append(float(mmg_state[3]))
        ys.append(float(mmg_state[4]))
        ts.append(t)

        next_state, _ = plant_step(ca.DM(mmg_state), ca.DM([delta, n]), current_dm, wave_dm)
        mmg_state = np.array(next_state).flatten()
        t += dt

        target_idx = select_active_waypoint(mmg_state[3], mmg_state[4], waypoints, target_idx, wp_radius)
        if target_idx == last_idx:
            goal = waypoints[last_idx]
            if np.hypot(mmg_state[3] - goal[0], mmg_state[4] - goal[1]) < wp_radius:
                status = "GOAL_REACHED"
                xs.append(float(mmg_state[3]))
                ys.append(float(mmg_state[4]))
                ts.append(t)
                break

    return {"x": np.array(xs), "y": np.array(ys), "t": np.array(ts), "status": status, "final_t": t,
            "x_hat": np.array(x_hats), "y_hat": np.array(y_hats)}


def plot_comparison(scenario: dict, log_clean: dict, log_noisy: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 9))

    waypoints = scenario["waypoints"]
    wp_x = [w[0] for w in waypoints]
    wp_y = [w[1] for w in waypoints]
    ax.plot(wp_y, wp_x, "g--", marker="x", markersize=8, linewidth=1, label="scenario waypoints")

    for ox, oy, orad in scenario.get("obstacles", []):
        ax.add_patch(plt.Circle((oy, ox), orad, color="tab:orange", alpha=0.3))

    ax.plot(log_clean["y"], log_clean["x"], "k-", linewidth=1.6,
             label=f"no noise ({log_clean['status']}, {log_clean['final_t']:.1f}s)")
    ax.plot(log_noisy["y"], log_noisy["x"], color="tab:red", alpha=0.8, linewidth=1.2,
             label=f"true position, noisy+UKF run ({log_noisy['status']}, {log_noisy['final_t']:.1f}s)")
    ax.plot(log_noisy["y_hat"], log_noisy["x_hat"], color="tab:blue", alpha=0.8, linewidth=1.1,
             label="UKF-ESTIMATED position (what NMPC actually steers on)")

    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title("Closed-loop path followed: no noise vs light sensor noise (UKF-filtered)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_scenario = os.path.join(get_package_share_directory("nmpc_sim_nodes"), "params", "scenario.json")
    parser.add_argument("--scenario-path", default=default_scenario, help="scenario.json to run (default: the package's own)")
    parser.add_argument("--seed", type=int, default=None,
                         help="override sensor_model RNG seed for the noisy run; "
                              "default: use sim_params.yaml's own sensor_seed")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="where to write the comparison plot")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)
    scenario = _load_scenario(args.scenario_path)

    print(f"--- Building acados solver (code-gen on first run)... ---")
    nmpc = _build_solver()

    print("--- Run 1/2: no noise ---")
    t0 = time.perf_counter()
    log_clean = run_closed_loop(nmpc, scenario, sensor_model=None)
    wall_clean = time.perf_counter() - t0
    print(f"  status={log_clean['status']}, sim_time={log_clean['final_t']:.1f}s, "
          f"wall_time={wall_clean:.1f}s, speedup={log_clean['final_t'] / wall_clean:.1f}x")

    _reset_solver(nmpc)
    preset_name, preset, sensor_seed = load_preset_and_seed(seed=args.seed)
    sensor_model = SensorModel(preset, DEFAULT_CONFIG.dt, sensor_seed)
    ukf, ukf_preset_name = _make_ukf(DEFAULT_CONFIG.dt, preset)
    if ukf_preset_name != preset_name:
        print(f"WARNING: mmg_node.sensor_preset={preset_name!r} != ukf_node.sensor_preset={ukf_preset_name!r} "
              f"in sim_params.yaml -- r_diag won't match the sensor model actually driving this run.")

    print(f"--- Run 2/2: sensor noise ({preset_name} preset), UKF-filtered ---")
    t0 = time.perf_counter()
    log_noisy = run_closed_loop(nmpc, scenario, sensor_model=sensor_model, ukf=ukf)
    wall_noisy = time.perf_counter() - t0
    print(f"  status={log_noisy['status']}, sim_time={log_noisy['final_t']:.1f}s, "
          f"wall_time={wall_noisy:.1f}s, speedup={log_noisy['final_t'] / wall_noisy:.1f}x")

    # True vs UKF-estimated position, under REAL closed-loop conditions (this
    # project's own scenario/obstacles/current/wave/NMPC feedback) -- unlike
    # test_ukf.py's canned maneuver, this is what NMPC actually saw and steered
    # on, so it's the most direct read on whatever x/y drift a live
    # bringup.launch.py run would show.
    n = min(len(log_noisy["x"]), len(log_noisy["x_hat"]))
    x_err = log_noisy["x_hat"][:n] - log_noisy["x"][:n]
    y_err = log_noisy["y_hat"][:n] - log_noisy["y"][:n]
    final_drift = float(np.hypot(x_err[-1], y_err[-1])) if n > 0 else float("nan")
    print(f"\n--- true vs UKF-estimated position (closed-loop, what NMPC steered on) ---")
    print(f"  x RMSE: {np.sqrt(np.mean(x_err ** 2)):.3f} m   y RMSE: {np.sqrt(np.mean(y_err ** 2)):.3f} m")
    print(f"  final-step drift: {final_drift:.3f} m")

    out_path = os.path.join(args.results_dir, "path_comparison.png")
    plot_comparison(scenario, log_clean, log_noisy, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
