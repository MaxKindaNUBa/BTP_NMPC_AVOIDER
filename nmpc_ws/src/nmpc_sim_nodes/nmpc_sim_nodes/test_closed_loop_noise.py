"""Standalone headless run of the full closed-loop pipeline (acados NMPC +
MMG plant integrator, driven exactly like mmg_node/nmpc_node do at runtime)
on the package's scenario.json, twice: once with sensor noise disabled, once
with sensor_node's light noise preset. Saves one final plot comparing the two
true (executed) paths against the scenario's waypoints/obstacles.

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

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from casadi_mmg_solver.casadi_mmg import make_casadi_integrator  # noqa: E402
from nmpc.params import DEFAULT_CONFIG  # noqa: E402
from nmpc.nmpc_acados import AcadosNMPC  # noqa: E402
from nmpc.path_following import compute_path_angle, select_active_waypoint  # noqa: E402
from sensor_model.config import load_preset_and_seed  # noqa: E402
from sensor_model.sensor_model import SensorModel  # noqa: E402

RESULTS_DIR = os.path.expanduser("~/nmpc_sim_logs/test_closed_loop_noise_results")


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


def run_closed_loop(nmpc: AcadosNMPC, scenario: dict, sensor_model) -> dict:
    """Runs one full closed-loop rollout: mmg_node's true-state integration +
    nmpc_node's solve, with sensor_model standing in for sensor_node (or None
    for the no-noise run, matching sensor_node's enabled=False pass-through).
    Stops on GOAL_REACHED (map_node's 'endpoint' SIM_TIME_MODE) or after the
    scenario's own 'sim_time' seconds, whichever comes first."""
    cfg = nmpc.config
    dt = cfg.dt
    plant_step = make_casadi_integrator(dt, method="rk4", sym_type=ca.SX)

    mmg_state = np.array(scenario["mmg_init"], dtype=float)
    delta, n = float(cfg.DELTA_TRIM), float(cfg.N_TRIM)

    waypoints = [tuple(wp) for wp in scenario["waypoints"]]
    obstacles = [tuple(o) for o in scenario.get("obstacles", [])]
    last_idx = len(waypoints) - 1
    target_idx = 1
    wp_radius = cfg.WP_RADIUS
    max_steps = int(float(scenario.get("sim_time", cfg.SIM_TIME_FIXED)) / dt)

    xs, ys, ts = [], [], []
    status = "MAX_STEPS_REACHED"
    t = 0.0

    for _ in range(max_steps):
        prev_wp, target_wp = waypoints[target_idx - 1], waypoints[target_idx]
        chi_p = compute_path_angle(prev_wp, target_wp)
        x_d, y_d = target_wp

        if sensor_model is not None:
            meas_state, meas_delta, meas_n = sensor_model.measure(list(mmg_state), delta, n)
        else:
            meas_state, meas_delta, meas_n = list(mmg_state), delta, n

        result = nmpc.solve(meas_state, meas_delta, meas_n, chi_p, x_d, y_d, obstacles=obstacles)
        delta, n = result["delta"], result["n"]

        xs.append(float(mmg_state[3]))
        ys.append(float(mmg_state[4]))
        ts.append(t)

        next_state, _ = plant_step(ca.DM(mmg_state), ca.DM([delta, n]))
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

    return {"x": np.array(xs), "y": np.array(ys), "t": np.array(ts), "status": status, "final_t": t}


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
             label=f"light noise ({log_noisy['status']}, {log_noisy['final_t']:.1f}s)")

    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title("Closed-loop path followed: no noise vs light sensor noise")
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

    print(f"--- Run 2/2: sensor noise ({preset_name} preset) ---")
    t0 = time.perf_counter()
    log_noisy = run_closed_loop(nmpc, scenario, sensor_model=sensor_model)
    wall_noisy = time.perf_counter() - t0
    print(f"  status={log_noisy['status']}, sim_time={log_noisy['final_t']:.1f}s, "
          f"wall_time={wall_noisy:.1f}s, speedup={log_noisy['final_t'] / wall_noisy:.1f}x")

    out_path = os.path.join(args.results_dir, "path_comparison.png")
    plot_comparison(scenario, log_clean, log_noisy, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
