"""Standalone headless run of the full closed-loop pipeline (acados NMPC + MMG
plant integrator, driven exactly like mmg_node/nmpc_node do at runtime) on the
package's scenario.json, four times: no disturbance, wave only, current only,
and current+wave -- using env_model's default CurrentModel/WaveModel driving
the plant. Saves one final plot comparing all four true (executed) paths
against the scenario's waypoints/obstacles. Directly analogous to
test_closed_loop_noise.py -- same structure, same conventions -- but for
WAVE_CURRENT_DISTURBANCE_PLAN.md instead of SENSOR_NOISE_MODEL.md.

No rclpy, no ROS graph, no env_node -- this bypasses env_node exactly like
test_closed_loop_noise.py bypasses sensor_node, calling CurrentModel/WaveModel
directly inside the loop, one .step(dt)/.force(psi) call per iteration, right
next to plant_step(...). No wall-clock throttling anywhere in this loop --
it runs every tick back-to-back as fast as the CPU allows, reported at the end
as an "effective speedup" versus real time, same as test_closed_loop_noise.py.

The current-only and wave-only runs each reuse the exact same seed (and
therefore the exact same random realization) as the current+wave run's
matching component, so the four paths isolate each disturbance's marginal
effect rather than differing because of unrelated RNG draws.

Run: ros2 run nmpc_sim_nodes test_closed_loop_env
     ros2 run nmpc_sim_nodes test_closed_loop_env --seed 7
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

from casadi_mmg_solver.casadi_mmg import make_casadi_integrator  # noqa: E402
from nmpc.params import DEFAULT_CONFIG  # noqa: E402
from nmpc.nmpc_acados import AcadosNMPC  # noqa: E402
from nmpc.path_following import compute_path_angle, select_active_waypoint  # noqa: E402
from env_model.config import load_current_config, load_wave_config  # noqa: E402
from env_model.current_model import CurrentModel  # noqa: E402
from env_model.wave_model import WaveModel  # noqa: E402

RESULTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "test_closed_loop_env_results")

# (label, use_current, use_wave, plot color) -- run order and plot styling.
_SCENARIOS = [
    ("no disturbance", False, False, "k"),
    ("wave only", False, True, "tab:green"),
    ("current only", True, False, "tab:purple"),
    ("current+wave", True, True, "tab:blue"),
]


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
    """Clears the SQP-RTI warm-start iterate so the next run doesn't inherit
    the previous run's internal solver state -- reuses the one compiled
    solver instead of paying acados' code-gen/compile cost four times."""
    nmpc.solver.reset()
    nmpc._last_delta = nmpc.config.DELTA_TRIM
    nmpc._last_n = nmpc.config.N_TRIM


def _make_current_model(seed) -> CurrentModel:
    """seed=None uses sim_params.yaml's own current_seed -- mean/Tc/sigma
    always come from the yaml, never a copy kept in this file."""
    return CurrentModel(load_current_config(seed=seed))


def _make_wave_model(seed, dt: float) -> WaveModel:
    """seed=None uses sim_params.yaml's own wave_seed -- hs/tp/gamma/... always
    come from the yaml, never a copy kept in this file."""
    return WaveModel(load_wave_config(seed=seed), dt)


def run_closed_loop(nmpc: AcadosNMPC, scenario: dict, current_model, wave_model) -> dict:
    """Runs one full closed-loop rollout: mmg_node's true-state integration +
    nmpc_node's solve, with current_model/wave_model standing in for env_node
    (either may be None, matching env_node's own enabled=False -> zero
    contribution philosophy). Stops on GOAL_REACHED (map_node's 'endpoint'
    SIM_TIME_MODE) or after the scenario's own 'sim_time' seconds, whichever
    comes first."""
    cfg = nmpc.config
    dt = cfg.dt
    with_env = current_model is not None or wave_model is not None
    plant_step = make_casadi_integrator(dt, method='rk4', sym_type=ca.SX, with_env=with_env)

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

        result = nmpc.solve(list(mmg_state), delta, n, chi_p, x_d, y_d, obstacles=obstacles)
        delta, n = result["delta"], result["n"]

        xs.append(float(mmg_state[3]))
        ys.append(float(mmg_state[4]))
        ts.append(t)

        if with_env:
            current = current_model.step(dt) if current_model is not None else (0.0, 0.0)
            wave_force = wave_model.force(float(mmg_state[5])) if wave_model is not None else (0.0, 0.0, 0.0)
            next_state, _ = plant_step(ca.DM(mmg_state), ca.DM([delta, n]), ca.DM(list(current)), ca.DM(list(wave_force)))
        else:
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


def plot_comparison(scenario: dict, logs: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 9))

    waypoints = scenario["waypoints"]
    wp_x = [w[0] for w in waypoints]
    wp_y = [w[1] for w in waypoints]
    ax.plot(wp_y, wp_x, "g--", marker="x", markersize=8, linewidth=1, label="scenario waypoints")

    for ox, oy, orad in scenario.get("obstacles", []):
        ax.add_patch(plt.Circle((oy, ox), orad, color="tab:orange", alpha=0.3))

    for label, _, _, color in _SCENARIOS:
        log = logs[label]
        ax.plot(log["y"], log["x"], color=color, alpha=0.85, linewidth=1.4,
                 label=f"{label} ({log['status']}, {log['final_t']:.1f}s)")

    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title("Closed-loop path followed: no disturbance vs wave-only vs current-only vs current+wave")
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
                         help="override CurrentModel RNG seed (WaveModel uses seed+1); "
                              "default: use sim_params.yaml's own current_seed/wave_seed")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="where to write the comparison plot")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)
    scenario = _load_scenario(args.scenario_path)

    print("--- Building acados solver (code-gen on first run)... ---")
    nmpc = _build_solver()

    logs = {}
    for i, (label, use_current, use_wave, _color) in enumerate(_SCENARIOS, start=1):
        if i > 1:
            _reset_solver(nmpc)

        wave_seed = (args.seed + 1) if args.seed is not None else None
        current_model = _make_current_model(args.seed) if use_current else None
        wave_model = _make_wave_model(wave_seed, DEFAULT_CONFIG.dt) if use_wave else None

        print(f"--- Run {i}/{len(_SCENARIOS)}: {label} ---")
        t0 = time.perf_counter()
        log = run_closed_loop(nmpc, scenario, current_model=current_model, wave_model=wave_model)
        wall = time.perf_counter() - t0
        print(f"  status={log['status']}, sim_time={log['final_t']:.1f}s, "
              f"wall_time={wall:.1f}s, speedup={log['final_t'] / wall:.1f}x")
        logs[label] = log

    out_path = os.path.join(args.results_dir, "path_comparison.png")
    plot_comparison(scenario, logs, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
