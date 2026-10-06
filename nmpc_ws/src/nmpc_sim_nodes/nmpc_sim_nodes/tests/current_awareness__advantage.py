"""current_awareness__advantage: Test orchestrator and analysis script comparing
Current-Aware NMPC vs. Current-Unaware NMPC (Current assumed 0).

Operates similarly to nmpc_ablation_runs by sequentially configuring sim_params.yaml
and launching `bringup.launch.py` for two benchmark runs:
  1. Current-Aware NMPC   : NMPC prediction model receives estimated/true ocean current.
  2. Current-Unaware NMPC : NMPC prediction model assumes zero current (current=(0,0)),
                            while the physical vessel still experiences real ocean current.

Outputs deliverables:
  - Trajectory comparison map (current_awareness_paths.png)
  - Animated looping GIF with prediction horizons and bow lines (current_awareness_animation.gif)
  - Heading, Course Reference, and Crab Angle time-series plot (heading_and_crab_angle_comparison.png)
  - Terminal performance summary comparison table
  - Offline data archive (current_awareness_results.npz)

Run:
  ros2 run nmpc_sim_nodes current_awareness__advantage
  ros2 run nmpc_sim_nodes current_awareness__advantage --scavenge
"""
import argparse
import csv
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from .. import _pkg_paths
except (ImportError, ValueError):
    _curr = os.path.dirname(os.path.abspath(__file__))
    _pkg_nmpc_sim_nodes = os.path.dirname(_curr)
    if _pkg_nmpc_sim_nodes not in sys.path:
        sys.path.insert(0, _pkg_nmpc_sim_nodes)
    import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.params import DEFAULT_CONFIG, load_nmpc_config

DEFAULT_RESULTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "current_awareness__advantage")
DEFAULT_EXPERIMENTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "experiments")

# 2 Benchmark Configurations for Current-Awareness Advantage Study
AWARENESS_CASES = [
    {
        "key": "current_aware",
        "label": "1. Current-Aware NMPC",
        "color": "#1F77B4",       # Solid Blue
        "linestyle": "-",
        "params": {
            "current_aware": True,
            "current_enabled": True,
        },
        "run_name": "current_aware_on",
    },
    {
        "key": "current_unaware",
        "label": "2. Current-Unaware NMPC (Current=0)",
        "color": "#D62728",       # Crimson Red
        "linestyle": "--",
        "params": {
            "current_aware": False,
            "current_enabled": True,
        },
        "run_name": "current_aware_off",
    },
]


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def update_sim_params(yaml_path: str, params: dict, run_name: Optional[str] = None):
    """Modifies flags and logger_node run_name in sim_params.yaml while preserving comments."""
    with open(yaml_path, "r") as f:
        content = f.read()

    for key, val in params.items():
        val_str = "true" if val else "false"
        pattern = rf"(^\s*{key}:\s*)(true|false)"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, rf"\g<1>{val_str}", content, flags=re.MULTILINE)
        else:
            # If key missing in mmg_node, insert it
            mmg_pat = r"(mmg_node:\s*\n\s*ros__parameters:\s*\n)"
            content = re.sub(mmg_pat, rf"\g<1>    {key}: {val_str}\n", content)

    if run_name:
        pattern = r'(^\s*run_name:\s*)"[^"]*"'
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, rf'\g<1>"{run_name}"', content, flags=re.MULTILINE)

    with open(yaml_path, "w") as f:
        f.write(content)


def restore_sim_params(yaml_path: str, original_content: str):
    """Restores the original sim_params.yaml content."""
    try:
        with open(yaml_path, "w") as f:
            f.write(original_content)
    except Exception as e:
        print(f"[WARN] Failed to restore {yaml_path}: {e}")


def load_scenario(path: str) -> dict:
    if not os.path.exists(path):
        return {"waypoints": [], "obstacles": [], "ellipses": [], "walls": [], "obstacle_ships": []}
    with open(path, "r") as f:
        return json.load(f)


def scavenge_experiment_run(run_dir: str, case_info: dict) -> dict:
    """Extracts trajectory, heading, reference course, crab angle, efforts, and KPI data from run directory."""
    summary_path = os.path.join(run_dir, "summary.json")
    telemetry_path = os.path.join(run_dir, "telemetry.csv")
    horizons_path = os.path.join(run_dir, "prediction_horizons.npz")

    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception as e:
            print(f"[WARN] Error reading {summary_path}: {e}")

    ts, xs, ys, psis, deltas, ns = [], [], [], [], [], []
    us, vs, chi_ps, crab_angles = [], [], [], []
    e_cts, e_psis, solve_times = [], [], []

    if os.path.exists(telemetry_path):
        with open(telemetry_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = float(row["time_s"])
                    x = float(row["true_x_m"])
                    y = float(row["true_y_m"])
                    psi = float(row["true_psi_rad"])
                    delta = float(row["true_delta_rad"])
                    n = float(row["true_n_rps"])
                    tgt_wp_str = row.get("target_wp_idx", "-1")
                    tgt_wp_idx = int(float(tgt_wp_str)) if tgt_wp_str not in ("", None) else -1
                    # Skip uninitialized startup samples before map_node publishes active reference
                    if tgt_wp_idx < 0:
                        continue

                    u = float(row["true_u_m_s"]) if "true_u_m_s" in row else 0.0
                    v = float(row["true_v_m_s"]) if "true_v_m_s" in row else 0.0
                    target_x = float(row["target_x_m"]) if row.get("target_x_m") else 0.0
                    target_y = float(row["target_y_m"]) if row.get("target_y_m") else 0.0
                    chi_p = float(row["chi_p_rad"]) if row.get("chi_p_rad") else 0.0
                    solve_time = float(row["solve_time_ms"]) * 1e-3 if row.get("solve_time_ms") else 0.0

                    # Hydrodynamic sideslip / crab angle beta = atan2(v, u)
                    crab = math.atan2(v, u) if (abs(u) > 1e-4 or abs(v) > 1e-4) else 0.0

                    # Compute tracking errors
                    e_ct = -(x - target_x) * math.sin(chi_p) + (y - target_y) * math.cos(chi_p)
                    e_psi = wrap_to_pi(psi - chi_p)

                    ts.append(t)
                    xs.append(x)
                    ys.append(y)
                    psis.append(psi)
                    deltas.append(delta)
                    ns.append(n)
                    us.append(u)
                    vs.append(v)
                    chi_ps.append(chi_p)
                    crab_angles.append(crab)
                    e_cts.append(e_ct)
                    e_psis.append(e_psi)
                    solve_times.append(solve_time)
                except (ValueError, KeyError):
                    continue

    # Load prediction horizons from prediction_horizons.npz
    horizons_x, horizons_y = [], []
    if os.path.exists(horizons_path):
        try:
            with np.load(horizons_path) as npz:
                if "xi_trajs" in npz:
                    xi = npz["xi_trajs"]  # (N_steps, 11, horizon_len + 1)
                    for k in range(len(xi)):
                        horizons_x.append(np.array(xi[k, 4, :], dtype=float))
                        horizons_y.append(np.array(xi[k, 5, :], dtype=float))
        except Exception as e:
            print(f"[WARN] Could not load prediction horizons from {horizons_path}: {e}")

    # Fallback padding if horizons count differs from telemetry samples
    if len(horizons_x) < len(xs):
        for k in range(len(horizons_x), len(xs)):
            horizons_x.append(np.array([xs[k]]))
            horizons_y.append(np.array([ys[k]]))

    dt = 0.1
    if len(ts) >= 2:
        dt = float(np.mean(np.diff(ts)))

    e_ct_arr = np.array(e_cts) if e_cts else np.array([0.0])
    e_psi_arr = np.array(e_psis) if e_psis else np.array([0.0])
    deltas_arr = np.array(deltas) if deltas else np.array([0.0])
    ns_arr = np.array(ns) if ns else np.array([0.0])

    max_crosstrack = float(np.max(np.abs(e_ct_arr)))
    crosstrack_rmse = float(np.sqrt(np.mean(e_ct_arr ** 2)))
    max_heading_err = float(np.degrees(np.max(np.abs(e_psi_arr))))
    heading_rmse = float(np.degrees(np.sqrt(np.mean(e_psi_arr ** 2))))

    rudder_effort = float(np.sum(deltas_arr ** 2) * dt)
    propeller_effort = float(np.sum(ns_arr ** 2) * dt)
    total_solve_time = float(np.sum(solve_times))

    solver_perf = summary.get("solver_performance", {})
    solver_failures = int(solver_perf.get("total_solves", 0) - solver_perf.get("successful_solves", 0))

    raw_status = summary.get("status", "")
    if raw_status in ("GOAL_REACHED", "COMPLETED"):
        status = "GOAL_REACHED"
    elif raw_status == "TIMEOUT":
        status = "TIMEOUT"
    elif raw_status == "INTERRUPTED":
        status = "TIMEOUT (Max Time)"
    else:
        status = "TIMEOUT" if len(xs) > 1 else "NOT_FOUND"

    return {
        "x": xs,
        "y": ys,
        "psi": psis,
        "t": ts,
        "u": us,
        "v": vs,
        "chi_p": chi_ps,
        "crab_angle": crab_angles,
        "delta": deltas,
        "n": ns,
        "e_ct": e_cts,
        "e_psi": e_psis,
        "horizons_x": horizons_x,
        "horizons_y": horizons_y,
        "max_crosstrack": max_crosstrack,
        "crosstrack_rmse": crosstrack_rmse,
        "max_heading_err": max_heading_err,
        "heading_rmse": heading_rmse,
        "rudder_effort": rudder_effort,
        "propeller_effort": propeller_effort,
        "total_solve_time": total_solve_time,
        "solver_failures": solver_failures,
        "total_steps": len(xs),
        "status": status,
        "final_t": ts[-1] if ts else 0.0,
        "run_dir": run_dir,
    }


def create_empty_result(label: str) -> dict:
    return {
        "x": [0.0],
        "y": [0.0],
        "psi": [0.0],
        "t": [0.0],
        "u": [0.0],
        "v": [0.0],
        "chi_p": [0.0],
        "crab_angle": [0.0],
        "delta": [0.0],
        "n": [0.0],
        "e_ct": [0.0],
        "e_psi": [0.0],
        "horizons_x": [np.array([0.0])],
        "horizons_y": [np.array([0.0])],
        "max_crosstrack": 0.0,
        "crosstrack_rmse": 0.0,
        "max_heading_err": 0.0,
        "heading_rmse": 0.0,
        "rudder_effort": 0.0,
        "propeller_effort": 0.0,
        "total_solve_time": 0.0,
        "solver_failures": 0,
        "total_steps": 0,
        "status": "NOT_FOUND",
        "final_t": 0.0,
        "run_dir": "",
    }


def scavenge_all_existing_runs(cases: list, experiments_dir: str) -> dict:
    """Scavenges the most recent matching experiment directories from experiments_dir."""
    results = {}
    if not os.path.isdir(experiments_dir):
        print(f"[WARN] Experiments directory does not exist: {experiments_dir}")
        for case in cases:
            results[case["key"]] = create_empty_result(case["label"])
        return results

    all_dirs = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]
    all_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(experiments_dir, d)), reverse=True)

    used_dirs = set()
    for case in cases:
        key = case["key"]
        run_name = case["run_name"]
        matched_dir = None

        for d in all_dirs:
            if d.startswith(run_name) and d not in used_dirs:
                summary_f = os.path.join(experiments_dir, d, "summary.json")
                telemetry_f = os.path.join(experiments_dir, d, "telemetry.csv")
                if os.path.exists(summary_f) or (os.path.exists(telemetry_f) and os.path.getsize(telemetry_f) > 100):
                    matched_dir = os.path.join(experiments_dir, d)
                    used_dirs.add(d)
                    break

        if matched_dir:
            print(f"[Scavenge] Found matching folder for {case['label']}: {matched_dir}")
            results[key] = scavenge_experiment_run(matched_dir, case)
        else:
            print(f"[WARN] No folder starting with '{run_name}' found for {case['label']}.")

    # Fallback to associate most recent runs if run_name was not prefixed
    missing_keys = [c["key"] for c in cases if c["key"] not in results]
    if missing_keys:
        unmatched_dirs = []
        for d in all_dirs:
            if d not in used_dirs:
                full_d = os.path.join(experiments_dir, d)
                if os.path.exists(os.path.join(full_d, "summary.json")):
                    unmatched_dirs.append(full_d)

        unmatched_dirs.sort(key=lambda d: os.path.getmtime(d))
        for key in missing_keys:
            case = next(c for c in cases if c["key"] == key)
            if unmatched_dirs:
                cand = unmatched_dirs.pop(0)
                used_dirs.add(os.path.basename(cand))
                print(f"[Scavenge] Associating {case['label']} with recent run: {cand}")
                results[key] = scavenge_experiment_run(cand, case)
            else:
                results[key] = create_empty_result(case["label"])

    # Automatically recover authentic scenario from matching experiment folders
    for case in cases:
        key = case["key"]
        run_dir = results.get(key, {}).get("run_dir", "")
        if run_dir and os.path.isdir(run_dir):
            scen_f = os.path.join(run_dir, "scenario_copy.json")
            if os.path.exists(scen_f):
                try:
                    results["__scenario__"] = load_scenario(scen_f)
                    break
                except Exception:
                    pass

    return results


def run_suite_via_bringup(cases: list, params_path: str, experiments_dir: str, timeout: Optional[float] = None, scenario_path: Optional[str] = None) -> dict:
    """Sequentially configures sim_params.yaml, runs bringup.launch.py, and collects results."""
    os.makedirs(experiments_dir, exist_ok=True)
    if timeout is None:
        scen_dict = load_scenario(scenario_path) if scenario_path else {}
        timeout = float(scen_dict.get("sim_time", 800.0))
        print(f"  [Timeout] Dynamically resolved from scenario 'sim_time': {timeout:.1f}s")

    with open(params_path, "r") as f:
        original_yaml = f.read()

    results = {}
    try:
        for i, case in enumerate(cases, start=1):
            key = case["key"]
            label = case["label"]
            run_name = case["run_name"]
            case_params = case["params"]

            print(f"\n==================================================================================")
            print(f"  [Run {i}/{len(cases)}] Launching bringup for {label}")
            print(f"  Configuring sim_params.yaml -> {case_params}")
            print(f"==================================================================================")

            update_sim_params(params_path, case_params, run_name=run_name)

            existing_dirs = set(os.listdir(experiments_dir)) if os.path.exists(experiments_dir) else set()

            cmd = ["ros2", "launch", "nmpc_sim_nodes", "bringup.launch.py", f"params_file:={params_path}"]
            print(f"  Executing: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )

            run_dir = None
            t_start = time.time()
            print("  Waiting for simulation run to complete...", end="", flush=True)

            while proc.poll() is None:
                elapsed = time.time() - t_start
                if elapsed > timeout:
                    print(f"\n  [WARN] Case '{label}' reached timeout limit ({timeout:.1f}s)!")
                    break

                current_dirs = set(os.listdir(experiments_dir))
                new_dirs = [d for d in (current_dirs - existing_dirs) if d.startswith(run_name)]
                if not new_dirs:
                    candidates = [d for d in current_dirs if d.startswith(run_name)]
                    if candidates:
                        new_dirs = sorted(candidates, key=lambda d: os.path.getmtime(os.path.join(experiments_dir, d)), reverse=True)[:1]

                for d in new_dirs:
                    cand_path = os.path.join(experiments_dir, d)
                    summary_file = os.path.join(cand_path, "summary.json")
                    if os.path.exists(summary_file) and os.path.getsize(summary_file) > 10:
                        run_dir = cand_path
                        break

                if run_dir is not None:
                    break

                time.sleep(1.0)
                print(".", end="", flush=True)

            print()

            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=8.0)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait(timeout=2.0)
                    except Exception:
                        pass

            time.sleep(2.0)

            # If run_dir was not resolved via summary.json during the wait loop (e.g. timeout occurred),
            # search for the run directory that was created and populated with telemetry during this run.
            if run_dir is None:
                candidates = [d for d in os.listdir(experiments_dir) if d.startswith(run_name)]
                if candidates:
                    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(experiments_dir, d)), reverse=True)
                    for cand in candidates:
                        cand_path = os.path.join(experiments_dir, cand)
                        telemetry_file = os.path.join(cand_path, "telemetry.csv")
                        if os.path.exists(telemetry_file) and os.path.getsize(telemetry_file) > 100:
                            run_dir = cand_path
                            print(f"  [Scavenge on Timeout] Recovered experiment telemetry from: {run_dir}")
                            break

            if run_dir and os.path.isdir(run_dir):
                print(f"  [OK] Run finalized and logged at: {run_dir}")
                res = scavenge_experiment_run(run_dir, case)
                results[key] = res
                print(f"  Status: {res['status']} | Sim Time: {res['final_t']:.1f}s | "
                      f"Max CT: {res['max_crosstrack']:.2f}m | CT RMSE: {res['crosstrack_rmse']:.2f}m")
            else:
                print(f"  [ERROR] No completed experiment folder detected for {label}!")
                results[key] = create_empty_result(label)

    finally:
        print("\n[INFO] Restoring original sim_params.yaml...")
        restore_sim_params(params_path, original_yaml)
        print("[INFO] sim_params.yaml restored successfully.")

    return results


def draw_scenario_obstacles(ax, scenario: dict):
    """Draws obstacles on the matplotlib axis."""
    for ox, oy, orad, *_ in scenario.get("obstacles", []):
        ax.add_patch(plt.Circle((oy, ox), orad, color="tab:orange", alpha=0.35, zorder=2))

    for ellipse in scenario.get("ellipses", []):
        xc, yc, a, b, theta = ellipse[:5]
        t_vals = np.linspace(0, 2.0 * math.pi, 100)
        bx = a * np.cos(t_vals)
        by = b * np.sin(t_vals)
        ell_x = xc + bx * math.cos(theta) - by * math.sin(theta)
        ell_y = yc + bx * math.sin(theta) + by * math.cos(theta)
        ax.fill(ell_y, ell_x, color="coral", alpha=0.35, zorder=2)
        ax.plot(ell_y, ell_x, color="chocolate", linewidth=1.2, zorder=2)

    for ship in scenario.get("obstacle_ships", []):
        xc, yc, a, b, theta = ship[:5]
        t_vals = np.linspace(0, 2.0 * math.pi, 100)
        bx = a * np.cos(t_vals)
        by = b * np.sin(t_vals)
        ship_x = xc + bx * math.cos(theta) - by * math.sin(theta)
        ship_y = yc + bx * math.sin(theta) + by * math.cos(theta)
        ax.fill(ship_y, ship_x, color="gray", alpha=0.4, zorder=2)
        ax.plot(ship_y, ship_x, color="dimgray", linewidth=1.2, zorder=2)

    for wall in scenario.get("walls", []):
        x0, y0, x1, y1, r_wall = wall[:5]
        ax.plot([y0, y1], [x0, x1], color="dimgray", linewidth=2.0 + 2.0 * r_wall, alpha=0.5, zorder=2)


def generate_static_graph(scenario: dict, results: dict, out_path: str, cases: list):
    """Generates a high-resolution graph showing both paths on the scenario map."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.94, bottom=0.08)

    waypoints = scenario.get("waypoints", [])
    if waypoints:
        wp_x = [w[0] for w in waypoints]
        wp_y = [w[1] for w in waypoints]
        ax.plot(wp_y, wp_x, "g--", marker="x", markersize=8, linewidth=1.2, label="Waypoints", zorder=3)
        ax.plot(wp_y[0], wp_x[0], "go", markersize=7, label="Start", zorder=3)
        ax.plot(wp_y[-1], wp_x[-1], "g*", markersize=11, label="Goal Endpoint", zorder=3)

    draw_scenario_obstacles(ax, scenario)

    for case in cases:
        key = case["key"]
        label = case["label"]
        color = case["color"]
        ls = case.get("linestyle", "-")
        res = results.get(key)
        if not res or len(res["x"]) <= 1:
            continue
        ax.plot(res["y"], res["x"], color=color, linestyle=ls, linewidth=2.0, alpha=0.9,
                label=f"{label} ({res['status']}, {res['final_t']:.1f}s)", zorder=4)

    ax.set_xlabel("Y [Cross-track offset] (m)", fontsize=11)
    ax.set_ylabel("X [Along track] (m)", fontsize=11)
    ax.set_title("Current Awareness Advantage: Trajectory Comparison", fontsize=12, fontweight="bold")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", fontsize=9.0, framealpha=0.95, facecolor="white", edgecolor="#999999")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Trajectory comparison graph saved: {out_path}")


def generate_looping_gif(scenario: dict, results: dict, out_path: str, bow_len: float = 2.9, subsample: int = 10, cases: list = None):
    """Generates an animated looping GIF showing live traversal, bow vectors, and predicted horizons."""
    if cases is None:
        cases = AWARENESS_CASES
    active_keys = [c["key"] for c in cases if c["key"] in results and len(results[c["key"]]["x"]) > 1]
    if not active_keys:
        print("[WARN] No trajectory data available to render GIF.")
        return

    print(f"--- Rendering looping animated GIF (subsample={subsample})... ---")
    max_frames = max(len(results[k]["x"]) for k in active_keys)
    frame_indices = list(range(0, max_frames, subsample))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    waypoints = scenario.get("waypoints", [])
    wp_x = [w[0] for w in waypoints] if waypoints else [0.0]
    wp_y = [w[1] for w in waypoints] if waypoints else [0.0]

    all_x = np.concatenate([results[k]["x"] for k in active_keys] + [wp_x])
    all_y = np.concatenate([results[k]["y"] for k in active_keys] + [wp_y])
    pad = 12.0
    x_min, x_max = float(np.min(all_x)) - pad, float(np.max(all_x)) + pad
    y_min, y_max = float(np.min(all_y)) - pad, float(np.max(all_y)) + pad

    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min) + 6.0
    fixed_xlim = (mid_y - half_span, mid_y + half_span)
    fixed_ylim = (mid_x - half_span, mid_x + half_span)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.93, bottom=0.08)

    raw_frames = []
    dt = 0.1

    for frame_idx, step_k in enumerate(frame_indices):
        ax.clear()
        sim_time_curr = step_k * dt

        ax.set_xlim(fixed_xlim)
        ax.set_ylim(fixed_ylim)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.35)

        if waypoints:
            ax.plot(wp_y, wp_x, "g--", marker="x", markersize=7, linewidth=1.0, label="Waypoints", zorder=2)
            ax.plot(wp_y[0], wp_x[0], "go", markersize=6, zorder=2)
            ax.plot(wp_y[-1], wp_x[-1], "g*", markersize=10, zorder=2)
        draw_scenario_obstacles(ax, scenario)

        for case in cases:
            key = case["key"]
            label = case["label"]
            color = case["color"]
            if key not in active_keys:
                continue

            res = results[key]
            n_samples = len(res["x"])
            idx = min(step_k, n_samples - 1)

            # 1. Past path traversed
            ax.plot(res["y"][:idx + 1], res["x"][:idx + 1], color=color, linewidth=1.6, alpha=0.85,
                    label=label)

            # 2. Current vessel position marker
            curr_x, curr_y, curr_psi = res["x"][idx], res["y"][idx], res["psi"][idx]
            ax.plot(curr_y, curr_x, marker="o", color=color, markersize=5, zorder=5)

            # 3. Heading bow line
            bow_x = curr_x + bow_len * math.cos(curr_psi)
            bow_y = curr_y + bow_len * math.sin(curr_psi)
            ax.plot([curr_y, bow_y], [curr_x, bow_x], color=color, linewidth=2.4, zorder=6)
            ax.plot(bow_y, bow_x, marker="^", color=color, markersize=4, zorder=6)

            # 4. Future state horizon prediction
            if step_k < n_samples and idx < len(res["horizons_x"]):
                hz_x = res["horizons_x"][idx]
                hz_y = res["horizons_y"][idx]
                if len(hz_x) > 1:
                    ax.plot(hz_y, hz_x, linestyle=":", color=color, linewidth=1.2, alpha=0.85, zorder=4)

        ax.set_xlabel("Y [Cross-track offset] (m)", fontsize=10)
        ax.set_ylabel("X [Along track] (m)", fontsize=10)
        ax.set_title(f"Current Awareness Live Traversal | Sim Time: {sim_time_curr:5.1f} s",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, facecolor="white", edgecolor="#999999")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        raw_frames.append(Image.open(buf).convert("RGB"))
        buf.close()

    plt.close(fig)

    if raw_frames:
        palette_img = raw_frames[-1].quantize(colors=256)
        p_images = [im.quantize(palette=palette_img, dither=Image.Dither.NONE) for im in raw_frames]

        p_images[0].save(
            out_path,
            save_all=True,
            append_images=p_images[1:],
            duration=65,
            loop=0,
        )
        print(f"[OK] Looping animated GIF saved ({len(p_images)} frames): {out_path}")


def generate_heading_and_crab_graph(results: dict, out_path: str):
    """Generates a 2-subplot time-series plot comparing:
       - Ship Heading (psi)
       - Reference Heading Angle (chi_p)
       - Crab Angle (beta = atan2(v, u))
       Top Subplot: Current-Aware NMPC
       Bottom Subplot: Current-Unaware NMPC (Current=0)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, dpi=150)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.08, hspace=0.25)

    res_aware = results.get("current_aware")
    res_unaware = results.get("current_unaware")

    def _plot_sub(ax, res, title, main_color):
        if not res or len(res["t"]) <= 1:
            ax.set_title(f"{title} (No data)", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.35)
            return

        ts = np.array(res["t"])
        # Wrap heading and reference angles to [-180, 180] deg
        psis_deg = np.degrees(np.array([wrap_to_pi(p) for p in res["psi"]]))
        chi_ps_deg = np.degrees(np.array([wrap_to_pi(c) for c in res["chi_p"]]))
        crabs_deg = np.degrees(np.array([wrap_to_pi(b) for b in res["crab_angle"]]))

        # Primary Y-axis: Heading & Reference Course Angle
        l1, = ax.plot(ts, chi_ps_deg, color="#2CA02C", linestyle="--", linewidth=1.6,
                      label="Reference Heading χ_p (deg)", zorder=3)
        l2, = ax.plot(ts, psis_deg, color=main_color, linestyle="-", linewidth=1.8,
                      label="Ship Heading ψ (deg)", zorder=4)

        ax.set_ylabel("Heading & Reference (°)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Secondary Y-axis: Crab Angle beta
        ax_crab = ax.twinx()
        l3, = ax_crab.plot(ts, crabs_deg, color="#FF7F0E", linestyle="-", linewidth=1.4,
                           alpha=0.9, label="Crab Angle β = atan2(v, u) (°)", zorder=2)
        ax_crab.axhline(0.0, color="#FF7F0E", linestyle=":", alpha=0.4)
        ax_crab.set_ylabel("Crab Angle β (°)", fontsize=10, color="#D95F02", fontweight="bold")
        ax_crab.tick_params(axis="y", labelcolor="#D95F02")

        # Combine legends from both axes
        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.92, facecolor="white")

    # Subplot 1: Current-Aware NMPC
    _plot_sub(ax1, res_aware, "Current-Aware NMPC: Heading, Reference Course & Crab Angle vs. Time", "#1F77B4")

    # Subplot 2: Current-Unaware NMPC
    _plot_sub(ax2, res_unaware, "Current-Unaware NMPC (Current Assumed = 0): Heading, Reference Course & Crab Angle vs. Time", "#D62728")

    ax2.set_xlabel("Time (s)", fontsize=11, fontweight="bold")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Heading and crab angle comparison plot saved: {out_path}")


def print_terminal_table(results: dict, cases: list):
    """Prints a clear summary table of tracking errors, control efforts, and solve latency."""
    border = "=" * 144
    divider = "-" * 144
    title = "NMPC CURRENT-AWARENESS ADVANTAGE: BENCHMARK STUDY RESULTS".center(144)
    header_cols = (
        f"{'Configuration':<34} "
        f"{'Max Crosstrack':>14} "
        f"{'Crosstrack RMSE':>16} "
        f"{'Max Heading Err':>17} "
        f"{'Heading RMSE':>14} "
        f"{'Rudder Effort':>15} "
        f"{'Propeller Effort':>17} "
        f"{'Total Solve Time':>15}"
    )
    print(f"\n{border}\n{title}\n{border}\n{header_cols}\n{divider}")

    for case in cases:
        key = case["key"]
        label = case["label"]
        res = results.get(key)
        if not res or res["total_steps"] <= 1:
            print(f"{label:<34} {'N/A':>14} {'N/A':>16} {'N/A':>17} {'N/A':>14} {'N/A':>15} {'N/A':>17} {'N/A':>15}")
            continue

        max_ct = f"{res['max_crosstrack']:.2f} m"
        ct_rmse = f"{res['crosstrack_rmse']:.2f} m"
        max_head = f"{res['max_heading_err']:.1f}°"
        head_rmse = f"{res['heading_rmse']:.1f}°"
        rudder = f"{res['rudder_effort']:.2f} rad²·s"
        prop = f"{res['propeller_effort']:.2f} rps²·s"
        solve_t = f"{res['total_solve_time']:.2f} s"

        row = (
            f"{label:<34} "
            f"{max_ct:>14} "
            f"{ct_rmse:>16} "
            f"{max_head:>17} "
            f"{head_rmse:>14} "
            f"{rudder:>15} "
            f"{prop:>17} "
            f"{solve_t:>15}"
        )
        print(row)

    print(f"{border}\n")


def save_results_npz(results: dict, npz_path: str, cases: list):
    npz_data = {}
    for case in cases:
        key = case["key"]
        res = results.get(key)
        if not res:
            continue
        for field in ["x", "y", "psi", "t", "u", "v", "chi_p", "crab_angle", "delta", "n", "e_ct", "e_psi"]:
            if field in res:
                npz_data[f"{key}_{field}"] = np.array(res[field])
        npz_data[f"{key}_horizons_x"] = np.array(res["horizons_x"], dtype=object)
        npz_data[f"{key}_horizons_y"] = np.array(res["horizons_y"], dtype=object)
        metrics = ["max_crosstrack", "crosstrack_rmse", "max_heading_err",
                   "heading_rmse", "rudder_effort", "propeller_effort", "total_solve_time", "final_t",
                   "solver_failures", "total_steps"]
        for metric in metrics:
            npz_data[f"{key}_{metric}"] = np.array([res[metric]])
        npz_data[f"{key}_status"] = np.array([res["status"]])

    # Persist the scenario used for these runs
    scen_to_save = None
    if "__scenario__" in results:
        scen_to_save = results["__scenario__"]
    elif cases and cases[0]["key"] in results:
        r_dir = results[cases[0]["key"]].get("run_dir", "")
        if r_dir and os.path.exists(os.path.join(r_dir, "scenario_copy.json")):
            try:
                scen_to_save = load_scenario(os.path.join(r_dir, "scenario_copy.json"))
            except Exception:
                pass

    if scen_to_save:
        npz_data["scenario_json"] = np.array([json.dumps(scen_to_save)])

    np.savez_compressed(npz_path, **npz_data)
    print(f"[OK] Compressed results archive saved: {npz_path}")


def load_results_npz(npz_path: str, cases: list) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    results = {}
    for case in cases:
        key = case["key"]
        if f"{key}_x" not in data:
            continue
        results[key] = {
            "x": data[f"{key}_x"].tolist(),
            "y": data[f"{key}_y"].tolist(),
            "psi": data[f"{key}_psi"].tolist(),
            "t": data[f"{key}_t"].tolist(),
            "u": data[f"{key}_u"].tolist() if f"{key}_u" in data else [],
            "v": data[f"{key}_v"].tolist() if f"{key}_v" in data else [],
            "chi_p": data[f"{key}_chi_p"].tolist() if f"{key}_chi_p" in data else [],
            "crab_angle": data[f"{key}_crab_angle"].tolist() if f"{key}_crab_angle" in data else [],
            "delta": data[f"{key}_delta"].tolist(),
            "n": data[f"{key}_n"].tolist(),
            "e_ct": data[f"{key}_e_ct"].tolist(),
            "e_psi": data[f"{key}_e_psi"].tolist(),
            "horizons_x": list(data[f"{key}_horizons_x"]),
            "horizons_y": list(data[f"{key}_horizons_y"]),
            "status": str(data[f"{key}_status"][0]),
            "final_t": float(data[f"{key}_final_t"][0]),
            "max_crosstrack": float(data[f"{key}_max_crosstrack"][0]),
            "crosstrack_rmse": float(data[f"{key}_crosstrack_rmse"][0]),
            "max_heading_err": float(data[f"{key}_max_heading_err"][0]),
            "heading_rmse": float(data[f"{key}_heading_rmse"][0]),
            "rudder_effort": float(data[f"{key}_rudder_effort"][0]),
            "propeller_effort": float(data[f"{key}_propeller_effort"][0]),
            "total_solve_time": float(data[f"{key}_total_solve_time"][0]),
            "solver_failures": int(data[f"{key}_solver_failures"][0]),
            "total_steps": int(data[f"{key}_total_steps"][0]),
        }

    if "scenario_json" in data:
        try:
            results["__scenario__"] = json.loads(str(data["scenario_json"][0]))
        except Exception:
            pass

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_scenario = os.path.join(
        _pkg_paths.repo_root(), "src", "nmpc_sim_nodes", "params", "scenario.json"
    )
    if not os.path.isfile(default_scenario):
        default_scenario = os.path.join(
            _pkg_paths.repo_root(), "nmpc_ws", "src", "nmpc_sim_nodes", "params", "scenario.json"
        )

    default_params = _pkg_paths.sim_params_path()

    parser.add_argument("--params-path", default=default_params,
                        help="Path to sim_params.yaml (default: active package sim_params.yaml)")
    parser.add_argument("--scenario-path", default=None,
                        help="Path to scenario.json (default: matched experiment scenario or active package scenario.json)")
    parser.add_argument("--experiments-dir", default=DEFAULT_EXPERIMENTS_DIR,
                        help="Directory where experiment logs are written by logger_node")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Output directory for generated graph, gif, and npz logs")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Step subsample rate for animated GIF generation")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Max timeout in seconds per simulation run (default: None -> read from scenario sim_time)")
    parser.add_argument("--scavenge", action="store_true",
                        help="Scavenge from existing runs in --experiments-dir instead of launching bringup")
    parser.add_argument("--render-only", action="store_true",
                        help="Re-render graphs and gif from existing current_awareness_results.npz if present")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)
    cfg = load_nmpc_config(args.params_path)
    bow_len = float(cfg.LPP)

    npz_path = os.path.join(args.results_dir, "current_awareness_results.npz")
    results = None
    explicit_scenario = args.scenario_path is not None
    active_scenario_path = args.scenario_path if explicit_scenario else default_scenario

    if args.render_only and os.path.isfile(npz_path):
        print(f"\n[INFO] Loading existing results from {npz_path}...")
        results = load_results_npz(npz_path, AWARENESS_CASES)
    elif args.scavenge:
        print(f"\n[INFO] Scavenging existing logs from {args.experiments_dir}...")
        results = scavenge_all_existing_runs(AWARENESS_CASES, args.experiments_dir)
        save_results_npz(results, npz_path, AWARENESS_CASES)
    else:
        print("==================================================================================")
        print("  RUNNING CURRENT-AWARENESS ADVANTAGE STUDY (AWARE VS. UNAWARE VIA BRINGUP)")
        print(f"  Params File     : {args.params_path}")
        print(f"  Scenario File   : {active_scenario_path}")
        print(f"  Experiments Dir : {args.experiments_dir}")
        print(f"  Results Dir     : {args.results_dir}")
        timeout_disp = f"{args.timeout:.1f}s (explicit CLI)" if args.timeout is not None else f"{load_scenario(active_scenario_path).get('sim_time', 800.0):.1f}s (from scenario)"
        print(f"  Per-Run Timeout : {timeout_disp}")
        print("==================================================================================")
        results = run_suite_via_bringup(AWARENESS_CASES, args.params_path, args.experiments_dir, timeout=args.timeout, scenario_path=active_scenario_path)
        save_results_npz(results, npz_path, AWARENESS_CASES)

    # Resolve scenario: if not explicitly overridden by CLI, prefer authentic scenario from experiment logs
    if not explicit_scenario and results and "__scenario__" in results:
        scenario = results["__scenario__"]
        print(f"[INFO] Using authentic scenario from experiment logs ({len(scenario.get('waypoints', []))} waypoints).")
    else:
        scenario = load_scenario(active_scenario_path)

    # 1. Print formatted terminal table
    print_terminal_table(results, AWARENESS_CASES)

    # 2. Generate static trajectory comparison graph
    graph_path = os.path.join(args.results_dir, "current_awareness_paths.png")
    generate_static_graph(scenario, results, graph_path, AWARENESS_CASES)

    # 3. Generate smooth looping animated GIF
    gif_path = os.path.join(args.results_dir, "current_awareness_animation.gif")
    generate_looping_gif(scenario, results, gif_path, bow_len=bow_len, subsample=args.subsample, cases=AWARENESS_CASES)

    # 4. Generate Heading and Crab Angle 2-subplot comparison time-series plot
    heading_crab_path = os.path.join(args.results_dir, "heading_and_crab_angle_comparison.png")
    generate_heading_and_crab_graph(results, heading_crab_path)

    print("\n==================================================================================")
    print("  CURRENT-AWARENESS ADVANTAGE STUDY COMPLETE")
    print(f"  1) Trajectory Graph   : {graph_path}")
    print(f"  2) Heading & Crab Plot: {heading_crab_path}")
    print(f"  3) Looping GIF        : {gif_path}")
    print(f"  4) Saved Results NPZ  : {npz_path}")
    print("==================================================================================\n")


if __name__ == "__main__":
    main()
