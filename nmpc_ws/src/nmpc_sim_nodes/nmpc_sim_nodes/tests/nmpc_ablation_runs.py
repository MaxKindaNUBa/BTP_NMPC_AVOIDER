"""nmpc_ablation_runs: Test orchestrator and analysis script for the 5-case
NMPC Ablation Study (Baseline, Current, Waves, Sensor Noise, Combined).

Features:
  1. Multi-seed randomized repetitions:
     - Configurable N rounds (default: 10 via --num-runs).
     - In each round, a common randomized disturbance set is generated:
         * wave_seed
         * current_seed
         * sensor_seed
         * current_mean_heading (rad in [0, 2*pi))
     - The disturbance set is applied to sim_params.yaml, and all 5 cases are
       executed under those identical environmental disturbance realizations.
     - Repeated for all N rounds.
  2. Absolute and Normalized Performance Tables:
     - Table 1 (Absolute): Displays mean ± std (when N > 1) or single values (when N = 1)
       for Crosstrack RMSE, Max Crosstrack, Heading RMSE, Max Heading Error,
       Rudder Effort, Propeller Effort, Sim Time, and Success Rate.
     - Table 2 (Normalized): Displays relative performance normalized against
       Case 1: Baseline (Calm Water) = 1.00x (ref), showing ratio and percentage
       change (+Δ%) for each metric.
     - Both tables are printed to the terminal and saved to ablation_table.txt.
  3. Representative Seed Visualization:
     - Selects the run closest to the median crosstrack RMSE across the repetitions
       to render the static trajectory comparison plot (ablation_paths.png) and
       the looping animated GIF with prediction horizons and bow lines (ablation_animation.gif).
  4. Scavenging and Offline Analysis:
     - Scavenges experiment runs from nmpc_sim_logs/experiments, with automatic
       telemetry recovery even if a simulation timed out.
     - Supports `--scavenge` to aggregate existing runs without re-running bringup.
     - Supports `--render-only` to reload ablation_results.npz.

Run:
  ros2 run nmpc_sim_nodes nmpc_ablation_runs --num-runs 10
  ros2 run nmpc_sim_nodes nmpc_ablation_runs --scavenge
"""
import argparse
import csv
from datetime import datetime
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

DEFAULT_RESULTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "nmpc_ablation_runs")
DEFAULT_EXPERIMENTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "experiments")

# 5 Ablation Study Configurations (only toggling current_enabled, wave_enabled, sensor_enabled; use_ukf is left untouched)
ABLATION_CASES = [
    {
        "key": "baseline",
        "label": "1. Baseline (Calm)",
        "color": "#111111",
        "linestyle": "-",
        "params": {
            "current_enabled": False,
            "wave_enabled": False,
            "sensor_enabled": False,
        },
        "run_name": "ablation_1_baseline",
    },
    {
        "key": "current_only",
        "label": "2. Current only",
        "color": "#8A2BE2",
        "linestyle": "-",
        "params": {
            "current_enabled": True,
            "wave_enabled": False,
            "sensor_enabled": False,
        },
        "run_name": "ablation_2_current",
    },
    {
        "key": "wave_only",
        "label": "3. Wave only",
        "color": "#2CA02C",
        "linestyle": "-",
        "params": {
            "current_enabled": False,
            "wave_enabled": True,
            "sensor_enabled": False,
        },
        "run_name": "ablation_3_wave",
    },
    {
        "key": "noise_only",
        "label": "4. Sensor noise only",
        "color": "#FF7F0E",
        "linestyle": "-",
        "params": {
            "current_enabled": False,
            "wave_enabled": False,
            "sensor_enabled": True,
        },
        "run_name": "ablation_4_noise",
    },
    {
        "key": "combined",
        "label": "5. Combined (All 3)",
        "color": "#1F77B4",
        "linestyle": "-",
        "params": {
            "current_enabled": True,
            "wave_enabled": True,
            "sensor_enabled": True,
        },
        "run_name": "ablation_5_combined",
    },
]


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def find_default_scenario_path() -> str:
    """Finds active scenario.json prioritizing workspace src tree, falling back to share directory."""
    cand_ws = os.path.join(_pkg_paths.repo_root(), "nmpc_ws", "src", "nmpc_sim_nodes", "params", "scenario.json")
    if os.path.isfile(cand_ws):
        return os.path.abspath(cand_ws)
    cand_repo = os.path.join(_pkg_paths.repo_root(), "src", "nmpc_sim_nodes", "params", "scenario.json")
    if os.path.isfile(cand_repo):
        return os.path.abspath(cand_repo)
    try:
        from ament_index_python.packages import get_package_share_directory
        share_scen = os.path.join(get_package_share_directory("nmpc_sim_nodes"), "params", "scenario.json")
        if os.path.isfile(share_scen):
            return os.path.abspath(share_scen)
    except Exception:
        pass
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cand_rel = os.path.normpath(os.path.join(curr_dir, "..", "..", "params", "scenario.json"))
    if os.path.isfile(cand_rel):
        return os.path.abspath(cand_rel)
    return cand_ws


def update_sim_params(yaml_path: str, params: dict, run_name: Optional[str] = None, scenario_path: Optional[str] = None):
    """Modifies mmg_node flags/seeds and logger_node run_name/scenario_path in sim_params.yaml while preserving comments."""
    with open(yaml_path, "r") as f:
        content = f.read()

    for key, val in params.items():
        if isinstance(val, bool):
            val_str = "true" if val else "false"
            pattern = rf"(^\s*{key}:\s*)(true|false)"
            content = re.sub(pattern, rf"\g<1>{val_str}", content, flags=re.MULTILINE | re.IGNORECASE)
        elif isinstance(val, (int, float)):
            val_str = str(val)
            pattern = rf"(^\s*{key}:\s*)([0-9eE\.\+\-]+)"
            content = re.sub(pattern, rf"\g<1>{val_str}", content, flags=re.MULTILINE)

    if run_name:
        pattern = r'(^\s*run_name:\s*)"[^"]*"'
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, rf'\g<1>"{run_name}"', content, flags=re.MULTILINE)
        else:
            pat_logger = r"(logger_node:\s*\n\s*ros__parameters:\s*\n)"
            content = re.sub(pat_logger, rf'\g<1>    run_name: "{run_name}"\n', content)

    if scenario_path:
        scen_norm = os.path.abspath(scenario_path)
        for section in ["map_node", "logger_node"]:
            pat_existing = rf"(^\s*{section}:\s*\n\s*ros__parameters:[\s\S]*?^\s*scenario_path:\s*)[^\n]+"
            if re.search(pat_existing, content, flags=re.MULTILINE):
                content = re.sub(pat_existing, rf'\g<1>"{scen_norm}"', content, flags=re.MULTILINE)
            else:
                pat_insert = rf"({section}:\s*\n\s*ros__parameters:\s*\n)"
                content = re.sub(pat_insert, rf'\g<1>    scenario_path: "{scen_norm}"\n', content)

    with open(yaml_path, "w") as f:
        f.write(content)


def restore_sim_params(yaml_path: str, original_content: str):
    """Restores the original sim_params.yaml content."""
    try:
        with open(yaml_path, "w") as f:
            f.write(original_content)
    except Exception as e:
        print(f"[WARN] Failed to restore {yaml_path}: {e}")


def generate_round_disturbances(num_rounds: int, master_seed: Optional[int] = 42) -> List[dict]:
    """Generates reproducible randomized disturbance settings for each repetition round."""
    rng = np.random.default_rng(master_seed)
    round_configs = []
    for r in range(1, num_rounds + 1):
        round_configs.append({
            "round": r,
            "wave_seed": int(rng.integers(100, 1_000_000)),
            "current_seed": int(rng.integers(100, 1_000_000)),
            "sensor_seed": int(rng.integers(100, 1_000_000)),
            "current_mean_heading": round(float(rng.uniform(0.0, 2.0 * math.pi)), 4),
        })
    return round_configs


def load_scenario(path: str) -> dict:
    if not os.path.exists(path):
        return {"waypoints": [], "obstacles": [], "ellipses": [], "walls": [], "obstacle_ships": []}
    with open(path, "r") as f:
        return json.load(f)


def scavenge_experiment_run(run_dir: str, case_info: dict) -> dict:
    """Extracts trajectory, horizon, error, effort, and KPI data from an experiment run directory."""
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
                    delta = float(row["true_delta_rad"]) if row.get("true_delta_rad") not in ("", None) else (float(row["cmd_delta_rad"]) if row.get("cmd_delta_rad") not in ("", None) else 0.0)
                    n = float(row["true_n_rps"]) if row.get("true_n_rps") not in ("", None) else (float(row["cmd_n_rps"]) if row.get("cmd_n_rps") not in ("", None) else 0.0)
                    tgt_wp_str = row.get("target_wp_idx", "-1")
                    tgt_wp_idx = int(float(tgt_wp_str)) if tgt_wp_str not in ("", None) else -1
                    # Skip uninitialized startup samples before map_node publishes the active reference
                    if tgt_wp_idx < 0:
                        continue

                    target_x = float(row["target_x_m"]) if row.get("target_x_m") else 0.0
                    target_y = float(row["target_y_m"]) if row.get("target_y_m") else 0.0
                    chi_p = float(row["chi_p_rad"]) if row.get("chi_p_rad") else 0.0
                    solve_time = float(row["solve_time_ms"]) * 1e-3 if row.get("solve_time_ms") else 0.0

                    # Compute tracking errors
                    e_ct = -(x - target_x) * math.sin(chi_p) + (y - target_y) * math.cos(chi_p)
                    e_psi = wrap_to_pi(psi - chi_p)

                    ts.append(t)
                    xs.append(x)
                    ys.append(y)
                    psis.append(psi)
                    deltas.append(delta)
                    ns.append(n)
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
                    # IDX_X = 4, IDX_Y = 5
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
        "key": case_info["key"],
        "label": case_info["label"],
        "x": xs,
        "y": ys,
        "psi": psis,
        "t": ts,
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


def create_empty_single_run(case_info: dict) -> dict:
    """Returns an empty trajectory result dict for a single run that failed or timed out."""
    return {
        "key": case_info["key"],
        "label": case_info["label"],
        "x": [0.0],
        "y": [0.0],
        "psi": [0.0],
        "t": [0.0],
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


def create_empty_result(label: str) -> dict:
    return {
        "key": "",
        "label": label,
        "num_runs": 0,
        "successful_runs": 0,
        "metrics": {
            m: {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "values": []}
            for m in [
                "crosstrack_rmse", "max_crosstrack", "heading_rmse", "max_heading_err",
                "rudder_effort", "propeller_effort", "total_solve_time", "final_t", "solver_failures"
            ]
        },
        "representative": {
            "x": [0.0],
            "y": [0.0],
            "psi": [0.0],
            "t": [0.0],
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
        },
        "x": [0.0],
        "y": [0.0],
        "psi": [0.0],
        "t": [0.0],
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


def aggregate_case_runs(runs: List[dict], case_info: dict) -> dict:
    """Aggregates multiple repetition runs for a single ablation case into statistics and picks a representative run."""
    valid_runs = [r for r in runs if r.get("total_steps", 0) > 1]
    if not valid_runs:
        return create_empty_result(case_info["label"])

    metric_keys = [
        "crosstrack_rmse", "max_crosstrack", "heading_rmse", "max_heading_err",
        "rudder_effort", "propeller_effort", "total_solve_time", "final_t",
        "solver_failures"
    ]

    metrics = {}
    for m in metric_keys:
        vals = [float(r[m]) for r in valid_runs]
        mean_val = float(np.mean(vals)) if vals else 0.0
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        metrics[m] = {
            "mean": mean_val,
            "std": std_val,
            "min": float(np.min(vals)) if vals else 0.0,
            "max": float(np.max(vals)) if vals else 0.0,
            "values": vals,
        }

    # Pick representative run: run whose crosstrack_rmse is closest to the median crosstrack_rmse
    ct_rmse_vals = [r["crosstrack_rmse"] for r in valid_runs]
    median_ct = float(np.median(ct_rmse_vals))
    rep_run = min(valid_runs, key=lambda r: abs(r["crosstrack_rmse"] - median_ct))

    # Calculate success count
    successful_runs = sum(1 for r in valid_runs if r.get("status") in ("GOAL_REACHED", "COMPLETED"))
    total_runs = len(valid_runs)

    res_agg = {
        "key": case_info["key"],
        "label": case_info["label"],
        "num_runs": total_runs,
        "successful_runs": successful_runs,
        "metrics": metrics,
        "representative": rep_run,
        "all_runs": runs,
        # Flat accessors mapped for backward compatibility
        "x": rep_run["x"],
        "y": rep_run["y"],
        "psi": rep_run["psi"],
        "t": rep_run["t"],
        "delta": rep_run["delta"],
        "n": rep_run["n"],
        "e_ct": rep_run["e_ct"],
        "e_psi": rep_run["e_psi"],
        "horizons_x": rep_run["horizons_x"],
        "horizons_y": rep_run["horizons_y"],
        "status": rep_run["status"],
        "final_t": metrics["final_t"]["mean"],
        "crosstrack_rmse": metrics["crosstrack_rmse"]["mean"],
        "max_crosstrack": metrics["max_crosstrack"]["mean"],
        "heading_rmse": metrics["heading_rmse"]["mean"],
        "max_heading_err": metrics["max_heading_err"]["mean"],
        "rudder_effort": metrics["rudder_effort"]["mean"],
        "propeller_effort": metrics["propeller_effort"]["mean"],
        "total_solve_time": metrics["total_solve_time"]["mean"],
        "solver_failures": int(round(metrics["solver_failures"]["mean"])),
        "total_steps": rep_run["total_steps"],
        "run_dir": rep_run.get("run_dir", ""),
    }
    return res_agg


def scavenge_all_existing_runs(cases: list, experiments_dir: str, max_runs_per_case: int = 10) -> dict:
    """Scavenges all matching experiment directories from experiments_dir, groups them per case,
    and returns aggregated statistics."""
    results = {}
    if not os.path.isdir(experiments_dir):
        print(f"[WARN] Experiments directory does not exist: {experiments_dir}")
        for case in cases:
            results[case["key"]] = create_empty_result(case["label"])
        return results

    all_dirs = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]

    for case in cases:
        key = case["key"]
        run_name = case["run_name"]
        matching_dirs = []

        for d in all_dirs:
            if d.startswith(run_name):
                full_d = os.path.join(experiments_dir, d)
                if (os.path.exists(os.path.join(full_d, "summary.json")) or
                    (os.path.exists(os.path.join(full_d, "telemetry.csv")) and os.path.getsize(os.path.join(full_d, "telemetry.csv")) > 100)):
                    matching_dirs.append(full_d)

        # Sort matching directories chronologically by modification time
        matching_dirs.sort(key=lambda d: os.path.getmtime(d))

        # Keep the most recent max_runs_per_case if set
        if max_runs_per_case > 0 and len(matching_dirs) > max_runs_per_case:
            matching_dirs = matching_dirs[-max_runs_per_case:]

        if not matching_dirs:
            print(f"[WARN] No matching experiment folders found for {case['label']}.")
            results[key] = create_empty_result(case["label"])
            continue

        print(f"[Scavenge] Found {len(matching_dirs)} run(s) for {case['label']}:")
        for md in matching_dirs:
            print(f"  - {os.path.basename(md)}")

        runs = [scavenge_experiment_run(md, case) for md in matching_dirs]
        results[key] = aggregate_case_runs(runs, case)

    # Automatically recover authentic scenario from matching experiment folders
    for case in cases:
        key = case["key"]
        rep = results.get(key, {}).get("representative", {})
        run_dir = rep.get("run_dir", "")
        if run_dir and os.path.isdir(run_dir):
            scen_f = os.path.join(run_dir, "scenario_copy.json")
            if os.path.exists(scen_f):
                try:
                    results["__scenario__"] = load_scenario(scen_f)
                    break
                except Exception:
                    pass

    return results


def run_suite_via_bringup(
    cases: list,
    params_path: str,
    scenario_path: str,
    experiments_dir: str,
    num_rounds: int = 1,
    master_seed: Optional[int] = 42,
    timeout: Optional[float] = None,
) -> dict:
    """Iterates through num_rounds of randomized disturbance seed sets.
    In each round, applies the common disturbance settings to sim_params.yaml,
    runs all 5 ablation cases sequentially one by one via bringup.launch.py with the explicit scenario_file,
    and collects results.
    """
    os.makedirs(experiments_dir, exist_ok=True)
    if timeout is None:
        scen_dict = load_scenario(scenario_path)
        timeout = float(scen_dict.get("sim_time", 800.0))
        print(f"  [Timeout] Dynamically resolved from scenario 'sim_time': {timeout:.1f}s")

    with open(params_path, "r") as f:
        original_yaml = f.read()

    round_disturbances = generate_round_disturbances(num_rounds, master_seed)
    case_runs = {c["key"]: [] for c in cases}
    authentic_scenario = None

    try:
        for r_idx, dist in enumerate(round_disturbances, start=1):
            print("\n" + "=" * 110)
            print(f"  >>> ABLATION STUDY REPETITION ROUND [{r_idx}/{num_rounds}] <<<")
            print(f"  Disturbance Realization:")
            print(f"    - Current Mean Heading : {dist['current_mean_heading']:.4f} rad ({math.degrees(dist['current_mean_heading']):.1f}°)")
            print(f"    - Current Drift Seed   : {dist['current_seed']}")
            print(f"    - Wave Seed            : {dist['wave_seed']}")
            print(f"    - Sensor Noise Seed    : {dist['sensor_seed']}")
            print("=" * 110)

            for i, case in enumerate(cases, start=1):
                key = case["key"]
                label = case["label"]
                base_run_name = case["run_name"]
                session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{base_run_name}_{session_tag}_r{r_idx:02d}" if num_rounds > 1 else f"{base_run_name}_{session_tag}"

                # Combine ablation disturbance toggles with this round's disturbance realization
                active_params = dict(case["params"])
                active_params.update({
                    "wave_seed": dist["wave_seed"],
                    "current_seed": dist["current_seed"],
                    "sensor_seed": dist["sensor_seed"],
                    "current_mean_heading": dist["current_mean_heading"],
                })

                print(f"\n----------------------------------------------------------------------------------")
                print(f"  [Round {r_idx}/{num_rounds} | Case {i}/{len(cases)}] Launching bringup for {label}")
                print(f"  Configuring sim_params.yaml -> run_name: '{run_name}'")
                print(f"  Scenario: '{scenario_path}'")
                print(f"  Parameters: {active_params}")
                print(f"----------------------------------------------------------------------------------")

                # 1. Update sim_params.yaml with params, run_name AND scenario_path
                update_sim_params(params_path, active_params, run_name=run_name, scenario_path=scenario_path)

                existing_dirs = set(os.listdir(experiments_dir)) if os.path.exists(experiments_dir) else set()

                # 2. Launch bringup.launch.py as an isolated process group passing explicit scenario_file
                cmd = [
                    "ros2", "launch", "nmpc_sim_nodes", "bringup.launch.py",
                    f"params_file:={params_path}",
                    f"scenario_file:={scenario_path}",
                ]
                print(f"  Executing: {' '.join(cmd)}")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                )

                # 3. Monitor for completion via summary.json in a newly created directory
                run_dir = None
                t_start = time.time()
                print("  Waiting for simulation run to complete...", end="", flush=True)

                while proc.poll() is None:
                    elapsed = time.time() - t_start
                    if elapsed > timeout:
                        print(f"\n  [WARN] Case '{label}' (Round {r_idx}) reached timeout limit ({timeout:.1f}s)!")
                        break

                    current_dirs = set(os.listdir(experiments_dir)) if os.path.exists(experiments_dir) else set()
                    new_dirs = [d for d in (current_dirs - existing_dirs) if d.startswith(run_name)]

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

                # 4. Gracefully terminate the launch process group
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

                # Settle pause for clean DDS node deregistration
                time.sleep(2.0)

                # 5. Fallback recovery on timeout: search for populated telemetry.csv created during THIS run
                if run_dir is None:
                    candidates = [
                        d for d in (os.listdir(experiments_dir) if os.path.exists(experiments_dir) else [])
                        if d.startswith(run_name) and os.path.getmtime(os.path.join(experiments_dir, d)) >= t_start - 2.0
                    ]
                    if candidates:
                        candidates.sort(key=lambda d: os.path.getmtime(os.path.join(experiments_dir, d)), reverse=True)
                        for cand in candidates:
                            cand_path = os.path.join(experiments_dir, cand)
                            telemetry_file = os.path.join(cand_path, "telemetry.csv")
                            if os.path.exists(telemetry_file) and os.path.getsize(telemetry_file) > 100:
                                run_dir = cand_path
                                print(f"  [Scavenge on Timeout] Recovered experiment telemetry from: {run_dir}")
                                break

                # 6. Scavenge the experiment data
                if run_dir and os.path.isdir(run_dir):
                    print(f"  [OK] Run finalized and logged at: {run_dir}")
                    res = scavenge_experiment_run(run_dir, case)
                    res["round"] = r_idx
                    res["disturbances"] = dist
                    case_runs[key].append(res)

                    scen_f = os.path.join(run_dir, "scenario_copy.json")
                    if os.path.exists(scen_f) and authentic_scenario is None:
                        try:
                            authentic_scenario = load_scenario(scen_f)
                        except Exception:
                            pass

                    print(f"  Status: {res['status']} | Sim Time: {res['final_t']:.1f}s | "
                          f"Max CT: {res['max_crosstrack']:.2f}m | CT RMSE: {res['crosstrack_rmse']:.2f}m")
                else:
                    print(f"  [ERROR] No completed experiment folder detected for {label} (Round {r_idx})!")
                    case_runs[key].append(create_empty_single_run(case))

    except KeyboardInterrupt:
        print("\n[WARN] Execution interrupted by user! Finalizing completed runs...")
    finally:
        print("\n[INFO] Restoring original sim_params.yaml...")
        restore_sim_params(params_path, original_yaml)
        print("[INFO] sim_params.yaml restored successfully.")

    # Aggregate all completed runs per case
    results = {}
    for case in cases:
        key = case["key"]
        runs = case_runs.get(key, [])
        runs.sort(key=lambda r: r.get("round", 1))
        results[key] = aggregate_case_runs(runs, case)

    if authentic_scenario:
        results["__scenario__"] = authentic_scenario
    elif os.path.isfile(scenario_path):
        results["__scenario__"] = load_scenario(scenario_path)

    return results


def compute_map_limits(scenario: dict, results: dict, cases: list, pad: float = 12.0):
    """Calculates unified, aspect-ratio preserved map limits covering waypoints, trajectories, and obstacles."""
    waypoints = scenario.get("waypoints", [])
    all_x = [float(w[0]) for w in waypoints] if waypoints else [0.0]
    all_y = [float(w[1]) for w in waypoints] if waypoints else [0.0]

    for c in cases:
        k = c["key"]
        res = results.get(k)
        if not res:
            continue
        rep = res.get("representative", res)
        if len(rep.get("x", [])) > 0:
            all_x.extend([float(v) for v in rep["x"]])
            all_y.extend([float(v) for v in rep["y"]])

    for obs in scenario.get("obstacles", []):
        ox, oy, orad = obs[:3]
        all_x.extend([ox - orad, ox + orad])
        all_y.extend([oy - orad, oy + orad])

    for wall in scenario.get("walls", []):
        x0, y0, x1, y1, rw = wall[:5]
        all_x.extend([x0 - rw, x0 + rw, x1 - rw, x1 + rw])
        all_y.extend([y0 - rw, y0 + rw, y1 - rw, y1 + rw])

    for ell in scenario.get("ellipses", []):
        xc, yc, a, b = ell[:4]
        max_r = max(a, b)
        all_x.extend([xc - max_r, xc + max_r])
        all_y.extend([yc - max_r, yc + max_r])

    for ship in scenario.get("obstacle_ships", []):
        if len(ship) > 5 and not ship[5]:
            continue
        xc, yc, a, b = ship[:4]
        max_r = max(a, b)
        all_x.extend([xc - max_r, xc + max_r])
        all_y.extend([yc - max_r, yc + max_r])

    x_min, x_max = float(min(all_x)) - pad, float(max(all_x)) + pad
    y_min, y_max = float(min(all_y)) - pad, float(max(all_y)) + pad

    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min) + 2.0
    fixed_xlim = (mid_y - half_span, mid_y + half_span)
    fixed_ylim = (mid_x - half_span, mid_x + half_span)
    return fixed_xlim, fixed_ylim


def draw_scenario_obstacles(ax, scenario: dict):
    """Draws circular, capsule (wall), elliptical, and obstacle ship obstacles from scenario.json on the plot."""
    for obs in scenario.get("obstacles", []):
        ox, oy, orad = obs[:3]
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
        enabled = bool(ship[5]) if len(ship) > 5 else True
        if not enabled:
            continue
        xc, yc, a, b, psi0 = ship[:5]
        t_vals = np.linspace(0, 2.0 * math.pi, 100)
        bx = a * np.cos(t_vals)
        by = b * np.sin(t_vals)
        ship_x = xc + bx * math.cos(psi0) - by * math.sin(psi0)
        ship_y = yc + bx * math.sin(psi0) + by * math.cos(psi0)
        ax.fill(ship_y, ship_x, color="steelblue", alpha=0.35, zorder=2)
        ax.plot(ship_y, ship_x, color="navy", linewidth=1.2, zorder=2)

    for wall in scenario.get("walls", []):
        x0, y0, x1, y1, r_wall = wall[:5]
        ex, ey = x1 - x0, y1 - y0
        length = float(math.hypot(ex, ey))
        if length > 1e-9:
            perp_x, perp_y = -ey / length, ex / length
            corners = [
                (y0 + perp_y * r_wall, x0 + perp_x * r_wall),
                (y1 + perp_y * r_wall, x1 + perp_x * r_wall),
                (y1 - perp_y * r_wall, x1 - perp_x * r_wall),
                (y0 - perp_y * r_wall, x0 - perp_x * r_wall),
            ]
            poly = plt.Polygon(corners, closed=True, color="firebrick", alpha=0.35, zorder=2)
            ax.add_patch(poly)
            ax.plot([c[0] for c in corners + [corners[0]]], [c[1] for c in corners + [corners[0]]],
                    color="darkred", linewidth=1.0, zorder=2)
        ax.add_patch(plt.Circle((y0, x0), r_wall, color="firebrick", alpha=0.35, zorder=2))
        ax.add_patch(plt.Circle((y1, x1), r_wall, color="firebrick", alpha=0.35, zorder=2))


def generate_static_graph(scenario: dict, results: dict, out_path: str, cases: list):
    """Generates a high-resolution graph showing all 5 ablation paths on the scenario map."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.94, bottom=0.08)

    fixed_xlim, fixed_ylim = compute_map_limits(scenario, results, cases)
    ax.set_xlim(fixed_xlim)
    ax.set_ylim(fixed_ylim)
    ax.set_aspect("equal")

    waypoints = scenario.get("waypoints", [])
    if waypoints:
        wp_x = [w[0] for w in waypoints]
        wp_y = [w[1] for w in waypoints]
        ax.plot(wp_y, wp_x, "g--", marker="x", markersize=8, linewidth=1.2, label="Waypoints", zorder=3)
        ax.plot(wp_y[0], wp_x[0], "go", markersize=7, label="Start", zorder=3)
        ax.plot(wp_y[-1], wp_x[-1], "g*", markersize=11, label="Goal Endpoint", zorder=3)

    draw_scenario_obstacles(ax, scenario)

    num_runs = 1
    for case in cases:
        key = case["key"]
        label = case["label"]
        color = case["color"]
        ls = case.get("linestyle", "-")
        res = results.get(key)
        if not res:
            continue
        rep = res.get("representative", res)
        if len(rep.get("x", [])) <= 1:
            continue
        num_runs = max(num_runs, res.get("num_runs", 1))
        rep_t = rep.get("final_t", res.get("final_t", 0.0))
        rep_status = rep.get("status", "COMPLETED")
        ax.plot(rep["y"], rep["x"], color=color, linestyle=ls, linewidth=1.8, alpha=0.9,
                label=f"{label} ({rep_status}, {rep_t:.1f}s)", zorder=4)

    title_suffix = f" (Representative Runs, N={num_runs} Seeds)" if num_runs > 1 else ""
    ax.set_xlabel("Y [Cross-track offset] (m)", fontsize=11)
    ax.set_ylabel("X [Along track] (m)", fontsize=11)
    ax.set_title(f"NMPC Waypoint Maneuvering - 5 Ablation Configurations{title_suffix}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, facecolor="white", edgecolor="#999999")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Static ablation trajectory graph saved: {out_path}")


def generate_looping_gif(scenario: dict, results: dict, out_path: str, bow_len: float = 2.9, subsample: int = 10, cases: list = None):
    """Generates an animated looping GIF showing live traversal, bow vectors, and predicted horizons."""
    if cases is None:
        cases = ABLATION_CASES
    active_keys = []
    reps = {}
    for c in cases:
        k = c["key"]
        if k in results:
            rep = results[k].get("representative", results[k])
            if len(rep.get("x", [])) > 1:
                active_keys.append(k)
                reps[k] = rep

    if not active_keys:
        print("[WARN] No trajectory data available to render GIF.")
        return

    num_runs = max((results[k].get("num_runs", 1) for k in active_keys), default=1)
    print(f"--- Rendering looping animated GIF (subsample={subsample})... ---")
    max_frames = max(len(reps[k]["x"]) for k in active_keys)
    frame_indices = list(range(0, max_frames, subsample))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    fixed_xlim, fixed_ylim = compute_map_limits(scenario, results, cases)

    waypoints = scenario.get("waypoints", [])
    wp_x = [w[0] for w in waypoints] if waypoints else [0.0]
    wp_y = [w[1] for w in waypoints] if waypoints else [0.0]

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

            rep = reps[key]
            n_samples = len(rep["x"])
            idx = min(step_k, n_samples - 1)

            # 1. Past path traversed
            ax.plot(rep["y"][:idx + 1], rep["x"][:idx + 1], color=color, linewidth=1.5, alpha=0.85,
                    label=label)

            # 2. Current vessel position marker
            curr_x, curr_y, curr_psi = rep["x"][idx], rep["y"][idx], rep["psi"][idx]
            ax.plot(curr_y, curr_x, marker="o", color=color, markersize=5, zorder=5)

            # 3. Heading bow line and directional arrow
            bow_x = curr_x + bow_len * math.cos(curr_psi)
            bow_y = curr_y + bow_len * math.sin(curr_psi)
            ax.plot([curr_y, bow_y], [curr_x, bow_x], color=color, linewidth=2.2, zorder=6)
            ax.annotate("", xy=(bow_y, bow_x), xytext=(curr_y, curr_x),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=12),
                        zorder=7)

            # 4. Future state horizon prediction
            if step_k < n_samples and idx < len(rep.get("horizons_x", [])):
                hz_x = rep["horizons_x"][idx]
                hz_y = rep["horizons_y"][idx]
                if len(hz_x) > 1:
                    ax.plot(hz_y, hz_x, linestyle=":", color=color, linewidth=1.2, alpha=0.85, zorder=4)

        title_suffix = f" (Representative Runs, N={num_runs} Seeds)" if num_runs > 1 else ""
        ax.set_xlabel("Y [Cross-track offset] (m)", fontsize=10)
        ax.set_ylabel("X [Along track] (m)", fontsize=10)
        ax.set_title(f"NMPC Live Traversal{title_suffix} | Sim Time: {sim_time_curr:5.1f} s",
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


def format_norm_cell(val: float, std: float, base_val: float, is_ref: bool = False) -> str:
    """Computes relative multiplier against baseline value with normalized standard deviation in parentheses."""
    if base_val <= 1e-9 or math.isnan(base_val) or math.isnan(val):
        return "N/A"
    ratio = val / base_val
    norm_std = std / base_val if (std is not None and not math.isnan(std) and base_val > 1e-9) else 0.0

    if norm_std > 1e-4:
        return f"{ratio:.2f}x (±{norm_std:.2f})"
    elif is_ref:
        return "1.00x (ref)"
    else:
        return f"{ratio:.2f}x"


def generate_summary_tables_text(results: dict, cases: list) -> str:
    """Generates formatted strings for:
    1. Table 1: Unified Performance (Method 1: Average Values with Relative % Change vs Baseline)
    2. Table 2: Detailed Statistical Dispersion (Mean ± Std, N runs per case)
    """
    num_runs = 1
    for c in cases:
        if c["key"] in results and "num_runs" in results[c["key"]]:
            num_runs = max(num_runs, results[c["key"]]["num_runs"])

    lines = []
    width = 186
    border = "=" * width
    divider = "-" * width

    base_res = results.get("baseline")
    base_m = base_res.get("metrics") if base_res else None

    def fmt_pct(num_val: float, base_val: float, fmt_spec: str, unit: str = "", is_ref: bool = False) -> str:
        formatted_val = format(num_val, fmt_spec)
        if is_ref:
            return f"{formatted_val}{unit} (ref)"
        if base_val == 0.0:
            return f"{formatted_val}{unit}"
        delta_pct = ((num_val - base_val) / base_val) * 100.0
        if delta_pct >= 0.05:
            pct_str = f"(+{delta_pct:.1f}%)"
        elif delta_pct <= -0.05:
            pct_str = f"({delta_pct:.1f}%)"
        else:
            pct_str = "(0.0%)"
        return f"{formatted_val}{unit} {pct_str}"

    # ---------------------------------------------------------
    # TABLE 1: UNIFIED ABSOLUTE & RELATIVE PERFORMANCE (METHOD 1)
    # ---------------------------------------------------------
    title1 = f"TABLE 1: NMPC ABLATION STUDY - UNIFIED PERFORMANCE (AVERAGE VALUES WITH RELATIVE % CHANGE VS BASELINE, N={num_runs})".center(width)
    header1 = (
        f"{'Configuration':<24} "
        f"{'Crosstrack RMSE':>18} "
        f"{'Max Crosstrack':>18} "
        f"{'Heading RMSE':>18} "
        f"{'Max Heading Err':>18} "
        f"{'Rudder Effort':>20} "
        f"{'Propeller Effort':>22} "
        f"{'Traversal Time':>18} "
        f"{'Success Rate':>14}"
    )

    lines.append("")
    lines.append(border)
    lines.append(title1)
    lines.append(border)
    lines.append(header1)
    lines.append(divider)

    for case in cases:
        key = case["key"]
        label = case["label"]
        res = results.get(key)
        if not res or res.get("num_runs", 0) == 0:
            lines.append(f"{label:<24} {'N/A':>18} {'N/A':>18} {'N/A':>18} {'N/A':>18} {'N/A':>20} {'N/A':>22} {'N/A':>18} {'N/A':>14}")
            continue

        m = res["metrics"]
        n_r = res.get("num_runs", 1)
        succ = res.get("successful_runs", n_r)
        is_ref = (key == "baseline")

        if base_m:
            s_ct = fmt_pct(m['crosstrack_rmse']['mean'], base_m['crosstrack_rmse']['mean'], ".2f", " m", is_ref)
            s_mct = fmt_pct(m['max_crosstrack']['mean'], base_m['max_crosstrack']['mean'], ".2f", " m", is_ref)
            s_head = fmt_pct(m['heading_rmse']['mean'], base_m['heading_rmse']['mean'], ".1f", "°", is_ref)
            s_mhead = fmt_pct(m['max_heading_err']['mean'], base_m['max_heading_err']['mean'], ".1f", "°", is_ref)
            s_rud = fmt_pct(m['rudder_effort']['mean'], base_m['rudder_effort']['mean'], ".1f", " rad²·s", is_ref)
            s_prop = fmt_pct(m['propeller_effort']['mean'], base_m['propeller_effort']['mean'], ".0f", " rps²·s", is_ref)
            s_time = fmt_pct(m['final_t']['mean'], base_m['final_t']['mean'], ".1f", " s", is_ref)
        else:
            s_ct = f"{m['crosstrack_rmse']['mean']:.2f} m"
            s_mct = f"{m['max_crosstrack']['mean']:.2f} m"
            s_head = f"{m['heading_rmse']['mean']:.1f}°"
            s_mhead = f"{m['max_heading_err']['mean']:.1f}°"
            s_rud = f"{m['rudder_effort']['mean']:.1f} rad²·s"
            s_prop = f"{m['propeller_effort']['mean']:.0f} rps²·s"
            s_time = f"{m['final_t']['mean']:.1f} s"

        succ_rate = f"{succ}/{n_r} ({100.0 * succ / n_r:.0f}%)"

        lines.append(
            f"{label:<24} "
            f"{s_ct:>18} "
            f"{s_mct:>18} "
            f"{s_head:>18} "
            f"{s_mhead:>18} "
            f"{s_rud:>20} "
            f"{s_prop:>22} "
            f"{s_time:>18} "
            f"{succ_rate:>14}"
        )

    lines.append(border)
    lines.append("")

    # ---------------------------------------------------------
    # TABLE 2: DETAILED STATISTICAL DISPERSION (MEAN ± STD)
    # ---------------------------------------------------------
    title2 = f"TABLE 2: NMPC ABLATION STUDY - STATISTICAL DISPERSION (MEAN ± STD, N={num_runs})".center(width)
    header2 = (
        f"{'Configuration':<24} "
        f"{'Crosstrack RMSE':>18} "
        f"{'Max RMSE':>14} "
        f"{'Max Crosstrack':>18} "
        f"{'Heading RMSE':>16} "
        f"{'Max Heading Err':>17} "
        f"{'Rudder Effort':>20} "
        f"{'Propeller Effort':>20} "
        f"{'Sim Time [s]':>15} "
        f"{'Success Rate':>14}"
    )

    lines.append(border)
    lines.append(title2)
    lines.append(border)
    lines.append(header2)
    lines.append(divider)

    for case in cases:
        key = case["key"]
        label = case["label"]
        res = results.get(key)
        if not res or res.get("num_runs", 0) == 0:
            lines.append(f"{label:<24} {'N/A':>18} {'N/A':>14} {'N/A':>18} {'N/A':>16} {'N/A':>17} {'N/A':>20} {'N/A':>20} {'N/A':>15} {'N/A':>14}")
            continue

        m = res["metrics"]
        n_r = res.get("num_runs", 1)
        succ = res.get("successful_runs", n_r)

        if n_r > 1:
            ct_rmse = f"{m['crosstrack_rmse']['mean']:.2f} ± {m['crosstrack_rmse']['std']:.2f} m"
            max_rmse = f"{m['crosstrack_rmse']['max']:.2f} m"
            max_ct = f"{m['max_crosstrack']['mean']:.2f} ± {m['max_crosstrack']['std']:.2f} m"
            head_rmse = f"{m['heading_rmse']['mean']:.1f} ± {m['heading_rmse']['std']:.1f}°"
            max_head = f"{m['max_heading_err']['mean']:.1f} ± {m['max_heading_err']['std']:.1f}°"
            rudder = f"{m['rudder_effort']['mean']:.1f} ± {m['rudder_effort']['std']:.1f} rad²·s"
            prop = f"{m['propeller_effort']['mean']:.1f} ± {m['propeller_effort']['std']:.1f} rps²·s"
            sim_t = f"{m['final_t']['mean']:.1f} ± {m['final_t']['std']:.1f} s"
        else:
            ct_rmse = f"{m['crosstrack_rmse']['mean']:.2f} m"
            max_rmse = f"{m['crosstrack_rmse']['mean']:.2f} m"
            max_ct = f"{m['max_crosstrack']['mean']:.2f} m"
            head_rmse = f"{m['heading_rmse']['mean']:.1f}°"
            max_head = f"{m['max_heading_err']['mean']:.1f}°"
            rudder = f"{m['rudder_effort']['mean']:.2f} rad²·s"
            prop = f"{m['propeller_effort']['mean']:.2f} rps²·s"
            sim_t = f"{m['final_t']['mean']:.1f} s"

        succ_rate = f"{succ}/{n_r} ({100.0 * succ / n_r:.0f}%)"

        lines.append(
            f"{label:<24} "
            f"{ct_rmse:>18} "
            f"{max_rmse:>14} "
            f"{max_ct:>18} "
            f"{head_rmse:>16} "
            f"{max_head:>17} "
            f"{rudder:>20} "
            f"{prop:>20} "
            f"{sim_t:>15} "
            f"{succ_rate:>14}"
        )

    lines.append(border)
    lines.append("")

    return "\n".join(lines)


def save_results_npz(results: dict, npz_path: str, cases: list):
    """Saves trajectory datasets, representative runs, and aggregated statistical metrics to an NPZ archive."""
    npz_data = {}
    for case in cases:
        key = case["key"]
        res = results.get(key)
        if not res:
            continue
        rep = res.get("representative", res)
        for field in ["x", "y", "psi", "t", "delta", "n", "e_ct", "e_psi"]:
            if field in rep:
                npz_data[f"{key}_{field}"] = np.array(rep[field])
        if "horizons_x" in rep:
            npz_data[f"{key}_horizons_x"] = np.array(rep["horizons_x"], dtype=object)
        if "horizons_y" in rep:
            npz_data[f"{key}_horizons_y"] = np.array(rep["horizons_y"], dtype=object)

        npz_data[f"{key}_num_runs"] = np.array([res.get("num_runs", 1)])
        npz_data[f"{key}_successful_runs"] = np.array([res.get("successful_runs", 1)])
        npz_data[f"{key}_status"] = np.array([rep.get("status", "COMPLETED")])
        npz_data[f"{key}_final_t"] = np.array([res.get("final_t", 0.0)])
        npz_data[f"{key}_total_steps"] = np.array([rep.get("total_steps", 0)])

        metrics = res.get("metrics", {})
        metric_names = [
            "crosstrack_rmse", "max_crosstrack", "heading_rmse", "max_heading_err",
            "rudder_effort", "propeller_effort", "total_solve_time", "solver_failures", "final_t"
        ]
        for metric in metric_names:
            if metric in metrics:
                m_info = metrics[metric]
                npz_data[f"{key}_{metric}_mean"] = np.array([m_info["mean"]])
                npz_data[f"{key}_{metric}_std"] = np.array([m_info["std"]])
                npz_data[f"{key}_{metric}_min"] = np.array([m_info["min"]])
                npz_data[f"{key}_{metric}_max"] = np.array([m_info["max"]])
                npz_data[f"{key}_{metric}_values"] = np.array(m_info["values"])
                npz_data[f"{key}_{metric}"] = np.array([m_info["mean"]])
            elif metric in res:
                npz_data[f"{key}_{metric}"] = np.array([res[metric]])

    # Persist the scenario used for these runs
    scen_to_save = None
    if "__scenario__" in results:
        scen_to_save = results["__scenario__"]
    elif cases and cases[0]["key"] in results:
        rep = results[cases[0]["key"]].get("representative", {})
        r_dir = rep.get("run_dir", "")
        if r_dir and os.path.exists(os.path.join(r_dir, "scenario_copy.json")):
            try:
                scen_to_save = load_scenario(os.path.join(r_dir, "scenario_copy.json"))
            except Exception:
                pass

    if scen_to_save:
        npz_data["scenario_json"] = np.array([json.dumps(scen_to_save)])

    np.savez_compressed(npz_path, **npz_data)
    print(f"[OK] Ablation results saved to archive: {npz_path}")


def load_results_npz(npz_path: str, cases: list) -> dict:
    """Loads trajectory datasets and aggregated statistical metrics from an NPZ archive."""
    data = np.load(npz_path, allow_pickle=True)
    results = {}
    metric_names = [
        "crosstrack_rmse", "max_crosstrack", "heading_rmse", "max_heading_err",
        "rudder_effort", "propeller_effort", "total_solve_time", "solver_failures", "final_t"
    ]

    for case in cases:
        key = case["key"]
        if f"{key}_x" not in data:
            continue

        num_runs = int(data[f"{key}_num_runs"][0]) if f"{key}_num_runs" in data else 1
        succ_runs = int(data[f"{key}_successful_runs"][0]) if f"{key}_successful_runs" in data else num_runs
        status_val = str(data[f"{key}_status"][0]) if f"{key}_status" in data else "COMPLETED"
        final_t_val = float(data[f"{key}_final_t"][0]) if f"{key}_final_t" in data else 0.0
        steps_val = int(data[f"{key}_total_steps"][0]) if f"{key}_total_steps" in data else 0

        metrics = {}
        for m in metric_names:
            if f"{key}_{m}_mean" in data:
                mean_v = float(data[f"{key}_{m}_mean"][0])
                std_v = float(data[f"{key}_{m}_std"][0]) if f"{key}_{m}_std" in data else 0.0
                min_v = float(data[f"{key}_{m}_min"][0]) if f"{key}_{m}_min" in data else mean_v
                max_v = float(data[f"{key}_{m}_max"][0]) if f"{key}_{m}_max" in data else mean_v
                vals = data[f"{key}_{m}_values"].tolist() if f"{key}_{m}_values" in data else [mean_v]
                metrics[m] = {"mean": mean_v, "std": std_v, "min": min_v, "max": max_v, "values": vals}
            elif f"{key}_{m}" in data:
                v = float(data[f"{key}_{m}"][0])
                metrics[m] = {"mean": v, "std": 0.0, "min": v, "max": v, "values": [v]}

        rep = {
            "x": data[f"{key}_x"].tolist(),
            "y": data[f"{key}_y"].tolist(),
            "psi": data[f"{key}_psi"].tolist(),
            "t": data[f"{key}_t"].tolist(),
            "delta": data[f"{key}_delta"].tolist() if f"{key}_delta" in data else [],
            "n": data[f"{key}_n"].tolist() if f"{key}_n" in data else [],
            "e_ct": data[f"{key}_e_ct"].tolist() if f"{key}_e_ct" in data else [],
            "e_psi": data[f"{key}_e_psi"].tolist() if f"{key}_e_psi" in data else [],
            "horizons_x": list(data[f"{key}_horizons_x"]) if f"{key}_horizons_x" in data else [],
            "horizons_y": list(data[f"{key}_horizons_y"]) if f"{key}_horizons_y" in data else [],
            "status": status_val,
            "final_t": final_t_val,
            "total_steps": steps_val,
        }

        results[key] = {
            "key": key,
            "label": case["label"],
            "num_runs": num_runs,
            "successful_runs": succ_runs,
            "metrics": metrics,
            "representative": rep,
            "x": rep["x"],
            "y": rep["y"],
            "psi": rep["psi"],
            "t": rep["t"],
            "horizons_x": rep["horizons_x"],
            "horizons_y": rep["horizons_y"],
            "status": rep["status"],
            "final_t": metrics.get("final_t", {}).get("mean", final_t_val),
            "crosstrack_rmse": metrics.get("crosstrack_rmse", {}).get("mean", 0.0),
            "max_crosstrack": metrics.get("max_crosstrack", {}).get("mean", 0.0),
            "heading_rmse": metrics.get("heading_rmse", {}).get("mean", 0.0),
            "max_heading_err": metrics.get("max_heading_err", {}).get("mean", 0.0),
            "rudder_effort": metrics.get("rudder_effort", {}).get("mean", 0.0),
            "propeller_effort": metrics.get("propeller_effort", {}).get("mean", 0.0),
            "total_solve_time": metrics.get("total_solve_time", {}).get("mean", 0.0),
            "solver_failures": int(round(metrics.get("solver_failures", {}).get("mean", 0))),
            "total_steps": steps_val,
        }

    if "scenario_json" in data:
        try:
            results["__scenario__"] = json.loads(str(data["scenario_json"][0]))
        except Exception:
            pass

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_scenario = find_default_scenario_path()
    default_params = _pkg_paths.sim_params_path()

    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of repetitions per ablation case with randomized disturbance seeds (default: 1)")
    parser.add_argument("--master-seed", type=int, default=42,
                        help="Master random seed for generating reproducible disturbance seed sets across rounds (default: 42)")
    parser.add_argument("--max-scavenge-runs", type=int, default=10,
                        help="Maximum runs per case to scavenge from experiments-dir (default: 10)")
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
                        help="Re-render graph and gif from existing ablation_results.npz if present")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)
    cfg = load_nmpc_config(args.params_path)
    bow_len = float(cfg.LPP)

    npz_path = os.path.join(args.results_dir, "ablation_results.npz")
    results = None
    explicit_scenario = bool(args.scenario_path and args.scenario_path.strip())
    active_scenario_path = os.path.abspath(args.scenario_path if explicit_scenario else default_scenario)
    if not os.path.isfile(active_scenario_path):
        raise FileNotFoundError(f"Scenario file not found at: {active_scenario_path}")

    if args.render_only and os.path.isfile(npz_path):
        print(f"\n[INFO] Loading existing results from {npz_path}...")
        results = load_results_npz(npz_path, ABLATION_CASES)
    elif args.scavenge:
        print(f"\n[INFO] Scavenging existing logs from {args.experiments_dir} (up to {args.max_scavenge_runs} runs per case)...")
        results = scavenge_all_existing_runs(ABLATION_CASES, args.experiments_dir, max_runs_per_case=args.max_scavenge_runs)
        save_results_npz(results, npz_path, ABLATION_CASES)
    else:
        print("==================================================================================")
        print("  RUNNING NMPC FUNCTION TESTING (5 ABLATION CONFIGURATIONS VIA BRINGUP SUITE)")
        print(f"  Reps per Case   : {args.num_runs}")
        print(f"  Master Seed     : {args.master_seed}")
        print(f"  Params File     : {args.params_path}")
        print(f"  Scenario File   : {active_scenario_path}")
        print(f"  Experiments Dir : {args.experiments_dir}")
        print(f"  Results Dir     : {args.results_dir}")
        timeout_disp = f"{args.timeout:.1f}s (explicit CLI)" if args.timeout is not None else f"{load_scenario(active_scenario_path).get('sim_time', 800.0):.1f}s (from scenario)"
        print(f"  Per-Run Timeout : {timeout_disp}")
        print("==================================================================================")
        results = run_suite_via_bringup(
            ABLATION_CASES,
            args.params_path,
            active_scenario_path,
            args.experiments_dir,
            num_rounds=args.num_runs,
            master_seed=args.master_seed,
            timeout=args.timeout,
        )
        save_results_npz(results, npz_path, ABLATION_CASES)

    # Resolve scenario: if not explicitly overridden by CLI, prefer authentic scenario from experiment logs
    if explicit_scenario:
        scenario = load_scenario(active_scenario_path)
        print(f"[INFO] Using explicitly requested scenario from CLI: {active_scenario_path}")
    elif results and "__scenario__" in results:
        scenario = results["__scenario__"]
        print(f"[INFO] Using authentic scenario recorded from simulation runs ({len(scenario.get('waypoints', []))} waypoints).")
    else:
        scenario = load_scenario(active_scenario_path)
        print(f"[INFO] Using resolved active scenario: {active_scenario_path}")

    # 1. Print formatted terminal tables (Absolute and Normalized) and save text report
    table_text_path = os.path.join(args.results_dir, "ablation_table.txt")
    summary_text = generate_summary_tables_text(results, ABLATION_CASES)
    print(summary_text)
    with open(table_text_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"[OK] Summary performance tables saved: {table_text_path}")

    # 2. Generate static trajectory comparison graph (using representative seeds)
    graph_path = os.path.join(args.results_dir, "ablation_paths.png")
    generate_static_graph(scenario, results, graph_path, ABLATION_CASES)

    # 3. Generate smooth looping animated GIF (using representative seeds)
    gif_path = os.path.join(args.results_dir, "ablation_animation.gif")
    generate_looping_gif(scenario, results, gif_path, bow_len=bow_len, subsample=args.subsample, cases=ABLATION_CASES)

    print("\n==================================================================================")
    print("  NMPC FUNCTION TESTING COMPLETE")
    print(f"  1) Comparison Graph : {graph_path}")
    print(f"  2) Looping GIF      : {gif_path}")
    print(f"  3) Saved Results NPZ: {npz_path}")
    print(f"  4) Tables Report    : {table_text_path}")
    print("==================================================================================\n")


if __name__ == "__main__":
    main()
