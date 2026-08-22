"""experiment_logger: Reusable experiment logger for NMPC simulation runs.
Saves complete run artifacts:
- metadata.json: Run context, Git state, and system parameters
- scenario_copy.json: Snapshot of waypoints, obstacles, and initial state
- telemetry.csv: High-frequency timeseries (true state, measured state, control commands,
                 guidance reference, environmental disturbances, solver status)
- prediction_horizons.npz: Compressed 3D tensors of predicted state & control horizons
- summary.json: Aggregate KPIs, solver latency statistics, and trajectory endpoints
"""

import csv
from datetime import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy types, tuples, and non-serializable objects to json-safe types."""
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean_for_json(obj.tolist())
    return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj


def _get_git_info(search_dir: str) -> Dict[str, Any]:
    """Extract current git commit, branch, and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=search_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=search_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=search_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {
            "commit": commit,
            "branch": branch,
            "dirty": len(status) > 0,
        }
    except Exception as e:
        return {"commit": "unknown", "branch": "unknown", "dirty": False, "error": str(e)}


def _find_repo_root(start_dir: str) -> str:
    """Traverse upwards from start_dir to find repo root containing .git or nmpc_ws."""
    curr = os.path.abspath(start_dir)
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".git")) or os.path.exists(os.path.join(curr, "nmpc_ws")):
            return curr
        curr = os.path.dirname(curr)
    return os.path.expanduser("~")


class ExperimentLogger:
    """Synchronous / streaming experiment logger for NMPC runs."""

    CSV_HEADER = [
        "time_s",
        # True State (Plant)
        "true_u_m_s",
        "true_v_m_s",
        "true_r_rad_s",
        "true_x_m",
        "true_y_m",
        "true_psi_rad",
        "true_psi_deg",
        "true_delta_rad",
        "true_delta_deg",
        "true_n_rps",
        # Measured State (Sensor model output)
        "meas_u_m_s",
        "meas_v_m_s",
        "meas_r_rad_s",
        "meas_x_m",
        "meas_y_m",
        "meas_psi_rad",
        "meas_psi_deg",
        "meas_delta_rad",
        "meas_delta_deg",
        "meas_n_rps",
        # Actuator Command (NMPC output)
        "cmd_delta_rad",
        "cmd_delta_deg",
        "cmd_n_rps",
        # Active Guidance Reference
        "target_wp_idx",
        "target_x_m",
        "target_y_m",
        "chi_p_rad",
        "chi_p_deg",
        # Environmental Disturbances
        "current_enabled",
        "current_vx_m_s",
        "current_vy_m_s",
        "current_speed_m_s",
        "current_heading_rad",
        "current_heading_deg",
        "wave_enabled",
        "wave_fx_N",
        "wave_fy_N",
        "wave_fn_Nm",
        # Solver Diagnostics
        "solver_success",
        "solver_return_status",
        "solve_time_ms",
    ]

    def __init__(
        self,
        log_dir: Optional[str] = None,
        run_name: str = "nmpc_experiment",
        parameters: Optional[Dict[str, Any]] = None,
        scenario: Optional[Dict[str, Any]] = None,
        save_horizons_npz: bool = True,
    ):
        self.repo_dir = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))
        if log_dir is None:
            self.base_log_dir = os.path.join(self.repo_dir, "nmpc_sim_logs", "experiments")
        else:
            self.base_log_dir = os.path.expanduser(log_dir)

        self.run_name = run_name
        self.save_horizons_npz = save_horizons_npz
        self.start_wall_time = time.time()
        self.start_iso = datetime.now().isoformat()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_log_dir, f"{run_name}_{timestamp_str}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.metadata = {
            "run_name": self.run_name,
            "start_time": self.start_iso,
            "log_dir": self.run_dir,
            "git": _get_git_info(self.repo_dir),
            "platform": platform.platform(),
            "python_version": sys.version,
            "parameters": _clean_for_json(parameters or {}),
        }
        self._write_json(os.path.join(self.run_dir, "metadata.json"), self.metadata)

        if scenario is not None:
            self.set_scenario(scenario)

        # Open Telemetry CSV
        self.csv_path = os.path.join(self.run_dir, "telemetry.csv")
        self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADER)
        self._csv_file.flush()

        # In-memory Horizon & Metric Buffers
        self.horizon_timestamps: List[float] = []
        self.xi_trajs: List[np.ndarray] = []
        self.control_horizons: List[np.ndarray] = []

        self.solve_times_ms: List[float] = []
        self.solver_successes: List[bool] = []
        self.trajectory_x: List[float] = []
        self.trajectory_y: List[float] = []
        self.sim_times: List[float] = []

        self.step_count = 0
        self._finalized = False

        print(f"[ExperimentLogger] Initialized run at: {self.run_dir}")

    def set_scenario(self, scenario: Dict[str, Any]):
        """Persist a copy of the scenario configuration."""
        scenario_clean = _clean_for_json(scenario)
        self._write_json(os.path.join(self.run_dir, "scenario_copy.json"), scenario_clean)

    def _write_json(self, filepath: str, data: Any):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def log_step(
        self,
        time_s: float,
        true_state: Tuple[float, float, float, float, float, float, float, float],
        meas_state: Optional[Tuple[float, float, float, float, float, float, float, float]] = None,
        cmd: Optional[Tuple[float, float]] = None,
        active_ref: Optional[Tuple[int, float, float, float]] = None,
        current: Optional[Tuple[bool, float, float, float, float]] = None,
        wave: Optional[Tuple[bool, float, float, float]] = None,
        solver_status: Optional[Tuple[bool, str, float]] = None,
        prediction_horizon: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """Record a single step of simulation telemetry and prediction data.

        true_state: (u, v, r, x, y, psi, delta, n)
        meas_state: (u, v, r, x, y, psi, delta, n) or None (falls back to true_state)
        cmd: (delta_cmd, n_cmd) or None
        active_ref: (target_idx, x_d, y_d, chi_p) or None
        current: (enabled, vx, vy, speed, heading) or None
        wave: (enabled, fx, fy, fn) or None
        solver_status: (success, return_status, solve_time_s) or None
        prediction_horizon: (xi_traj [11, N+1], control_horizon [2, N]) or None
        """
        if self._finalized:
            return

        u_t, v_t, r_t, x_t, y_t, psi_t, delta_t, n_t = true_state
        psi_t_deg = math.degrees(psi_t)
        delta_t_deg = math.degrees(delta_t)

        if meas_state is not None:
            u_m, v_m, r_m, x_m, y_m, psi_m, delta_m, n_m = meas_state
        else:
            u_m, v_m, r_m, x_m, y_m, psi_m, delta_m, n_m = true_state
        psi_m_deg = math.degrees(psi_m)
        delta_m_deg = math.degrees(delta_m)

        if cmd is not None:
            cmd_delta, cmd_n = cmd
            cmd_delta_deg = math.degrees(cmd_delta)
        else:
            cmd_delta, cmd_delta_deg, cmd_n = delta_t, delta_t_deg, n_t

        if active_ref is not None:
            tgt_idx, x_d, y_d, chi_p = active_ref
            chi_p_deg = math.degrees(chi_p)
        else:
            tgt_idx, x_d, y_d, chi_p, chi_p_deg = -1, 0.0, 0.0, 0.0, 0.0

        if current is not None:
            c_en, c_vx, c_vy, c_spd, c_hdg = current
            c_hdg_deg = math.degrees(c_hdg)
        else:
            c_en, c_vx, c_vy, c_spd, c_hdg, c_hdg_deg = False, 0.0, 0.0, 0.0, 0.0, 0.0

        if wave is not None:
            w_en, w_fx, w_fy, w_fn = wave
        else:
            w_en, w_fx, w_fy, w_fn = False, 0.0, 0.0, 0.0

        if solver_status is not None:
            s_succ, s_stat, s_time = solver_status
            s_time_ms = s_time * 1000.0
        else:
            s_succ, s_stat, s_time_ms = True, "OK", 0.0

        row = [
            f"{time_s:.4f}",
            f"{u_t:.6f}", f"{v_t:.6f}", f"{r_t:.6f}", f"{x_t:.6f}", f"{y_t:.6f}",
            f"{psi_t:.6f}", f"{psi_t_deg:.4f}", f"{delta_t:.6f}", f"{delta_t_deg:.4f}", f"{n_t:.6f}",
            f"{u_m:.6f}", f"{v_m:.6f}", f"{r_m:.6f}", f"{x_m:.6f}", f"{y_m:.6f}",
            f"{psi_m:.6f}", f"{psi_m_deg:.4f}", f"{delta_m:.6f}", f"{delta_m_deg:.4f}", f"{n_m:.6f}",
            f"{cmd_delta:.6f}", f"{cmd_delta_deg:.4f}", f"{cmd_n:.6f}",
            tgt_idx, f"{x_d:.6f}", f"{y_d:.6f}", f"{chi_p:.6f}", f"{chi_p_deg:.4f}",
            int(c_en), f"{c_vx:.6f}", f"{c_vy:.6f}", f"{c_spd:.6f}", f"{c_hdg:.6f}", f"{c_hdg_deg:.4f}",
            int(w_en), f"{w_fx:.6f}", f"{w_fy:.6f}", f"{w_fn:.6f}",
            int(s_succ), s_stat, f"{s_time_ms:.4f}",
        ]

        self._csv_writer.writerow(row)
        if self.step_count % 10 == 0:
            self._csv_file.flush()

        self.sim_times.append(time_s)
        self.trajectory_x.append(x_t)
        self.trajectory_y.append(y_t)
        self.solve_times_ms.append(s_time_ms)
        self.solver_successes.append(bool(s_succ))

        if self.save_horizons_npz and prediction_horizon is not None:
            xi_traj, u_opt = prediction_horizon
            self.horizon_timestamps.append(time_s)
            self.xi_trajs.append(np.asarray(xi_traj, dtype=np.float32))
            self.control_horizons.append(np.asarray(u_opt, dtype=np.float32))

        self.step_count += 1

    def finalize(self, status: str = "UNKNOWN", goal_pos: Optional[Tuple[float, float]] = None,
                 extra: Optional[Dict[str, Any]] = None):
        """Flush and finalize experiment log files.

        extra: optional dict merged as top-level keys into summary.json before
        it's written -- e.g. logger_node passes ControllerEffortLogger.finalize()'s
        {"costs_errors": {...}} fragment here, so costs_errors.csv's min/max/sum/avg
        breakdown lands in the same summary.json write this method already does,
        instead of a second file or a separate cross-process merge."""
        if self._finalized:
            return
        self._finalized = True

        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass

        # Save prediction horizons NPZ if enabled
        if self.save_horizons_npz and len(self.xi_trajs) > 0:
            npz_path = os.path.join(self.run_dir, "prediction_horizons.npz")
            try:
                np.savez_compressed(
                    npz_path,
                    timestamps=np.array(self.horizon_timestamps, dtype=np.float32),
                    xi_trajs=np.array(self.xi_trajs, dtype=np.float32),
                    control_horizons=np.array(self.control_horizons, dtype=np.float32),
                )
            except Exception as e:
                print(f"[ExperimentLogger] Failed to save prediction_horizons.npz: {e}")

        # Compute Summary
        wall_duration = time.time() - self.start_wall_time
        final_sim_time = self.sim_times[-1] if self.sim_times else 0.0

        n_solves = len(self.solve_times_ms)
        n_succ = sum(1 for s in self.solver_successes if s)
        succ_rate = (n_succ / n_solves * 100.0) if n_solves > 0 else 0.0

        solve_mean = float(np.mean(self.solve_times_ms)) if n_solves > 0 else 0.0
        solve_std = float(np.std(self.solve_times_ms)) if n_solves > 0 else 0.0
        solve_min = float(np.min(self.solve_times_ms)) if n_solves > 0 else 0.0
        solve_max = float(np.max(self.solve_times_ms)) if n_solves > 0 else 0.0
        solve_median = float(np.median(self.solve_times_ms)) if n_solves > 0 else 0.0

        start_pos = [self.trajectory_x[0], self.trajectory_y[0]] if self.trajectory_x else None
        final_pos = [self.trajectory_x[-1], self.trajectory_y[-1]] if self.trajectory_x else None
        dist_to_goal = None
        if final_pos is not None and goal_pos is not None:
            dist_to_goal = float(math.hypot(final_pos[0] - goal_pos[0], final_pos[1] - goal_pos[1]))

        summary = {
            "run_name": self.run_name,
            "status": status,
            "start_time": self.start_iso,
            "end_time": datetime.now().isoformat(),
            "wall_duration_s": round(wall_duration, 3),
            "sim_time_s": round(final_sim_time, 3),
            "total_steps": self.step_count,
            "effective_speedup": round(final_sim_time / max(wall_duration, 1e-6), 2),
            "solver_performance": {
                "total_solves": n_solves,
                "successful_solves": n_succ,
                "success_rate_pct": round(succ_rate, 2),
                "mean_solve_time_ms": round(solve_mean, 3),
                "std_solve_time_ms": round(solve_std, 3),
                "min_solve_time_ms": round(solve_min, 3),
                "max_solve_time_ms": round(solve_max, 3),
                "median_solve_time_ms": round(solve_median, 3),
            },
            "trajectory": {
                "start_position": start_pos,
                "final_position": final_pos,
                "goal_position": list(goal_pos) if goal_pos else None,
                "final_distance_to_goal_m": round(dist_to_goal, 3) if dist_to_goal is not None else None,
            },
        }

        if extra:
            summary.update(extra)

        self._write_json(os.path.join(self.run_dir, "summary.json"), summary)

        print("\n" + "=" * 70)
        print(f"[ExperimentLogger] Experiment Run Finalized: {self.run_name}")
        print(f"[ExperimentLogger] Output Folder : {self.run_dir}")
        print(f"[ExperimentLogger] Final Status  : {status}")
        print(f"[ExperimentLogger] Steps / Sim T : {self.step_count} steps | {final_sim_time:.1f}s (wall: {wall_duration:.1f}s, speedup: {summary['effective_speedup']}x)")
        print(f"[ExperimentLogger] Solver Latency: avg {solve_mean:.2f}ms (min {solve_min:.2f}ms, max {solve_max:.2f}ms, succ {succ_rate:.1f}%)")
        if dist_to_goal is not None:
            print(f"[ExperimentLogger] Goal Distance : {dist_to_goal:.2f}m")
        print("=" * 70 + "\n")
