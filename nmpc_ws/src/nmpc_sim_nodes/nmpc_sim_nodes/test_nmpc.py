"""
Phase 2 validation harness — closed-loop simulation with NO obstacles.
Drives the true (numeric) MMG plant with the NMPC's optimal first control
action each step, exactly as an outer control loop would in deployment.

Run: ros2 run nmpc_sim_nodes test_nmpc
"""
import os
import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from casadi_mmg_solver.casadi_mmg import make_casadi_integrator  # noqa: E402
from nmpc.config import DEFAULT_CONFIG  # noqa: E402
from nmpc.path_following import (  # noqa: E402
    compute_path_angle, compute_cross_track_error, compute_sideslip,
    compute_course_angle, compute_course_error, select_active_waypoint,
)

RESULTS_DIR = os.path.expanduser("~/nmpc_sim_logs/test_nmpc_results")


def run_scenario(nmpc_solver, waypoints, mmg_init, sim_time, config=DEFAULT_CONFIG,
                  delta_init=0.0, n_init=None, label=""):
    """Closed-loop rollout. nmpc_solver must expose .solve(mmg_state, delta, n,
    chi_p, x_d, y_d, obstacles=[]) -> dict with u_opt/delta/n/solve_time/success."""
    if n_init is None:
        n_init = config.N_TRIM

    plant_step = make_casadi_integrator(config.dt, method="rk4", sym_type=ca.SX)

    mmg_state = np.array(mmg_init, dtype=float)
    delta, n = float(delta_init), float(n_init)
    target_idx = 1

    n_steps = int(sim_time / config.dt)
    log = {k: [] for k in ["t", "x", "y", "psi", "u", "v", "r",
                            "e_y", "psi_e", "delta", "n", "solve_time", "success"]}

    print(f"--- Running {label}: {n_steps} steps, dt={config.dt}s, T={sim_time}s ---")
    t0_wall = time.perf_counter()
    for step in range(n_steps):
        prev_wp = waypoints[target_idx - 1]
        target_wp = waypoints[target_idx]
        chi_p = compute_path_angle(prev_wp, target_wp)
        x_d, y_d = target_wp

        # ask the NMPC for the next actuator command given the true current state
        result = nmpc_solver.solve(mmg_state.tolist(), delta, n, chi_p, x_d, y_d, obstacles=[])
        delta, n = result["delta"], result["n"]

        # step the REAL (numeric) MMG plant forward with that command
        state_ca = ca.DM(mmg_state)
        control_ca = ca.DM([delta, n])
        next_state, _ = plant_step(state_ca, control_ca)
        mmg_state = np.array(next_state).flatten()
        u, v, r, x, y, psi = mmg_state

        e_y = compute_cross_track_error(x, y, x_d, y_d, chi_p)
        beta = compute_sideslip(u, v)
        chi = compute_course_angle(psi, beta)
        psi_e = compute_course_error(chi, chi_p)

        log["t"].append(step * config.dt)
        log["x"].append(x); log["y"].append(y); log["psi"].append(psi)
        log["u"].append(u); log["v"].append(v); log["r"].append(r)
        log["e_y"].append(e_y); log["psi_e"].append(psi_e)
        log["delta"].append(delta); log["n"].append(n)
        log["solve_time"].append(result["solve_time"])
        log["success"].append(result["success"])

        target_idx = select_active_waypoint(x, y, waypoints, target_idx, config.WP_RADIUS)  # advance if wp reached

        if not result["success"]:
            print(f"  [step {step}] solver FAILED: {result['return_status']}")

    wall = time.perf_counter() - t0_wall
    avg_solve = np.mean(log["solve_time"])
    print(f"  done in {wall:.1f}s wall time, avg solve_time={avg_solve*1000:.1f}ms, "
          f"max={max(log['solve_time'])*1000:.1f}ms, "
          f"success_rate={100*np.mean(log['success']):.1f}%")

    return {k: np.array(v) for k, v in log.items()}


# ------------------------------------------------------------------ plotting
def plot_trajectory(log, waypoints, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    wp = np.array(waypoints)
    ax.plot(wp[:, 1], wp[:, 0], "k:", alpha=0.5, label="Reference path")
    ax.plot(wp[:, 1], wp[:, 0], "kx", markersize=8)
    ax.plot(log["y"], log["x"], "b-", linewidth=1.5, label="Ship trajectory")
    ax.plot(log["y"][0], log["x"][0], "go", markersize=9, label="Start")
    ax.plot(log["y"][-1], log["x"][-1], "r*", markersize=12, label="End")
    ax.set_xlabel("Y (m)"); ax.set_ylabel("X (m)")
    ax.set_title(title); ax.axis("equal"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_errors(log, out_path, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(log["t"], log["e_y"], "b-")
    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1.set_ylabel("e_y (m)"); ax1.grid(True, linestyle="--", alpha=0.5)
    ax2.plot(log["t"], np.rad2deg(log["psi_e"]), "r-")
    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_ylabel("psi_e (deg)"); ax2.set_xlabel("Time (s)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_controls(log, out_path, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(log["t"], np.rad2deg(log["delta"]), "g-")
    ax1.set_ylabel("Rudder delta (deg)"); ax1.grid(True, linestyle="--", alpha=0.5)
    ax2.plot(log["t"], log["n"], "m-")
    ax2.set_ylabel("Propeller n (rps)"); ax2.set_xlabel("Time (s)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_speed(log, u_ref, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log["t"], log["u"], "b-", label="u (surge)")
    ax.axhline(u_ref, color="r", linestyle="--", label="U_ref")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("u (m/s)")
    ax.set_title(title); ax.grid(True, linestyle="--", alpha=0.5); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ tests
def test1_straight_aligned(nmpc_solver, config=DEFAULT_CONFIG):
    """Straight path, ship starts on it. Checks basic tracking stability."""
    waypoints = [(0.0, 0.0), (0.0, -30.0)]
    # mmg_init = [u, v, r, x, y, psi]
    mmg_init = [0.78, 1.5, 0.0, 0.0, 0.0, 0.0]
    log = run_scenario(nmpc_solver, waypoints, mmg_init, sim_time=450.0, config=config, label="Test 1")

    plot_trajectory(log, waypoints, os.path.join(RESULTS_DIR, "test1_trajectory.png"),
                     "Test 1: Straight line, aligned start")
    plot_errors(log, os.path.join(RESULTS_DIR, "test1_errors.png"), "Test 1: Errors")
    plot_controls(log, os.path.join(RESULTS_DIR, "test1_controls.png"), "Test 1: Controls")

    ok = bool(np.all(np.abs(log["e_y"][log["t"] > 15.0]) < 1.0))
    print(f"Test 1 [e_y < 1.0m after settling]: {'PASS' if ok else 'FAIL'} "
          f"(max |e_y| after t=15s: {np.max(np.abs(log['e_y'][log['t']>15.0])):.3f}m)")
    return log, ok


def test2_offset_convergence(nmpc_solver, config=DEFAULT_CONFIG):
    """Ship starts 1.5m off the path. Checks it corrects back within ~10s."""
    waypoints = [(0.0, 0.0), (0.0, +30.0)]
    # mmg_init = [u, v, r, x, y, psi]
    mmg_init = [0.78, 0.0, 0.0, 15.5, 0.0, 0.0]
    log = run_scenario(nmpc_solver, waypoints, mmg_init, sim_time=250.0, config=config, label="Test 2")

    plot_trajectory(log, waypoints, os.path.join(RESULTS_DIR, "test2_convergence.png"),
                     "Test 2: Offset start (1.5m) convergence")

    mask_10s = log["t"] >= 10.0
    ok = bool(np.any(mask_10s) and np.max(np.abs(log["e_y"][mask_10s])) < 0.3)
    print(f"Test 2 [|e_y| < 0.3m by t=10s]: {'PASS' if ok else 'FAIL'} "
          f"(|e_y| at t=10s: {np.abs(log['e_y'][mask_10s][0]) if np.any(mask_10s) else float('nan'):.3f}m)")
    return log, ok


def test3_two_waypoint_turn(nmpc_solver, config=DEFAULT_CONFIG):
    """Two-leg path with a turn. Checks waypoint switching + turn tracking."""
    waypoints = [(0.0, 0.0), (5.0, -10.0), (5.0, -30.0)]
    # mmg_init = [u, v, r, x, y, psi]
    mmg_init = [0.78, 0.0, 0.0, 0.0, 0.0, 0.0]
    log = run_scenario(nmpc_solver, waypoints, mmg_init, sim_time=550.0, config=config, label="Test 3")

    plot_trajectory(log, waypoints, os.path.join(RESULTS_DIR, "test3_turn.png"),
                     "Test 3: Two-waypoint turn")

    ok = bool(np.all(np.abs(log["e_y"][log["t"] > 45.0]) < 2.0))
    print(f"Test 3 [smooth transition, e_y < 2.0m near end]: {'PASS' if ok else 'FAIL'} "
          f"(max |e_y| after t=45s: {np.max(np.abs(log['e_y'][log['t']>45.0])):.3f}m)")
    return log, ok


def test4_speed_tracking(log1, config=DEFAULT_CONFIG):
    """Reuses Test 1's log to check surge speed converges to U_REF."""
    plot_speed(log1, config.U_REF, os.path.join(RESULTS_DIR, "test4_speed.png"),
               "Test 4: Speed tracking (reuses Test 1 rollout)")

    mask = log1["t"] > 15.0
    ok = bool(np.max(np.abs(log1["u"][mask] - config.U_REF)) < 0.15)
    print(f"Test 4 [u tracks U_ref within 0.15 m/s after settling]: {'PASS' if ok else 'FAIL'} "
          f"(max |u-U_ref| after t=15s: {np.max(np.abs(log1['u'][mask]-config.U_REF)):.3f})")
    return ok


def main(nmpc_solver=None, config=DEFAULT_CONFIG, solver_name="AcadosNMPC"):
    """Runs all 4 tests."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if nmpc_solver is None:
        from nmpc.nmpc_acados import AcadosNMPC
        nmpc_solver = AcadosNMPC(config)

    print(f"\n=========== NMPC validation: {solver_name} ===========\n")

    results = {}
    log1, ok1 = test1_straight_aligned(nmpc_solver, config)
    results["test1"] = ok1
    log2, ok2 = test2_offset_convergence(nmpc_solver, config)
    results["test2"] = ok2
    log3, ok3 = test3_two_waypoint_turn(nmpc_solver, config)
    results["test3"] = ok3
    ok4 = test4_speed_tracking(log1, config)
    results["test4"] = ok4

    print("\n=========== SUMMARY ===========")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_pass = all(results.values())
    print(f"  ALL: {'PASS' if all_pass else 'FAIL'}")
    return results, {"test1": log1, "test2": log2, "test3": log3}


if __name__ == "__main__":
    main()
