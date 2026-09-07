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

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from casadi_mmg_solver.casadi_mmg import make_casadi_integrator  # noqa: E402
from nmpc.params import DEFAULT_CONFIG  # noqa: E402
from nmpc.path_following import (  # noqa: E402
    compute_path_angle, compute_cross_track_error, compute_sideslip,
    compute_course_angle, compute_course_error, select_active_waypoint,
    segments_from_waypoints, SegmentQueue,
)

RESULTS_DIR = os.path.join(_pkg_paths.repo_root(), "nmpc_sim_logs", "test_nmpc_results")


def run_scenario(nmpc_solver, waypoints, mmg_init, sim_time, config=DEFAULT_CONFIG,
                  delta_init=0.0, n_init=None, label="", obstacles=None, walls=None, ellipses=None):
    """Closed-loop rollout. nmpc_solver must expose .solve(mmg_state, delta, n,
    segments, obstacles=[], walls=[], ellipses=[]) -> dict with
    u_opt/delta/n/solve_time/success, where segments is an ordered
    [(chi_p, end_x, end_y), ...] list (see SegmentQueue). obstacles/walls/ellipses
    default to none, matching every test in this file except test7/test8."""
    if n_init is None:
        n_init = config.N_TRIM
    if obstacles is None:
        obstacles = []
    if walls is None:
        walls = []
    if ellipses is None:
        ellipses = []

    plant_step = make_casadi_integrator(config.dt, method="rk4", sym_type=ca.SX)

    mmg_state = np.array(mmg_init, dtype=float)
    delta, n = float(delta_init), float(n_init)
    target_idx = 1
    # Live queue of active path segments the NMPC horizon previews -- mirrors
    # map_node's own SegmentQueue usage (nmpc/README.md item 8), seeded once
    # with the whole remaining path and popped from the front as waypoints
    # are reached, in lockstep with target_idx below.
    segment_queue = SegmentQueue(segments_from_waypoints(waypoints, target_idx))

    n_steps = int(sim_time / config.dt)
    log = {k: [] for k in ["t", "x", "y", "psi", "u", "v", "r",
                            "e_y", "psi_e", "delta", "n", "solve_time", "success",
                            "target_idx", "n_active_segments"]}

    print(f"--- Running {label}: {n_steps} steps, dt={config.dt}s, T={sim_time}s ---")
    t0_wall = time.perf_counter()
    for step in range(n_steps):
        prev_wp = waypoints[target_idx - 1]
        target_wp = waypoints[target_idx]
        chi_p = compute_path_angle(prev_wp, target_wp)
        x_d, y_d = target_wp

        # ask the NMPC for the next actuator command given the true current state
        result = nmpc_solver.solve(mmg_state.tolist(), delta, n, segment_queue.segments,
                                    obstacles=obstacles, walls=walls, ellipses=ellipses)
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
        segment_queue.pop_crossed(x, y, config.WP_RADIUS)  # kept in lockstep, see SegmentQueue docstring
        log["target_idx"].append(target_idx)
        log["n_active_segments"].append(len(segment_queue))

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
    # mmg_init = [u, v, r, x, y, psi] -- psi=-pi/2 (not 0) to actually match
    # this leg's own path heading (compute_path_angle((0,0),(0,-30)) = -90deg);
    # psi=0 was a 90deg heading/path mismatch present since this file's very
    # first commit (6c68d8d) despite the test's name/intent ("aligned start").
    # NOTE: fixing psi alone does not make this test pass -- verified via a
    # standalone rollout that with the aligned heading, e_y tracks well for
    # the first ~9s then a separate divergence kicks in (rudder/thrust pin at
    # max, e_y grows past -5m by t=29s) well before the ship is anywhere near
    # the target waypoint. That divergence traces to this test's large
    # v=1.5 m/s initial sway (test2, which has v=0, does not show it even
    # with its own similar heading-mismatch bug fixed) -- a real, separate,
    # pre-existing issue, not something this session's sharp-turn/
    # waypoint-passage work touches. Left as-is (not chased further here);
    # this fix only corrects the unambiguous heading-mismatch typo.
    mmg_init = [0.78, 1.5, 0.0, 0.0, 0.0, -np.pi / 2]
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
    # mmg_init = [u, v, r, x, y, psi] -- two fixes vs the original, both
    # present since this file's very first commit (6c68d8d):
    # (1) x=1.5 (not 15.5) to actually match this test's own docstring/name
    #     ("starts 1.5m off the path") -- x=15.5 was a typo; a 15.5m offset
    #     made the 10s/0.3m convergence check below unmeetable regardless of
    #     controller quality (verified: it diverges to e_y=-10m by t=14s).
    # (2) psi=pi/2 (not 0) to match this leg's own path heading
    #     (compute_path_angle((0,0),(0,30)) = +90deg) -- same class of
    #     heading/path mismatch bug as test1. With BOTH fixed, the rollout is
    #     smooth and stable (delta stays ~3deg the whole time, no
    #     oscillation) but converges slower than the 10s/0.3m threshold
    #     expects (e_y only -1.33m by t=14s) -- so this still won't PASS its
    #     own check, but for a legitimate "threshold is optimistic" reason,
    #     not a real instability like test1's residual issue.
    mmg_init = [0.78, 0.0, 0.0, 1.5, 0.0, np.pi / 2]
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


def test5_sharp_turn_lookahead(nmpc_solver, config=DEFAULT_CONFIG):
    """Regression test for nmpc/README.md item 8: scenario.json's real
    wp0->wp1->wp2 sequence, which turns ~117 deg at wp1 -- the exact
    situation observed (via nmpc_experiment_20260826_222937's telemetry) to
    leave the OLD single-fixed-target guidance with true_delta_deg pegged at
    +45 deg and true_psi_deg stalled ~-70 deg (a >140 deg heading error) for
    20+ seconds after crossing wp1. With the horizon-preview fix, the
    solver should see wp2's very different heading coming and complete the
    turn -- checked here as psi_e (course error to the ACTIVE leg) dropping
    well below the old stalled magnitude within a bounded time of the
    wp1 crossing, rather than staying stuck near it."""
    waypoints = [(-3.7209, 2.3256), (40.0, -19.8837), (40.1163, 20.0)]
    mmg_init = [0.1, 0.0, 0.0, -3.7209, 2.3256, 0.0]
    log = run_scenario(nmpc_solver, waypoints, mmg_init, sim_time=170.0, config=config, label="Test 5")

    plot_trajectory(log, waypoints, os.path.join(RESULTS_DIR, "test5_sharp_turn.png"),
                     "Test 5: Sharp-turn lookahead (scenario.json wp0->wp1->wp2)")
    plot_controls(log, os.path.join(RESULTS_DIR, "test5_controls.png"), "Test 5: Controls")

    switched = np.where(log["target_idx"] >= 2)[0]
    if len(switched) == 0:
        print("Test 5 [turn completes within budget]: FAIL (never reached wp1 within sim_time)")
        return log, False
    t_switch = log["t"][switched[0]]
    window = (log["t"] >= t_switch + 30.0) & (log["t"] <= t_switch + 40.0)
    if not np.any(window):
        print("Test 5 [turn completes within budget]: FAIL (sim_time too short to check post-turn window)")
        return log, False
    max_psi_e_deg = float(np.max(np.abs(np.rad2deg(log["psi_e"][window]))))
    ok = bool(max_psi_e_deg < 30.0)
    print(f"Test 5 [|psi_e| < 30 deg by 30-40s after wp1 crossing]: {'PASS' if ok else 'FAIL'} "
          f"(max |psi_e| in that window: {max_psi_e_deg:.1f} deg; old behavior stalled >140 deg for 20+s)")
    return log, ok


def test6_packed_waypoints(nmpc_solver, config=DEFAULT_CONFIG):
    """Waypoints spaced ~1-2 ship-lengths apart (LPP=2.902m -> 4m spacing
    here), including a 90-deg corner, so several legs fall inside one
    horizon at once (nmpc/README.md item 8's motivating "future LIDAR
    rolling window" case). Checks the controller doesn't thrash/destabilize
    when build_horizon_references previews multiple close-together
    segments simultaneously."""
    waypoints = [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (12.0, 0.0), (12.0, 4.0), (12.0, 8.0)]
    mmg_init = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    log = run_scenario(nmpc_solver, waypoints, mmg_init, sim_time=60.0, config=config, label="Test 6")

    plot_trajectory(log, waypoints, os.path.join(RESULTS_DIR, "test6_packed.png"),
                     "Test 6: Closely-packed waypoints")

    max_n_segments = int(np.max(log["n_active_segments"]))
    solver_ok = bool(np.all(log["success"]))
    e_y_bounded = bool(np.all(np.abs(log["e_y"]) < 3.0))
    ok = solver_ok and e_y_bounded
    print(f"Test 6 [no thrashing on packed waypoints]: {'PASS' if ok else 'FAIL'} "
          f"(max active segments seen in one horizon: {max_n_segments}, "
          f"all solves succeeded: {solver_ok}, max |e_y|: {np.max(np.abs(log['e_y'])):.2f}m)")
    return log, ok


def test7_real_obstacle_collision_regression(config=DEFAULT_CONFIG):
    """Regression test for a real reported collision (nmpc_experiment_
    20260903_004637, user-drawn scenario via scenario_editor.py's new Wall
    mode): with a single real circle obstacle (10.0, -14.7674, r=5.9074)
    sitting almost directly on the ship's first leg, the ship passed
    STRAIGHT THROUGH it -- true position penetrated 3.44m past the obstacle
    boundary, with the solver reporting success=True on every single step
    the whole time. Root cause: the unified soft-min obstacle constraint
    (research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md) made h a LINEAR
    distance (meters) instead of the old per-circle constraint's SQUARED
    distance (meters^2); W_SLACK=50 (tuned for the old squared units) no
    longer penalized a large violation enough to make it costlier than a
    real detour under the tracking cost. Fixed by raising W_SLACK to 5000
    (sim_params.yaml) -- confirmed here to keep clearance positive with
    zero solver failures. Reproduces deterministically outside ROS (no
    UKF/sensor noise involved), so this checks the exact scenario, not just
    a synthetic one, unlike test5/test6 above.

    Deliberately builds its OWN fresh AcadosNMPC instance rather than
    reusing main()'s shared one: SQP-RTI's internal iterate persists across
    solve() calls as a warm start (nmpc_acados.py's own comment on this),
    so chaining this scenario after test1-6's completely unrelated ones
    starts it from a stale, irrelevant warm start and produces a WORSE
    (or better) result than a real deployment ever would -- confirmed by
    running this exact check first with a fresh solver (min_clearance
    +0.53m, matching a real `ros2 launch bringup.launch.py` process, which
    also builds exactly one fresh solver) vs. chained after test6 in the
    suite (min_clearance -0.41m, a suite artifact, not a real bug). A fresh
    solver here is the one that actually matches deployment."""
    from nmpc.nmpc_acados import AcadosNMPC
    nmpc_solver = AcadosNMPC(config)

    waypoints = [(-9.6512, -13.6047), (30.0, -27.2093), (30.0, 49.3023), (14.7674, 19.8837)]
    mmg_init = [0.1, 0.0, 0.0, -9.6512, -13.6047, 0.0]
    obstacles = [(10.0, -14.7674, 5.9074)]
    walls = [(30.1163, -8.8372, 30.0, 30.9302, 1.5)]
    log = run_scenario(nmpc_solver, waypoints, mmg_init, sim_time=70.0, config=config,
                        label="Test 7", obstacles=obstacles, walls=walls)

    plot_trajectory(log, waypoints, os.path.join(RESULTS_DIR, "test7_obstacle.png"),
                     "Test 7: Real obstacle collision regression")

    ox, oy, orad = obstacles[0]
    r_c = orad + config.R_ASV
    clearance = np.hypot(log["x"] - ox, log["y"] - oy) - r_c
    min_clearance = float(np.min(clearance))
    solver_ok = bool(np.all(log["success"]))
    ok = solver_ok and min_clearance >= -config.SIGMA - 1e-3
    print(f"Test 7 [real obstacle actually avoided, no collision]: {'PASS' if ok else 'FAIL'} "
          f"(min clearance to obstacle boundary: {min_clearance:.3f}m, SIGMA={config.SIGMA}m, "
          f"all solves succeeded: {solver_ok}; old bug: -3.44m penetration)")
    return log, ok


def _true_ellipse_boundary_distance(x, y, xc, yc, a, b, theta, n=1440):
    """Brute-force parametric ground truth (fine sampling of the ellipse
    boundary), used ONLY here for verification -- not the solver's own
    ellipse_distance_casadi, which is an approximation (see
    tests/test_ellipse_distance.py and research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md).
    Mirrors test7's pattern of checking TRUE geometric clearance, not just
    solver success -- test7's own lesson was that success=True alone doesn't
    guarantee no collision."""
    dx = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
    dy = -(x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    bx, by = a * np.cos(t), b * np.sin(t)
    return float(np.min(np.hypot(dx - bx, dy - by)))


def test8_ellipse_obstacle_avoidance(config=DEFAULT_CONFIG):
    """New-feature acceptance test for elliptical obstacles (not a real-
    incident regression like test7 -- there is no real ellipse deployment
    yet). A single ellipse (center (20,0), semi-major a=8 along EAST i.e.
    theta=90deg, semi-minor b=3 along North) sits directly across a straight
    North-traveling leg, wide enough that a real detour is unavoidable.
    Checks TRUE geometric clearance against the padded ellipse boundary via
    brute-force parametric sampling (_true_ellipse_boundary_distance), not
    just success=True -- per test7's own lesson that solver success alone
    doesn't guarantee no collision. Deliberately builds its own fresh
    AcadosNMPC instance, same SQP-RTI warm-start-contamination rationale as
    test7."""
    from nmpc.nmpc_acados import AcadosNMPC
    nmpc_solver = AcadosNMPC(config)

    waypoints = [(0.0, 0.0), (40.0, 0.0)]
    mmg_init = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    ellipses = [(20.0, 0.0, 8.0, 3.0, np.pi / 2.0)]
    log = run_scenario(nmpc_solver, waypoints, mmg_init, sim_time=60.0, config=config,
                        label="Test 8", ellipses=ellipses)

    plot_trajectory(log, waypoints, os.path.join(RESULTS_DIR, "test8_ellipse.png"),
                     "Test 8: Elliptical obstacle avoidance")

    xc, yc, a, b, theta = ellipses[0]
    clearance = np.array([_true_ellipse_boundary_distance(x, y, xc, yc, a, b, theta)
                           for x, y in zip(log["x"], log["y"])]) - config.R_ASV
    min_clearance = float(np.min(clearance))
    solver_ok = bool(np.all(log["success"]))
    ok = solver_ok and min_clearance >= -config.SIGMA - 1e-3
    print(f"Test 8 [elliptical obstacle actually avoided, no collision]: {'PASS' if ok else 'FAIL'} "
          f"(min clearance to padded ellipse boundary: {min_clearance:.3f}m, SIGMA={config.SIGMA}m, "
          f"all solves succeeded: {solver_ok})")
    return log, ok


def main(nmpc_solver=None, config=DEFAULT_CONFIG, solver_name="AcadosNMPC"):
    """Runs all tests."""
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
    log5, ok5 = test5_sharp_turn_lookahead(nmpc_solver, config)
    results["test5"] = ok5
    log6, ok6 = test6_packed_waypoints(nmpc_solver, config)
    results["test6"] = ok6
    log7, ok7 = test7_real_obstacle_collision_regression(config)
    results["test7"] = ok7
    log8, ok8 = test8_ellipse_obstacle_avoidance(config)
    results["test8"] = ok8

    print("\n=========== SUMMARY ===========")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_pass = all(results.values())
    print(f"  ALL: {'PASS' if all_pass else 'FAIL'}")
    return results, {"test1": log1, "test2": log2, "test3": log3, "test5": log5, "test6": log6, "test7": log7,
                      "test8": log8}


if __name__ == "__main__":
    main()
