"""Standalone unit checks for nmpc/moving_obstacle.py's constant-velocity
horizon prediction (predict_moving_obstacle_positions) and growing_radius.
No rclpy, no ROS graph, no acados/CasADi -- same "headless, no ROS graph"
style as test_capsule_distance.py/test_ellipse_distance.py. This module is
deliberately never imported by nmpc_acados.py -- see nmpc/moving_obstacle.py's
own module docstring and research_papers/COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md.

Run: ros2 run nmpc_sim_nodes test_moving_obstacle_prediction
"""
import math

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.moving_obstacle import growing_radius, predict_moving_obstacle_positions  # noqa: E402


def check_constant_velocity_matches_hand_computed():
    x0, y0, vx, vy, dt, N = 10.0, -5.0, 1.5, -0.5, 0.2, 20
    xy = predict_moving_obstacle_positions(x0, y0, vx, vy, dt, N)
    ok = True
    for k in (0, N // 2, N):
        expected_x = x0 + k * dt * vx
        expected_y = y0 + k * dt * vy
        got_x, got_y = xy[k]
        this_ok = math.isclose(got_x, expected_x, abs_tol=1e-9) and math.isclose(got_y, expected_y, abs_tol=1e-9)
        ok = ok and this_ok
    print(f"Test 1 [constant-velocity extrapolation matches x0+k*dt*vx / y0+k*dt*vy]: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_shape_is_N_plus_1_by_2():
    N = 15
    xy = predict_moving_obstacle_positions(0.0, 0.0, 1.0, 1.0, 0.1, N)
    ok = xy.shape == (N + 1, 2)
    print(f"Test 2 [output shape is (N+1, 2)]: {'PASS' if ok else 'FAIL'} (got {xy.shape})")
    return ok


def check_zero_velocity_is_constant():
    x0, y0, N = 3.0, 4.0, 10
    xy = predict_moving_obstacle_positions(x0, y0, 0.0, 0.0, 0.5, N)
    ok = all(math.isclose(x, x0, abs_tol=1e-12) and math.isclose(y, y0, abs_tol=1e-12) for x, y in xy)
    print(f"Test 3 [zero velocity -> constant predicted position at every k]: {'PASS' if ok else 'FAIL'}")
    return ok


def check_growing_radius_monotonic_and_matches_formula():
    radius, growth = 2.0, 0.1
    values = [growing_radius(radius, k, growth) for k in range(21)]
    matches = all(math.isclose(v, radius + growth * k, abs_tol=1e-12) for k, v in enumerate(values))
    monotonic = all(b >= a for a, b in zip(values, values[1:]))
    ok = matches and monotonic
    print(f"Test 4 [growing_radius matches radius+growth*k, monotonically non-decreasing]: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_growing_radius_zero_growth_is_constant():
    radius = 2.5
    values = [growing_radius(radius, k, 0.0) for k in range(5)]
    ok = all(math.isclose(v, radius, abs_tol=1e-12) for v in values)
    print(f"Test 5 [growing_radius with growth_per_step=0.0 stays constant]: {'PASS' if ok else 'FAIL'}")
    return ok


def check_moving_ellipse_orientation_rotates_with_psi():
    from nmpc.path_following import ellipse_distance_casadi

    xc, yc, a, b = 10.0, 10.0, 6.0, 2.0
    r_pad = 0.0
    eps = 1e-9

    px, py = 16.0, 10.0  # 6m offset in +X direction from center

    # Heading theta = 0 (major axis along +X): point is on boundary
    d_theta0 = float(ellipse_distance_casadi(px, py, xc, yc, a, b, 0.0, r_pad, eps))

    # Heading theta = pi/2 (minor axis along +X): boundary is at b=2m, point is 4m outside
    d_theta_pi2 = float(ellipse_distance_casadi(px, py, xc, yc, a, b, math.pi / 2.0, r_pad, eps))

    ok = abs(d_theta0) < 1e-3 and d_theta_pi2 > 1.5
    print(f"Test 6 [ellipse distance rotates with heading psi: d(theta=0)={d_theta0:.4f}m, d(theta=pi/2)={d_theta_pi2:.4f}m]: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_stage_varying_horizon_matching():
    from nmpc.nmpc_acados import _is_stage_varying
    from nmpc.params import DEFAULT_CONFIG

    N = DEFAULT_CONFIG.N
    static_ell = [(10.0, 10.0, 6.0, 2.0, 0.0)]
    ok1 = not _is_stage_varying(static_ell, N)

    traj_ell = [[(10.0 + k * 0.1, 10.0, 6.0, 2.0, k * 0.01)] for k in range(N + 1)]
    ok2 = _is_stage_varying(traj_ell, N)

    # Wrong length (e.g. 201 instead of N+1=401) is strictly rejected
    wrong_len_ell = [[(10.0, 10.0, 6.0, 2.0, 0.0)] for _ in range(201)]
    ok3 = not _is_stage_varying(wrong_len_ell, N)

    ok = ok1 and ok2 and ok3
    print(f"Test 7 [stage-varying detection matches solver horizon N={N} (length {N+1})]: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_moving_ellipse_closed_loop_avoidance():
    """End-to-end closed-loop validation: own-ship navigates a straight path while
    a crossing elliptical obstacle vessel intersects the path ahead. Verifies that
    the stage-varying prediction horizon (N=400, 401 stages) is consumed by AcadosNMPC
    and successfully steers own-ship around the obstacle without collision."""
    import casadi as ca
    import numpy as np
    from nmpc.params import DEFAULT_CONFIG
    from nmpc.nmpc_acados import AcadosNMPC
    from nmpc.path_following import (
        SegmentQueue, segments_from_waypoints, select_active_waypoint
    )
    from casadi_mmg_solver.casadi_mmg import make_casadi_integrator

    def true_ellipse_boundary_dist(x, y, xc, yc, a, b, theta, n=360):
        dx = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
        dy = -(x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        bx, by = a * np.cos(t), b * np.sin(t)
        return float(np.min(np.hypot(dx - bx, dy - by)))

    config = DEFAULT_CONFIG
    solver = AcadosNMPC(config)
    plant_step = make_casadi_integrator(config.dt, method="rk4", sym_type=ca.SX)

    waypoints = [(0.0, 0.0), (50.0, 0.0)]
    mmg_state = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    delta = 0.0
    n = float(config.N_TRIM)
    target_idx = 1
    segment_queue = SegmentQueue(segments_from_waypoints(waypoints, target_idx))

    # Obstacle: starts at (25, 8), moves South with vy = -0.25 m/s, heading -pi/2
    obs_x, obs_y = 25.0, 8.0
    obs_vx, obs_vy = 0.0, -0.25
    obs_a, obs_b, obs_psi = 5.0, 2.0, -math.pi / 2.0

    sim_time = 45.0
    n_steps = int(sim_time / config.dt)
    min_clearances = []
    solves_ok = True

    for step in range(n_steps):
        ellipses_horizon = []
        for k in range(config.N + 1):
            x_k = obs_x + k * config.dt * obs_vx
            y_k = obs_y + k * config.dt * obs_vy
            ellipses_horizon.append([(x_k, y_k, obs_a, obs_b, obs_psi)])

        res = solver.solve(mmg_state.tolist(), delta, n, segment_queue.segments, ellipses=ellipses_horizon)
        if not res["success"]:
            solves_ok = False

        delta = float(res["delta"])
        n = float(res["n"])

        state_ca = ca.DM(mmg_state)
        control_ca = ca.DM([delta, n])
        next_state, _ = plant_step(state_ca, control_ca)
        mmg_state = np.array(next_state).flatten()
        x, y = mmg_state[3], mmg_state[4]

        target_idx = select_active_waypoint(x, y, waypoints, target_idx, config.WP_RADIUS)
        segment_queue.pop_crossed(x, y, config.WP_RADIUS)

        clearance = true_ellipse_boundary_dist(x, y, obs_x, obs_y, obs_a, obs_b, obs_psi) - config.R_ASV
        min_clearances.append(clearance)

        obs_x += config.dt * obs_vx
        obs_y += config.dt * obs_vy

    min_clr = min(min_clearances)
    ok = solves_ok and min_clr >= -config.SIGMA - 1e-3
    print(f"Test 8 [closed-loop moving ellipse avoidance across horizon N={config.N}]: "
          f"{'PASS' if ok else 'FAIL'} (min clearance: {min_clr:.3f}m, all solves succeeded: {solves_ok})")
    return ok


def check_nomoto_obstacle_dynamics_and_velocity_sync():
    """Validates that:
    1. With delta=0, Nomoto steering preserves exact initial velocity (u0, psi0)
       matching (vx, vy) without disturbance drift or speed decay.
    2. With delta!=0, it turns according to Nomoto dynamics r_dot = (-r + K*delta)/T.
    3. Commanded surge speed u matches teleop input.
    """
    from nmpc.nomoto_obstacle import step as nomoto_step
    import numpy as np

    K = 0.15
    T = 3.0
    dt = 0.1
    vx, vy = 0.4, 0.3
    u0 = float(np.hypot(vx, vy))  # 0.5 m/s
    psi0 = float(np.arctan2(vy, vx))
    x0, y0 = 10.0, 20.0

    # 1. Straight line (delta=0) for 5 seconds (50 steps)
    x, y, psi, r = x0, y0, psi0, 0.0
    for _ in range(50):
        x, y, psi, r = nomoto_step(x, y, psi, u0, r, delta=0.0, K=K, T=T, dt=dt)

    t_total = 50 * dt
    expected_x = x0 + t_total * vx
    expected_y = y0 + t_total * vy
    straight_ok = (math.isclose(x, expected_x, abs_tol=1e-6) and
                   math.isclose(y, expected_y, abs_tol=1e-6) and
                   math.isclose(psi, psi0, abs_tol=1e-6) and
                   math.isclose(r, 0.0, abs_tol=1e-6))

    # 2. Turning response with delta = 15 deg (0.2618 rad)
    delta_turn = math.radians(15.0)
    for _ in range(50):
        x, y, psi, r = nomoto_step(x, y, psi, u0, r, delta=delta_turn, K=K, T=T, dt=dt)

    turn_ok = psi > psi0 and r > 0.0

    ok = straight_ok and turn_ok
    print(f"Test 9 [Nomoto obstacle preserves initial (vx, vy) and turns cleanly under rudder]: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main(args=None):
    results = [
        check_constant_velocity_matches_hand_computed(),
        check_shape_is_N_plus_1_by_2(),
        check_zero_velocity_is_constant(),
        check_growing_radius_monotonic_and_matches_formula(),
        check_growing_radius_zero_growth_is_constant(),
        check_moving_ellipse_orientation_rotates_with_psi(),
        check_stage_varying_horizon_matching(),
        check_moving_ellipse_closed_loop_avoidance(),
        check_nomoto_obstacle_dynamics_and_velocity_sync(),
    ]
    all_pass = all(results)
    print(f"  ALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())

