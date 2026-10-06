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


def main(args=None):
    results = [
        check_constant_velocity_matches_hand_computed(),
        check_shape_is_N_plus_1_by_2(),
        check_zero_velocity_is_constant(),
        check_growing_radius_monotonic_and_matches_formula(),
        check_growing_radius_zero_growth_is_constant(),
    ]
    all_pass = all(results)
    print(f"  ALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
