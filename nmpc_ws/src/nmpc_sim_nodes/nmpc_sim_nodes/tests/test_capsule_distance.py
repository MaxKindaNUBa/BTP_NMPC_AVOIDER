"""Standalone unit checks for nmpc/path_following.py's capsule_distance_casadi
and softmin_casadi -- the closed-form point-to-capsule distance (circles are
the degenerate zero-length-segment case) and its soft-min aggregation that
nmpc_acados.py's single obstacle constraint row is built from. No rclpy, no
ROS graph, no acados solver -- headless, standalone math checks only. See
research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md.

Run: ros2 run nmpc_sim_nodes test_capsule_distance
"""
import math

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.path_following import capsule_distance_casadi, softmin_casadi  # noqa: E402

EPS = 1e-6


def _d(x, y, p0x, p0y, p1x, p1y, r):
    return float(capsule_distance_casadi(x, y, p0x, p0y, p1x, p1y, r, EPS))


def check_degenerate_circle_matches_old_formula():
    # p0 == p1 -> a circle at (0, 0), radius 2. Old formula: sqrt(x^2+y^2) - r.
    d = _d(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0)
    expected = math.hypot(5.0, 0.0) - 2.0
    ok = abs(d - expected) < 1e-3
    print(f"Test 1 [degenerate capsule (p0==p1) matches old circle formula]: {'PASS' if ok else 'FAIL'} "
          f"(got {d:.4f}, expected {expected:.4f})")
    return ok


def check_beyond_endpoint_clamps():
    # segment (0,0)->(10,0), r=1. A point past p1 must clamp to p1, not the
    # infinite line.
    d_past_p1 = _d(15.0, 0.0, 0.0, 0.0, 10.0, 0.0, 1.0)
    ok_p1 = abs(d_past_p1 - 4.0) < 1e-3  # |15-10| - 1

    # symmetric check on the p0 side
    d_past_p0 = _d(-5.0, 0.0, 0.0, 0.0, 10.0, 0.0, 1.0)
    ok_p0 = abs(d_past_p0 - 4.0) < 1e-3  # |0-(-5)| - 1

    ok = ok_p1 and ok_p0
    print(f"Test 2 [beyond-endpoint distance clamps to nearest endpoint]: {'PASS' if ok else 'FAIL'} "
          f"(past p1={d_past_p1:.4f}, past p0={d_past_p0:.4f}, both expected 4.0)")
    return ok


def check_perpendicular_midsegment():
    # segment (0,0)->(0,10) (along y), r=1. A point at (3, 5) is beside the
    # segment's midpoint -> perpendicular distance 3, minus r.
    d = _d(3.0, 5.0, 0.0, 0.0, 0.0, 10.0, 1.0)
    expected = 3.0 - 1.0
    ok = abs(d - expected) < 1e-3
    print(f"Test 3 [perpendicular distance to segment interior]: {'PASS' if ok else 'FAIL'} "
          f"(got {d:.4f}, expected {expected:.4f})")
    return ok


def check_softmin_safety_property_and_convergence():
    # two known distances: true min = 2.0. Soft-min must never exceed it
    # (the safety property: D_hat <= true min, proven via sum >= max term),
    # and must converge closer to it as K grows.
    d1, d2 = 2.0, 5.0
    true_min = min(d1, d2)

    softmin_low_k = float(softmin_casadi([d1, d2], 1.0))
    softmin_default_k = float(softmin_casadi([d1, d2], 3.0))
    softmin_high_k = float(softmin_casadi([d1, d2], 50.0))

    ok_never_exceeds = (softmin_low_k <= true_min + 1e-9 and
                        softmin_default_k <= true_min + 1e-9 and
                        softmin_high_k <= true_min + 1e-9)
    ok_converges = (softmin_low_k < softmin_default_k < softmin_high_k <= true_min)
    ok_tight_at_high_k = abs(softmin_high_k - true_min) < 1e-3

    ok = ok_never_exceeds and ok_converges and ok_tight_at_high_k
    print(f"Test 4 [soft-min <= true min always, tightens as K grows]: {'PASS' if ok else 'FAIL'} "
          f"(K=1:{softmin_low_k:.4f}, K=3:{softmin_default_k:.4f}, K=50:{softmin_high_k:.4f}, "
          f"true min={true_min:.4f})")
    return ok


def check_softmin_no_underflow_with_all_dummy_slots():
    # Regression test: every obstacle/wall slot is an unused, far-away dummy
    # (matches pad_obstacles/pad_walls' dummy_pos=1.0e3 convention) -- with
    # the ship near the origin, every d_i is ~1413m. Before the per-term
    # _SOFTMIN_DISTANCE_CAP clip, exp(-k*1413) underflowed to exactly 0.0 for
    # every term, making softmin_casadi return -log(0.0)/k == +inf and
    # poisoning every solve (confirmed via ros2 run nmpc_sim_nodes test_nmpc:
    # 100% QP failures on the zero-obstacle case before this fix). Must now
    # return a finite value anchored at the cap instead.
    ship_x, ship_y = 0.0, 0.0
    dummy_r_c = 1.451  # R_ASV, matches nmpc/params.py's default LPP*0.5
    circle_ds = [_d(ship_x, ship_y, 1.0e3, 1.0e3, 1.0e3, 1.0e3, dummy_r_c) for _ in range(20)]
    wall_ds = [_d(ship_x, ship_y, 1.0e3, 1.0e3, 1.0e3, 1.0e3, dummy_r_c) for _ in range(5)]
    D_hat = float(softmin_casadi(circle_ds + wall_ds, 3.0))

    # every dummy distance (~1413m) clips to _SOFTMIN_DISTANCE_CAP (100.0)
    # before exponentiating, so with all 25 slots tied post-clip, D_hat ==
    # CAP - log(25)/k (same tie-breaking conservatism as an unclipped tie,
    # just anchored at the cap instead of the true distance).
    cap = 100.0
    expected = cap - math.log(25) / 3.0
    ok_finite = math.isfinite(D_hat)
    ok_close = ok_finite and abs(D_hat - expected) < 1e-3

    ok = ok_finite and ok_close
    print(f"Test 5 [soft-min stays finite with all-dummy (far-away) obstacle slots]: {'PASS' if ok else 'FAIL'} "
          f"(D_hat={D_hat}, expected~={expected:.4f})")
    return ok


def main(args=None):
    results = [
        check_degenerate_circle_matches_old_formula(),
        check_beyond_endpoint_clamps(),
        check_perpendicular_midsegment(),
        check_softmin_safety_property_and_convergence(),
        check_softmin_no_underflow_with_all_dummy_slots(),
    ]
    all_pass = all(results)
    print(f"  ALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
