"""Standalone unit checks for nmpc/path_following.py's ellipse_distance_casadi --
the gradient-normalized APPROXIMATE point-to-ellipse distance (no closed form
exists for a true one) that nmpc_acados.py's single obstacle constraint row
folds ellipse terms into, alongside circle/wall terms. No rclpy, no ROS graph,
no acados solver -- same "headless, no ROS graph" style as
test_capsule_distance.py. See research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md.

Unlike the capsule primitive, there is NO closed form to check the ellipse
formula against analytically for the general case -- so this file includes one
extra *kind* of check beyond test_capsule_distance.py's pattern: a brute-force
numerical sweep verifying the safety-critical property (never overestimates
clearance), since only a numerical ground truth is available for that.

Run: ros2 run nmpc_sim_nodes test_ellipse_distance
"""
import math

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.path_following import ellipse_distance_casadi, softmin_casadi  # noqa: E402

EPS = 1e-9


def _d(x, y, xc, yc, a, b, theta, r_pad=0.0):
    return float(ellipse_distance_casadi(x, y, xc, yc, a, b, theta, r_pad, EPS))


def _true_boundary_distance(x, y, xc, yc, a, b, theta, n=3600):
    """Brute-force ground truth: fine parametric sampling of the (unpadded)
    ellipse boundary in its own frame, min Euclidean distance back to world
    frame. Only used by this test file, never by the solver (too slow/not a
    static CasADi graph) -- exactly why the gradient-normalized approximation
    is needed in the first place."""
    dx = (x - xc) * math.cos(theta) + (y - yc) * math.sin(theta)
    dy = -(x - xc) * math.sin(theta) + (y - yc) * math.cos(theta)
    best = float("inf")
    for i in range(n):
        t = 2.0 * math.pi * i / n
        bx, by = a * math.cos(t), b * math.sin(t)
        dist = math.hypot(dx - bx, dy - by)
        if dist < best:
            best = dist
    inside = (dx / a) ** 2 + (dy / b) ** 2 < 1.0
    return -best if inside else best


def check_degenerate_axisaligned_matches_circle_formula():
    # a=b=r=3 (a circle), center at origin. The formula reduces EXACTLY (self-
    # consistency, not a different ground truth) to (D^2-r^2)/(2D).
    xc, yc, r = 0.0, 0.0, 3.0
    x, y = 7.0, 0.0
    D = math.hypot(x - xc, y - yc)
    expected = (D ** 2 - r ** 2) / (2.0 * D)
    got = _d(x, y, xc, yc, r, r, 0.0)
    ok = abs(got - expected) < 1e-6
    print(f"Test 1 [degenerate a=b=r matches closed-form (D^2-r^2)/(2D)]: {'PASS' if ok else 'FAIL'} "
          f"(got {got:.6f}, expected {expected:.6f})")
    return ok


def check_boundary_first_order_exact():
    # Points placed exactly on the (unpadded) ellipse boundary, at several
    # angles including rotation -- d_ellipse should read ~0 there (first-
    # order exact at g=0, per the design doc), and with r_pad subtracted,
    # exactly -r_pad.
    xc, yc, a, b, theta = 5.0, -2.0, 4.0, 2.0, math.radians(30.0)
    ok = True
    for t_deg in (0, 45, 90, 135, 180, 270):
        t = math.radians(t_deg)
        # boundary point in ellipse-local frame, rotated+translated to world
        lx, ly = a * math.cos(t), b * math.sin(t)
        wx = xc + lx * math.cos(theta) - ly * math.sin(theta)
        wy = yc + lx * math.sin(theta) + ly * math.cos(theta)
        d0 = _d(wx, wy, xc, yc, a, b, theta)
        d_padded = _d(wx, wy, xc, yc, a, b, theta, r_pad=1.5)
        this_ok = abs(d0) < 1e-3 and abs(d_padded - (-1.5)) < 1e-3
        ok = ok and this_ok
    print(f"Test 2 [boundary points read ~0 (first-order exact), r_pad subtracts cleanly]: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_conservative_vs_bruteforce():
    # THE safety-critical property (mirrors softmin_casadi's own "D_hat <=
    # true min" property, extended to this primitive): ellipse_distance_casadi
    # must NEVER overestimate clearance. Swept across eccentricities, angles,
    # and ranges -- same spirit sweep as the one done during design (960
    # points there; a smaller grid here for fast test runtime, same
    # methodology).
    a_b_pairs = [(3.0, 3.0), (5.0, 3.0), (8.0, 2.0), (10.0, 1.0)]
    angles_deg = list(range(0, 360, 30))
    range_factors = [1.05, 1.5, 3.0, 8.0]
    worst_violation = 0.0
    n_checked = 0
    for (a, b) in a_b_pairs:
        for ang_deg in angles_deg:
            ang = math.radians(ang_deg)
            for rf in range_factors:
                # query point at range_factor * a along a direction ang from
                # the (axis-aligned, theta=0) ellipse center -- deliberately
                # NOT restricted to the major/minor axes, to exercise off-axis
                # behavior.
                x, y = rf * a * math.cos(ang), rf * a * math.sin(ang)
                true_dist = _true_boundary_distance(x, y, 0.0, 0.0, a, b, 0.0, n=1440)
                if true_dist <= 0:
                    continue  # skip points inside the ellipse -- conservatism is
                    # about not UNDERSTATING danger from outside; the sign
                    # convention inside is a separate, non-safety-critical detail
                approx = _d(x, y, 0.0, 0.0, a, b, 0.0)
                violation = approx - true_dist
                worst_violation = max(worst_violation, violation)
                n_checked += 1
    ok = worst_violation < 1e-6
    print(f"Test 3 [conservative vs brute-force truth: approx <= true distance always]: "
          f"{'PASS' if ok else 'FAIL'} (checked {n_checked} points, worst violation={worst_violation:.6f}m)")
    return ok


def check_dummy_padding_no_underflow():
    # Mirrors test_capsule_distance.py's Test 5: MAX_ELLIPSES dummy slots
    # (far-away center, a=b=1.0 per pad_ellipses' convention) mixed into the
    # same d_list softmin_casadi aggregates circles/walls into -- confirm no
    # underflow/inf, matching the established per-term-clip discipline.
    dummy_ds = [_d(0.0, 0.0, 1.0e3, 1.0e3, 1.0, 1.0, 0.0) for _ in range(5)]
    D_hat = float(softmin_casadi(dummy_ds, 3.0))
    ok = math.isfinite(D_hat)
    print(f"Test 4 [dummy ellipse slots (far-away, a=b=1.0) don't underflow softmin]: "
          f"{'PASS' if ok else 'FAIL'} (D_hat={D_hat})")
    return ok


def check_far_field_degradation_characterized():
    # Characterization test (not a correctness assertion): pins the known,
    # documented far-field accuracy degradation so a future silent change to
    # the formula is caught. For a circle (a=b=r), far-field ratio to true
    # distance settles ~0.5 (NOT 1.0 -- an earlier design draft's claim of
    # asymptotic exactness was checked and found wrong, see the design doc).
    r = 3.0
    D = 500.0 * r  # deep into the far field
    true_dist = D - r
    approx = _d(D, 0.0, 0.0, 0.0, r, r, 0.0)
    ratio = approx / true_dist
    ok = 0.45 < ratio < 0.55
    print(f"Test 5 [far-field ratio for a circle settles ~0.5, not 1.0 -- characterization]: "
          f"{'PASS' if ok else 'FAIL'} (ratio={ratio:.4f})")
    return ok


def main(args=None):
    results = [
        check_degenerate_axisaligned_matches_circle_formula(),
        check_boundary_first_order_exact(),
        check_conservative_vs_bruteforce(),
        check_dummy_padding_no_underflow(),
        check_far_field_degradation_characterized(),
    ]
    all_pass = all(results)
    print(f"  ALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
