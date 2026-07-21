"""
Pure NumPy path-following geometry (no CasADi).
Computes cross-track error, course angle and error, waypoint switching and the
NMPC reference state xi_ref. Called once per NMPC step from the outer loop.

Formulas: main.pdf Eq. (1)/(17-19), idekewfewf.pdf Eq. (51).
"""
import os
import sys
import numpy as np
import casadi as ca

# allow running this file directly (python nmpc/path_following.py) by putting
# the repo root on sys.path, so `nmpc` resolves as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nmpc.config import (
    DEFAULT_CONFIG, STATE_DIM,
    IDX_EY, IDX_SPSI, IDX_CPSI, IDX_R, IDX_X, IDX_Y, IDX_PSI, IDX_U, IDX_V, IDX_DELTA, IDX_N,
)


def wrap_to_pi(angle: float) -> float:
    """Wraps any angle into [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap180_casadi(theta):
    """Symbolic (CasADi/acados-safe) twin of wrap_to_pi: smooth wrap into (-pi, pi].
    Built from atan2(sin,cos) so it's differentiable everywhere except the single
    +-pi branch cut (same as any angle wrap) — safe to use inside NLP cost terms."""
    return ca.atan2(ca.sin(theta), ca.cos(theta))


def compute_path_angle(wp_a, wp_b) -> float:
    """Heading of the straight-line segment wp_a -> wp_b."""
    return float(np.arctan2(wp_b[1] - wp_a[1], wp_b[0] - wp_a[0]))


def compute_cross_track_error(x, y, x_d, y_d, chi_p) -> float:
    """Perpendicular distance from (x,y) to the path line, signed."""
    return float(-(x - x_d) * np.sin(chi_p) + (y - y_d) * np.cos(chi_p))


def compute_sideslip(u, v) -> float:
    """Drift angle between heading and actual velocity direction.
    atan2(-v,u), matching casadi_mmg.py's own internal convention exactly
    (NOT asin(v/U) — that form is sign-blind to reverse thrust: it can't
    tell u>0 from u<0, so it silently breaks once the ship goes astern)."""
    return float(np.arctan2(-v, u))


def compute_course_angle(psi, beta) -> float:
    """Actual direction of travel (heading + sideslip)."""
    return wrap_to_pi(psi + beta)


def compute_course_error(chi, chi_p) -> float:
    """How far off course the ship is relative to the path."""
    return wrap_to_pi(chi - chi_p)


def get_reference_state(chi_p, x_d, y_d, U_ref, delta_trim=None, n_trim=None,
                         config=DEFAULT_CONFIG) -> np.ndarray:
    """Builds xi_ref: zero error, desired heading = chi_p, desired speed = U_ref."""
    if delta_trim is None:
        delta_trim = config.DELTA_TRIM  # fall back to config default
    if n_trim is None:
        n_trim = config.N_TRIM

    xi_ref = np.zeros(STATE_DIM)
    xi_ref[IDX_EY] = 0.0
    xi_ref[IDX_SPSI] = 0.0
    xi_ref[IDX_CPSI] = 1.0
    xi_ref[IDX_R] = 0.0
    xi_ref[IDX_X] = x_d
    xi_ref[IDX_Y] = y_d
    xi_ref[IDX_PSI] = chi_p
    xi_ref[IDX_U] = U_ref
    xi_ref[IDX_V] = 0.0
    xi_ref[IDX_DELTA] = delta_trim
    xi_ref[IDX_N] = n_trim
    return xi_ref


def select_active_waypoint(x, y, waypoints, current_idx, wp_radius: float = None,
                            config=DEFAULT_CONFIG) -> int:
    """Advances current_idx to current_idx+1 once the ship has either come within
    wp_radius of the target waypoint, OR crossed the perpendicular "gate" plane
    through it (projection of (x,y) onto the prev_wp->wp leg direction is past wp).

    The radius-only check can fail to fire forever: a large initial heading error
    makes the turning transient converge onto the path's line well beyond the
    target point, so Euclidean distance to that point never dips below wp_radius
    again (see test3_waypoint_switching_issue.md). The along-track gate test
    catches this because it only cares about progress along the leg direction,
    not lateral offset, so it fires exactly once as the ship passes abeam of the
    waypoint no matter how wide the turn was.
    """
    if wp_radius is None:
        wp_radius = config.WP_RADIUS

    last_idx = len(waypoints) - 1
    current_idx = min(current_idx, last_idx)
    wp = waypoints[current_idx]
    dist = np.sqrt((x - wp[0]) ** 2 + (y - wp[1]) ** 2)
    reached = dist < wp_radius

    if not reached and current_idx > 0:
        prev_wp = waypoints[current_idx - 1]
        leg_dx, leg_dy = wp[0] - prev_wp[0], wp[1] - prev_wp[1]
        leg_len = np.hypot(leg_dx, leg_dy)
        if leg_len > 1e-9:
            # along-track position of (x,y) relative to wp, projected onto the
            # leg direction; >= 0 once (x,y) has crossed the plane through wp
            # perpendicular to the leg, regardless of cross-track offset there.
            along = ((x - wp[0]) * leg_dx + (y - wp[1]) * leg_dy) / leg_len
            reached = along >= 0.0

    if reached and current_idx < last_idx:
        current_idx += 1   # close enough, or already passed it -> move to next leg
    return min(current_idx, last_idx)


def build_xi_from_mmg(mmg_state, chi_p, x_d, y_d) -> np.ndarray:
    """mmg_state = [u, v, r, x, y, psi] -> augmented xi, delta/n left at 0
    (mmg_state has no actuator entries; use build_xi_full to set them)."""
    u, v, r, x, y, psi = mmg_state

    e_y = compute_cross_track_error(x, y, x_d, y_d, chi_p)
    beta = compute_sideslip(u, v)
    chi = compute_course_angle(psi, beta)
    psi_e = compute_course_error(chi, chi_p)

    xi = np.zeros(STATE_DIM)
    xi[IDX_EY] = e_y
    xi[IDX_SPSI] = np.sin(psi_e)
    xi[IDX_CPSI] = np.cos(psi_e)
    xi[IDX_R] = r
    xi[IDX_X] = x
    xi[IDX_Y] = y
    xi[IDX_PSI] = psi
    xi[IDX_U] = u
    xi[IDX_V] = v
    xi[IDX_DELTA] = 0.0
    xi[IDX_N] = 0.0
    return xi


def build_xi_full(mmg_state, delta, n, chi_p, x_d, y_d) -> np.ndarray:
    """Same as build_xi_from_mmg but with actuator states filled in
    (delta/n are tracked externally, not part of mmg_state)."""
    xi = build_xi_from_mmg(mmg_state, chi_p, x_d, y_d)
    xi[IDX_DELTA] = delta
    xi[IDX_N] = n
    return xi


def pad_obstacles(obstacles, max_obstacles: int, dummy_pos: float = 1.0e3) -> np.ndarray:
    """Pads/truncates obstacle list to a fixed length so the NLP/OCP obstacle
    block has a constant size. Unused slots get a far-away dummy obstacle
    (zero radius) so their constraint is always trivially satisfied."""
    obs = list(obstacles)[:max_obstacles]  # truncate if more than max_obstacles given
    flat = []
    for (ox, oy, orad) in obs:
        flat.extend([ox, oy, orad])
    for _ in range(max_obstacles - len(obs)):
        flat.extend([dummy_pos, dummy_pos, 0.0])  # pad remaining slots
    return np.array(flat, dtype=float)


if __name__ == "__main__":
    # Standalone sanity checks for the e_y formula and helpers.
    wp_a, wp_b = (0.0, 0.0), (0.0, -30.0)
    chi_p = compute_path_angle(wp_a, wp_b)
    print(f"chi_p (path pointing -y) = {np.rad2deg(chi_p):.2f} deg (expected -90)")
    assert np.isclose(chi_p, -np.pi / 2)

    # On the path -> e_y should be 0
    e_y_on = compute_cross_track_error(0.0, -10.0, 0.0, -30.0, chi_p)
    print(f"e_y on path = {e_y_on:.4f} (expected 0)")
    assert np.isclose(e_y_on, 0.0, atol=1e-9)

    # Ship offset to +x (right of a ship travelling in -y direction) -> should be e_y > 0
    e_y_right = compute_cross_track_error(1.5, -10.0, 0.0, -30.0, chi_p)
    print(f"e_y offset +x by 1.5m = {e_y_right:.4f} (expected +1.5)")
    assert np.isclose(e_y_right, 1.5, atol=1e-9)

    e_y_left = compute_cross_track_error(-1.5, -10.0, 0.0, -30.0, chi_p)
    print(f"e_y offset -x by 1.5m = {e_y_left:.4f} (expected -1.5)")
    assert np.isclose(e_y_left, -1.5, atol=1e-9)

    beta = compute_sideslip(0.78, 0.0)
    print(f"beta at v=0 = {beta:.6f} (expected 0)")
    assert np.isclose(beta, 0.0, atol=1e-6)

    chi = compute_course_angle(chi_p, beta)
    psi_e = compute_course_error(chi, chi_p)
    print(f"psi_e when psi==chi_p = {psi_e:.6f} (expected 0)")
    assert np.isclose(psi_e, 0.0, atol=1e-9)

    xi_ref = get_reference_state(chi_p, 0.0, -30.0, 0.78)
    print(f"xi_ref = {xi_ref}")
    assert xi_ref.shape == (STATE_DIM,)

    waypoints = [(0.0, 0.0), (5.0, -10.0), (5.0, -30.0)]
    idx = select_active_waypoint(0.0, 0.0, waypoints, current_idx=0, wp_radius=1.0)
    print(f"waypoint idx near wp0 = {idx} (expected 1, switches)")
    assert idx == 1
    idx2 = select_active_waypoint(2.0, -4.0, waypoints, current_idx=1, wp_radius=1.0)
    print(f"waypoint idx before wp1 gate = {idx2} (expected 1, stays)")
    assert idx2 == 1

    # Regression test for test3_waypoint_switching_issue.md: a wide turning
    # transient overshoots leg 1's line, landing far (Euclidean) from wp1 but
    # past its perpendicular gate -> must still switch via the along-track test.
    idx3 = select_active_waypoint(0.0, -50.0, waypoints, current_idx=1, wp_radius=1.0)
    print(f"waypoint idx past wp1 gate (overshoot) = {idx3} (expected 2, switches via gate)")
    assert idx3 == 2

    mmg_state = [0.78, 0.0, 0.0, 1.5, 0.0, 0.0]
    xi = build_xi_full(mmg_state, delta=0.0, n=10.0, chi_p=chi_p, x_d=0.0, y_d=-30.0)
    print(f"xi from mmg_state = {xi}")
    assert np.isclose(xi[IDX_EY], 1.5, atol=1e-9)

    print("\nAll path_following.py self-tests PASSED.")
