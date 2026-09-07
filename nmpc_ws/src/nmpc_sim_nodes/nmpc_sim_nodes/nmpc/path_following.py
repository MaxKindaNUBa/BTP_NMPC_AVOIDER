"""
Pure NumPy path-following geometry (no CasADi).
Computes cross-track error, course angle and error, waypoint switching and the
NMPC reference state xi_ref. Called once per NMPC step from the outer loop.

Formulas: main.pdf Eq. (1)/(17-19), idekewfewf.pdf Eq. (51).
"""
import collections
import os
import sys
import numpy as np
import casadi as ca

# allow running this file directly (python nmpc/path_following.py) by putting
# the repo root on sys.path, so `nmpc` resolves as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nmpc.config import (
    STATE_DIM,
    IDX_EY, IDX_SPSI, IDX_CPSI, IDX_R, IDX_X, IDX_Y, IDX_PSI, IDX_U, IDX_V, IDX_DELTA, IDX_N,
)
from nmpc.params import DEFAULT_CONFIG


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


def _brake_ramp(dist, U_ref, config=DEFAULT_CONFIG) -> float:
    """Linear braking ramp: full U_ref outside config.BRAKE_DISTANCE, ramping down
    to config.U_REF_MIN right at the target. Shared core of compute_effective_u_ref
    (single active-target case, real distance) and build_horizon_references
    (per-stage, multi-segment case, predicted remaining distance)."""
    if dist >= config.BRAKE_DISTANCE:
        return float(U_ref)
    frac = dist / config.BRAKE_DISTANCE
    return float(config.U_REF_MIN + frac * (U_ref - config.U_REF_MIN))


def compute_effective_u_ref(x, y, x_d, y_d, U_ref, config=DEFAULT_CONFIG) -> float:
    """Linear braking ramp: full U_ref outside config.BRAKE_DISTANCE, ramping down
    to config.U_REF_MIN right at the target. Recomputed fresh every solve() call
    from the ship's true current distance to the active target (x_d, y_d) — this
    is what actually produces deceleration near arrival, since Q[x]/Q[y] (meters^2)
    dwarfs Q[u] ((m/s)^2) at any real distance, so a constant U_ref never gets
    "discovered" as needing to shrink by the cost weights alone."""
    return _brake_ramp(float(np.hypot(x - x_d, y - y_d)), U_ref, config)


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


def is_segment_crossed(x, y, seg_start, seg_end, wp_radius) -> bool:
    """True once (x,y) has passed seg_end's perpendicular gate along the
    seg_start->seg_end line, OR is directly within wp_radius of seg_end --
    the same along-track/cross-track/direct-hit test select_active_waypoint()
    uses to decide a waypoint is "reached", factored out here as a standalone
    predicate so it can drive an arbitrary segment queue's pop/drop logic
    (SegmentQueue.pop_crossed, build_horizon_references), not just the next
    waypoint in a fixed index-ordered sequence. See select_active_waypoint's
    own docstring for the geometric rationale (wide-turn overshoot + the
    cross-track bound added 2026-08-25)."""
    dist = float(np.hypot(x - seg_end[0], y - seg_end[1]))
    if dist < wp_radius:
        return True
    leg_dx, leg_dy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]
    leg_len = np.hypot(leg_dx, leg_dy)
    if leg_len < 1e-9:
        return False
    along = ((x - seg_end[0]) * leg_dx + (y - seg_end[1]) * leg_dy) / leg_len
    cross = np.sqrt(max(0.0, dist ** 2 - along ** 2))
    return bool(along >= 0.0 and cross <= wp_radius)


def select_active_waypoint(x, y, waypoints, current_idx, wp_radius: float = None,
                            config=DEFAULT_CONFIG) -> int:
    """Advances current_idx to current_idx+1 once the ship has either come within
    wp_radius of the target waypoint, OR crossed the perpendicular "gate" plane
    through it WITHIN wp_radius of it laterally too (projection of (x,y) onto
    the prev_wp->wp leg direction is past wp, AND the cross-track/perpendicular
    offset from the leg line at that point is <= wp_radius).

    The radius-only check can fail to fire forever: a large initial heading error
    makes the turning transient converge onto the path's line well beyond the
    target point, so Euclidean distance to that point never dips below wp_radius
    again (see test3_waypoint_switching_issue.md). The along-track gate test
    catches this because it only cares about progress along the leg direction,
    not lateral offset, so it fires exactly once as the ship passes abeam of the
    waypoint no matter how wide the turn was.

    CROSS-TRACK BOUND ADDED 2026-08-25: the gate used to accept ANY lateral
    offset once past the waypoint's along-track position -- fine for a wide
    turning transient (which stays close to the line), but a real bug for an
    obstacle-avoidance detour: this project's scenarios route obstacles with
    radii up to ~6m right next to waypoints (see scenario.json), so a detour
    swinging 6-10m off the path line crosses the gate plane FAR from the
    actual waypoint and was being accepted as "reached" -- observed live via
    rviz_node.py: the active-waypoint marker jumped straight to the final
    endpoint while the ship was still visibly nowhere near the middle
    waypoint. Bounding the gate to wp_radius laterally (the same radius
    already used for the direct-hit check) closes that off while leaving the
    original wide-turning-transient fix intact, since that case has small
    cross-track offset by construction.
    """
    if wp_radius is None:
        wp_radius = config.WP_RADIUS

    last_idx = len(waypoints) - 1
    current_idx = min(current_idx, last_idx)
    wp = waypoints[current_idx]
    # prev_wp == wp (degenerate zero-length leg) when current_idx==0: is_segment_crossed's
    # leg_len<1e-9 guard then falls back to the direct-hit radius check alone, matching this
    # function's original behavior of skipping the along-track test with no previous leg.
    prev_wp = waypoints[current_idx - 1] if current_idx > 0 else wp
    reached = is_segment_crossed(x, y, prev_wp, wp, wp_radius)

    if reached and current_idx < last_idx:
        current_idx += 1   # close enough, or already passed it -> move to next leg
    return min(current_idx, last_idx)


def segments_from_waypoints(waypoints, target_idx):
    """Default segment producer for a static waypoint path (e.g. scenario.json):
    the ordered chain waypoints[target_idx] -> waypoints[target_idx+1] -> ... ->
    last waypoint, each entry as (chi_p, end_x, end_y), suitable for seeding a
    SegmentQueue. segments[0]'s chi_p is fixed to the REAL current leg's line
    (prev_wp->waypoints[target_idx]), matching select_active_waypoint's own
    convention -- not recomputed from the ship's live position. A future
    dynamic producer (e.g. a LIDAR rolling window) would supply its own list
    in this same (chi_p, end_x, end_y) shape instead of calling this."""
    last_idx = len(waypoints) - 1
    idx = min(target_idx, last_idx)
    prev_wp = waypoints[idx - 1] if idx > 0 else waypoints[idx]
    segs = [(compute_path_angle(prev_wp, waypoints[idx]), waypoints[idx][0], waypoints[idx][1])]
    i = idx
    while i < last_idx:
        segs.append((compute_path_angle(waypoints[i], waypoints[i + 1]),
                      waypoints[i + 1][0], waypoints[i + 1][1]))
        i += 1
    return segs


class SegmentQueue:
    """Ordered, mutable queue of active path segments -- (chi_p, end_x, end_y)
    tuples -- forming the chained reference an NMPC horizon previews (see
    build_horizon_references). Newer segments are appended to the back; a
    segment is popped from the front once its endpoint is crossed by the
    ship's live position. Exists standalone (not baked into map_node) so a
    future segment PRODUCER -- e.g. a LIDAR-derived rolling window of
    locally-sensed segments -- can drive the same queue via
    append()/pop_crossed(), with no change on the NMPC-consuming side."""

    def __init__(self, initial_segments=()):
        self._segments = collections.deque(initial_segments)

    def append(self, chi_p, end_x, end_y):
        self._segments.append((chi_p, end_x, end_y))

    def pop_crossed(self, x, y, wp_radius) -> int:
        """Pops segments from the front while their endpoint has been crossed
        by (x,y) -- each successive check's "start" is the previous popped
        segment's endpoint, or (x,y) itself for the first pending one,
        matching how build_horizon_references re-chains survivors. Returns
        how many were popped (0 most ticks). Never pops the LAST remaining
        segment, even if crossed -- mirrors select_active_waypoint's own
        current_idx < last_idx guard (it never advances past the final
        waypoint either): once the queue is down to one segment, that's the
        final destination, and there's nothing to replace it with, so it
        stays referenced (station-keeping) instead of leaving the queue
        empty for build_horizon_references to choke on."""
        popped = 0
        seg_start = (x, y)
        while len(self._segments) > 1:
            chi_p, ex, ey = self._segments[0]
            if not is_segment_crossed(x, y, seg_start, (ex, ey), wp_radius):
                break
            self._segments.popleft()
            seg_start = (ex, ey)
            popped += 1
        return popped

    def __len__(self):
        return len(self._segments)

    @property
    def segments(self):
        return list(self._segments)


def build_horizon_references(x, y, segments, N, dt, speed_est, U_ref,
                              config=DEFAULT_CONFIG, max_segments=None):
    """Walks an ordered list of candidate path segments [(chi_p, end_x,
    end_y), ...] -- typically SegmentQueue(...).segments, assumed chained
    (each segment's start is the previous segment's end, or the ship's
    current position for the first surviving one) -- at a constant
    speed_est, and returns 5 length-(N+1) arrays (chi_p, x_d, y_d, u_ref_eff,
    dist_to_corner): the reference every OCP stage k=0..N should track;
    dist_to_corner is that stage's predicted arclength distance to the
    nearest real waypoint (see below). Handles however many segments
    actually fall inside the horizon, not just one -- matters when segments
    are packed closer together than the horizon's travel distance (e.g.
    ~1-2 ship-lengths apart, see nmpc/README.md item 8), and works whether
    `segments` came from a long static queue or a short dynamic one.

    HEADING vs POSITION are deliberately decoupled: chi_p_arr previews
    ahead across however many segments fit in the horizon, but
    x_d_arr/y_d_arr stay pinned to survivors[0]'s endpoint -- the single
    REAL current target -- for every stage, never advancing to a later
    segment's endpoint just because the internal arclength walk predicts a
    stage is "past" it. Only the actual gate (SegmentQueue.pop_crossed,
    between solve() calls) changes what the real current target is.

    max_segments: cap on how many segments (after dropping already-crossed
    ones) are ever considered, independent of how many would geometrically
    fit in the horizon. None (default) means no cap -- a future caller (e.g.
    a perception source with its own confidence window) can pass a smaller
    value without needing a different function.

    Any segment whose endpoint (x,y) has already passed (is_segment_crossed,
    against that segment's own start/end -- the previous surviving segment's
    endpoint, or (x,y) itself for the first one) is dropped before the walk,
    then survivors are re-chained starting from (x,y). This is a no-op when
    `segments` already came from a queue that's had pop_crossed() applied
    against the same (x,y) (map_node's normal usage) -- kept here too as a
    defensive, general guarantee for any caller, including one that doesn't
    pre-filter.

    Segments past the horizon's max reach (N*dt*speed_est) are dropped too --
    never referenced by any of the returned per-stage arrays, so a caller
    that then does solver.set(k, ...) for k=0..N can never end up pushing an
    out-of-horizon segment into the solver.

    Once segments are exhausted, remaining stages hold the last surviving
    segment's leg (today's terminal-approach behavior, unchanged).
    """
    segments = list(segments)
    start = (x, y)
    survivors = []
    for chi_p, ex, ey in segments:
        if is_segment_crossed(x, y, start, (ex, ey), config.WP_RADIUS):
            continue    # stale -- drop, don't advance `start` past it
        survivors.append((chi_p, ex, ey))
        start = (ex, ey)
        if max_segments is not None and len(survivors) >= max_segments:
            break

    if not survivors and segments:
        # every candidate looked "crossed" (e.g. the ship is sitting right on
        # top of the final destination, station-keeping there -- mirrors
        # SegmentQueue.pop_crossed's own "never drop the last one" guard):
        # fall back to the last entry in the given list rather than leaving
        # nothing to track at all.
        survivors = [segments[-1]]

    cum_dist = [0.0]
    prev_pt = (x, y)
    for _, ex, ey in survivors:
        cum_dist.append(cum_dist[-1] + float(np.hypot(ex - prev_pt[0], ey - prev_pt[1])))
        prev_pt = (ex, ey)
        if cum_dist[-1] >= N * dt * speed_est:   # stop once past max horizon reach
            break
    cum_dist = np.array(cum_dist)
    survivors = survivors[:len(cum_dist) - 1]   # segments past max horizon reach are dropped
    # here -- never referenced by any solver.set(k, ...) call below, since only stages k=0..N
    # ever get built, and no `seg` index resolved from `cum_dist` can point past `survivors`.

    if not survivors:
        raise ValueError("build_horizon_references: no segment to track -- `segments` was "
                          "empty, or every entry was already crossed. Callers must always "
                          "supply at least one not-yet-crossed segment (e.g. nmpc_acados.py's "
                          "solve() falls back to a degenerate single-segment list built from "
                          "its own chi_p/x_d/y_d args when the live queue is momentarily empty).")

    # Every cum_dist[1:] entry is a real waypoint the survivor chain passes
    # through (cum_dist[0]=0 is just the ship's own current position, not a
    # waypoint to gate through). Used below to build dist_to_corner_arr --
    # how far each stage's predicted position is from the NEAREST such
    # waypoint, in either direction -- so a caller can locally boost
    # point-tracking weight right around a crossing (see
    # nmpc_acados.py's WAYPOINT_PASSAGE_DIST/_XY_BOOST) without touching the
    # cost anywhere else along the leg.
    waypoint_arclens = cum_dist[1:]

    chi_p_arr = np.empty(N + 1)
    x_d_arr = np.empty(N + 1)
    y_d_arr = np.empty(N + 1)
    u_ref_arr = np.empty(N + 1)
    dist_to_corner_arr = np.empty(N + 1)
    for k in range(N + 1):
        arclen_k = k * dt * speed_est
        seg = max(min(int(np.searchsorted(cum_dist, arclen_k, side='right')) - 1,
                       len(survivors) - 1), 0)
        chi_p_arr[k] = survivors[seg][0]
        # Position target stays pinned to the REAL current target (survivors[0],
        # the queue front) for every stage -- deliberately NOT survivors[seg] --
        # see the decoupling note in this function's docstring.
        x_d_arr[k], y_d_arr[k] = survivors[0][1], survivors[0][2]
        remaining = max(cum_dist[seg + 1] - arclen_k, 0.0) if seg + 1 < len(cum_dist) else 0.0
        u_ref_arr[k] = _brake_ramp(remaining, U_ref, config)
        dist_to_corner_arr[k] = float(np.min(np.abs(waypoint_arclens - arclen_k)))
    return chi_p_arr, x_d_arr, y_d_arr, u_ref_arr, dist_to_corner_arr


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


def pad_walls(walls, max_walls: int, dummy_pos: float = 1.0e3) -> np.ndarray:
    """pad_obstacles' twin for wall (capsule) obstacles: pads/truncates a list
    of (x0, y0, x1, y1, radius) segments to a fixed length. Unused slots get a
    zero-length, zero-radius dummy segment far away, same spirit as
    pad_obstacles' dummy circle -- trivially satisfied, never binds the
    soft-min aggregate (see softmin_casadi)."""
    w = list(walls)[:max_walls]
    flat = []
    for (x0, y0, x1, y1, r) in w:
        flat.extend([x0, y0, x1, y1, r])
    for _ in range(max_walls - len(w)):
        flat.extend([dummy_pos, dummy_pos, dummy_pos, dummy_pos, 0.0])
    return np.array(flat, dtype=float)


def capsule_distance_casadi(x, y, p0x, p0y, p1x, p1y, r, eps):
    """Symbolic (CasADi) signed distance from point (x, y) to a capsule --
    a line segment [p0, p1] padded by radius r. A circle is the degenerate
    case p0 == p1: the segment-length denominator below is guarded by `eps`
    so t safely reads as 0/eps == 0 (numerator is also exactly 0 when
    p1 == p0, since it's a dot product against the same zero vector), giving
    closest == p0 exactly -- the correct circle-distance result, no branching
    needed. See research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md."""
    ex, ey = p1x - p0x, p1y - p0y
    t_raw = ((x - p0x) * ex + (y - p0y) * ey) / (ex * ex + ey * ey + eps)
    t = ca.fmin(ca.fmax(t_raw, 0.0), 1.0)
    closest_x = p0x + t * ex
    closest_y = p0y + t * ey
    return ca.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2 + eps) - r


_SOFTMIN_DISTANCE_CAP = 100.0  # meters -- see softmin_casadi's docstring


def softmin_casadi(distances, k):
    """Smooth soft-min of a list of CasADi scalars: -1/k * log(sum(exp(-k*d_i))).
    Always <= true min(distances) (sum >= max single term), so the resulting
    constraint is never less conservative than the true nearest-obstacle
    distance -- see research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md for
    the full derivation and error-vs-k tradeoff.

    Each d_i is independently clipped to _SOFTMIN_DISTANCE_CAP before
    exponentiating (ca.fmin, not ca.mmin) -- this prevents exp(-k*d_i)
    underflowing to a hard 0.0 in float64 for a far-away/dummy (padded,
    unused) obstacle slot: with dummies ~1400m away and k=3, k*d_i ~ 4200,
    far past exp()'s ~-745 underflow floor, so if EVERY slot is a dummy (no
    real obstacles at all) the unclipped sum is exactly 0.0 and log(0.0) =
    -inf, poisoning every QP solve.

    Deliberately NOT a shared ca.mmin(d_stack) shift (the more textbook
    log-sum-exp stabilization): that was tried first and is WORSE here --
    padded slots are all placed at the exact same dummy coordinate, so their
    distance functions are literally identical for every (x, y), not just
    numerically tied at one point. Differentiating a shared hard-min over
    exactly-identical functions is degenerate everywhere (not a rare edge
    case), and corrupted the SQP-RTI gradient badly enough to break ordinary
    path tracking even with zero real obstacles present (confirmed via
    ros2 run nmpc_sim_nodes test_nmpc regressing all 4 tracking tests).
    Per-term clipping has no such cross-term coupling: each d_i's kink at
    the cap is independent of every other slot, so identical/tied dummies
    are simply identical/tied clipped terms -- no shared subgradient to get
    wrong. The cap only ever discards precision far past any real safety
    margin (see the docstring's tail on the preserved D_hat <= true_min
    property), never near where the constraint could plausibly bind."""
    terms = [ca.exp(-k * ca.fmin(d, _SOFTMIN_DISTANCE_CAP)) for d in distances]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return -ca.log(total) / k


def pad_ellipses(ellipses, max_ellipses: int, dummy_pos: float = 1.0e3) -> np.ndarray:
    """pad_obstacles/pad_walls' twin for elliptical obstacles: pads/truncates a list
    of (xc, yc, a, b, theta) tuples to a fixed length. Unused slots get a=b=1.0 (NOT
    0.0 -- avoids a 0/0-shaped denominator in ellipse_distance_casadi's gradient
    magnitude) at a far-away dummy center, theta=0."""
    e = list(ellipses)[:max_ellipses]
    flat = []
    for (xc, yc, a, b, theta) in e:
        flat.extend([xc, yc, a, b, theta])
    for _ in range(max_ellipses - len(e)):
        flat.extend([dummy_pos, dummy_pos, 1.0, 1.0, 0.0])
    return np.array(flat, dtype=float)


def ellipse_distance_casadi(x, y, xc, yc, a, b, theta, r_pad, eps):
    """Symbolic (CasADi) APPROXIMATE signed distance from point (x, y) to an
    elliptical obstacle (center (xc, yc), semi-axes a/b, rotation theta, radians,
    world frame) padded by r_pad. Unlike a circle or capsule, an ellipse has NO
    closed-form Euclidean distance (would require solving a quartic) -- see
    research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md for the full derivation,
    numerical verification, and the accuracy caveat summarized below.

    Computed via GRADIENT NORMALIZATION of the ellipse's implicit algebraic
    equation g(x,y) = (dx/a)^2 + (dy/b)^2 - 1 (0 on the boundary, <0 inside, >0
    outside; dx/dy are (x,y) rotated into the ellipse's own frame) -- the same
    family of technique as Rimon-Koditschek navigation-function potentials:
    d = g / |grad g| is a first-order-accurate approximate distance, exact right
    at the boundary (g=0), which is exactly where the SIGMA-bounded soft
    constraint actually cares.

    VERIFIED CONSERVATIVE (never overestimates clearance -- d <= true nearest-
    boundary distance) via a 960-point numerical sweep across 5 eccentricities x
    24 angles x 8 range factors during design, with zero violations -- the same
    safety property softmin_casadi's own docstring establishes for the aggregate,
    so an ellipse term drops directly into the same d_list with no special
    handling. Accuracy DOES degrade with range and off-axis eccentricity (the
    verified far-field ratio to true distance settles ~0.5 for a circle, as low
    as ~0.26-0.36 off-axis for a 10:1 aspect-ratio ellipse -- NOT asymptotically
    exact, an earlier draft's claim to the contrary was checked and found wrong).
    This degradation only ever makes the constraint react a bit earlier than
    strictly necessary to a distant, eccentric ellipse, never later -- for
    scenarios needing tight long-range accuracy against a very elongated
    ellipse, prefer a chain of capsules instead.

    r_pad is subtracted from the normalized value (not added to a/b before
    normalizing -- Minkowski-padding an ellipse by a disk isn't itself an
    ellipse, no simple closed form). Note the resulting calling convention
    deliberately differs from capsule_distance_casadi's: an ellipse's own size
    is already fully encoded in a/b, so callers should pass r_pad=config.R_ASV
    alone (ship-only padding), NOT object_radius + config.R_ASV like circles/
    walls do."""
    dx = (x - xc) * ca.cos(theta) + (y - yc) * ca.sin(theta)
    dy = -(x - xc) * ca.sin(theta) + (y - yc) * ca.cos(theta)
    g = (dx / a) ** 2 + (dy / b) ** 2 - 1.0
    grad_mag = 2.0 * ca.sqrt((dx / a ** 2) ** 2 + (dy / b ** 2) ** 2)
    return g / (grad_mag + eps) - r_pad


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
    # Position chosen ON the wp0->wp1 line, 5m past wp1 along-track (cross=0),
    # matching the 5b cross-track-bound fix's own stated assumption that this
    # case "has small cross-track offset by construction" (path_following.py's
    # select_active_waypoint docstring) -- the original (0.0, -50.0) position
    # here had a ~22m cross-track offset, already failing this assertion on
    # master before any of the horizon-preview changes (confirmed via `git
    # stash`), so this is a stale-fixture fix, not a behavior change.
    idx3 = select_active_waypoint(7.236, -14.472, waypoints, current_idx=1, wp_radius=1.0)
    print(f"waypoint idx past wp1 gate (overshoot) = {idx3} (expected 2, switches via gate)")
    assert idx3 == 2

    mmg_state = [0.78, 0.0, 0.0, 1.5, 0.0, 0.0]
    xi = build_xi_full(mmg_state, delta=0.0, n=10.0, chi_p=chi_p, x_d=0.0, y_d=-30.0)
    print(f"xi from mmg_state = {xi}")
    assert np.isclose(xi[IDX_EY], 1.5, atol=1e-9)

    # ---- build_horizon_references / SegmentQueue self-tests ----

    # (a) Backward-compat: a widely-spaced single-leg scenario reduces to
    # today's constant-reference, single-target compute_effective_u_ref behavior.
    N_h, dt_h, speed_h = 200, 0.1, 0.7
    single_seg_waypoints = [(0.0, 0.0), (0.0, -100.0)]
    single_segs = segments_from_waypoints(single_seg_waypoints, target_idx=1)
    chi_a, xd_a, yd_a, uref_a, _ = build_horizon_references(
        0.0, 0.0, single_segs, N_h, dt_h, speed_h, 0.7, DEFAULT_CONFIG)
    print(f"(a) single-leg: chi_p const={np.allclose(chi_a, chi_a[0])}, "
          f"x_d const={np.allclose(xd_a, 0.0)}, y_d const={np.allclose(yd_a, -100.0)}, "
          f"u_ref[0]={uref_a[0]:.4f}")
    assert np.allclose(chi_a, compute_path_angle((0.0, 0.0), (0.0, -100.0)))
    assert np.allclose(xd_a, 0.0) and np.allclose(yd_a, -100.0)
    assert np.isclose(uref_a[0], compute_effective_u_ref(0.0, 0.0, 0.0, -100.0, 0.7, DEFAULT_CONFIG))

    # (b) + (f): waypoints packed ~1-2 ship-lengths apart (~4-5m), each leg at
    # a distinct heading so multi-segment HEADING preview is unambiguous.
    # chi_p_arr should show several distinct headings inside the horizon
    # (heading preview), while x_d_arr/y_d_arr stay pinned to the single
    # REAL current target (position/heading are decoupled -- see this
    # function's docstring) and any segment past the horizon's max reach
    # must never appear in chi_p_arr at all.
    packed_waypoints = [(0.0, 0.0), (4.0, 0.0), (8.0, 2.0), (11.0, 6.0), (11.0, 10.0), (8.0, 13.0)]
    packed_segs = segments_from_waypoints(packed_waypoints, target_idx=1)
    chi_b, xd_b, yd_b, _, _ = build_horizon_references(0.0, 0.0, packed_segs, N_h, dt_h, speed_h, 0.7, DEFAULT_CONFIG)
    distinct_headings_b = {round(float(c), 4) for c in chi_b}
    last_leg_heading = round(compute_path_angle(packed_waypoints[4], packed_waypoints[5]), 4)
    print(f"(b) packed waypoints: distinct headings previewed in horizon = {len(distinct_headings_b)} "
          f"({sorted(distinct_headings_b)}), x_d/y_d pinned={np.allclose(xd_b, 4.0) and np.allclose(yd_b, 0.0)}")
    assert len(distinct_headings_b) > 2, "expected more than 2 distinct headings inside the horizon"
    assert last_leg_heading not in distinct_headings_b, "out-of-horizon-reach segment must never be previewed"
    assert np.allclose(xd_b, 4.0) and np.allclose(yd_b, 0.0), \
        "position target must stay pinned to the real current target (survivors[0]), not advance with chi_p"

    # (c) max_segments caps how many segments are ever considered, independent
    # of how many would geometrically fit -- checked via chi_p_arr now that
    # x_d_arr/y_d_arr no longer vary by segment.
    chi_c, xd_c, yd_c, _, _ = build_horizon_references(0.0, 0.0, packed_segs, N_h, dt_h, speed_h, 0.7,
                                                         DEFAULT_CONFIG, max_segments=2)
    distinct_headings_c = {round(float(c), 4) for c in chi_c}
    expected_headings_c = {round(compute_path_angle(packed_waypoints[0], packed_waypoints[1]), 4),
                            round(compute_path_angle(packed_waypoints[1], packed_waypoints[2]), 4)}
    print(f"(c) max_segments=2: distinct headings previewed = {len(distinct_headings_c)} ({sorted(distinct_headings_c)})")
    assert distinct_headings_c == expected_headings_c
    assert np.allclose(xd_c, 4.0) and np.allclose(yd_c, 0.0)

    # (d) a hand-built, out-of-order/overlapping segment list: the ship's
    # live position is already effectively at the SECOND segment's endpoint
    # even though the first segment (earlier in the list) hasn't been
    # reached yet -- the stale second segment must be dropped, not referenced.
    overlapping_segs = [(0.0, 10.0, 0.0), (np.pi, 0.0, 0.0)]
    chi_d, xd_d, yd_d, _, _ = build_horizon_references(0.0, 0.5, overlapping_segs, 50, dt_h, speed_h, 0.7, DEFAULT_CONFIG)
    print(f"(d) overlapping segments: x_d unique={set(np.round(xd_d, 6))}, y_d unique={set(np.round(yd_d, 6))}, "
          f"chi_p unique={set(np.round(chi_d, 6))}")
    assert np.allclose(xd_d, 10.0) and np.allclose(yd_d, 0.0), "the already-crossed segment must be dropped"
    assert np.allclose(chi_d, 0.0), "the dropped segment's heading (pi) must never be previewed either"

    # (e) SegmentQueue: append() then pop_crossed() pops exactly one once the
    # front segment's endpoint has been passed; a not-yet-reached position pops zero.
    q = SegmentQueue([(0.0, 10.0, 0.0)])
    popped_none = q.pop_crossed(0.0, 0.0, wp_radius=2.0)
    print(f"(e) pop_crossed before reaching front segment: popped={popped_none} (expected 0)")
    assert popped_none == 0 and len(q) == 1
    q.append(np.pi, 0.0, 0.0)
    popped_one = q.pop_crossed(9.9, 0.0, wp_radius=2.0)
    print(f"(e) pop_crossed just past front segment's endpoint: popped={popped_one} (expected 1), "
          f"remaining={q.segments}")
    assert popped_one == 1 and q.segments == [(np.pi, 0.0, 0.0)]

    # (f2) SegmentQueue never pops its LAST remaining segment, even once
    # crossed -- station-keeps at the final destination instead of emptying.
    q_final = SegmentQueue([(0.0, 0.0, 0.0)])
    popped_final = q_final.pop_crossed(0.0, 0.0, wp_radius=2.0)  # sitting right on top of it
    print(f"(f2) pop_crossed on the last remaining segment: popped={popped_final} (expected 0), "
          f"remaining={q_final.segments}")
    assert popped_final == 0 and len(q_final) == 1

    # (g) build_horizon_references falls back to the last given segment
    # rather than raising when the ship is already sitting on top of it
    # (the single-remaining-waypoint / station-keeping case).
    chi_g, xd_g, yd_g, _, _ = build_horizon_references(0.0, 0.0, [(0.0, 0.0, 0.0)], 20, dt_h, speed_h, 0.7, DEFAULT_CONFIG)
    print(f"(g) ship on final waypoint: x_d const={np.allclose(xd_g, 0.0)}, y_d const={np.allclose(yd_g, 0.0)}")
    assert np.allclose(xd_g, 0.0) and np.allclose(yd_g, 0.0)

    # (h) dist_to_corner_arr: for a single 10m leg walked at 1 m/s, stage k=0
    # (arclen=0) should be 10m from the only waypoint, and it should shrink
    # monotonically to 0 exactly at the stage whose arclength reaches 10m.
    _, _, _, _, dist_h = build_horizon_references(0.0, 0.0, [(0.0, 10.0, 0.0)], 20, 1.0, 1.0, 0.7, DEFAULT_CONFIG)
    print(f"(h) dist_to_corner: k=0 -> {dist_h[0]:.2f}m (expected 10.0), "
          f"k=10 -> {dist_h[10]:.2f}m (expected 0.0), monotonic decreasing over 0..10: "
          f"{bool(np.all(np.diff(dist_h[:11]) <= 1e-9))}")
    assert np.isclose(dist_h[0], 10.0) and np.isclose(dist_h[10], 0.0)
    assert np.all(np.diff(dist_h[:11]) <= 1e-9)

    print("\nAll path_following.py self-tests PASSED.")
