"""Pure NumPy constant-velocity prediction, over the NMPC's own future
horizon, for any obstacle primitive (circle/Obstacle, wall/WallObstacle,
ellipse/Ellipse) that has been given a velocity -- see
research_papers/COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md (theory-only design
note) section 2 for why constant-velocity extrapolation is the standard
treatment, and section 3 for the growing-margin rationale. Velocity is a
per-obstacle (vx, vy) attribute on the SAME message types the static case
already uses (Obstacle.msg/WallObstacle.msg/Ellipse.msg) -- there is no
separate "moving obstacle" primitive; a circle/wall/ellipse with vx=vy=0.0 is
simply stationary, the default for every obstacle placed before this field
existed.

DELIBERATELY OUT OF SCOPE (see the same doc, and the plan this module was
built from): this file has no CasADi/acados dependency and is never imported
by nmpc_acados.py or wired into AcadosNMPC.solve()'s obstacles=/walls=/
ellipses= kwargs -- nmpc_node.py's _on_obstacles/_on_walls/_on_ellipses
callbacks explicitly extract only (x, y, radius) / (x0, y0, x1, y1, radius) /
(xc, yc, a, b, theta) from each message, so vx/vy never reach the solver's
cache regardless of what's published on /map/obstacles et al. Predicting an
obstacle's future path is not the same as the NMPC knowing how to react to
it -- feeding a predicted-but-unhandled moving obstacle into the existing
static-obstacle constraint would be worse than not seeing it at all (a stale
position, not a live one). This module is prediction/visualization only.
"""
import numpy as np


def predict_moving_obstacle_positions(x: float, y: float, vx: float, vy: float,
                                       dt: float, N: int) -> np.ndarray:
    """Constant-velocity extrapolation of one moving obstacle over the horizon.

    Returns an (N+1, 2) array; row k is (x + k*dt*vx, y + k*dt*vy) for
    k = 0..N -- the same stage indexing as the NMPC's own xi_traj, so a
    caller with an obstacle's live (x, y) can predict exactly as far ahead as
    one solve's horizon reaches.
    """
    k = np.arange(N + 1, dtype=float)
    xs = x + k * dt * vx
    ys = y + k * dt * vy
    return np.stack([xs, ys], axis=1)


def growing_radius(radius: float, k: int, growth_per_step: float) -> float:
    """Linear growth of a moving obstacle's effective safety radius with
    horizon stage k -- the simple mitigation for constant-velocity
    extrapolation error compounding further into the horizon (see the design
    doc's section 3). growth_per_step=0.0 recovers a constant radius."""
    return radius + growth_per_step * k
