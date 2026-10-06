"""1st-order Nomoto steering model for externally-observed moving obstacles
(e.g. obstacle_ship_node's live-tracked vessel) -- pure NumPy, no CasADi, never
imported by nmpc_acados.py/AcadosNMPC.solve() -- see nmpc/moving_obstacle.py's
own module docstring for why prediction and solver-consumption are kept
strictly separate, and nmpc/README.md's "Planned: Nomoto-driven moving
obstacles" section for the design this implements.

r_dot = (-r + K*delta) / T is the standard 1st-order Nomoto relation between
commanded rudder angle and yaw rate; x_dot = u*cos(psi), y_dot = u*sin(psi) is
plain kinematic integration at a constant surge speed (no sway) -- adequate
for a turning-inertia-aware obstacle proxy, not a maneuvering-accurate model
of the tracked vessel's own actual dynamics (that's whatever produced the
observed (x, y, psi, u, r, delta) in the first place, e.g. obstacle_ship_node's
real MMG simulation).

rollout_frozen_rudder() is the only entry point map_node.py needs: given a
moving obstacle's CURRENTLY OBSERVED state, it extrapolates forward assuming
the rudder stays at its last-observed value -- "assume it keeps doing what
it's doing right now," the standard assumption for a vessel whose actual
control intent is unknown.
"""
import numpy as np


def step(x: float, y: float, psi: float, u: float, r: float, delta: float,
         K: float, T: float, dt: float):
    """One RK4 step of [x_dot, y_dot, psi_dot, r_dot] with u and delta held
    constant across the step (T, K are the Nomoto time-constant/gain)."""
    def deriv(psi_, r_):
        return (u * np.cos(psi_), u * np.sin(psi_), r_, (-r_ + K * delta) / T)

    k1 = deriv(psi, r)
    k2 = deriv(psi + dt / 2.0 * k1[2], r + dt / 2.0 * k1[3])
    k3 = deriv(psi + dt / 2.0 * k2[2], r + dt / 2.0 * k2[3])
    k4 = deriv(psi + dt * k3[2], r + dt * k3[3])

    x_next = x + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    y_next = y + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    psi_next = psi + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    r_next = r + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
    return float(x_next), float(y_next), float(psi_next), float(r_next)


def rollout_frozen_rudder(x: float, y: float, psi: float, u: float, r: float, delta: float,
                           K: float, T: float, dt: float, N: int) -> np.ndarray:
    """Returns an (N+1, 3) array of (x, y, psi); row 0 is the given state
    itself (k=0, "now"), rows 1..N are the Nomoto rollout with delta and u
    both held fixed at their observed values for the whole horizon."""
    out = np.empty((N + 1, 3), dtype=float)
    out[0] = (x, y, psi)
    xk, yk, psik, rk = x, y, psi, r
    for k in range(1, N + 1):
        xk, yk, psik, rk = step(xk, yk, psik, u, rk, delta, K, T, dt)
        out[k] = (xk, yk, psik)
    return out
