"""
CasADi symbolic augmented dynamics — core of the NMPC prediction model.

xi = [e_y, sin(psi_e), cos(psi_e), r, x, y, psi, u, v, delta, n]   (11,)
u_aug = [delta_dot, n_dot]                                         (2,)

Section 1.3 of nmpc_Action_plan.md. The MMG accelerations/kinematics
(u_dot, v_dot, r_dot, x_dot, y_dot, psi_dot) come directly from
MMG_Time_Derivative_casadi in casadi_mmg_solver/casadi_mmg.py (untouched,
imported as-is); everything else (e_y, sin/cos(psi_e), delta, n rows) is
new bookkeeping layered on top per the path-following augmentation.

Discretization note: make_rk4_step() below performs a standard vector RK4
of the full 11-state augmented ODE — the same scheme acados' generic ERK
integrator applies to model.f_expl_expr. This keeps this module's own RK4 and
AcadosNMPC's numerically consistent with each other. It is NOT the same discretization
casadi_mmg.py's own make_casadi_integrator() uses internally (that helper
does a specialized "integrate in body frame, rotate once" RK4 trick for
[x, y, psi] only) — see the note in test_augmented_vs_mmg() below.
"""
import os
import sys
import casadi as ca
import numpy as np

# allow running this file directly (python nmpc/state_augmentation.py) by putting
# the repo root on sys.path, so `casadi_mmg_solver` and `nmpc` resolve as packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from casadi_mmg_solver.casadi_mmg import MMG_Time_Derivative_casadi, make_casadi_integrator
from nmpc.config import (
    STATE_DIM, CONTROL_DIM, MMG_STATE_DIM,
    IDX_EY, IDX_SPSI, IDX_CPSI, IDX_R, IDX_X, IDX_Y, IDX_PSI, IDX_U, IDX_V, IDX_DELTA, IDX_N,
    IDX_DDELTA, IDX_DN,
)
from nmpc.params import DEFAULT_CONFIG


def augmented_dynamics_casadi(xi_sym, u_aug_sym, chi_p_sym, current_sym=(0.0, 0.0)):
    """xi_dot = f_aug(xi, u_aug, chi_p, current) — Section 1.3 row by row.

    current_sym: earth-frame (vcx, vcy) [m/s], held constant over whatever
    horizon/step this is evaluated at (frozen-disturbance approximation --
    see CURRENT_AWARE_NMPC_PAPERS.md). Passed straight through into the MMG
    sub-block so the NMPC's own dynamics-continuity constraint reflects the
    same current-aware physics the plant/UKF already use.
    """
    # unpack the 11-state vector by name for readability
    e_y = xi_sym[IDX_EY]
    s_psi = xi_sym[IDX_SPSI]
    c_psi = xi_sym[IDX_CPSI]
    r = xi_sym[IDX_R]
    x = xi_sym[IDX_X]
    y = xi_sym[IDX_Y]
    psi = xi_sym[IDX_PSI]
    u = xi_sym[IDX_U]
    v = xi_sym[IDX_V]
    delta = xi_sym[IDX_DELTA]
    n = xi_sym[IDX_N]

    ddelta = u_aug_sym[IDX_DDELTA]
    dn = u_aug_sym[IDX_DN]

    # hand the 6-state sub-block to the untouched MMG model for the physics
    mmg_state = ca.vertcat(u, v, r, x, y, psi)
    mmg_control = ca.vertcat(delta, n)
    mmg_dot = MMG_Time_Derivative_casadi(mmg_state, mmg_control, current=current_sym)
    u_dot, v_dot, r_dot = mmg_dot[0], mmg_dot[1], mmg_dot[2]
    x_dot, y_dot, psi_dot = mmg_dot[3], mmg_dot[4], mmg_dot[5]

    # ey_dot = U sin(chi_p - chi); expressed via x_dot, y_dot (main.pdf Eq. 4)
    e_y_dot = -x_dot * ca.sin(chi_p_sym) + y_dot * ca.cos(chi_p_sym)

    # psi_e_dot = chi_dot - 0 = r + beta_dot ; beta_dot small for slow ships -> approximate by r
    psi_e_dot = r
    s_psi_dot = c_psi * psi_e_dot
    c_psi_dot = -s_psi * psi_e_dot

    # delta/n rows are trivial integrators: their "dynamics" IS the control input
    xi_dot = ca.vertcat(
        e_y_dot, s_psi_dot, c_psi_dot, r_dot,
        x_dot, y_dot, psi_dot,
        u_dot, v_dot,
        ddelta, dn,
    )
    return xi_dot


def make_rk4_step(dt: float, sym_type=ca.MX) -> ca.Function:
    """Builds ca.Function('rk4_step', [xi, u_aug, chi_p, current], [xi_next])
    via standard 4th-order vector Runge-Kutta of augmented_dynamics_casadi.
    current is held constant across all 4 RK stages, same convention chi_p
    already uses (and the same convention casadi_mmg.py's own integrator
    uses for current/wave_force with with_env=True)."""
    xi = sym_type.sym("xi", STATE_DIM)
    u_aug = sym_type.sym("u_aug", CONTROL_DIM)
    chi_p = sym_type.sym("chi_p")   # path angle, held constant over one RK4 step
    current = sym_type.sym("current", 2)   # [vcx, vcy], held constant over one RK4 step

    def f(xi_, u_):
        return augmented_dynamics_casadi(xi_, u_, chi_p, current)

    # standard 4-stage Runge-Kutta
    k1 = f(xi, u_aug)
    k2 = f(xi + dt / 2.0 * k1, u_aug)
    k3 = f(xi + dt / 2.0 * k2, u_aug)
    k4 = f(xi + dt * k3, u_aug)
    xi_next = xi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ca.Function("rk4_step", [xi, u_aug, chi_p, current], [xi_next],
                        ["xi", "u_aug", "chi_p", "current"], ["xi_next"])


def test_augmented_vs_mmg(sim_time: float = 200.0, dt: float = 0.05,
                           delta_deg: float = 35.0, rps: float = 18.2,
                           tol: float = 1e-6) -> bool:
    """
    Rolls out a constant-rudder turning circle two ways and checks that the
    MMG sub-block (u, v, r, x, y, psi) of the augmented dynamics is a faithful
    reproduction of the raw MMG_Time_Derivative_casadi model:

      (a) f_mmg_only : standard vector RK4 built directly on
                        MMG_Time_Derivative_casadi (6-state, no augmentation)
      (b) f_aug       : make_rk4_step() on the full 11-state augmented ODE,
                        reading back its [u, v, r, x, y, psi] rows

    Both (a) and (b) use the *same* RK4 discretization, so they must agree to
    numerical noise (< tol) — this isolates bugs in the augmentation (wrong
    index, sign, missing term) from RK4-scheme differences.

    A third, informational-only comparison is run against
    casadi_mmg.make_casadi_integrator() (the position-rotation "local frame"
    RK4 trick used elsewhere in this repo). That scheme is a *different*
    valid 4th-order discretization of the same continuous ODE, so it is
    expected to diverge slowly from (a)/(b) — the divergence is reported but
    not asserted against `tol`.
    """
    control_val = ca.DM([np.deg2rad(delta_deg), rps])
    N = int(sim_time / dt)

    # --- (a) raw MMG, standard vector RK4 ---
    state_sym = ca.MX.sym("state", MMG_STATE_DIM)
    control_sym = ca.MX.sym("control", 2)

    def f_mmg(s, c):
        return MMG_Time_Derivative_casadi(s, c)

    k1 = f_mmg(state_sym, control_sym)
    k2 = f_mmg(state_sym + dt / 2.0 * k1, control_sym)
    k3 = f_mmg(state_sym + dt / 2.0 * k2, control_sym)
    k4 = f_mmg(state_sym + dt * k3, control_sym)
    state_next = state_sym + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    f_mmg_only = ca.Function("f_mmg_only", [state_sym, control_sym], [state_next])

    # --- (b) augmented dynamics RK4 ---
    f_aug = make_rk4_step(dt)

    # --- (c) casadi_mmg.py's own specialized integrator (informational) ---
    f_local_frame = make_casadi_integrator(dt, method="rk4", sym_type=ca.MX)

    s_mmg = ca.DM([0.78, 0.0, 0.0, 0.0, 0.0, 0.0])
    xi = np.zeros(STATE_DIM)
    xi[IDX_U] = 0.78
    xi[IDX_DELTA] = np.deg2rad(delta_deg)
    xi[IDX_N] = rps
    xi[IDX_CPSI] = 1.0
    xi = ca.DM(xi)
    s_local = ca.DM([0.78, 0.0, 0.0, 0.0, 0.0, 0.0])

    u_aug_val = ca.DM([0.0, 0.0])   # no rudder/prop rate change: delta, n held constant
    chi_p_val = 0.0
    current_val = ca.DM([0.0, 0.0])   # disturbance-free, matches f_mmg_only/f_local_frame's defaults

    max_diff_ab = np.zeros(MMG_STATE_DIM)
    for _ in range(N):
        s_mmg = f_mmg_only(s_mmg, control_val)
        xi = f_aug(xi, u_aug_val, chi_p_val, current_val)
        s_local, _ = f_local_frame(s_local, control_val)

        xi_np = np.array(xi).flatten()
        aug_mmg_block = np.array([xi_np[IDX_U], xi_np[IDX_V], xi_np[IDX_R],
                                   xi_np[IDX_X], xi_np[IDX_Y], xi_np[IDX_PSI]])
        diff_ab = np.abs(np.array(s_mmg).flatten() - aug_mmg_block)
        max_diff_ab = np.maximum(max_diff_ab, diff_ab)

    diff_ac = np.abs(np.array(s_mmg).flatten() - np.array(s_local).flatten())

    names = ["u", "v", "r", "x", "y", "psi"]
    print("=== test_augmented_vs_mmg ===")
    print(f"(a) vs (b) [same RK4 scheme, must be < {tol:.0e}]:")
    for name, d in zip(names, max_diff_ab):
        print(f"    {name:>3}: {d:.3e}")
    print("(a) vs (c) [casadi_mmg.py local-frame RK4, informational only]:")
    for name, d in zip(names, diff_ac):
        print(f"    {name:>3}: {d:.3e}")

    passed = bool(np.all(max_diff_ab < tol))
    print("PASSED" if passed else "FAILED", "\n")
    return passed


if __name__ == "__main__":
    ok = test_augmented_vs_mmg()
    if not ok:
        raise SystemExit(1)
