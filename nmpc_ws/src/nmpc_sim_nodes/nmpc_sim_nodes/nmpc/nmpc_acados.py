"""
NMPC problem formulation, implemented with Acados OCP for real-time SQP-RTI --
the only solver backend nmpc_node uses.
"""
import os
import sys
import time
import numpy as np
import casadi as ca

# allow running this file directly (python nmpc/nmpc_acados.py) by putting
# the repo root on sys.path, so `nmpc` resolves as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("ACADOS_SOURCE_DIR", "/home/chandran/acados")
_acados_lib_dir = "/home/chandran/acados/lib"
if os.path.exists(_acados_lib_dir):
    # preload shared libs so LD_LIBRARY_PATH doesn't need to be set beforehand
    import ctypes
    try:
        mode = ctypes.RTLD_GLOBAL
        for _lib in ["libqdldl.so", "libosqp.so", "libqpOASES_e.so",
                     "libblasfeo.so", "libhpipm.so", "libacados.so"]:
            ctypes.CDLL(os.path.join(_acados_lib_dir, _lib), mode=mode)
    except Exception as e:
        print(f"Warning: programmatically loading acados libraries failed: {e}")

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from nmpc.config import (
    STATE_DIM, CONTROL_DIM,
    IDX_EY, IDX_SPSI, IDX_CPSI, IDX_R, IDX_X, IDX_Y, IDX_PSI, IDX_U, IDX_V, IDX_DELTA, IDX_N,
    IDX_DDELTA, IDX_DN,
)
from nmpc.params import DEFAULT_CONFIG
from nmpc.state_augmentation import augmented_dynamics_casadi
from nmpc.path_following import (
    build_xi_full, pad_obstacles, get_reference_state, wrap180_casadi, compute_effective_u_ref,
)


def _param_vector(config):
    """Length of the runtime parameter vector p:
    p = [chi_p, x_d, y_d, vcx, vcy, x_obs_1, y_obs_1, r_obs_1, ...]"""
    return 5 + 3 * config.MAX_OBSTACLES


def build_acados_ocp(config=DEFAULT_CONFIG) -> AcadosOcp:
    """Builds the acados OCP definition once, ahead of time (compiled to C).
    Runtime values are set later via solver.set(...) in AcadosNMPC.solve()."""
    N, n_obs = config.N, config.MAX_OBSTACLES

    # ---- model: state, control, params, and the ODE right-hand side ----
    model = AcadosModel()
    model.name = "nmpc_mmg_augmented"

    xi = ca.SX.sym("xi", STATE_DIM)
    u_aug = ca.SX.sym("u_aug", CONTROL_DIM)
    p = ca.SX.sym("p", _param_vector(config))

    chi_p = p[0]
    x_d = p[1]
    y_d = p[2]
    current_p = p[3:5]  # [vcx, vcy], frozen over the horizon (frozen-disturbance approximation)
    obs_p = p[5:]

    model.x = xi
    model.u = u_aug
    model.p = p
    model.f_expl_expr = augmented_dynamics_casadi(xi, u_aug, chi_p, current_p)

    # obstacle constraint expressions, one row per fixed obstacle slot:
    # (x_k - x_obs_i)^2 + (y_k - y_obs_i)^2 - r_c_i^2 + s_i >= 0
    h_list = []
    for i in range(n_obs):
        ox, oy, orad = obs_p[3 * i], obs_p[3 * i + 1], obs_p[3 * i + 2]
        r_c = orad + config.R_ASV
        h_list.append((xi[IDX_X] - ox) ** 2 + (xi[IDX_Y] - oy) ** 2 - r_c ** 2)
    model.con_h_expr = ca.vertcat(*h_list) if n_obs > 0 else ca.SX.zeros(0)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N

    # ---- cost: NONLINEAR_LS, y = [xi with psi row wrapped; u_aug] tracked to yref ----
    # (psi row is wrapped before squaring: raw psi never wraps back to (-pi,pi] but chi_p does, so a
    # long rollout can read an otherwise-fine heading as a huge error at this row's
    # dominant Q weight (main.pdf Q[psi]=30). LINEAR_LS can't express a wrapped
    # residual (not affine in xi), hence NONLINEAR_LS here instead of the plain
    # Vx/Vu selection-matrix form used for every other state.)
    ny = STATE_DIM + CONTROL_DIM
    ny_e = STATE_DIM

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_0 = "NONLINEAR_LS"  # explicit stage-0 cost type, avoids acados defaulting elsewhere
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    y_state = ca.vertcat(*[
        wrap180_casadi(xi[i] - chi_p) if i == IDX_PSI else xi[i]
        for i in range(STATE_DIM)
    ])
    model.cost_y_expr = ca.vertcat(y_state, u_aug)
    model.cost_y_expr_0 = model.cost_y_expr
    model.cost_y_expr_e = y_state

    ocp.cost.W = _block_diag(config.Q, config.R)     # stage weight = blockdiag(Q, R)
    ocp.cost.W_0 = _block_diag(config.Q, config.R)
    ocp.cost.W_e = config.Qe                          # terminal weight

    # placeholder yref (overwritten every solve() call, but acados requires a valid initial value).
    # psi's target is 0 here (not chi_p) because cost_y_expr already turns that row into
    # the wrapped residual (psi - chi_p) directly — the wrap+subtraction is baked into y,
    # so the target for it is simply "zero residual", same convention _yref_wrap_psi() uses at runtime.
    xi_ref0 = get_reference_state(0.0, 0.0, 0.0, config.U_REF, config.DELTA_TRIM, config.N_TRIM, config)
    yref0 = _yref_wrap_psi(xi_ref0)
    ocp.cost.yref = np.concatenate([yref0, np.zeros(CONTROL_DIM)])
    ocp.cost.yref_0 = np.concatenate([yref0, np.zeros(CONTROL_DIM)])
    ocp.cost.yref_e = yref0

    # ---- obstacle (soft) constraints: h >= 0, relaxable by slack down to -SIGMA ----
    if n_obs > 0:
        ocp.constraints.lh = np.zeros(n_obs)
        ocp.constraints.uh = np.full(n_obs, 1e8)  # "no upper bound"; kept << ACADOS_INFTY (1e10)
        ocp.constraints.idxsh = np.arange(n_obs)   # marks all h rows as soft (slacked)
        ocp.cost.zl = np.zeros(n_obs)
        ocp.cost.zu = np.zeros(n_obs)
        ocp.cost.Zl = config.W_SLACK * np.ones(n_obs)   # quadratic slack penalty weight
        ocp.cost.Zu = config.W_SLACK * np.ones(n_obs)
        ocp.constraints.lsh = -config.SIGMA * np.ones(n_obs)  # max allowed relaxation
        ocp.constraints.ush = np.zeros(n_obs)

    # ---- state/control bounds ----
    ocp.constraints.x0 = np.array(xi_ref0)  # placeholder; overwritten every solve() call

    ocp.constraints.idxbx = np.array([IDX_DELTA, IDX_N])   # only delta/n states are bounded
    ocp.constraints.lbx = np.array([config.DELTA_MIN, config.RPS_MIN])
    ocp.constraints.ubx = np.array([config.DELTA_MAX, config.RPS_MAX])

    ocp.constraints.idxbx_e = np.array([IDX_DELTA, IDX_N])  # same bounds at the terminal stage
    ocp.constraints.lbx_e = np.array([config.DELTA_MIN, config.RPS_MIN])
    ocp.constraints.ubx_e = np.array([config.DELTA_MAX, config.RPS_MAX])

    ocp.constraints.idxbu = np.array([IDX_DDELTA, IDX_DN])  # rate limits on both controls
    ocp.constraints.lbu = np.array([config.DELTA_DOT_MIN, config.RPS_DOT_MIN])
    ocp.constraints.ubu = np.array([config.DELTA_DOT_MAX, config.RPS_DOT_MAX])

    ocp.parameter_values = np.zeros(_param_vector(config))  # placeholder, overwritten every solve() call

    # ---- solver options ----
    ocp.solver_options.nlp_solver_type = config.ACADOS_NLP_SOLVER   # SQP_RTI = one Newton step per solve() call
    ocp.solver_options.integrator_type = config.ACADOS_INTEGRATOR
    ocp.solver_options.sim_method_num_stages = config.ACADOS_NUM_STAGES
    ocp.solver_options.sim_method_num_steps = config.ACADOS_NUM_STEPS
    ocp.solver_options.qp_solver = config.ACADOS_QP_SOLVER
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.tf = config.T_horizon
    ocp.code_export_directory = config.ACADOS_CODE_EXPORT_DIR  # where the generated C solver lands

    return ocp


def _yref_wrap_psi(xi_ref: np.ndarray) -> np.ndarray:
    """xi_ref with the psi entry zeroed — pairs with the NONLINEAR_LS cost_y_expr
    above, which already turns that row into the wrapped residual (psi - chi_p),
    so the target for it is just zero (not chi_p, which is baked into y itself)."""
    out = xi_ref.copy()
    out[IDX_PSI] = 0.0
    return out


def _block_diag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combines two square matrices into one block-diagonal matrix."""
    na, nb = a.shape[0], b.shape[0]
    out = np.zeros((na + nb, na + nb))
    out[:na, :na] = a
    out[na:, na:] = b
    return out


class AcadosNMPC:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.N = config.N
        self.dt = config.dt
        self.n_obs = config.MAX_OBSTACLES

        ocp = build_acados_ocp(config)
        json_path = os.path.join(os.path.dirname(__file__), "..", config.ACADOS_JSON_FILE)
        self.solver = AcadosOcpSolver(ocp, json_file=json_path)  # triggers C code-gen + compile on first call

        self._last_delta = config.DELTA_TRIM  # fallback command if a solve ever fails
        self._last_n = config.N_TRIM

    def solve(self, mmg_state, delta, n, chi_p, x_d, y_d, obstacles=None, current=(0.0, 0.0)):
        cfg = self.config
        if obstacles is None:
            obstacles = []

        xi_0 = build_xi_full(mmg_state, delta, n, chi_p, x_d, y_d)
        obs_flat = pad_obstacles(obstacles, self.n_obs)
        # current is set once per solve() call and held fixed across all N
        # stages below via the per-stage "p" set loop -- the frozen-disturbance
        # approximation (we only have a single online estimate, not a
        # horizon-length forecast).
        params = np.concatenate([[chi_p, x_d, y_d], current, obs_flat])  # matches _param_vector() order

        u_ref_eff = compute_effective_u_ref(mmg_state[3], mmg_state[4], x_d, y_d, cfg.U_REF, cfg)
        xi_ref = get_reference_state(chi_p, x_d, y_d, u_ref_eff, cfg.DELTA_TRIM, cfg.N_TRIM, cfg)
        # xi_ref (unwrapped, psi row = chi_p) is kept as-is for the return dict below
        # (controller-effort logging); acados' own NONLINEAR_LS cost needs the
        # separate psi-zeroed copy instead (wrapped residual baked into cost_y_expr).
        xi_ref_cost = _yref_wrap_psi(xi_ref)
        yref = np.concatenate([xi_ref_cost, np.zeros(CONTROL_DIM)])

        # pin the initial state to the actual current state (standard MPC shrinking-horizon setup)
        self.solver.set(0, "lbx", xi_0)
        self.solver.set(0, "ubx", xi_0)

        # push the same reference/parameters into every stage of the horizon
        for k in range(self.N):
            self.solver.set(k, "yref", yref)
            self.solver.set(k, "p", params)
        self.solver.set(self.N, "yref", xi_ref_cost)
        self.solver.set(self.N, "p", params)

        t0 = time.perf_counter()
        status = self.solver.solve()  # one SQP-RTI iteration; internal iterate persists across calls (warm start)
        solve_time = time.perf_counter() - t0

        success = (status == 0)

        u0 = self.solver.get(0, "u")
        xi_traj = np.array([self.solver.get(k, "x") for k in range(self.N + 1)]).T
        u_traj = np.array([self.solver.get(k, "u") for k in range(self.N)]).T

        if success:
            # integrate the first optimal rate one dt step to get the new actuator command
            delta_dot, n_dot = u0[IDX_DDELTA], u0[IDX_DN]
            delta_new = np.clip(delta + delta_dot * self.dt, cfg.DELTA_MIN, cfg.DELTA_MAX)
            n_new = np.clip(n + n_dot * self.dt, cfg.RPS_MIN, cfg.RPS_MAX)
            self._last_delta, self._last_n = delta_new, n_new
        else:
            # fall back to holding the previous actuator command
            delta_new, n_new = self._last_delta, self._last_n

        return {
            "u_opt": u_traj,
            "xi_traj": xi_traj,
            "xi_ref": xi_ref,
            "delta": float(delta_new),
            "n": float(n_new),
            "solve_time": solve_time,
            "success": success,
            "return_status": status,
        }
