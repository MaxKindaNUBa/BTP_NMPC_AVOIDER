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

# "No upper bound" sentinel for a one-sided (lower-only) state bound below --
# comfortably under acados' own infinity threshold (1e10, see the obstacle
# dummy-distance fix in nmpc/README.md's bug list) but far past any realistic
# surge speed, so it never binds.
_U_NO_UPPER_BOUND = 1e3
from nmpc.params import DEFAULT_CONFIG
from nmpc.state_augmentation import augmented_dynamics_casadi
from nmpc.path_following import (
    build_xi_full, pad_obstacles, pad_walls, pad_ellipses, get_reference_state, wrap180_casadi,
    build_horizon_references, capsule_distance_casadi, ellipse_distance_casadi, softmin_casadi,
)


def _param_vector(config):
    """Length of the runtime parameter vector p:
    p = [chi_p, x_d, y_d, vcx, vcy,
         x_obs_1, y_obs_1, r_obs_1, ...,                      (config.MAX_OBSTACLES circle slots)
         x0_wall_1, y0_wall_1, x1_wall_1, y1_wall_1, r_wall_1, ...,  (config.MAX_WALLS capsule slots)
         xc_ell_1, yc_ell_1, a_ell_1, b_ell_1, theta_ell_1, ...]  (config.MAX_ELLIPSES ellipse slots)
    See research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md."""
    return 5 + 3 * config.MAX_OBSTACLES + 5 * config.MAX_WALLS + 5 * config.MAX_ELLIPSES


def build_acados_ocp(config=DEFAULT_CONFIG) -> AcadosOcp:
    """Builds the acados OCP definition once, ahead of time (compiled to C).
    Runtime values are set later via solver.set(...) in AcadosNMPC.solve()."""
    N, n_obs, n_walls, n_ell = config.N, config.MAX_OBSTACLES, config.MAX_WALLS, config.MAX_ELLIPSES
    n_prim = n_obs + n_walls + n_ell  # total obstacle primitives (circles + wall capsules + ellipses)

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
    obs_p = p[5:5 + 3 * n_obs]
    wall_p = p[5 + 3 * n_obs:5 + 3 * n_obs + 5 * n_walls]
    ellipse_p = p[5 + 3 * n_obs + 5 * n_walls:]

    model.x = xi
    model.u = u_aug
    model.p = p
    model.f_expl_expr = augmented_dynamics_casadi(xi, u_aug, chi_p, current_p)

    # Obstacle avoidance: every circle/wall (capsule_distance_casadi -- a circle
    # is the degenerate capsule p0==p1) and ellipse (ellipse_distance_casadi --
    # an APPROXIMATE, gradient-normalized distance, see its own docstring; no
    # closed form exists for a true point-to-ellipse distance) primitive's
    # point-to-surface distance is aggregated via a smooth soft-min
    # (softmin_casadi) into ONE scalar D_hat, which is always <= the true
    # nearest-obstacle distance (safe-direction approximation, verified
    # numerically for ellipses too). Single constraint row, single slack --
    # see research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md for the full
    # derivation (this replaces the old one-row-per-circle formulation).
    d_list = []
    for i in range(n_obs):
        ox, oy, orad = obs_p[3 * i], obs_p[3 * i + 1], obs_p[3 * i + 2]
        r_c = orad + config.R_ASV
        d_list.append(capsule_distance_casadi(xi[IDX_X], xi[IDX_Y], ox, oy, ox, oy, r_c, config.EPS))
    for i in range(n_walls):
        x0, y0, x1, y1, wrad = (wall_p[5 * i], wall_p[5 * i + 1], wall_p[5 * i + 2],
                                 wall_p[5 * i + 3], wall_p[5 * i + 4])
        r_c = wrad + config.R_ASV
        d_list.append(capsule_distance_casadi(xi[IDX_X], xi[IDX_Y], x0, y0, x1, y1, r_c, config.EPS))
    for i in range(n_ell):
        xc, yc, a, b, theta = (ellipse_p[5 * i], ellipse_p[5 * i + 1], ellipse_p[5 * i + 2],
                                ellipse_p[5 * i + 3], ellipse_p[5 * i + 4])
        # NOTE: r_pad=config.R_ASV only (ship-only padding) -- unlike circles/walls,
        # an ellipse's own size is already fully encoded in a/b, so no object-radius
        # term is added here. See ellipse_distance_casadi's docstring.
        d_list.append(ellipse_distance_casadi(xi[IDX_X], xi[IDX_Y], xc, yc, a, b, theta,
                                               config.R_ASV, config.EPS))
    D_hat = softmin_casadi(d_list, config.SOFTMIN_K) if n_prim > 0 else None
    model.con_h_expr = ca.vertcat(D_hat) if n_prim > 0 else ca.SX.zeros(0)

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

    # ---- obstacle (soft) constraint: single row, h = D_hat >= 0, relaxable by
    # slack down to -SIGMA (same SIGMA/W_SLACK values as the old per-circle
    # scheme -- only the single nearest obstacle ever bound in practice before,
    # so this collapses cleanly without retuning). See
    # research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md.
    if n_prim > 0:
        ocp.constraints.lh = np.zeros(1)
        ocp.constraints.uh = np.full(1, 1e8)  # "no upper bound"; kept << ACADOS_INFTY (1e10)
        ocp.constraints.idxsh = np.array([0])   # marks the h row as soft (slacked)
        ocp.cost.zl = np.zeros(1)
        ocp.cost.zu = np.zeros(1)
        ocp.cost.Zl = config.W_SLACK * np.ones(1)   # quadratic slack penalty weight
        ocp.cost.Zu = config.W_SLACK * np.ones(1)
        ocp.constraints.lsh = -config.SIGMA * np.ones(1)  # max allowed relaxation
        ocp.constraints.ush = np.zeros(1)

    # ---- state/control bounds ----
    ocp.constraints.x0 = np.array(xi_ref0)  # placeholder; overwritten every solve() call

    # IDX_U carries a hard lower bound (U_REF_MIN, deliberately kept just above
    # zero) so the optimizer can never drive surge speed to ~0 mid-horizon: at
    # u≈0 with v also small, casadi_mmg.py's U=sqrt(ur^2+vr^2) divisor behind
    # v_ndm/r_ndm collapses and r_ndm blows up for any nonzero yaw rate,
    # poisoning the QP -- observed in practice during a low-speed pivot near a
    # target (braking ramp active). u_val is floored inside the MMG model
    # itself (ca.fmax), but that's a smoothing floor, not something the
    # optimizer is constrained to respect; this bound is what actually keeps
    # the solver out of that region. See nmpc/README.md's "Known open issue".
    ocp.constraints.idxbx = np.array([IDX_U, IDX_DELTA, IDX_N])
    ocp.constraints.lbx = np.array([config.U_REF_MIN, config.DELTA_MIN, config.RPS_MIN])
    ocp.constraints.ubx = np.array([_U_NO_UPPER_BOUND, config.DELTA_MAX, config.RPS_MAX])

    ocp.constraints.idxbx_e = np.array([IDX_U, IDX_DELTA, IDX_N])  # same bounds at the terminal stage
    ocp.constraints.lbx_e = np.array([config.U_REF_MIN, config.DELTA_MIN, config.RPS_MIN])
    ocp.constraints.ubx_e = np.array([_U_NO_UPPER_BOUND, config.DELTA_MAX, config.RPS_MAX])

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


def _is_stage_varying(items, N: int) -> bool:
    """True if items is a per-stage container of length N+1 whose elements are
    per-stage lists of obstacle tuples, rather than a single static list."""
    if items is None or len(items) != N + 1:
        return False
    if len(items) == 0:
        return False
    first = items[0]
    if isinstance(first, (list, tuple)):
        return len(first) == 0 or isinstance(first[0], (list, tuple, np.ndarray))
    return False


class AcadosNMPC:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.N = config.N
        self.dt = config.dt
        self.n_obs = config.MAX_OBSTACLES
        self.n_walls = config.MAX_WALLS
        self.n_ellipses = config.MAX_ELLIPSES

        ocp = build_acados_ocp(config)
        json_path = os.path.join(os.path.dirname(__file__), "..", config.ACADOS_JSON_FILE)
        self.solver = AcadosOcpSolver(ocp, json_file=json_path)  # triggers C code-gen + compile on first call

        self._last_delta = config.DELTA_TRIM  # fallback command if a solve ever fails
        self._last_n = config.N_TRIM

        # Waypoint-passage weight boost (nmpc/README.md item 9): a second,
        # per-stage-selectable W/W_e pair with Q[x]/Q[y] scaled up, applied
        # via acados' runtime cost_set(stage, "W", ...) -- NOT an OCP rebuild
        # -- only to the handful of stages predicted near a waypoint crossing
        # (build_horizon_references' dist_to_corner_arr), so it pulls the
        # trajectory through the actual point there without raising cost
        # anywhere else along a leg.
        self._W_default = _block_diag(config.Q, config.R)
        self._We_default = config.Qe
        Q_boosted = config.Q.copy()
        Q_boosted[IDX_X, IDX_X] *= config.WAYPOINT_PASSAGE_XY_BOOST
        Q_boosted[IDX_Y, IDX_Y] *= config.WAYPOINT_PASSAGE_XY_BOOST
        self._W_boosted = _block_diag(Q_boosted, config.R)
        self._We_boosted = config.QE_SCALE * Q_boosted

    def solve(self, mmg_state, delta, n, segments, obstacles=None, walls=None, ellipses=None,
              current=(0.0, 0.0)):
        """segments: ordered list of active path segments [(chi_p, end_x, end_y), ...]
        -- typically SegmentQueue(...).segments -- previewed into the horizon by
        build_horizon_references(), which may reference several upcoming segments
        across the horizon stages, not just one (see nmpc/README.md item 8).
        obstacles: static list of (x, y, radius) tuples, OR length-(N+1) sequence of per-stage lists.
        walls: static list of (x0, y0, x1, y1, radius) tuples, OR length-(N+1) sequence of per-stage lists.
        ellipses: static list of (xc, yc, a, b, theta) tuples, OR length-(N+1) sequence of per-stage lists
        (e.g. moving vessels with live predicted positions and orientations across the horizon)."""
        cfg = self.config
        if obstacles is None:
            obstacles = []
        if walls is None:
            walls = []
        if ellipses is None:
            ellipses = []

        obs_varying = _is_stage_varying(obstacles, self.N)
        wall_varying = _is_stage_varying(walls, self.N)
        ell_varying = _is_stage_varying(ellipses, self.N)

        if not obs_varying:
            obs_flat = pad_obstacles(obstacles, self.n_obs)
        if not wall_varying:
            wall_flat = pad_walls(walls, self.n_walls)
        if not ell_varying:
            ellipse_flat = pad_ellipses(ellipses, self.n_ellipses)

        # speed_est: coarse forward-walk speed used only to decide which segment
        # each stage's predicted arclength position falls into -- recomputed fresh
        # every solve() call from the ship's own current surge speed (floored to
        # avoid a blown-up ETA near-zero speed), so it self-corrects each tick
        # (receding-horizon fashion) rather than needing to be precise.
        speed_est = max(abs(float(mmg_state[0])), 0.1)
        chi_p_arr, x_d_arr, y_d_arr, u_ref_arr, dist_to_corner_arr = build_horizon_references(
            mmg_state[3], mmg_state[4], segments, self.N, self.dt, speed_est, cfg.U_REF, cfg)
        passage_mask = dist_to_corner_arr <= cfg.WAYPOINT_PASSAGE_DIST

        xi_0 = build_xi_full(mmg_state, delta, n, chi_p_arr[0], x_d_arr[0], y_d_arr[0])

        # pin the initial state to the actual current state (standard MPC shrinking-horizon setup)
        self.solver.set(0, "lbx", xi_0)
        self.solver.set(0, "ubx", xi_0)

        # push each stage's OWN segment reference/parameters -- current is set once
        # per solve() call and held fixed across all N stages (the frozen-disturbance
        # approximation: we only have a single online estimate, not a horizon-length
        # forecast), but chi_p/x_d/y_d/u_ref now vary per stage via the arrays above.
        # xi_ref (unwrapped, psi row = chi_p, stage k=0) is kept for the return dict
        # below (controller-effort logging: "reference used in this solve's stage
        # cost") -- acados' own NONLINEAR_LS cost needs the separate psi-zeroed copy
        # instead (wrapped residual baked into cost_y_expr).
        xi_ref = None
        for k in range(self.N):
            xi_ref_k = get_reference_state(chi_p_arr[k], x_d_arr[k], y_d_arr[k], u_ref_arr[k],
                                            cfg.DELTA_TRIM, cfg.N_TRIM, cfg)
            if k == 0:
                xi_ref = xi_ref_k
            yref_k = np.concatenate([_yref_wrap_psi(xi_ref_k), np.zeros(CONTROL_DIM)])
            obs_flat_k = pad_obstacles(obstacles[k], self.n_obs) if obs_varying else obs_flat
            wall_flat_k = pad_walls(walls[k], self.n_walls) if wall_varying else wall_flat
            ellipse_flat_k = pad_ellipses(ellipses[k], self.n_ellipses) if ell_varying else ellipse_flat
            params_k = np.concatenate([[chi_p_arr[k], x_d_arr[k], y_d_arr[k]], current, obs_flat_k, wall_flat_k,
                                        ellipse_flat_k])
            self.solver.set(k, "yref", yref_k)
            self.solver.set(k, "p", params_k)
            self.solver.cost_set(k, "W", self._W_boosted if passage_mask[k] else self._W_default)

        xi_ref_N = get_reference_state(chi_p_arr[self.N], x_d_arr[self.N], y_d_arr[self.N], u_ref_arr[self.N],
                                        cfg.DELTA_TRIM, cfg.N_TRIM, cfg)
        obs_flat_N = pad_obstacles(obstacles[self.N], self.n_obs) if obs_varying else obs_flat
        wall_flat_N = pad_walls(walls[self.N], self.n_walls) if wall_varying else wall_flat
        ellipse_flat_N = pad_ellipses(ellipses[self.N], self.n_ellipses) if ell_varying else ellipse_flat
        params_N = np.concatenate([[chi_p_arr[self.N], x_d_arr[self.N], y_d_arr[self.N]], current, obs_flat_N,
                                    wall_flat_N, ellipse_flat_N])
        self.solver.set(self.N, "yref", _yref_wrap_psi(xi_ref_N))
        self.solver.set(self.N, "p", params_N)
        self.solver.cost_set(self.N, "W", self._We_boosted if passage_mask[self.N] else self._We_default)

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
