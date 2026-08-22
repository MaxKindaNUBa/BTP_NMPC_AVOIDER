"""
Step A — complete multiple-shooting NMPC using CasADi + IPOPT.
Slow but fully debuggable; validates the formulation before porting to Acados.
"""
import os
import sys
import time
import numpy as np
import casadi as ca

# allow running this file directly (python nmpc/nmpc_casadi.py) by putting
# the repo root on sys.path, so `nmpc` resolves as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nmpc.config import (
    STATE_DIM, CONTROL_DIM,
    IDX_EY, IDX_SPSI, IDX_CPSI, IDX_R, IDX_X, IDX_Y, IDX_PSI, IDX_U, IDX_V, IDX_DELTA, IDX_N,
    IDX_DDELTA, IDX_DN,
)
from nmpc.params import DEFAULT_CONFIG
from nmpc.state_augmentation import make_rk4_step
from nmpc.path_following import (
    build_xi_full, pad_obstacles, wrap180_casadi, compute_effective_u_ref, get_reference_state,
)


class CasadiNMPC:
    """Multiple-shooting NMPC with a fixed obstacle-slot budget (config.MAX_OBSTACLES).
    Unused obstacle slots are padded with a far-away dummy so the constraint is
    always trivially inactive — this keeps the NLP structure fixed regardless of
    how many real obstacles are passed to solve(), and mirrors the Acados OCP
    (Step B), which must have a fixed parameter/constraint size at code-gen time.
    """

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.N = config.N
        self.dt = config.dt
        self.n_obs = config.MAX_OBSTACLES

        # which horizon steps get an obstacle constraint (skips k=0, see config.OBSTACLE_START_K)
        self.obs_start_k = min(config.OBSTACLE_START_K, self.N + 1)
        self.slack_ks = list(range(self.obs_start_k, self.N + 1))
        self.n_slack_steps = len(self.slack_ks)

        self._rk4_step = make_rk4_step(self.dt)  # one discrete dynamics step, reused in every shooting constraint

        self._build_nlp()  # constructs self.solver and the fixed lbx/ubx/lbg/ubg arrays

        self._prev_z = None  # decision vector from last solve, for warm-starting
        self._last_delta = 0.0
        self._last_n = 0.0

    # ------------------------------------------------------------------
    def _param_layout(self):
        """Length of the runtime parameter vector p passed into solve():
        p = [xi_0(11), chi_p(1), x_d(1), y_d(1), u_ref(1), obs(3*n_obs)]"""
        n = STATE_DIM + 4 + 3 * self.n_obs
        return n

    def _build_nlp(self):
        """Builds the multiple-shooting NLP once at construction time. Runtime
        values (current state, target waypoint, obstacles) are NOT baked in
        here — they're passed in later as the parameter vector p in solve()."""
        cfg = self.config
        N, n_obs = self.N, self.n_obs

        # decision variables: state trajectory X, control trajectory U, obstacle slacks S
        X = ca.MX.sym("X", STATE_DIM, N + 1)
        U = ca.MX.sym("U", CONTROL_DIM, N)
        S = ca.MX.sym("S", n_obs, self.n_slack_steps) if self.n_slack_steps > 0 else ca.MX.sym("S", 0, 0)

        # runtime parameters, filled in fresh on every solve() call
        p = ca.MX.sym("p", self._param_layout())
        xi0_p = p[0:STATE_DIM]
        chi_p = p[STATE_DIM]
        x_d = p[STATE_DIM + 1]
        y_d = p[STATE_DIM + 2]
        u_ref_p = p[STATE_DIM + 3]  # distance-scaled speed target, computed fresh in solve()
        obs_p = p[STATE_DIM + 4:]  # (3*n_obs,) flattened [x_i, y_i, r_i, ...]

        # build xi_ref symbolically from the parameters (chi_p, x_d, y_d, u_ref_p change every solve)
        xi_ref = ca.MX.zeros(STATE_DIM)
        xi_ref[IDX_EY] = 0.0
        xi_ref[IDX_SPSI] = 0.0
        xi_ref[IDX_CPSI] = 1.0
        xi_ref[IDX_R] = 0.0
        xi_ref[IDX_X] = x_d
        xi_ref[IDX_Y] = y_d
        xi_ref[IDX_PSI] = chi_p
        xi_ref[IDX_U] = u_ref_p
        xi_ref[IDX_V] = 0.0
        xi_ref[IDX_DELTA] = cfg.DELTA_TRIM
        xi_ref[IDX_N] = cfg.N_TRIM

        Q = ca.DM(cfg.Q)
        R = ca.DM(cfg.R)
        Qe = ca.DM(cfg.Qe)

        # ---- cost: sum of stage costs + terminal cost + slack penalty ----
        # psi row of dx is wrapped into (-pi,pi] before squaring: raw psi drifts
        # unbounded (never wraps) while chi_p stays in (-pi,pi], so late in a long
        # rollout an otherwise-fine heading can read as a huge, spurious error at
        # this row's dominant Q weight. Wrapping keeps the residual meaningful
        # without changing the paper's weight structure (main.pdf Q[psi]=30).
        cost = 0
        for k in range(N):
            dx = X[:, k] - xi_ref
            dx[IDX_PSI] = wrap180_casadi(dx[IDX_PSI])
            cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([U[:, k].T, R, U[:, k]])
        dxN = X[:, N] - xi_ref
        dxN[IDX_PSI] = wrap180_casadi(dxN[IDX_PSI])
        cost += ca.mtimes([dxN.T, Qe, dxN])
        if self.n_slack_steps > 0:
            cost += cfg.W_SLACK * ca.sumsqr(S)  # penalize obstacle-constraint violations

        # ---- constraints g(z; p), all built as g >= 0 style bounds below ----
        g_list = [X[:, 0] - xi0_p]                      # initial state, eq (pins X[:,0] to xi_0)
        lbg_list = [np.zeros(STATE_DIM)]
        ubg_list = [np.zeros(STATE_DIM)]

        for k in range(N):                               # dynamics continuity, eq (multiple shooting)
            g_list.append(X[:, k + 1] - self._rk4_step(X[:, k], U[:, k], chi_p))
            lbg_list.append(np.zeros(STATE_DIM))
            ubg_list.append(np.zeros(STATE_DIM))

        for si, k in enumerate(self.slack_ks):            # obstacle avoidance, ineq (soft via slack S)
            for i in range(n_obs):
                ox, oy, orad = obs_p[3 * i], obs_p[3 * i + 1], obs_p[3 * i + 2]
                r_c = orad + cfg.R_ASV  # collision radius = obstacle radius + ship radius
                expr = (X[IDX_X, k] - ox) ** 2 + (X[IDX_Y, k] - oy) ** 2 - (r_c ** 2 - S[i, si])
                g_list.append(expr)
                lbg_list.append(np.zeros(1))
                ubg_list.append(np.full(1, np.inf))

        g = ca.vertcat(*g_list)
        lbg = np.concatenate(lbg_list)
        ubg = np.concatenate(ubg_list)

        # stack all decision variables into one flat vector z, in [X; U; S] order
        z = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            ca.reshape(S, -1, 1) if self.n_slack_steps > 0 else ca.MX.zeros(0, 1),
        )

        # ---- variable bounds lbx/ubx, matching the [X; U; S] layout of z ----
        n_x, n_u, n_s = STATE_DIM * (N + 1), CONTROL_DIM * N, n_obs * self.n_slack_steps
        lbx = np.full(n_x + n_u + n_s, -np.inf)
        ubx = np.full(n_x + n_u + n_s, np.inf)

        # state bounds: only delta/n (actuator states) are constrained, rest unbounded
        X_lb = np.full((STATE_DIM, N + 1), -np.inf)
        X_ub = np.full((STATE_DIM, N + 1), np.inf)
        X_lb[IDX_DELTA, :] = cfg.DELTA_MIN
        X_ub[IDX_DELTA, :] = cfg.DELTA_MAX
        X_lb[IDX_N, :] = cfg.RPS_MIN
        X_ub[IDX_N, :] = cfg.RPS_MAX
        lbx[:n_x] = X_lb.flatten(order="F")  # 'F' order matches X's (STATE_DIM, N+1) column-major layout
        ubx[:n_x] = X_ub.flatten(order="F")

        # control bounds: both delta_dot and n_dot are rate-limited
        U_lb = np.zeros((CONTROL_DIM, N))
        U_ub = np.zeros((CONTROL_DIM, N))
        U_lb[IDX_DDELTA, :] = cfg.DELTA_DOT_MIN
        U_ub[IDX_DDELTA, :] = cfg.DELTA_DOT_MAX
        U_lb[IDX_DN, :] = cfg.RPS_DOT_MIN
        U_ub[IDX_DN, :] = cfg.RPS_DOT_MAX
        lbx[n_x:n_x + n_u] = U_lb.flatten(order="F")
        ubx[n_x:n_x + n_u] = U_ub.flatten(order="F")

        if n_s > 0:
            lbx[n_x + n_u:] = -cfg.SIGMA  # slack can go negative (relax constraint) up to SIGMA
            ubx[n_x + n_u:] = 0.0          # never positive (never tighten beyond the hard limit)

        # stash sizes/bounds needed later by solve() and _initial_guess()
        self._n_x, self._n_u, self._n_s = n_x, n_u, n_s
        self._lbx, self._ubx = lbx, ubx
        self._lbg, self._ubg = lbg, ubg

        nlp = {"x": z, "f": cost, "g": g, "p": p}
        opts = {
            "ipopt.max_iter": cfg.IPOPT_MAX_ITER,
            "ipopt.tol": cfg.IPOPT_TOL,
            "ipopt.print_level": cfg.IPOPT_PRINT_LEVEL,
            "print_time": 0,
            "ipopt.sb": "yes",  # suppress IPOPT banner
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # ------------------------------------------------------------------
    def _initial_guess(self, xi_0):
        """Warm-start: shift last solution one step forward (standard MPC trick)
        so IPOPT starts near the true solution instead of from scratch."""
        if self._prev_z is not None:
            X_prev = self._prev_z[: self._n_x].reshape(STATE_DIM, self.N + 1, order="F")
            U_prev = self._prev_z[self._n_x: self._n_x + self._n_u].reshape(CONTROL_DIM, self.N, order="F")
            S_prev = self._prev_z[self._n_x + self._n_u:]

            # drop the oldest step, repeat the last one to keep the same length
            X_guess = np.hstack([X_prev[:, 1:], X_prev[:, -1:]])
            X_guess[:, 0] = xi_0  # first entry is always pinned to the actual current state
            U_guess = np.hstack([U_prev[:, 1:], U_prev[:, -1:]]) if self.N > 0 else U_prev
            z0 = np.concatenate([X_guess.flatten(order="F"), U_guess.flatten(order="F"), S_prev])
            return z0

        # first-ever call: no history yet, guess a constant trajectory
        X_guess = np.tile(np.array(xi_0).reshape(-1, 1), (1, self.N + 1))
        U_guess = np.zeros((CONTROL_DIM, self.N))
        S_guess = np.zeros(self._n_s)
        return np.concatenate([X_guess.flatten(order="F"), U_guess.flatten(order="F"), S_guess])

    def solve(self, mmg_state, delta, n, chi_p, x_d, y_d, obstacles=None):
        """
        mmg_state: [u, v, r, x, y, psi]
        delta, n : current actuator states (tracked externally between calls)
        Returns dict: u_opt (2,N), xi_traj (11,N+1), delta, n, solve_time, success
        """
        if obstacles is None:
            obstacles = []

        xi_0 = build_xi_full(mmg_state, delta, n, chi_p, x_d, y_d)  # current state -> initial-state constraint
        obs_flat = pad_obstacles(obstacles, self.n_obs)
        u_ref_eff = compute_effective_u_ref(mmg_state[3], mmg_state[4], x_d, y_d, self.config.U_REF, self.config)
        # numeric twin of the symbolic xi_ref built inside _build_nlp (same inputs,
        # same formula) -- needed only for controller-effort logging (nmpc_node
        # publishes it raw); the solve itself never uses this numpy copy.
        xi_ref = get_reference_state(chi_p, x_d, y_d, u_ref_eff, self.config.DELTA_TRIM, self.config.N_TRIM, self.config)
        p_val = np.concatenate([xi_0, [chi_p, x_d, y_d, u_ref_eff], obs_flat])  # matches _param_layout() order
        z0 = self._initial_guess(xi_0)

        t0 = time.perf_counter()
        sol = self.solver(x0=z0, lbx=self._lbx, ubx=self._ubx,
                           lbg=self._lbg, ubg=self._ubg, p=p_val)
        solve_time = time.perf_counter() - t0

        stats = self.solver.stats()
        success = bool(stats.get("success", False))

        z_opt = np.array(sol["x"]).flatten()
        X_opt = z_opt[: self._n_x].reshape(STATE_DIM, self.N + 1, order="F")
        U_opt = z_opt[self._n_x: self._n_x + self._n_u].reshape(CONTROL_DIM, self.N, order="F")

        if success:
            self._prev_z = z_opt
        # if the solve failed, keep the previous warm-start (do not overwrite with garbage)

        # integrate the first optimal rate one dt step to get the new actuator command
        delta_dot, n_dot = U_opt[IDX_DDELTA, 0], U_opt[IDX_DN, 0]
        delta_new = np.clip(delta + delta_dot * self.dt, self.config.DELTA_MIN, self.config.DELTA_MAX)
        n_new = np.clip(n + n_dot * self.dt, self.config.RPS_MIN, self.config.RPS_MAX)

        self._last_delta, self._last_n = delta_new, n_new

        return {
            "u_opt": U_opt,
            "xi_traj": X_opt,
            "xi_ref": xi_ref,
            "delta": float(delta_new),
            "n": float(n_new),
            "solve_time": solve_time,
            "success": success,
            "return_status": stats.get("return_status", "unknown"),
        }
