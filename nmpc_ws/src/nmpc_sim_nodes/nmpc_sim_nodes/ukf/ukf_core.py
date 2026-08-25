"""UnscentedKalmanFilter: pure math, no rclpy import -- independently testable
(see test_ukf.py) and reusable from ukf_node.py's ROS glue.

State ordering (fixed, STATE_DIM=12):
[u, v, r, x, y, psi, vcx, vcy, ax_bias, ay_bias, pos_bias_x, pos_bias_y] --
the MMG state, augmented with earth-frame current per the current-estimation
UKF design discussed for this project (state's own u,v,r,x,y,psi stay
absolute/ground-relative throughout, matching casadi_mmg.py's convention;
current only enters the hydrodynamic force terms via relative-velocity
substitution inside the process model), further augmented with two slowly-
drifting sensor bias pairs: the IMU accelerometer's (ax_bias, ay_bias) and
the GPS position's (pos_bias_x, pos_bias_y). Both bias pairs exist for the
same underlying reason -- each is the *only* channel a specific hidden state
is observable through, so an unmodeled bias in that channel gets
misattributed to that state almost entirely: accel bias -> current (a
mismatch between predicted and measured acceleration can only be "explained"
by wrong current or wrong bias; empirically, RNG-seed-dependent current
error from -0.03 to +0.27 m/s against a true 0.1 m/s, explained almost
completely by which accel-bias value that seed's Gauss-Markov draw happened
to produce); GPS bias -> current too, via a different path (a drifting
position bias looks identical, through the kinematics, to the vessel's
actual velocity being slightly wrong -- empirically the dominant remaining
error after augmenting accel bias alone: isolating each untracked bias
source showed pos_bias_sigma alone accounted for ~4.6x more of the residual
current error than psi_bias_sigma or r_bias_sigma, which changed nothing
measurable when disabled individually). Widening R to account for either
bias variance (rather than tracking it as a state) was tried first for accel
and barely helped, since R only discounts trust in each individual sample --
it can't represent that the error is correlated across time. Explicitly
modeling and subtracting each bias is the actual fix.

Measurement ordering (fixed, MEAS_DIM=6): [x, y, psi, r, ax, ay] -- GPS
position, compass heading, rate gyro, IMU accelerometer. No direct u,v
measurement -- reconstructing them is this filter's whole job.

Process model: the project's own `make_casadi_integrator(..., with_env=True)`
(the caller builds and passes this in, so ukf_node and mmg_node share the
exact same dynamics, not a re-derived copy) for state[0:6]. Current
propagates as an identity random walk (no OU mean-reversion term in the
filter's own model -- justified since current_time_constant=21600s >> dt,
making the per-step mean-reversion contribution negligible next to Q's
driving noise). ax_bias/ay_bias and pos_bias_x/pos_bias_y each propagate as
their own first-order Gauss-Markov AR(1) process (bias_next = decay *
bias_prev, using accel_bias_decay/pos_bias_decay respectively -- two
different coefficients, since accel_bias_tau and pos_bias_tau differ, e.g.
300s vs 120s in the LIGHT preset), the SAME model
sensor_model.noise_primitives.GaussMarkov uses to generate the real biases
(see ukf/config.py's accel_bias_decay()/pos_bias_decay()) -- not an identity
random walk, since the real processes genuinely mean-revert and matching
that (rather than assuming they're static) is what lets the filter track
them instead of just absorbing whatever value they started at. Wave force is
deliberately not modeled (unobserved/stochastic, not part of the state).

Measurement model: [x, y] add that sigma point's own pos_bias_x/pos_bias_y
to the propagated position; [psi, r] are a direct linear selection of the
propagated state; [ax, ay] are NOT -- they come from re-evaluating the MMG
dynamics' acceleration output (`make_casadi_accel_function`) at each sigma
point plus that sigma point's own ax_bias/ay_bias, since a real accelerometer
senses net specific force plus its own drifting zero-offset, not a state
component. These nonlinear/bias-augmented measurement rows are why this
project uses a UKF rather than a plain (linear) KF.

Angle handling: psi (state index 5) and the measurement's psi row (index 2)
wrap at +-pi. All means and residuals touching those indices are
angle-aware (circular mean via atan2 of weighted sin/cos; wrapped
subtraction) -- plain arithmetic would corrupt the filter near the wrap.

Adaptive R/Q (Sage-Husa): tuning Q/R as fixed constants (ukf/config.py,
tune_ukf.py) hits a hard ceiling -- two independent NEES-consistency tuning
passes (3 and 8 training seeds) each converged their training seeds to
mean_NEES ~ 12 (perfectly calibrated) but held-out seeds still ranged from
~6 to ~46, because each RNG seed's Gauss-Markov bias draws wander a
genuinely different amount, and one fixed number can't describe every
realization equally well. Fixed Q/R describe the *average* case; NEES
measures calibration on *this specific run*. Sage-Husa recursively re-
estimates R (and, if enabled, Q) from the actual innovation sequence each
tick, so the filter can be tighter on calm runs and looser on wild ones
instead of splitting the difference. See config.adapt_r/adapt_q/
forgetting_factor (ukf/config.py) and the papers cited in this project's
memory/commit history for the algorithm and its typical use cases (AUV/USV
navigation, target tracking with unknown/time-varying noise).
"""
import numpy as np

STATE_DIM = 12
MEAS_DIM = 6
IDX_PSI_STATE = 5
IDX_AX_BIAS = 8
IDX_AY_BIAS = 9
IDX_POS_BIAS_X = 10
IDX_POS_BIAS_Y = 11
IDX_PSI_MEAS = 2


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return float(np.arctan2(s, c))


class UnscentedKalmanFilter:
    def __init__(self, config, predict_integrator, accel_fn, r_diag: np.ndarray,
                 accel_bias_decay: float, pos_bias_decay: float):
        """predict_integrator: ca.Function (state[6], control[2], current[2],
        wave_force[3]) -> (next_state[6], r_dot_a) -- e.g.
        make_casadi_integrator(dt, method='rk4', sym_type=ca.SX, with_env=True).
        accel_fn: ca.Function (state[6], control[2], current[2], wave_force[3])
        -> accel[2] -- make_casadi_accel_function(sym_type=ca.SX).
        r_diag: MEAS_DIM-length measurement-noise diagonal (config.r_diag if
        set, else config's caller should resolve via default_r_diag()).
        accel_bias_decay: the ax_bias/ay_bias AR(1) coefficient -- see
        ukf/config.py's accel_bias_decay().
        pos_bias_decay: the pos_bias_x/pos_bias_y AR(1) coefficient -- see
        ukf/config.py's pos_bias_decay().
        """
        import casadi as ca  # local import: keep this module importable (e.g. by
        self._ca = ca         # test harnesses) without requiring casadi at module load

        self.integrator = predict_integrator
        self.accel_fn = accel_fn
        self.accel_bias_decay = float(accel_bias_decay)
        self.pos_bias_decay = float(pos_bias_decay)

        n = STATE_DIM
        self.alpha, self.beta, self.kappa = config.alpha, config.beta, config.kappa
        self.lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        c = n + self.lambda_
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
        self.Wc = np.full(2 * n + 1, 1.0 / (2.0 * c))
        self.Wm[0] = self.lambda_ / c
        self.Wc[0] = self.lambda_ / c + (1.0 - self.alpha ** 2 + self.beta)
        self._gamma = float(np.sqrt(c))

        self.Q0 = np.diag(np.asarray(config.q_diag, dtype=float))
        self.P0 = np.diag(np.asarray(config.p0_diag, dtype=float))
        self.R0 = np.diag(np.asarray(r_diag, dtype=float))
        self.Q = self.Q0.copy()
        self.R = self.R0.copy()

        # Sage-Husa adaptive noise estimation -- see module docstring and
        # ukf/config.py's adapt_r/adapt_q/forgetting_factor. Q0/R0 above are
        # what R/Q reset to and what the recursion is centered around, not
        # dead values -- tune_ukf.py's static search still matters, it's
        # tuning the *prior* the adaptive filter starts from and reverts
        # toward under weak evidence, not being made irrelevant by adaptation.
        self.adapt_r = bool(getattr(config, "adapt_r", False))
        self.adapt_q = bool(getattr(config, "adapt_q", False))
        self.forgetting_factor = float(getattr(config, "forgetting_factor", 0.98))
        self._k = 0  # step counter for the time-varying forgetting weight d_k

        self.x = np.zeros(STATE_DIM)
        self.P = self.P0.copy()

    def reset(self, x0: np.ndarray, P0: np.ndarray = None):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = (self.P0 if P0 is None else np.asarray(P0, dtype=float)).copy()
        self.Q = self.Q0.copy()
        self.R = self.R0.copy()
        self._k = 0

    # ------------------------------------------------------------------
    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        L = np.linalg.cholesky(P)  # raises LinAlgError if P not PD -- caller catches
        X = np.zeros((2 * STATE_DIM + 1, STATE_DIM))
        X[0] = x
        for i in range(STATE_DIM):
            X[1 + i] = x + self._gamma * L[:, i]
            X[1 + STATE_DIM + i] = x - self._gamma * L[:, i]
        return X

    def _weighted_mean(self, points: np.ndarray, weights: np.ndarray, angle_idx: int) -> np.ndarray:
        mean = np.sum(weights[:, None] * points, axis=0)
        mean[angle_idx] = _circular_mean(points[:, angle_idx], weights)
        return mean

    def _weighted_cov(self, points: np.ndarray, mean: np.ndarray, weights: np.ndarray,
                       angle_idx: int, other_points=None, other_mean=None, other_angle_idx=None) -> np.ndarray:
        d1 = points - mean
        d1[:, angle_idx] = _wrap(d1[:, angle_idx])
        if other_points is None:
            d2 = d1
        else:
            d2 = other_points - other_mean
            d2[:, other_angle_idx] = _wrap(d2[:, other_angle_idx])
        return (weights[:, None, None] * d1[:, :, None] * d2[:, None, :]).sum(axis=0)

    # ------------------------------------------------------------------
    def predict(self, control):
        """Returns (x_pred, P_pred, Y) -- Y (the propagated sigma points) is
        reused by correct() so the caller must pass it straight through."""
        ca = self._ca
        X = self._sigma_points(self.x, self.P)

        Y = np.zeros_like(X)
        control_dm = ca.DM(list(control))
        wave_dm = ca.DM([0.0, 0.0, 0.0])
        for i in range(X.shape[0]):
            state6 = ca.DM(X[i, :6].tolist())
            current_dm = ca.DM(X[i, 6:8].tolist())
            next6, _ = self.integrator(state6, control_dm, current_dm, wave_dm)
            Y[i, :6] = np.array(next6).flatten()
            Y[i, 6:8] = X[i, 6:8]  # current: identity random walk
            Y[i, IDX_AX_BIAS] = self.accel_bias_decay * X[i, IDX_AX_BIAS]  # accel bias: AR(1) decay
            Y[i, IDX_AY_BIAS] = self.accel_bias_decay * X[i, IDX_AY_BIAS]
            Y[i, IDX_POS_BIAS_X] = self.pos_bias_decay * X[i, IDX_POS_BIAS_X]  # pos bias: AR(1) decay
            Y[i, IDX_POS_BIAS_Y] = self.pos_bias_decay * X[i, IDX_POS_BIAS_Y]

        x_pred = self._weighted_mean(Y, self.Wm, IDX_PSI_STATE)
        P_pred = self._weighted_cov(Y, x_pred, self.Wc, IDX_PSI_STATE) + self.Q
        return x_pred, P_pred, Y

    def correct(self, x_pred, P_pred, Y, z_meas, control):
        """Returns (x_hat, P, success, return_status). On a non-PD Pzz,
        skips the update and returns the predicted (x_pred, P_pred) unchanged
        with success=False -- caller (ukf_node) surfaces this as a filter
        divergence warning rather than corrupting the state with a bad update."""
        ca = self._ca
        control_dm = ca.DM(list(control))
        wave_dm = ca.DM([0.0, 0.0, 0.0])

        Z = np.zeros((Y.shape[0], MEAS_DIM))
        for i in range(Y.shape[0]):
            state6 = ca.DM(Y[i, :6].tolist())
            current_dm = ca.DM(Y[i, 6:8].tolist())
            accel = np.array(self.accel_fn(state6, control_dm, current_dm, wave_dm)).flatten()
            Z[i, 0] = Y[i, 3] + Y[i, IDX_POS_BIAS_X]  # x = true position + this sigma point's GPS bias estimate
            Z[i, 1] = Y[i, 4] + Y[i, IDX_POS_BIAS_Y]  # y
            Z[i, 2] = Y[i, 5]  # psi
            Z[i, 3] = Y[i, 2]  # r
            Z[i, 4] = accel[0] + Y[i, IDX_AX_BIAS]  # ax = true accel + this sigma point's bias estimate
            Z[i, 5] = accel[1] + Y[i, IDX_AY_BIAS]  # ay

        z_pred = self._weighted_mean(Z, self.Wm, IDX_PSI_MEAS)

        try:
            Pzz = self._weighted_cov(Z, z_pred, self.Wc, IDX_PSI_MEAS) + self.R
            Pxz = self._weighted_cov(Y, x_pred, self.Wc, IDX_PSI_STATE,
                                      other_points=Z, other_mean=z_pred, other_angle_idx=IDX_PSI_MEAS)
            K = np.linalg.solve(Pzz.T, Pxz.T).T  # Pxz @ inv(Pzz), solved rather than inverted directly
        except np.linalg.LinAlgError:
            return x_pred, P_pred, False, "CHOLESKY_FAILED"

        innovation = np.asarray(z_meas, dtype=float) - z_pred
        innovation[IDX_PSI_MEAS] = _wrap(innovation[IDX_PSI_MEAS])

        x_hat = x_pred + K @ innovation
        x_hat[IDX_PSI_STATE] = _wrap(x_hat[IDX_PSI_STATE])

        P = P_pred - K @ Pzz @ K.T
        P = 0.5 * (P + P.T)  # numerical-robustness symmetrization against float drift

        if self.adapt_r or self.adapt_q:
            self._k += 1
            b = self.forgetting_factor
            # Time-varying forgetting weight: d_k=1 on the first call
            # (self._k==1 -- a pure sample estimate, nothing to blend with
            # yet), decaying toward the steady-state (1-b) as k grows -- the
            # standard Sage-Husa recursive noise-statistics estimator (see
            # module docstring).
            d_k = (1.0 - b) / (1.0 - b ** self._k)

            # Diagonal-only, range-bounded around the prior (R0/Q0) -- NOT
            # the textbook full-matrix Sage-Husa recursion. That version was
            # tried first and diverged within ~30s: R's x/y rows collapsed
            # toward 0 (the raw sample estimate innov_outer - (Pzz - R) goes
            # negative whenever P has grown, so PSD-projection clips it
            # straight to the floor) while Q's pos_bias rows grew unbounded
            # -- a runaway feedback loop (Q too big -> P too big -> Pzz too
            # big -> R driven to ~0 -> filter over-trusts noisy measurements
            # -> correction overshoots -> Q estimate grows further). This is
            # the well-documented instability of naive joint Sage-Husa
            # adaptation that motivates "robust"/bounded variants in the
            # literature (see module docstring's cited papers). Restricting
            # each diagonal entry to stay within [0.05x, 20x] of its prior
            # value keeps the same online-adaptation idea (tighter on calm
            # runs, looser on wild ones) without the unbounded-growth failure
            # mode -- R/Q simply can't wander far enough from a sane prior to
            # start this feedback loop.
            if self.adapt_r:
                innov_sq = innovation ** 2
                pzz_state_only = np.diag(Pzz) - np.diag(self.R)
                r_sample = innov_sq - pzz_state_only
                r_new = (1.0 - d_k) * np.diag(self.R) + d_k * r_sample
                r0 = np.diag(self.R0)
                r_new = np.clip(r_new, 0.05 * r0, 20.0 * r0)
                self.R = np.diag(r_new)

            if self.adapt_q:
                Ky = K @ innovation
                q_sample = Ky ** 2 + np.diag(P_pred) - np.diag(P)
                q_new = (1.0 - d_k) * np.diag(self.Q) + d_k * q_sample
                q0 = np.diag(self.Q0)
                q_new = np.clip(q_new, 0.05 * q0, 20.0 * q0)
                self.Q = np.diag(q_new)

        return x_hat, P, True, "OK"

    def step(self, control, z_meas):
        """Convenience: predict() then correct(), updating self.x/self.P in
        place and returning (x_hat, P, success, return_status)."""
        try:
            x_pred, P_pred, Y = self.predict(control)
        except np.linalg.LinAlgError:
            return self.x, self.P, False, "CHOLESKY_FAILED"

        x_hat, P, success, status = self.correct(x_pred, P_pred, Y, z_meas, control)
        if success:
            self.x, self.P = x_hat, P
        return x_hat, P, success, status
