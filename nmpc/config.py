"""
Single source of truth for all NMPC parameters.
Every other file in nmpc/ imports its constants from here — nothing is
hardcoded elsewhere. Instantiate `NMPCConfig()` (or override any field) to
get a tunable parameter set; `DEFAULT_CONFIG` is the ready-to-use instance.
"""
import os
import sys
import numpy as np
from dataclasses import dataclass, field

# allow running this file directly (python nmpc/config.py) by putting
# the repo root on sys.path, so `nmpc` resolves as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Structural constants (NOT tunable). Change these only if reordering the
# state layout — and update augmented_dynamics_casadi() in state_augmentation.py
# to match, since the ODE is built row-by-row in this order.
# ---------------------------------------------------------------------------
STATE_DIM = 11    # xi = [e_y, sin(psi_e), cos(psi_e), r, x, y, psi, u, v, delta, n]
CONTROL_DIM = 2   # u_aug = [delta_dot, n_dot]
MMG_STATE_DIM = 6  # mmg_state = [u, v, r, x, y, psi]

# --- indices into the 11-dim augmented state xi ---
IDX_EY = 0      # cross-track error [m]
IDX_SPSI = 1    # sin(course angle error)
IDX_CPSI = 2    # cos(course angle error)
IDX_R = 3       # yaw rate [rad/s]
IDX_X = 4       # inertial-frame x position [m]
IDX_Y = 5       # inertial-frame y position [m]
IDX_PSI = 6     # heading angle [rad]
IDX_U = 7       # surge speed [m/s]
IDX_V = 8       # sway speed [m/s]
IDX_DELTA = 9   # rudder angle [rad]
IDX_N = 10      # propeller speed [rps]

# --- indices into the 2-dim augmented control u_aug ---
IDX_DDELTA = 0   # rudder rate [rad/s]
IDX_DN = 1       # propeller rate [rps/s]

# For printing/debugging only (plot legends etc).
STATE_NAMES = ["e_y", "sin(psi_e)", "cos(psi_e)", "r", "x", "y", "psi", "u", "v", "delta", "n"]
CONTROL_NAMES = ["delta_dot", "n_dot"]


@dataclass
class NMPCConfig:
    """All tunable NMPC parameters. Override via NMPCConfig(field=value, ...)."""

    # ---------------- Horizon parameters ----------------
    N: int = 200                 # prediction horizon steps
    dt: float = 0.1             # sampling time [s]

    # ---------------- Ship geometry ----------------
    LPP: float = 2.902          # length between perpendiculars [m]
    R_ASV_FACTOR: float = 0.5   # collision radius = R_ASV_FACTOR * LPP
    R_ASV: float = field(init=False)  # derived in __post_init__, not settable directly

    # ---------------- Actuator bounds ----------------
    DELTA_MIN: float = -np.deg2rad(45.0)   # rudder angle lower bound [rad]
    DELTA_MAX: float = np.deg2rad(45.0)    # rudder angle upper bound [rad]
    DELTA_DOT_MIN: float = -np.deg2rad(30.0)  # rudder rate lower bound [rad/s]
    DELTA_DOT_MAX: float = np.deg2rad(30.0)   # rudder rate upper bound [rad/s]

    RPS_MIN: float = -18.2      # propeller speed lower bound [rps] (bidirectional)
    RPS_MAX: float = 18.2       # propeller speed upper bound [rps]
    RPS_DOT_MIN: float = -5.0  # propeller acceleration lower bound [rps/s]
    RPS_DOT_MAX: float = 5.0   # propeller acceleration upper bound [rps/s]

    # ---------------- Reference / trim ----------------
    U_REF: float = 0.78        # desired surge speed [m/s] (Preliminary_func.py initial condition)
    DELTA_TRIM: float = 0.0     # trim rudder angle [rad]
    N_TRIM: float = 10.0        # trim propeller speed [rps] (roughly consistent with U_REF)

    # ---------------- Waypoint switching ----------------
    WP_RADIUS: float = 1.0      # switch to next waypoint when within this distance [m]

    # ---------------- Speed-reference braking ramp ----------------
    # Q[x]/Q[y] (position error, meters^2) dwarfs Q[u] (speed error, (m/s)^2) at any
    # real distance from the target, so a constant U_REF never gets "slowed down" by
    # the optimizer on its own -- it keeps chasing U_REF at full weight-imbalance
    # speed right up until it's almost on top of the target, then can't decelerate
    # in time (RPS_DOT_MIN) and overshoots. Fix: shrink the SPEED REFERENCE itself
    # as distance-to-target shrinks (see compute_effective_u_ref in path_following.py),
    # instead of relying on the cost weights to discover braking near arrival.
    BRAKE_DISTANCE: float = 8.0  # [m] start ramping U_ref down within this range of the target
    U_REF_MIN: float = 0.05      # [m/s] floor speed at zero distance -- kept nonzero to stay
                                  # clear of the u,v->0 singularity in casadi_mmg.py's U=sqrt(u^2+v^2)
                                  # denominator (v_ndm, r_ndm blow up as U->0)

    # ---------------- Numerical safety ----------------
    EPS: float = 1e-6           # generic numerical-safety guard (small-denominator clamps etc)

    # ---------------- Obstacle avoidance (soft constraint) ----------------
    SIGMA: float = 0.2          # max allowed soft-constraint violation [m]
    W_SLACK: float = 50.0       # quadratic penalty weight on slack violation, per obstacle
    OBSTACLE_START_K: int = 1   # first horizon step at which obstacle constraints are enforced
    MAX_OBSTACLES: int = 5      # fixed upper bound on obstacle count (acados needs a static param size)

    # ---------------- Weight matrices (Section 1.6) ----------------
    # Order: [e_y, sin(psi_e), cos(psi_e), r, x, y, psi, u, v, delta, n]
    # NOTE: paper's own starting value has e_y weight = 0, which leaves cross-track
    # error unpenalized (heading aligns with the path but never nulls the offset).
    # Bumped to a small non-zero value (5.0) purely for closed-loop stability during

    Q_DIAG: tuple = (20.0, 2.0, 2.0, 0.5, 5.0, 5.0, 30.0, 1000.0, 5.0, 0.001, 0.001)
    R_DIAG: tuple = (0.01, 0.01)          # penalizes [delta_dot, n_dot]
    QE_SCALE: float = 2.0                   # terminal cost = QE_SCALE * stage cost (standard choice)

    # ---------------- IPOPT (CasADi debug solver) ----------------
    IPOPT_MAX_ITER: int = 200
    IPOPT_TOL: float = 1e-6
    IPOPT_PRINT_LEVEL: int = 0

    # ---------------- Acados (deploy solver) ----------------
    ACADOS_QP_SOLVER: str = "PARTIAL_CONDENSING_HPIPM"
    ACADOS_NLP_SOLVER: str = "SQP_RTI"
    ACADOS_INTEGRATOR: str = "ERK"
    ACADOS_NUM_STAGES: int = 4
    ACADOS_NUM_STEPS: int = 1
    ACADOS_CODE_EXPORT_DIR: str = "c_generated_code_nmpc"
    ACADOS_JSON_FILE: str = "acados_nmpc.json"

    def __post_init__(self):
        # keeps R_ASV in sync whenever LPP/R_ASV_FACTOR are overridden
        self.R_ASV = self.R_ASV_FACTOR * self.LPP

    # turn the flat *_DIAG tuples into diagonal matrices for the solvers
    @property
    def Q(self) -> np.ndarray:
        return np.diag(self.Q_DIAG)

    @property
    def R(self) -> np.ndarray:
        return np.diag(self.R_DIAG)

    @property
    def Qe(self) -> np.ndarray:
        return self.QE_SCALE * self.Q

    @property
    def T_horizon(self) -> float:
        return self.N * self.dt


# shared default instance imported by every other nmpc/ file
DEFAULT_CONFIG = NMPCConfig()
