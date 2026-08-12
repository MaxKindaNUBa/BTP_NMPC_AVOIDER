"""Single source of truth for all TUNABLE NMPC/simulation parameters.

Every value lives only in params/sim_params.yaml -- this module's sole job is
to read that file and hand back a typed NMPCConfig. There are no hardcoded
Python defaults here or anywhere else: a ROS2 node launched with
sim_params.yaml and a standalone script calling load_nmpc_config() directly
both end up reading the exact same numbers from the exact same file, so the
two can never drift out of sync with each other.

Structural (non-tunable) constants -- state/control layout, index constants
-- live in nmpc/config.py instead, since editing those without also updating
the dynamics code would silently break the model; they aren't "config" in
the tunable sense this module covers.
"""
import os
import sys
import numpy as np
import yaml
from dataclasses import dataclass, field

# allow running this file directly (python nmpc/params.py) by putting
# the repo root on sys.path, so `nmpc` resolves as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _sim_params_path() -> str:
    """Resolves params/sim_params.yaml: package share dir when installed
    (the ros2 run / launch case), source-tree-relative fallback when a
    module's own __main__ self-test runs directly without ROS2 sourced."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("nmpc_sim_nodes"), "params", "sim_params.yaml")
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "params", "sim_params.yaml")


def _load_yaml(path: str = None) -> dict:
    with open(path or _sim_params_path(), "r") as f:
        return yaml.safe_load(f)


@dataclass
class NMPCConfig:
    """All tunable NMPC parameters, as loaded from sim_params.yaml. Construct
    via load_nmpc_config() -- never by hand with field defaults, since there
    are deliberately none: every field must come from the yaml file."""

    # ---------------- Horizon parameters ----------------
    N: int
    dt: float

    # ---------------- Ship geometry ----------------
    LPP: float
    R_ASV_FACTOR: float
    R_ASV: float = field(init=False)  # derived in __post_init__, not settable directly

    # ---------------- Actuator bounds ----------------
    DELTA_MIN: float
    DELTA_MAX: float
    DELTA_DOT_MIN: float
    DELTA_DOT_MAX: float

    RPS_MIN: float
    RPS_MAX: float
    RPS_DOT_MIN: float
    RPS_DOT_MAX: float

    # ---------------- Reference / trim ----------------
    U_REF: float
    DELTA_TRIM: float
    N_TRIM: float

    # ---------------- Waypoint switching (owned by map_node's yaml section) ----------------
    WP_RADIUS: float

    # ---------------- Simulation duration (owned by map_node's yaml section) ----------------
    SIM_TIME_MODE: str
    SIM_TIME_FIXED: float

    # ---------------- Speed-reference braking ramp ----------------
    BRAKE_DISTANCE: float
    U_REF_MIN: float

    # ---------------- Numerical safety ----------------
    EPS: float

    # ---------------- Obstacle avoidance (soft constraint) ----------------
    SIGMA: float
    W_SLACK: float
    OBSTACLE_START_K: int
    MAX_OBSTACLES: int

    # ---------------- Weight matrices ----------------
    Q_DIAG: tuple
    R_DIAG: tuple
    QE_SCALE: float

    # ---------------- IPOPT (CasADi debug solver) ----------------
    IPOPT_MAX_ITER: int
    IPOPT_TOL: float
    IPOPT_PRINT_LEVEL: int

    # ---------------- Acados (deploy solver) ----------------
    ACADOS_QP_SOLVER: str
    ACADOS_NLP_SOLVER: str
    ACADOS_INTEGRATOR: str
    ACADOS_NUM_STAGES: int
    ACADOS_NUM_STEPS: int
    # Deliberately NOT read from the yaml (ROS2_CONVERSION_PLAN.md section 2.6
    # packaging gotcha): these must be absolute, not cwd- or install-tree-
    # relative, or acados' first-solve code-gen fails to find/write its
    # output. Sourced from _pkg_paths' fixed ~/.ros/... defaults instead.
    ACADOS_CODE_EXPORT_DIR: str
    ACADOS_JSON_FILE: str

    def __post_init__(self):
        self.R_ASV = self.R_ASV_FACTOR * self.LPP

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


def load_nmpc_config(path: str = None) -> NMPCConfig:
    """Reads sim_params.yaml's nmpc_node/map_node sections and builds an
    NMPCConfig. `path` overrides the resolved sim_params.yaml location
    (mainly for tests); normally leave it unset."""
    import _pkg_paths  # top-level sibling module (see this file's sys.path.insert above)

    data = _load_yaml(path)
    nmpc_p = data["nmpc_node"]["ros__parameters"]
    map_p = data["map_node"]["ros__parameters"]

    return NMPCConfig(
        N=int(nmpc_p["N"]),
        dt=float(nmpc_p["dt"]),
        LPP=float(nmpc_p["LPP"]),
        R_ASV_FACTOR=float(nmpc_p["R_ASV_FACTOR"]),
        DELTA_MIN=float(nmpc_p["DELTA_MIN"]),
        DELTA_MAX=float(nmpc_p["DELTA_MAX"]),
        DELTA_DOT_MIN=float(nmpc_p["DELTA_DOT_MIN"]),
        DELTA_DOT_MAX=float(nmpc_p["DELTA_DOT_MAX"]),
        RPS_MIN=float(nmpc_p["RPS_MIN"]),
        RPS_MAX=float(nmpc_p["RPS_MAX"]),
        RPS_DOT_MIN=float(nmpc_p["RPS_DOT_MIN"]),
        RPS_DOT_MAX=float(nmpc_p["RPS_DOT_MAX"]),
        U_REF=float(nmpc_p["U_REF"]),
        DELTA_TRIM=float(nmpc_p["DELTA_TRIM"]),
        N_TRIM=float(nmpc_p["N_TRIM"]),
        WP_RADIUS=float(map_p["wp_radius"]),
        SIM_TIME_MODE=str(map_p["sim_time_mode"]),
        SIM_TIME_FIXED=float(map_p["sim_time_fixed"]),
        BRAKE_DISTANCE=float(nmpc_p["BRAKE_DISTANCE"]),
        U_REF_MIN=float(nmpc_p["U_REF_MIN"]),
        EPS=float(nmpc_p["EPS"]),
        SIGMA=float(nmpc_p["SIGMA"]),
        W_SLACK=float(nmpc_p["W_SLACK"]),
        OBSTACLE_START_K=int(nmpc_p["OBSTACLE_START_K"]),
        MAX_OBSTACLES=int(nmpc_p["MAX_OBSTACLES"]),
        Q_DIAG=tuple(float(v) for v in nmpc_p["Q_DIAG"]),
        R_DIAG=tuple(float(v) for v in nmpc_p["R_DIAG"]),
        QE_SCALE=float(nmpc_p["QE_SCALE"]),
        IPOPT_MAX_ITER=int(nmpc_p["IPOPT_MAX_ITER"]),
        IPOPT_TOL=float(nmpc_p["IPOPT_TOL"]),
        IPOPT_PRINT_LEVEL=int(nmpc_p["IPOPT_PRINT_LEVEL"]),
        ACADOS_QP_SOLVER=str(nmpc_p["ACADOS_QP_SOLVER"]),
        ACADOS_NLP_SOLVER=str(nmpc_p["ACADOS_NLP_SOLVER"]),
        ACADOS_INTEGRATOR=str(nmpc_p["ACADOS_INTEGRATOR"]),
        ACADOS_NUM_STAGES=int(nmpc_p["ACADOS_NUM_STAGES"]),
        ACADOS_NUM_STEPS=int(nmpc_p["ACADOS_NUM_STEPS"]),
        ACADOS_CODE_EXPORT_DIR=_pkg_paths.ACADOS_CODE_EXPORT_DIR_DEFAULT,
        ACADOS_JSON_FILE=_pkg_paths.ACADOS_JSON_FILE_DEFAULT,
    )


# shared default instance, loaded once at import time from sim_params.yaml --
# imported by every other nmpc/ file exactly like the old config.DEFAULT_CONFIG was.
DEFAULT_CONFIG = load_nmpc_config()
