"""
Structural (NOT tunable) constants: state/control vector layout and the
index constants used throughout nmpc/ to address rows of xi/u_aug directly
in the CasADi-symbolic dynamics and cost code (e.g. xi[IDX_EY]).

These are fixed architecture, not configuration -- change them only if
reordering the state layout, and update augmented_dynamics_casadi() in
state_augmentation.py to match, since the ODE is built row-by-row in this
order. For that reason they live here as plain Python constants rather than
in params/sim_params.yaml: nothing type-checks a YAML edit against the array
indexing this code depends on, so an accidental edit there would silently
corrupt the model instead of failing loudly.

All TUNABLE NMPC parameters (N, dt, weights, bounds, ...) live in
params/sim_params.yaml and are loaded via nmpc/params.py's DEFAULT_CONFIG --
not here.
"""
import os
import sys

# allow running this file directly (python nmpc/config.py) by putting
# the repo root on sys.path, so `nmpc` resolves as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
