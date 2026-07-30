"""Makes the true-moved nmpc/, casadi_mmg_solver/, mpc_visualization/
subpackages (which live right next to this file) resolve as top-level imports
-- "from nmpc.config import ...", "from casadi_mmg_solver.casadi_mmg import
...", "from mpc_bridge import ..." -- exactly as their original, unmodified
source expects, without touching a single import statement inside those files.

This package is now fully self-contained: nothing outside nmpc_ws is imported
at runtime. (Earlier versions of this module resolved an external repo path;
that's gone now that the code actually lives here.)
"""
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MPC_VISUALIZATION_DIR = os.path.join(_PKG_DIR, "mpc_visualization")

# acados code-gen output: a build artifact, not source, so it's kept outside
# the src/ tree entirely (same reasoning as section 2.6 of
# ROS2_CONVERSION_PLAN.md's packaging-gotcha note: never cwd- or
# install-tree-relative, or first-solve code-gen fails to find/write output).
ACADOS_CODE_EXPORT_DIR_DEFAULT = os.path.expanduser("~/.ros/nmpc_sim_nodes/acados_generated/c_generated_code_nmpc")
ACADOS_JSON_FILE_DEFAULT = os.path.expanduser("~/.ros/nmpc_sim_nodes/acados_generated/acados_nmpc.json")


def ensure_on_path():
    """Idempotent: safe to call from every node's module-level import block."""
    if _PKG_DIR not in sys.path:
        sys.path.insert(0, _PKG_DIR)
    # mpc_bridge.py / visualizer.py use flat imports ("from mpc_bridge import ..."),
    # not package-relative ones, so the directory itself must be on sys.path too.
    if _MPC_VISUALIZATION_DIR not in sys.path:
        sys.path.insert(0, _MPC_VISUALIZATION_DIR)
