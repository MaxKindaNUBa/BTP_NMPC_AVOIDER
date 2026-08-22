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
from typing import Optional

import yaml

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


def sim_params_path() -> str:
    """Resolves params/sim_params.yaml: package share dir when installed (the
    ros2 run / launch case), source-tree-relative fallback when a standalone
    script runs directly without ROS2 sourced. Same resolution nmpc/params.py
    uses for its own copy of this logic -- kept here too (rather than having
    nmpc/params.py import this module) so this file has no dependency on the
    nmpc subpackage."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("nmpc_sim_nodes"), "params", "sim_params.yaml")
    except Exception:
        return os.path.join(_PKG_DIR, "..", "params", "sim_params.yaml")


def load_sim_params(path: Optional[str] = None) -> dict:
    """Parses sim_params.yaml into a plain dict (top-level keys: mmg_node,
    nmpc_node, map_node, logger_node, each a {'ros__parameters': {...}}
    block). The one shared entry point every standalone script/config loader
    should use instead of hand-rolling its own path resolution + yaml.safe_load
    (or, worse, hardcoding a private copy of a value that already lives in
    this file) -- see env_model/config.py's and sensor_model/config.py's own
    load_*_config() functions for the typed wrappers built on top of this."""
    with open(path or sim_params_path(), "r") as f:
        return yaml.safe_load(f)
