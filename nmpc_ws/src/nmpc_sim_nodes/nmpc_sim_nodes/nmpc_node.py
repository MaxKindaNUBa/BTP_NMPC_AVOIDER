"""nmpc_node: pure optimizer (section 2.6 of ROS2_CONVERSION_PLAN.md). Serves
/nmpc/solve, wrapping AcadosNMPC.solve()/CasadiNMPC.solve() unchanged; caches
the latest /map/active_reference and /map/obstacles for use inside the next
solve() call; also broadcasts the same result on three topics for passive
consumers (viz_node, diagnostics).
"""
import dataclasses

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.config import DEFAULT_CONFIG  # noqa: E402

from nmpc_interfaces.msg import (  # noqa: E402
    ActiveReference, ControlCommand, ObstacleArray, PredictionHorizon, SolverStatus,
)
from nmpc_interfaces.srv import SolveNMPC  # noqa: E402

# matches map_node's /map/obstacles QoS (section 2.5: "transient-local, depth 1 -- latched")
_OBSTACLES_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)
# matches map_node's /map/active_reference QoS (section 2.5: "reliable, depth 1 (keep-last)")
_REFERENCE_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
)

# NMPCConfig fields (nmpc/config.py:48-149) that pass straight through to a ROS2
# parameter of the same name. WP_RADIUS/SIM_TIME_MODE/SIM_TIME_FIXED are owned by
# map_node instead (section 2.4/2.6) so they're deliberately excluded here.
_SCALAR_CONFIG_FIELDS = [
    'N', 'dt', 'LPP', 'R_ASV_FACTOR',
    'DELTA_MIN', 'DELTA_MAX', 'DELTA_DOT_MIN', 'DELTA_DOT_MAX',
    'RPS_MIN', 'RPS_MAX', 'RPS_DOT_MIN', 'RPS_DOT_MAX',
    'U_REF', 'DELTA_TRIM', 'N_TRIM',
    'BRAKE_DISTANCE', 'U_REF_MIN',
    'EPS',
    'SIGMA', 'W_SLACK', 'OBSTACLE_START_K', 'MAX_OBSTACLES',
    'QE_SCALE',
    'IPOPT_MAX_ITER', 'IPOPT_TOL', 'IPOPT_PRINT_LEVEL',
    'ACADOS_QP_SOLVER', 'ACADOS_NLP_SOLVER', 'ACADOS_INTEGRATOR',
    'ACADOS_NUM_STAGES', 'ACADOS_NUM_STEPS',
]


def _clean(value):
    """ROS2 parameter type inference chokes on numpy scalar types (several
    NMPCConfig fields are np.float64, e.g. DELTA_MIN = -np.deg2rad(45.0));
    coerce to plain python int/float first."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


class NmpcNode(Node):
    def __init__(self):
        super().__init__('nmpc_node')

        self.config = self._declare_and_build_config()

        self.declare_parameter('solver_backend', 'acados')
        backend = self.get_parameter('solver_backend').value
        self.get_logger().info(f'building NMPC solver (backend={backend})... this triggers code-gen on first run')
        if backend == 'acados':
            from nmpc.nmpc_acados import AcadosNMPC
            self.solver = AcadosNMPC(self.config)
        elif backend == 'casadi':
            from nmpc.nmpc_casadi import CasadiNMPC
            self.solver = CasadiNMPC(self.config)
        else:
            raise ValueError(f"Unknown solver_backend {backend!r}; expected 'acados' or 'casadi'")
        self.get_logger().info('NMPC solver ready')

        self._active_reference = None   # nmpc_interfaces.msg.ActiveReference, cached
        self._obstacles_cache = []      # list of (x, y, radius) tuples, cached

        self.control_pub = self.create_publisher(ControlCommand, '/nmpc/control_command', 10)
        self.horizon_pub = self.create_publisher(PredictionHorizon, '/nmpc/prediction_horizon', 10)
        self.status_pub = self.create_publisher(SolverStatus, '/nmpc/solver_status', 10)

        self.create_subscription(ActiveReference, '/map/active_reference', self._on_active_reference, _REFERENCE_QOS)
        self.create_subscription(ObstacleArray, '/map/obstacles', self._on_obstacles, _OBSTACLES_QOS)

        self.create_service(SolveNMPC, '/nmpc/solve', self._handle_solve)

        self.get_logger().info('nmpc_node up, /nmpc/solve ready')

    # ------------------------------------------------------------------
    def _declare_and_build_config(self):
        overrides = {}
        for field_name in _SCALAR_CONFIG_FIELDS:
            default = _clean(getattr(DEFAULT_CONFIG, field_name))
            self.declare_parameter(field_name, default)
            overrides[field_name] = self.get_parameter(field_name).value

        self.declare_parameter('Q_DIAG', [float(v) for v in DEFAULT_CONFIG.Q_DIAG])
        self.declare_parameter('R_DIAG', [float(v) for v in DEFAULT_CONFIG.R_DIAG])
        overrides['Q_DIAG'] = tuple(self.get_parameter('Q_DIAG').value)
        overrides['R_DIAG'] = tuple(self.get_parameter('R_DIAG').value)

        # absolute paths (section 2.6 packaging gotcha): must not be relative to
        # ros2 run's cwd, or acados' first-solve code-gen fails to find/write output.
        self.declare_parameter('ACADOS_CODE_EXPORT_DIR', _pkg_paths.ACADOS_CODE_EXPORT_DIR_DEFAULT)
        self.declare_parameter('ACADOS_JSON_FILE', _pkg_paths.ACADOS_JSON_FILE_DEFAULT)
        overrides['ACADOS_CODE_EXPORT_DIR'] = self.get_parameter('ACADOS_CODE_EXPORT_DIR').value
        overrides['ACADOS_JSON_FILE'] = self.get_parameter('ACADOS_JSON_FILE').value

        return dataclasses.replace(DEFAULT_CONFIG, **overrides)

    # ------------------------------------------------------------------
    def _on_active_reference(self, msg: ActiveReference):
        self._active_reference = msg

    def _on_obstacles(self, msg: ObstacleArray):
        self._obstacles_cache = [(o.x, o.y, o.radius) for o in msg.obstacles]

    def _pack_horizon(self, result, stamp) -> PredictionHorizon:
        xi_traj = result['xi_traj']  # (STATE_DIM, N+1)
        u_opt = result['u_opt']      # (CONTROL_DIM, N)
        msg = PredictionHorizon()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.n_states = int(xi_traj.shape[0])
        msg.n_controls = int(u_opt.shape[0])
        msg.horizon_len = int(u_opt.shape[1])
        msg.xi_traj = np.asarray(xi_traj, dtype=float).flatten(order='C').tolist()
        msg.control_horizon = np.asarray(u_opt, dtype=float).T.flatten(order='C').tolist()
        return msg

    def _handle_solve(self, request: SolveNMPC.Request, response: SolveNMPC.Response) -> SolveNMPC.Response:
        state = request.state
        mmg_state = [state.u, state.v, state.r, state.x, state.y, state.psi]
        delta, n = float(state.delta), float(state.n)

        if self._active_reference is not None:
            ref = self._active_reference
            chi_p, x_d, y_d = ref.chi_p, ref.x_d, ref.y_d
        else:
            # startup race: /map/active_reference hasn't arrived yet -- hold the
            # current pose as a safe placeholder target for this one call only.
            self.get_logger().warn('no /map/active_reference received yet, holding current pose', throttle_duration_sec=2.0)
            chi_p, x_d, y_d = state.psi, state.x, state.y

        result = self.solver.solve(mmg_state, delta, n, chi_p, x_d, y_d, obstacles=self._obstacles_cache)

        stamp = self.get_clock().now().to_msg()

        response.command.header.stamp = stamp
        response.command.header.frame_id = 'map'
        response.command.delta = float(result['delta'])
        response.command.n = float(result['n'])
        response.horizon = self._pack_horizon(result, stamp)
        response.success = bool(result['success'])
        response.return_status = str(result['return_status'])
        response.solve_time = float(result['solve_time'])

        self.control_pub.publish(response.command)
        self.horizon_pub.publish(response.horizon)

        status_msg = SolverStatus()
        status_msg.header.stamp = stamp
        status_msg.header.frame_id = 'map'
        status_msg.success = response.success
        status_msg.return_status = response.return_status
        status_msg.solve_time = response.solve_time
        self.status_pub.publish(status_msg)

        if not response.success:
            self.get_logger().warn(f'solver FAILED: {response.return_status}')

        return response


def main(args=None):
    rclpy.init(args=args)
    node = NmpcNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
