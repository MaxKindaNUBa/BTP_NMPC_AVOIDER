"""nmpc_node: pure optimizer (section 2.6 of ROS2_CONVERSION_PLAN.md). Serves
/nmpc/solve, wrapping AcadosNMPC.solve() (SQP-RTI) unchanged; caches
the latest /map/active_reference and /map/obstacles for use inside the next
solve() call; also broadcasts the same result on three topics for passive
consumers (viz_node, diagnostics).

/nmpc/solve's request.state is always the *measured* state -- mmg_node runs
the sensor noise model (SENSOR_NOISE_MODEL.md) in-process before calling this
service, so "the controller only ever sees corrupted measurements" holds by
construction: this node has no path to the plant's true state at all (no
/mmg/state subscription, no true-state field on the request), regardless of
whether mmg_node's sensor noise is enabled or disabled.
"""
import dataclasses
import queue
import threading

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.params import DEFAULT_CONFIG  # noqa: E402

from nmpc_interfaces.msg import (  # noqa: E402
    ActiveReference, ControlCommand, ControllerEffortSample, ObstacleArray, PredictionHorizon, SolverStatus,
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
# Controller-effort raw stream (consumed by logger_node's ControllerEffortLogger,
# see controller_effort_logger.py): BEST_EFFORT + shallow KEEP_LAST so a slow or
# absent subscriber can never back-pressure this publisher -- a full DDS queue
# just drops the oldest sample instead of blocking publish().
_EFFORT_QOS = QoSProfile(
    depth=200,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
)
# Bound on nmpc_node's own pre-publish queue (see _effort_publish_worker below);
# independent of _EFFORT_QOS's depth, since this one guards against the publish
# call itself (message construction/serialization) ever running on the solve thread.
_EFFORT_QUEUE_MAXSIZE = 200

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

        self.get_logger().info('building NMPC solver (acados SQP-RTI)... this triggers code-gen on first run')
        from nmpc.nmpc_acados import AcadosNMPC
        self.solver = AcadosNMPC(self.config)
        self.get_logger().info('NMPC solver ready')

        self._active_reference = None   # nmpc_interfaces.msg.ActiveReference, cached
        self._obstacles_cache = []      # list of (x, y, radius) tuples, cached

        self.control_pub = self.create_publisher(ControlCommand, '/nmpc/control_command', 10)
        self.horizon_pub = self.create_publisher(PredictionHorizon, '/nmpc/prediction_horizon', 10)
        self.status_pub = self.create_publisher(SolverStatus, '/nmpc/solver_status', 10)
        self.effort_pub = self.create_publisher(ControllerEffortSample, '/nmpc/controller_effort_raw', _EFFORT_QOS)

        # Controller-effort raw samples never get built/published from _handle_solve
        # itself (the real-time solve thread) -- they're handed off via this
        # non-blocking queue to a dedicated background thread instead (see
        # _effort_publish_worker), so message construction/serialization and the
        # publish() call both happen off the solve path. put_nowait() drops (and
        # counts) on backpressure rather than ever blocking the solve thread.
        self._effort_queue = queue.Queue(maxsize=_EFFORT_QUEUE_MAXSIZE)
        self._effort_drop_count = 0
        self._solve_step = 0
        self._effort_stop = threading.Event()
        self._effort_thread = threading.Thread(target=self._effort_publish_worker, daemon=True)
        self._effort_thread.start()

        self.create_subscription(ActiveReference, '/map/active_reference', self._on_active_reference, _REFERENCE_QOS)
        self.create_subscription(ObstacleArray, '/map/obstacles', self._on_obstacles, _OBSTACLES_QOS)

        # Own callback group + MultiThreadedExecutor so /map/active_reference
        # and /map/obstacles subscriptions can still update the cache while a
        # solve is in flight, instead of queuing behind it.
        self._solve_cbg = MutuallyExclusiveCallbackGroup()

        self.create_service(SolveNMPC, '/nmpc/solve', self._handle_solve, callback_group=self._solve_cbg)

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

    def _effort_publish_worker(self):
        """Runs on its own thread: drains the queue _handle_solve enqueues onto
        and does the actual message build + publish, so neither ever happens on
        the solve thread. get(timeout=...) instead of a sentinel keeps shutdown
        simple -- the loop just exits within one timeout tick of _effort_stop."""
        while not self._effort_stop.is_set():
            try:
                sample = self._effort_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            step, dt, u, x, xi_ref, solve_time, success, drops = sample
            msg = ControllerEffortSample()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.step = step
            msg.dt = dt
            msg.u = u.tolist()
            msg.x = x.tolist()
            msg.x_ref = xi_ref.tolist()
            msg.solve_time = solve_time
            msg.success = success
            msg.upstream_drops = drops
            self.effort_pub.publish(msg)

    def _handle_solve(self, request: SolveNMPC.Request, response: SolveNMPC.Response) -> SolveNMPC.Response:
        state = request.state  # already the measured state -- see module docstring
        mmg_state = [state.u, state.v, state.r, state.x, state.y, state.psi]
        delta, n = float(state.delta), float(state.n)
        current = (float(request.current.vx), float(request.current.vy))

        if self._active_reference is not None:
            ref = self._active_reference
            chi_p, x_d, y_d = ref.chi_p, ref.x_d, ref.y_d
        else:
            # startup race: /map/active_reference hasn't arrived yet -- hold the
            # current pose as a safe placeholder target for this one call only.
            self.get_logger().warn('no /map/active_reference received yet, holding current pose', throttle_duration_sec=2.0)
            chi_p, x_d, y_d = state.psi, state.x, state.y

        result = self.solver.solve(mmg_state, delta, n, chi_p, x_d, y_d,
                                    obstacles=self._obstacles_cache, current=current)

        # Raw controller-effort sample: cheap numpy slicing only (arrays already
        # exist in `result`), then a non-blocking hand-off -- no metric arithmetic,
        # no message construction, no I/O happens here (see _effort_publish_worker).
        self._solve_step += 1
        try:
            self._effort_queue.put_nowait((
                self._solve_step,
                self.config.dt,
                np.asarray(result['u_opt'])[:, 0],
                np.asarray(result['xi_traj'])[:, 0],
                np.asarray(result['xi_ref']),
                float(result['solve_time']),
                bool(result['success']),
                self._effort_drop_count,
            ))
        except queue.Full:
            self._effort_drop_count += 1

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
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node._effort_stop.set()
        node._effort_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
