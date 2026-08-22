"""logger_node: Synchronous ROS2 experiment recorder.
Subscribes to all simulation topics and persists complete experiment logs via ExperimentLogger.
Launched automatically in bringup.launch.py.
"""

import json
import os
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from controller_effort_logger import ControllerEffortLogger  # noqa: E402
from experiment_logger import ExperimentLogger  # noqa: E402

from nmpc.config import CONTROL_NAMES, IDX_PSI, STATE_NAMES  # noqa: E402
from nmpc.params import DEFAULT_CONFIG  # noqa: E402

from nmpc_interfaces.msg import (  # noqa: E402
    ActiveReference, ControlCommand, ControllerEffortSample, CurrentState, ObstacleArray,
    PredictionHorizon, SimStatus, SolverStatus, VesselState, WaveState,
)

_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)
_REFERENCE_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
)
# must match nmpc_node's _EFFORT_QOS exactly (BEST_EFFORT publisher requires a
# BEST_EFFORT-or-looser subscription, or the two are QoS-incompatible and never match)
_EFFORT_QOS = QoSProfile(
    depth=200,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')

        default_scenario = os.path.join(
            get_package_share_directory('nmpc_sim_nodes'), 'params', 'scenario.json'
        )

        self.declare_parameter('log_dir', '')
        self.declare_parameter('run_name', 'nmpc_experiment')
        self.declare_parameter('save_horizons_npz', True)
        self.declare_parameter('scenario_path', default_scenario)

        log_dir = self.get_parameter('log_dir').value or None
        run_name = self.get_parameter('run_name').value
        save_horizons_npz = bool(self.get_parameter('save_horizons_npz').value)
        scenario_path = self.get_parameter('scenario_path').value

        scenario_dict = None
        self._goal_pos = None
        if scenario_path and os.path.exists(scenario_path):
            try:
                with open(scenario_path, 'r') as f:
                    scenario_dict = json.load(f)
                    if 'waypoints' in scenario_dict and len(scenario_dict['waypoints']) > 0:
                        self._goal_pos = tuple(scenario_dict['waypoints'][-1])
            except Exception as e:
                self.get_logger().warn(f'Could not load scenario from {scenario_path}: {e}')

        # Initialize ExperimentLogger
        self.logger = ExperimentLogger(
            log_dir=log_dir,
            run_name=run_name,
            scenario=scenario_dict,
            save_horizons_npz=save_horizons_npz,
        )

        # Controller-effort / cost-breakdown logger -- writes costs_errors.csv into
        # the SAME run_dir ExperimentLogger just created, on its own background
        # thread (see controller_effort_logger.py's module docstring). Q_DIAG/R_DIAG/
        # rate bounds are read from the same sim_params.yaml nmpc_node itself reads
        # (DEFAULT_CONFIG), not redeclared here, so they can't drift from whatever
        # weights the solver is actually running with; R_DELTA_DIAG has no solver
        # meaning (there is no rate-of-rate term in the actual NMPC cost) so it's a
        # logger_node-only diagnostic parameter, defaulting to zero (no contribution).
        self.declare_parameter('R_DELTA_DIAG', [0.0] * len(DEFAULT_CONFIG.R_DIAG))
        self.declare_parameter('saturation_tol', 1e-3)
        self.declare_parameter('rolling_window', 50)
        self.declare_parameter('effort_queue_maxsize', 500)
        self.declare_parameter('costs_errors_flush_every', 10)

        self.effort_logger = ControllerEffortLogger(
            run_dir=self.logger.run_dir,
            state_names=STATE_NAMES,
            control_names=CONTROL_NAMES,
            q_diag=DEFAULT_CONFIG.Q_DIAG,
            r_diag=DEFAULT_CONFIG.R_DIAG,
            r_delta_diag=self.get_parameter('R_DELTA_DIAG').value,
            u_min=[DEFAULT_CONFIG.DELTA_DOT_MIN, DEFAULT_CONFIG.RPS_DOT_MIN],
            u_max=[DEFAULT_CONFIG.DELTA_DOT_MAX, DEFAULT_CONFIG.RPS_DOT_MAX],
            psi_index=IDX_PSI,
            saturation_tol=float(self.get_parameter('saturation_tol').value),
            rolling_window=int(self.get_parameter('rolling_window').value),
            queue_maxsize=int(self.get_parameter('effort_queue_maxsize').value),
            flush_every=int(self.get_parameter('costs_errors_flush_every').value),
        )
        self.effort_logger.start()

        # State Caches
        self._latest_measured_state: Optional[VesselState] = None
        self._latest_control_cmd: Optional[ControlCommand] = None
        self._latest_prediction_horizon: Optional[tuple] = None
        self._latest_solver_status: Optional[SolverStatus] = None
        self._latest_active_ref: Optional[ActiveReference] = None
        self._latest_current: Optional[CurrentState] = None
        self._latest_wave: Optional[WaveState] = None
        self._start_stamp = None
        self._is_finalized = False

        # Subscriptions
        self.create_subscription(VesselState, '/mmg/state', self._on_mmg_state, 10)
        self.create_subscription(VesselState, '/sensor/measured_state', self._on_measured_state, 10)
        self.create_subscription(ControlCommand, '/nmpc/control_command', self._on_control_command, 10)
        self.create_subscription(PredictionHorizon, '/nmpc/prediction_horizon', self._on_prediction_horizon, 10)
        self.create_subscription(SolverStatus, '/nmpc/solver_status', self._on_solver_status, 10)
        self.create_subscription(ActiveReference, '/map/active_reference', self._on_active_reference, _REFERENCE_QOS)
        self.create_subscription(ObstacleArray, '/map/obstacles', self._on_obstacles, _LATCHED_QOS)
        self.create_subscription(VesselState, '/map/initial_state', self._on_initial_state, _LATCHED_QOS)
        self.create_subscription(SimStatus, '/map/sim_status', self._on_sim_status, _LATCHED_QOS)
        self.create_subscription(CurrentState, '/env/current_state', self._on_current_state, 10)
        self.create_subscription(WaveState, '/env/wave_state', self._on_wave_state, 10)
        self.create_subscription(ControllerEffortSample, '/nmpc/controller_effort_raw',
                                  self._on_controller_effort, _EFFORT_QOS)

        self.get_logger().info(f'logger_node active. Logging to: {self.logger.run_dir}')

    # ------------------------------------------------------------------ Callbacks
    def _elapsed_time(self) -> float:
        if self._start_stamp is None:
            return 0.0
        return (self.get_clock().now() - self._start_stamp).nanoseconds * 1e-9

    def _on_measured_state(self, msg: VesselState):
        self._latest_measured_state = msg

    def _on_control_command(self, msg: ControlCommand):
        self._latest_control_cmd = msg

    def _on_solver_status(self, msg: SolverStatus):
        self._latest_solver_status = msg

    def _on_prediction_horizon(self, msg: PredictionHorizon):
        try:
            n_states, n_controls, horizon_len = msg.n_states, msg.n_controls, msg.horizon_len
            xi_traj = np.array(msg.xi_traj, dtype=float).reshape((n_states, horizon_len + 1), order='C')
            control_horizon = np.array(msg.control_horizon, dtype=float).reshape((horizon_len, n_controls), order='C').T
            self._latest_prediction_horizon = (xi_traj, control_horizon)
        except Exception as e:
            self.get_logger().warn(f'Failed to reshape prediction horizon: {e}')
            self._latest_prediction_horizon = None

    def _on_active_reference(self, msg: ActiveReference):
        self._latest_active_ref = msg

    def _on_obstacles(self, msg: ObstacleArray):
        pass  # already saved in scenario if available

    def _on_initial_state(self, msg: VesselState):
        pass

    def _on_current_state(self, msg: CurrentState):
        self._latest_current = msg

    def _on_wave_state(self, msg: WaveState):
        self._latest_wave = msg

    def _on_controller_effort(self, msg: ControllerEffortSample):
        # O(1) non-blocking hand-off only -- all cost/effort arithmetic and the
        # costs_errors.csv write happen on ControllerEffortLogger's own thread.
        self.effort_logger.log_sample_async(
            step=msg.step, dt=msg.dt, u=msg.u, x=msg.x, x_ref=msg.x_ref,
            solve_time=msg.solve_time, success=msg.success, upstream_drops=msg.upstream_drops,
        )

    def _on_sim_status(self, msg: SimStatus):
        if msg.status != SimStatus.RUNNING and not self._is_finalized:
            status_map = {
                SimStatus.GOAL_REACHED: 'GOAL_REACHED',
                SimStatus.TIMEOUT: 'TIMEOUT',
            }
            status_str = status_map.get(msg.status, f'STATUS_{msg.status}')
            self.get_logger().info(f'Simulation finished with status: {status_str}')
            self.finalize(status=status_str)

    def _on_mmg_state(self, msg: VesselState):
        if self._is_finalized:
            return

        if self._start_stamp is None:
            self._start_stamp = self.get_clock().now()

        t_sim = self._elapsed_time()
        true_state = (msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi, msg.delta, msg.n)

        meas_state = None
        if self._latest_measured_state is not None:
            m = self._latest_measured_state
            meas_state = (m.u, m.v, m.r, m.x, m.y, m.psi, m.delta, m.n)

        cmd = None
        if self._latest_control_cmd is not None:
            cmd = (self._latest_control_cmd.delta, self._latest_control_cmd.n)

        active_ref = None
        if self._latest_active_ref is not None:
            r = self._latest_active_ref
            active_ref = (r.target_idx, r.x_d, r.y_d, r.chi_p)

        current = None
        if self._latest_current is not None:
            c = self._latest_current
            current = (c.enabled, c.vx, c.vy, c.speed, c.heading)

        wave = None
        if self._latest_wave is not None:
            w = self._latest_wave
            wave = (w.enabled, w.fx, w.fy, w.fn)

        solver_status = None
        if self._latest_solver_status is not None:
            s = self._latest_solver_status
            solver_status = (s.success, s.return_status, s.solve_time)

        horizon = self._latest_prediction_horizon

        self.logger.log_step(
            time_s=t_sim,
            true_state=true_state,
            meas_state=meas_state,
            cmd=cmd,
            active_ref=active_ref,
            current=current,
            wave=wave,
            solver_status=solver_status,
            prediction_horizon=horizon,
        )

    def finalize(self, status: str = 'INTERRUPTED'):
        if self._is_finalized:
            return
        self._is_finalized = True
        # Stop+drain the effort logger first so its summary fragment is ready,
        # then fold it into ExperimentLogger's single summary.json write --
        # sequential within this one process, so there's no cross-process merge.
        cost_summary = self.effort_logger.finalize()
        self.logger.finalize(status=status, goal_pos=self._goal_pos, extra=cost_summary)


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize(status='INTERRUPTED')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
