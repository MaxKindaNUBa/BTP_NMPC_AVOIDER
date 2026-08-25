"""viz_node: hosts the live matplotlib visualizer (mpc_visualization/), split
out of map_node so it can be started/stopped independently -- `ros2 run
nmpc_sim_nodes viz_node`, separate from the map/nmpc/mmg nodes that actually
run the simulation. Gets scenario data (waypoints, obstacles, initial state)
via the /map/get_scenario service -- built for exactly this "late-joining
tool" purpose -- then follows the live simulation over topics only, the same
way any other passive consumer would.
"""
import math
import os
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.config import IDX_U, IDX_V, IDX_R, IDX_X, IDX_Y, IDX_PSI, IDX_DELTA, IDX_N  # noqa: E402

from mpc_bridge import CurrentReading, MPCBridge, Obstacle as BridgeObstacle, WaveReading  # noqa: E402
from visualizer import MPCVisualizer  # noqa: E402

from nmpc_interfaces.msg import (  # noqa: E402
    ActiveReference, CurrentState, ObstacleArray, PredictionHorizon, SimStatus, VesselState, WaveState,
)
from nmpc_interfaces.srv import GetScenario  # noqa: E402

# order the bridge expects ship_state/predicted_trajectory rows in: [u, v, r, x, y, psi]
_MMG_ROWS = [IDX_U, IDX_V, IDX_R, IDX_X, IDX_Y, IDX_PSI]

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


class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')

        self.declare_parameter('log_dir', os.path.expanduser('~/nmpc_sim_logs'))
        log_dir = self.get_parameter('log_dir').value

        scenario = self._fetch_scenario()
        waypoints = list(zip(scenario.waypoints_x, scenario.waypoints_y))
        obstacles = [BridgeObstacle(id=o.id, x=o.x, y=o.y, radius=o.radius) for o in scenario.obstacles]

        self.bridge = MPCBridge(start=waypoints[0], goal=waypoints[-1])
        self.bridge._visualizer_active = True
        self.bridge._reference_path = np.array(waypoints, dtype=float)
        self.bridge.set_obstacles(obstacles)

        self.visualizer = MPCVisualizer(self.bridge, update_interval_ms=20, run_name='viz_node', log_dir=log_dir)
        wp = np.array(waypoints)
        y_pad, x_pad = 40.0, 60.0
        self.visualizer.ax_map.set_xlim(wp[:, 1].min() - y_pad, wp[:, 1].max() + y_pad)
        self.visualizer.ax_map.set_ylim(wp[:, 0].min() - x_pad, wp[:, 0].max() + x_pad)

        self._start_stamp = None  # first /mmg/state message defines t=0 for this node's own display/log timestamps

        self.create_subscription(VesselState, '/mmg/state', self._on_mmg_state, 10)
        self.create_subscription(PredictionHorizon, '/nmpc/prediction_horizon', self._on_prediction_horizon, 10)
        self.create_subscription(ObstacleArray, '/map/obstacles', self._on_obstacles, _LATCHED_QOS)
        self.create_subscription(ActiveReference, '/map/active_reference', self._on_active_reference, _REFERENCE_QOS)
        self.create_subscription(SimStatus, '/map/sim_status', self._on_sim_status, _LATCHED_QOS)
        self.create_subscription(CurrentState, '/env/current_state', self._on_current_state, 10)
        self.create_subscription(WaveState, '/env/wave_state', self._on_wave_state, 10)
        self.create_subscription(CurrentState, '/ukf/estimated_current', self._on_ukf_current, 10)
        self.create_subscription(VesselState, '/ukf/estimated_state', self._on_ukf_state, 10)

        self.get_logger().info(f'viz_node up: {len(waypoints)} waypoints, {len(obstacles)} obstacles, log_dir={log_dir}')

    # ------------------------------------------------------------------
    def _fetch_scenario(self) -> GetScenario.Response:
        client = self.create_client(GetScenario, '/map/get_scenario')
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('waiting for /map/get_scenario (map_node not up yet?)...')
        future = client.call_async(GetScenario.Request())
        rclpy.spin_until_future_complete(self, future)  # safe here: runs once, before this node's own executor.spin() starts
        return future.result()

    # ------------------------------------------------------------------
    def _elapsed_time(self) -> float:
        if self._start_stamp is None:
            return 0.0
        return (self.get_clock().now() - self._start_stamp).nanoseconds * 1e-9

    def _on_mmg_state(self, msg: VesselState):
        if self._start_stamp is None:
            self._start_stamp = self.get_clock().now()
        mmg_state = np.array([msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi], dtype=float)
        self.bridge.update_ship_state(mmg_state, self._elapsed_time())

    def _on_prediction_horizon(self, msg: PredictionHorizon):
        n_states, horizon_len = msg.n_states, msg.horizon_len
        xi_traj = np.array(msg.xi_traj, dtype=float).reshape(n_states, horizon_len + 1, order='C')

        # bridge expects the SAME extraction the original run_live.py fed it
        # directly: MMG rows [u,v,r,x,y,psi] for the spatial overlay, and the
        # actuator VALUE columns [delta, n] (not the rate control
        # u_opt/[delta_dot,n_dot] that PredictionHorizon.control_horizon
        # itself carries) for the control panel.
        predicted_trajectory = xi_traj[_MMG_ROWS, :].T             # (horizon_len+1, 6)
        control_horizon = xi_traj[[IDX_DELTA, IDX_N], :-1].T       # (horizon_len, 2)

        self.bridge.update_prediction(predicted_trajectory, control_horizon)

    def _on_obstacles(self, msg: ObstacleArray):
        obstacles = [BridgeObstacle(id=o.id, x=o.x, y=o.y, radius=o.radius) for o in msg.obstacles]
        self.bridge.set_obstacles(obstacles)

    def _on_active_reference(self, msg: ActiveReference):
        self.bridge.set_active_waypoint((msg.x_d, msg.y_d))

    def _on_current_state(self, msg: CurrentState):
        self.bridge.update_current(CurrentReading(
            vx=msg.vx, vy=msg.vy, speed=msg.speed, heading=msg.heading, enabled=msg.enabled))

    def _on_ukf_current(self, msg: CurrentState):
        self.bridge.update_ukf_current(CurrentReading(
            vx=msg.vx, vy=msg.vy, speed=msg.speed, heading=msg.heading, enabled=msg.enabled))

    def _on_ukf_state(self, msg: VesselState):
        self.bridge.update_ukf_position((msg.x, msg.y))

    def _on_wave_state(self, msg: WaveState):
        # msg.fx/fy are body-frame (see CurrentState.msg/WaveState.msg); rotate into
        # earth-frame using the latest known heading for the compass HUD, same
        # rotation rviz_node.py's _on_wave_state applies for its ship-anchored arrow.
        psi = float(self.bridge.snapshot().ship_state[5])  # bridge's own fixed [u,v,r,x,y,psi] order
        earth_fx = msg.fx * math.cos(psi) - msg.fy * math.sin(psi)
        earth_fy = msg.fx * math.sin(psi) + msg.fy * math.cos(psi)
        self.bridge.update_wave(WaveReading(fx=earth_fx, fy=earth_fy, fn=msg.fn, enabled=msg.enabled))

    def _on_sim_status(self, msg: SimStatus):
        if msg.status != SimStatus.RUNNING:
            names = {SimStatus.GOAL_REACHED: 'GOAL_REACHED', SimStatus.TIMEOUT: 'TIMEOUT'}
            self.get_logger().info(f'sim_status -> {names.get(msg.status, msg.status)} at t={msg.sim_time:.1f}s')


def _spin_in_background(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass  # normal on rclpy.shutdown() racing this daemon thread


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()

    spin_thread = threading.Thread(target=_spin_in_background, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # blocks in THIS (main) thread until the window is closed; matplotlib
        # requires the main thread, same constraint the original run_live.py had.
        node.visualizer.start()
    finally:
        node.bridge._visualizer_active = False
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
