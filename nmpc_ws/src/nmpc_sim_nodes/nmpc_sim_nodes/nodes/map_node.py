"""map_node: environment (section 2.5 of ROS2_CONVERSION_PLAN.md, minus the
visualizer -- that's viz_node.py's job now). Owns scenario data, active-
waypoint bookkeeping, and run-termination logic.
"""
import json
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.params import DEFAULT_CONFIG  # noqa: E402
from nmpc.path_following import compute_path_angle, select_active_waypoint  # noqa: E402

from nmpc_interfaces.msg import (  # noqa: E402
    ActiveReference, ControlCommand, Obstacle, ObstacleArray, PredictionHorizon, SimStatus, VesselState,
)
from nmpc_interfaces.srv import GetScenario  # noqa: E402

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


class MapNode(Node):
    def __init__(self):
        super().__init__('map_node')

        default_scenario = os.path.join(get_package_share_directory('nmpc_sim_nodes'), 'params', 'scenario.json')
        self.declare_parameter('scenario_path', default_scenario)
        self.declare_parameter('wp_radius', float(DEFAULT_CONFIG.WP_RADIUS))
        self.declare_parameter('sim_time_mode', DEFAULT_CONFIG.SIM_TIME_MODE)
        self.declare_parameter('sim_time_fixed', float(DEFAULT_CONFIG.SIM_TIME_FIXED))
        self.declare_parameter('max_obstacles', int(DEFAULT_CONFIG.MAX_OBSTACLES))

        scenario_path = self.get_parameter('scenario_path').value
        self.wp_radius = float(self.get_parameter('wp_radius').value)
        self.sim_time_mode = self.get_parameter('sim_time_mode').value
        self.sim_time_fixed = float(self.get_parameter('sim_time_fixed').value)
        max_obstacles = int(self.get_parameter('max_obstacles').value)

        if self.sim_time_mode not in ('infinite', 'fixed', 'endpoint'):
            raise ValueError(f"Unknown sim_time_mode {self.sim_time_mode!r}; expected 'infinite', 'fixed', or 'endpoint'")

        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f'scenario file not found: {scenario_path}')
        with open(scenario_path, 'r') as f:
            scenario = json.load(f)

        self.waypoints = [tuple(wp) for wp in scenario['waypoints']]
        self.mmg_init = list(scenario['mmg_init'])
        self.scenario_obstacles = list(scenario.get('obstacles', []))  # [(x, y, radius), ...]
        if len(self.scenario_obstacles) > max_obstacles:
            self.get_logger().warn(
                f'scenario has {len(self.scenario_obstacles)} obstacles but nmpc_node max_obstacles={max_obstacles}; '
                'excess ones will be silently truncated by pad_obstacles() on the solver side -- '
                'raise nmpc_node\'s max_obstacles parameter to match.')

        self.last_idx = len(self.waypoints) - 1
        self.target_idx = 1  # matches the original run_live.py's fixed starting leg

        # ---- publishers ----
        self.obstacles_pub = self.create_publisher(ObstacleArray, '/map/obstacles', _LATCHED_QOS)
        self.initial_state_pub = self.create_publisher(VesselState, '/map/initial_state', _LATCHED_QOS)
        self.active_reference_pub = self.create_publisher(ActiveReference, '/map/active_reference', _REFERENCE_QOS)
        # latched: mmg_node's tick loop is gated on seeing RUNNING at least once
        # (see mmg_node.py); a late subscriber must still get the last status.
        self.sim_status_pub = self.create_publisher(SimStatus, '/map/sim_status', _LATCHED_QOS)

        # ---- subscriptions ----
        self.create_subscription(VesselState, '/mmg/state', self._on_mmg_state, 10)

        # ---- service ----
        self.create_service(GetScenario, '/map/get_scenario', self._handle_get_scenario)

        self._sim_status = SimStatus.RUNNING
        self._start_stamp = None  # set on first /mmg/state message, for "fixed"-mode elapsed-time tracking

        self._publish_obstacles()
        self._publish_initial_state()
        self._publish_active_reference()  # initial leg, before any /mmg/state has arrived
        self._publish_sim_status()

        self.get_logger().info(f'map_node up: {len(self.waypoints)} waypoints, {len(self.scenario_obstacles)} obstacles, '
                                f'sim_time_mode={self.sim_time_mode!r}')

    # ------------------------------------------------------------------
    def _publish_obstacles(self):
        msg = ObstacleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.obstacles = [Obstacle(id=f'obs_{i}', x=float(x), y=float(y), radius=float(r))
                          for i, (x, y, r) in enumerate(self.scenario_obstacles)]
        self.obstacles_pub.publish(msg)

    def _publish_initial_state(self):
        msg = VesselState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi = [float(v) for v in self.mmg_init]
        msg.delta = float(DEFAULT_CONFIG.DELTA_TRIM)
        msg.n = float(DEFAULT_CONFIG.N_TRIM)
        self.initial_state_pub.publish(msg)

    def _current_leg_reference(self):
        prev_wp = self.waypoints[self.target_idx - 1]
        target_wp = self.waypoints[self.target_idx]
        chi_p = compute_path_angle(prev_wp, target_wp)
        x_d, y_d = target_wp
        return chi_p, x_d, y_d

    def _publish_active_reference(self):
        chi_p, x_d, y_d = self._current_leg_reference()
        msg = ActiveReference()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.chi_p = float(chi_p)
        msg.x_d = float(x_d)
        msg.y_d = float(y_d)
        msg.target_idx = int(self.target_idx)
        self.active_reference_pub.publish(msg)

    def _publish_sim_status(self):
        msg = SimStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.status = self._sim_status
        msg.sim_time = self._elapsed_sim_time()
        self.sim_status_pub.publish(msg)

    def _elapsed_sim_time(self) -> float:
        if self._start_stamp is None:
            return 0.0
        return (self.get_clock().now() - self._start_stamp).nanoseconds * 1e-9

    # ------------------------------------------------------------------
    def _on_mmg_state(self, msg: VesselState):
        if self._start_stamp is None:
            self._start_stamp = self.get_clock().now()

        if self._sim_status != SimStatus.RUNNING:
            return  # already terminated; stop advancing waypoint/termination logic

        x, y = msg.x, msg.y
        t = self._elapsed_sim_time()

        self.target_idx = select_active_waypoint(x, y, self.waypoints, self.target_idx, self.wp_radius)
        self._publish_active_reference()

        if self.sim_time_mode == 'fixed':
            if t >= self.sim_time_fixed:
                self._sim_status = SimStatus.TIMEOUT
                self.get_logger().info(f'sim_time_fixed ({self.sim_time_fixed}s) reached -> TIMEOUT')
        elif self.sim_time_mode == 'endpoint':
            if self.target_idx == self.last_idx:
                goal = self.waypoints[self.last_idx]
                dist_to_goal = float(np.hypot(x - goal[0], y - goal[1]))
                if dist_to_goal < self.wp_radius:
                    self._sim_status = SimStatus.GOAL_REACHED
                    self.get_logger().info(f'endpoint reached (dist={dist_to_goal:.2f}m) -> GOAL_REACHED')
        # 'infinite': never sets a terminal status on its own (matches the original run_live.py)

        self._publish_sim_status()

    # ------------------------------------------------------------------
    def _handle_get_scenario(self, request: GetScenario.Request, response: GetScenario.Response) -> GetScenario.Response:
        response.initial_state = VesselState()
        (response.initial_state.u, response.initial_state.v, response.initial_state.r,
         response.initial_state.x, response.initial_state.y, response.initial_state.psi) = [float(v) for v in self.mmg_init]
        response.initial_state.delta = float(DEFAULT_CONFIG.DELTA_TRIM)
        response.initial_state.n = float(DEFAULT_CONFIG.N_TRIM)
        response.waypoints_x = [float(wp[0]) for wp in self.waypoints]
        response.waypoints_y = [float(wp[1]) for wp in self.waypoints]
        response.obstacles = [Obstacle(id=f'obs_{i}', x=float(x), y=float(y), radius=float(r))
                               for i, (x, y, r) in enumerate(self.scenario_obstacles)]
        response.sim_time_fixed = float(self.sim_time_fixed)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
