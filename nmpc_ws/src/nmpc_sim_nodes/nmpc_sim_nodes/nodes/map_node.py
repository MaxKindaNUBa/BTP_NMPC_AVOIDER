"""map_node: environment (section 2.5 of ROS2_CONVERSION_PLAN.md, minus the
visualizer -- that's rviz_node.py's/hud_node.py's job now). Owns scenario
data, active-waypoint bookkeeping, and run-termination logic.
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
from nmpc.path_following import (  # noqa: E402
    compute_path_angle, select_active_waypoint, segments_from_waypoints, SegmentQueue,
)

from nmpc_interfaces.msg import (  # noqa: E402
    ActiveReference, ControlCommand, Ellipse, EllipseArray, Obstacle, ObstacleArray, PredictionHorizon, Segment,
    SegmentArray, SimStatus, VesselState, WallObstacle, WallObstacleArray,
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
        self.declare_parameter('max_walls', int(DEFAULT_CONFIG.MAX_WALLS))
        self.declare_parameter('max_ellipses', int(DEFAULT_CONFIG.MAX_ELLIPSES))

        scenario_path = self.get_parameter('scenario_path').value
        self.wp_radius = float(self.get_parameter('wp_radius').value)
        self.sim_time_mode = self.get_parameter('sim_time_mode').value
        self.sim_time_fixed = float(self.get_parameter('sim_time_fixed').value)
        max_obstacles = int(self.get_parameter('max_obstacles').value)
        max_walls = int(self.get_parameter('max_walls').value)
        max_ellipses = int(self.get_parameter('max_ellipses').value)

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
        # [(x0, y0, x1, y1, radius), ...] capsule (wall) obstacles -- see
        # research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md
        self.scenario_walls = list(scenario.get('walls', []))
        if len(self.scenario_walls) > max_walls:
            self.get_logger().warn(
                f'scenario has {len(self.scenario_walls)} walls but nmpc_node max_walls={max_walls}; '
                'excess ones will be silently truncated by pad_walls() on the solver side -- '
                'raise nmpc_node\'s max_walls parameter to match.')
        # [(xc, yc, a, b, theta), ...] elliptical obstacles, e.g. other vessels --
        # theta in radians, world frame -- see research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md
        self.scenario_ellipses = list(scenario.get('ellipses', []))
        if len(self.scenario_ellipses) > max_ellipses:
            self.get_logger().warn(
                f'scenario has {len(self.scenario_ellipses)} ellipses but nmpc_node max_ellipses={max_ellipses}; '
                'excess ones will be silently truncated by pad_ellipses() on the solver side -- '
                'raise nmpc_node\'s max_ellipses parameter to match.')

        self.last_idx = len(self.waypoints) - 1
        self.target_idx = 1  # matches the original run_live.py's fixed starting leg

        # Live queue of active path segments the NMPC horizon previews (see
        # nmpc/README.md item 8): seeded here with the entire remaining static
        # path -- today's "let all segments go in" default -- and popped from
        # the front in lockstep with target_idx as waypoints are reached
        # (_on_mmg_state below). A future dynamic segment producer (e.g. a
        # LIDAR rolling window) would drive this same queue via append()/
        # pop_crossed() instead of being seeded from the static waypoint list.
        self._segment_queue = SegmentQueue(segments_from_waypoints(self.waypoints, self.target_idx))

        # ---- publishers ----
        self.obstacles_pub = self.create_publisher(ObstacleArray, '/map/obstacles', _LATCHED_QOS)
        self.walls_pub = self.create_publisher(WallObstacleArray, '/map/walls', _LATCHED_QOS)
        self.ellipses_pub = self.create_publisher(EllipseArray, '/map/ellipses', _LATCHED_QOS)
        self.initial_state_pub = self.create_publisher(VesselState, '/map/initial_state', _LATCHED_QOS)
        self.active_reference_pub = self.create_publisher(ActiveReference, '/map/active_reference', _REFERENCE_QOS)
        # streams every tick (queue genuinely changes over a run, unlike the
        # static obstacle set) -- same QoS as /map/active_reference, not the
        # transient-local "publish once" pattern used for /map/obstacles.
        self.active_segments_pub = self.create_publisher(SegmentArray, '/map/active_segments', _REFERENCE_QOS)
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
        self._publish_walls()
        self._publish_ellipses()
        self._publish_initial_state()
        self._publish_active_reference()  # initial leg, before any /mmg/state has arrived
        self._publish_active_segments()   # initial queue contents, same timing as above
        self._publish_sim_status()

        self.get_logger().info(f'map_node up: {len(self.waypoints)} waypoints, {len(self.scenario_obstacles)} obstacles, '
                                f'{len(self.scenario_walls)} walls, {len(self.scenario_ellipses)} ellipses, '
                                f'sim_time_mode={self.sim_time_mode!r}')

    # ------------------------------------------------------------------
    def _publish_obstacles(self):
        msg = ObstacleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.obstacles = [Obstacle(id=f'obs_{i}', x=float(x), y=float(y), radius=float(r))
                          for i, (x, y, r) in enumerate(self.scenario_obstacles)]
        self.obstacles_pub.publish(msg)

    def _publish_walls(self):
        msg = WallObstacleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.walls = [WallObstacle(id=f'wall_{i}', x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                                   radius=float(r))
                     for i, (x0, y0, x1, y1, r) in enumerate(self.scenario_walls)]
        self.walls_pub.publish(msg)

    def _publish_ellipses(self):
        msg = EllipseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.ellipses = [Ellipse(id=f'ellipse_{i}', x=float(xc), y=float(yc), a=float(a), b=float(b),
                                 theta=float(theta))
                        for i, (xc, yc, a, b, theta) in enumerate(self.scenario_ellipses)]
        self.ellipses_pub.publish(msg)

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

    def _publish_active_segments(self):
        msg = SegmentArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.segments = [Segment(chi_p=float(chi_p), end_x=float(ex), end_y=float(ey))
                         for chi_p, ex, ey in self._segment_queue.segments]
        self.active_segments_pub.publish(msg)

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
        # both driven by the same is_segment_crossed predicate (see path_following.py),
        # so target_idx and the queue front advance in lockstep by construction.
        self._segment_queue.pop_crossed(x, y, self.wp_radius)
        self._publish_active_reference()
        self._publish_active_segments()

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
        response.walls = [WallObstacle(id=f'wall_{i}', x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                                        radius=float(r))
                           for i, (x0, y0, x1, y1, r) in enumerate(self.scenario_walls)]
        response.ellipses = [Ellipse(id=f'ellipse_{i}', x=float(xc), y=float(yc), a=float(a), b=float(b),
                                      theta=float(theta))
                              for i, (xc, yc, a, b, theta) in enumerate(self.scenario_ellipses)]
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
