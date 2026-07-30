"""rviz_node: converts the sim's own topics (VesselState, ObstacleArray,
PredictionHorizon, ActiveReference) into visualization_msgs/MarkerArray on
/viz/markers, purely so RViz2 -- which only understands a fixed set of
standard message types -- can render the simulation. This node changes no
data, it's a read-only side channel: same subscriptions viz_node.py uses,
same source of truth, just re-published as shapes. Run alongside `ros2 run
rviz2 rviz2 -d $(ros2 pkg prefix nmpc_sim_nodes)/share/nmpc_sim_nodes/rviz/sim_view.rviz`,
or via `ros2 run nmpc_sim_nodes rviz_node`.
"""
import collections
import math

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.config import IDX_X, IDX_Y  # noqa: E402

from nmpc_interfaces.msg import ActiveReference, ObstacleArray, PredictionHorizon, SimStatus, VesselState  # noqa: E402
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

_TRAIL_MAX_POINTS = 5000
_SHIP_LENGTH = 2.902  # LPP; matches nmpc.config.NMPCConfig default
_SHIP_WIDTH = 0.7 * _SHIP_LENGTH * 0.25


def _color(r, g, b, a=1.0):
    return ColorRGBA(r=r, g=g, b=b, a=a)


def _plot_point(state_x, state_y, z=0.0):
    # mpc_visualization/visualizer.py plots everything as set_data(state_y, state_x)
    # (X axis on screen = East/state_y, Y axis on screen = North/state_x) -- match
    # that convention here so RViz renders the same layout as the matplotlib tool,
    # instead of ROS's usual "world X = right" default.
    return Point(x=float(state_y), y=float(state_x), z=float(z))


def _yaw_quat(psi):
    # the (state_y, state_x) swap above is orientation-reversing (a reflection),
    # so a heading of psi in state space must be re-expressed as (pi/2 - psi) in
    # the swapped plot frame for an ARROW marker (which points along +X at yaw=0)
    # to visually match the matplotlib ship polygon's heading.
    plot_yaw = math.pi / 2.0 - psi
    return Quaternion(x=0.0, y=0.0, z=math.sin(plot_yaw / 2.0), w=math.cos(plot_yaw / 2.0))


class RvizNode(Node):
    def __init__(self):
        super().__init__('rviz_node')

        self.markers_pub = self.create_publisher(MarkerArray, '/viz/markers', 10)
        self.status_pub = self.create_publisher(String, '/viz/status_text', 10)

        scenario = self._fetch_scenario()
        waypoints = list(zip(scenario.waypoints_x, scenario.waypoints_y))

        self._path_markers = self._build_path_markers(waypoints)
        self._obstacle_markers = [self._build_obstacle_marker(o) for o in scenario.obstacles]
        self._ship_marker = None
        self._trail_marker = None
        self._prediction_marker = None
        self._active_wp_marker = None
        self._trail_points = collections.deque(maxlen=_TRAIL_MAX_POINTS)

        # cached, used by the /viz/status_text telemetry block (mirrors visualizer.py's info_text)
        self._goal = waypoints[-1]
        self._active_waypoint = None
        self._obstacles_xyr = [(o.x, o.y, o.radius) for o in scenario.obstacles]

        # clear any markers left over from a previous run/session before publishing fresh ones
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        self.markers_pub.publish(MarkerArray(markers=[clear]))
        self._publish_all()

        self.create_subscription(VesselState, '/mmg/state', self._on_mmg_state, 10)
        self.create_subscription(PredictionHorizon, '/nmpc/prediction_horizon', self._on_prediction_horizon, 10)
        self.create_subscription(ObstacleArray, '/map/obstacles', self._on_obstacles, _LATCHED_QOS)
        self.create_subscription(ActiveReference, '/map/active_reference', self._on_active_reference, _REFERENCE_QOS)
        self.create_subscription(SimStatus, '/map/sim_status', self._on_sim_status, _LATCHED_QOS)

        self.get_logger().info(f'rviz_node up: {len(waypoints)} waypoints, {len(self._obstacle_markers)} obstacles; '
                                f'publishing MarkerArray on /viz/markers, telemetry on /viz/status_text')

    # ------------------------------------------------------------------
    def _fetch_scenario(self) -> GetScenario.Response:
        client = self.create_client(GetScenario, '/map/get_scenario')
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('waiting for /map/get_scenario (map_node not up yet?)...')
        future = client.call_async(GetScenario.Request())
        rclpy.spin_until_future_complete(self, future)  # safe here: one-shot, before this node's own spin() starts
        return future.result()

    # ---- static / slow-changing markers --------------------------------
    def _build_path_markers(self, waypoints):
        path = Marker()
        path.header.frame_id = 'map'
        path.ns, path.id = 'path', 0
        path.type, path.action = Marker.LINE_STRIP, Marker.ADD
        path.scale = Vector3(x=0.3, y=0.0, z=0.0)
        path.color = _color(0.6, 0.6, 0.6, 0.8)
        path.points = [_plot_point(x, y) for x, y in waypoints]

        wp_dots = Marker()
        wp_dots.header.frame_id = 'map'
        wp_dots.ns, wp_dots.id = 'waypoints', 0
        wp_dots.type, wp_dots.action = Marker.SPHERE_LIST, Marker.ADD
        wp_dots.scale = Vector3(x=1.0, y=1.0, z=1.0)
        wp_dots.color = _color(0.6, 0.6, 1.0, 0.9)
        wp_dots.points = [_plot_point(x, y) for x, y in waypoints]

        return [path, wp_dots]

    def _build_obstacle_marker(self, obstacle):
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id = 'obstacles', hash(obstacle.id) & 0x7FFFFFFF
        m.type, m.action = Marker.CYLINDER, Marker.ADD
        m.pose = Pose(position=_plot_point(obstacle.x, obstacle.y, 0.0), orientation=Quaternion(w=1.0))
        d = 2.0 * float(obstacle.radius)
        m.scale = Vector3(x=d, y=d, z=0.5)
        m.color = _color(0.9, 0.2, 0.2, 0.5)
        return m

    # ---- live callbacks --------------------------------------------------
    def _on_mmg_state(self, msg: VesselState):
        ship = Marker()
        ship.header.frame_id = 'map'
        ship.ns, ship.id = 'ship', 0
        ship.type, ship.action = Marker.ARROW, Marker.ADD
        ship.pose = Pose(position=_plot_point(msg.x, msg.y, 0.0), orientation=_yaw_quat(msg.psi))
        ship.scale = Vector3(x=_SHIP_LENGTH, y=_SHIP_WIDTH, z=_SHIP_WIDTH)
        ship.color = _color(0.1, 0.6, 1.0, 1.0)
        self._ship_marker = ship

        self._trail_points.append(_plot_point(msg.x, msg.y))
        trail = Marker()
        trail.header.frame_id = 'map'
        trail.ns, trail.id = 'trail', 0
        trail.type, trail.action = Marker.LINE_STRIP, Marker.ADD
        trail.scale = Vector3(x=0.15, y=0.0, z=0.0)
        trail.color = _color(0.1, 0.6, 1.0, 0.6)
        trail.points = list(self._trail_points)
        self._trail_marker = trail

        self.status_pub.publish(String(data=self._build_status_text(msg)))
        self._publish_all()

    def _on_prediction_horizon(self, msg: PredictionHorizon):
        n_states, horizon_len = msg.n_states, msg.horizon_len
        xi_traj = [msg.xi_traj[k * (horizon_len + 1):(k + 1) * (horizon_len + 1)] for k in range(n_states)]

        pred = Marker()
        pred.header.frame_id = 'map'
        pred.ns, pred.id = 'prediction', 0
        pred.type, pred.action = Marker.LINE_STRIP, Marker.ADD
        pred.scale = Vector3(x=0.2, y=0.0, z=0.0)
        pred.color = _color(0.2, 0.9, 0.2, 0.9)
        pred.points = [_plot_point(x, y) for x, y in zip(xi_traj[IDX_X], xi_traj[IDX_Y])]
        self._prediction_marker = pred

        self._publish_all()

    def _on_obstacles(self, msg: ObstacleArray):
        self._obstacle_markers = [self._build_obstacle_marker(o) for o in msg.obstacles]
        self._obstacles_xyr = [(o.x, o.y, o.radius) for o in msg.obstacles]
        self._publish_all()

    def _on_active_reference(self, msg: ActiveReference):
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id = 'active_waypoint', 0
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose = Pose(position=_plot_point(msg.x_d, msg.y_d, 0.0), orientation=Quaternion(w=1.0))
        m.scale = Vector3(x=1.6, y=1.6, z=1.6)
        m.color = _color(1.0, 0.7, 0.0, 0.95)
        self._active_wp_marker = m
        self._active_waypoint = (msg.x_d, msg.y_d)
        self._publish_all()

    def _build_status_text(self, msg: VesselState) -> str:
        # mirrors mpc_visualization/visualizer.py's info_text block (SHIP STATUS /
        # CONTROLS / TARGETS), just as a plain string instead of a matplotlib overlay.
        ship_pos = (msg.x, msg.y)
        goal_dist = math.hypot(ship_pos[0] - self._goal[0], ship_pos[1] - self._goal[1])
        target = self._active_waypoint if self._active_waypoint is not None else self._goal
        active_wp_dist = math.hypot(ship_pos[0] - target[0], ship_pos[1] - target[1])
        nearest_obstacle_dist = min(
            (math.hypot(ship_pos[0] - ox, ship_pos[1] - oy) - orad for ox, oy, orad in self._obstacles_xyr),
            default=float('inf'))

        return (
            f"=== SHIP STATUS ===\n"
            f"u (Surge) : {msg.u:6.3f} m/s\n"
            f"v (Sway)  : {msg.v:6.3f} m/s\n"
            f"r (Yaw Rt): {msg.r:6.3f} rad/s\n"
            f"Heading   : {math.degrees(msg.psi):6.1f} deg\n\n"
            f"=== CONTROLS ===\n"
            f"Rudder    : {math.degrees(msg.delta):6.1f} deg\n"
            f"Propeller : {msg.n:6.1f} rps\n\n"
            f"=== TARGETS ===\n"
            f"Active WP : {active_wp_dist:6.2f} m\n"
            f"Goal Dist : {goal_dist:6.2f} m\n"
            f"Obs Dist  : {nearest_obstacle_dist:6.2f} m"
        )

    def _on_sim_status(self, msg: SimStatus):
        if msg.status != SimStatus.RUNNING:
            names = {SimStatus.GOAL_REACHED: 'GOAL_REACHED', SimStatus.TIMEOUT: 'TIMEOUT'}
            self.get_logger().info(f'sim_status -> {names.get(msg.status, msg.status)} at t={msg.sim_time:.1f}s')

    # ------------------------------------------------------------------
    def _publish_all(self):
        markers = list(self._path_markers) + list(self._obstacle_markers)
        for m in (self._active_wp_marker, self._trail_marker, self._prediction_marker, self._ship_marker):
            if m is not None:
                markers.append(m)
        now = self.get_clock().now().to_msg()
        for m in markers:
            m.header.stamp = now
        self.markers_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = RvizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
