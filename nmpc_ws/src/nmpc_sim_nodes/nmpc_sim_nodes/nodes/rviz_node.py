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

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.config import IDX_X, IDX_Y  # noqa: E402
from nmpc.params import DEFAULT_CONFIG  # noqa: E402

from nmpc_interfaces.msg import (ActiveReference, CurrentState, Ellipse, EllipseArray, Obstacle,  # noqa: E402
                                  ObstacleArray, PredictedPathArray, PredictionHorizon, SimStatus, VesselState,
                                  WallObstacle, WaveState, WallObstacleArray)
from nmpc_interfaces.srv import GetScenario  # noqa: E402
from rviz_2d_overlay_msgs.msg import OverlayText  # noqa: E402

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
_SHIP_LENGTH = DEFAULT_CONFIG.LPP  # sim_params.yaml's nmpc_node.LPP, never a separate copy
_SHIP_WIDTH = 0.7 * _SHIP_LENGTH * 0.25

# Display-only scale factors: current speed [m/s] and wave force [N] are both
# far too small in magnitude to read as arrow lengths at ship scale, so both
# get a fixed visual gain -- these do not affect the underlying data, only
# how long the RViz arrow is drawn.
_CURRENT_ARROW_GAIN = 20.0
_WAVE_ARROW_GAIN = 40.0
_MIN_ARROW_LENGTH = 0.3

_VELOCITY_EPS = 1e-9  # below this speed [m/s], treat an obstacle/wall/ellipse as stationary


def _color(r, g, b, a=1.0):
    return ColorRGBA(r=r, g=g, b=b, a=a)


def _overlay_text(text, horizontal_alignment, vertical_alignment, horizontal_distance, vertical_distance,
                   width, height, text_size=12.0):
    # rviz_2d_overlay_plugins/TextOverlay: a screen-pixel-anchored text box (unlike
    # every other HUD element in this file, which is a world-space Marker) -- stays
    # put regardless of camera pan/zoom. width/height must be set explicitly or the
    # plugin renders a zero-size texture.
    m = OverlayText()
    m.action = OverlayText.ADD
    m.width = width
    m.height = height
    m.horizontal_distance = horizontal_distance
    m.vertical_distance = vertical_distance
    m.horizontal_alignment = horizontal_alignment
    m.vertical_alignment = vertical_alignment
    m.bg_color = _color(0.0, 0.0, 0.0, 0.5)
    m.fg_color = _color(0.9, 0.9, 0.9, 0.95)
    m.line_width = 2
    m.text_size = text_size
    m.font = 'DejaVu Sans Mono'
    m.text = text
    return m


def _point_to_segment_dist(px, py, x0, y0, x1, y1):
    """Plain-python twin of nmpc/path_following.py's capsule_distance_casadi
    (minus the radius padding, minus the CasADi symbolics) -- used only for
    the HUD's nearest-obstacle readout, not the solver."""
    ex, ey = x1 - x0, y1 - y0
    denom = ex * ex + ey * ey
    t = 0.0 if denom < 1e-9 else max(0.0, min(1.0, ((px - x0) * ex + (py - y0) * ey) / denom))
    cx, cy = x0 + t * ex, y0 + t * ey
    return math.hypot(px - cx, py - cy)


def _point_ellipse_distance(px, py, xc, yc, a, b, theta):
    """Plain-python twin of nmpc/path_following.py's ellipse_distance_casadi
    (minus r_pad, minus the CasADi symbolics) -- APPROXIMATE (gradient-
    normalized, not exact -- no closed form exists), same formula, used only
    for the HUD's nearest-obstacle readout, not the solver. See
    research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md."""
    dx = (px - xc) * math.cos(theta) + (py - yc) * math.sin(theta)
    dy = -(px - xc) * math.sin(theta) + (py - yc) * math.cos(theta)
    g = (dx / a) ** 2 + (dy / b) ** 2 - 1.0
    grad_mag = 2.0 * math.sqrt((dx / a ** 2) ** 2 + (dy / b ** 2) ** 2)
    return g / (grad_mag + 1e-9)


def _plot_point(state_x, state_y, z=0.0):
    # mpc_visualization/visualizer.py plots everything as set_data(state_y, state_x)
    # (X axis on screen = East/state_y, Y axis on screen = North/state_x) -- match
    # that convention here so RViz renders the same layout as the matplotlib tool,
    # instead of ROS's usual "world X = right" default.
    return Point(x=float(state_y), y=float(state_x), z=float(z))


def _arrow_marker(ns, marker_id, x, y, heading, length, color):
    m = Marker()
    m.header.frame_id = 'map'
    m.ns, m.id = ns, marker_id
    m.type, m.action = Marker.ARROW, Marker.ADD
    m.pose = Pose(position=_plot_point(x, y, 0.05), orientation=_yaw_quat(heading))
    m.scale = Vector3(x=max(length, _MIN_ARROW_LENGTH), y=0.15, z=0.15)
    m.color = color
    return m


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
        self.status_overlay_pub = self.create_publisher(OverlayText, '/viz/status_overlay', 10)
        self.env_overlay_pub = self.create_publisher(OverlayText, '/viz/env_overlay', 10)

        scenario = self._fetch_scenario()
        waypoints = list(zip(scenario.waypoints_x, scenario.waypoints_y))

        self._path_markers = self._build_path_markers(waypoints)
        # Original (scenario-authored / latest-published) msg objects, keyed implicitly
        # by list order -- id (e.g. "obs_3") is what actually ties an obstacle/wall/
        # ellipse to its /map/predicted_paths entry (see _rebuild_obstacle_markers).
        # A stationary one (vx=vy=0.0) never gets a predicted-path entry at all, so it
        # just renders at its own x/y forever, same as before vx/vy existed.
        self._obstacles_msgs = list(scenario.obstacles)
        self._walls_msgs = list(scenario.walls)
        self._ellipses_msgs = list(scenario.ellipses)
        # Kept entirely SEPARATE from _ellipses_msgs/_ellipse_markers/_ellipses_xyabtheta
        # on purpose -- an obstacle ship must never feed the HUD's nearest-obstacle
        # readout (it isn't a real, avoidable obstacle -- see this feature's plan).
        self._obstacle_ship_specs = list(scenario.obstacle_ships)
        self._live_pos = {}  # id -> (x, y, psi_or_None), latest k=0 point from /map/predicted_paths
        self._obstacle_markers = []
        self._wall_markers = []
        self._ellipse_markers = []
        self._obstacle_ship_markers = []
        self._predicted_path_markers = []
        self._rebuild_obstacle_markers()  # builds the lists above from _obstacles_msgs et al.
        self._ship_marker = None
        self._trail_marker = None
        self._prediction_marker = None
        self._active_wp_marker = None
        self._current_marker = None
        self._ukf_current_marker = None  # UKF-predicted current arrow, alongside the actual one
        self._wave_marker = None
        self._last_vessel = None  # (x, y, psi), for current/wave arrows -- set by _on_mmg_state
        self._trail_points = collections.deque(maxlen=_TRAIL_MAX_POINTS)
        self._current_state_msg = None  # latest /env/current_state, for the top-right env_overlay
        self._wave_state_msg = None     # latest /env/wave_state, for the top-right env_overlay
        self._ukf_current_msg = None    # latest /ukf/estimated_current, for env_overlay's "| predicted" column
        self._ukf_state_msg = None      # latest /ukf/estimated_state, for status_overlay's x/y "| predicted" column

        # cached, used by the /viz/status_text telemetry block (mirrors visualizer.py's info_text)
        # -- self._obstacles_xyr/_walls_xyr/_ellipses_xyabtheta are (re)populated by
        # _rebuild_obstacle_markers() above, using LIVE positions when available.
        self._goal = waypoints[-1]
        self._active_waypoint = None

        # clear any markers left over from a previous run/session before publishing fresh ones
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        self.markers_pub.publish(MarkerArray(markers=[clear]))
        self._publish_all()

        self.create_subscription(VesselState, '/mmg/state', self._on_mmg_state, 10)
        self.create_subscription(PredictionHorizon, '/nmpc/prediction_horizon', self._on_prediction_horizon, 10)
        self.create_subscription(ObstacleArray, '/map/obstacles', self._on_obstacles, _LATCHED_QOS)
        self.create_subscription(WallObstacleArray, '/map/walls', self._on_walls, _LATCHED_QOS)
        self.create_subscription(EllipseArray, '/map/ellipses', self._on_ellipses, _LATCHED_QOS)
        self.create_subscription(PredictedPathArray, '/map/predicted_paths', self._on_predicted_paths, _REFERENCE_QOS)
        self.create_subscription(ActiveReference, '/map/active_reference', self._on_active_reference, _REFERENCE_QOS)
        self.create_subscription(SimStatus, '/map/sim_status', self._on_sim_status, _LATCHED_QOS)
        self.create_subscription(CurrentState, '/env/current_state', self._on_current_state, 10)
        self.create_subscription(WaveState, '/env/wave_state', self._on_wave_state, 10)
        self.create_subscription(CurrentState, '/ukf/estimated_current', self._on_ukf_current, 10)
        self.create_subscription(VesselState, '/ukf/estimated_state', self._on_ukf_state, 10)

        n_moving = sum(1 for o in self._obstacles_msgs if math.hypot(o.vx, o.vy) > _VELOCITY_EPS) + \
            sum(1 for w in self._walls_msgs if math.hypot(w.vx, w.vy) > _VELOCITY_EPS) + \
            sum(1 for e in self._ellipses_msgs if math.hypot(e.vx, e.vy) > _VELOCITY_EPS)
        self.get_logger().info(f'rviz_node up: {len(waypoints)} waypoints, {len(self._obstacle_markers)} obstacles, '
                                f'{len(self._wall_markers)} walls, {len(self._ellipse_markers)} ellipses '
                                f'({n_moving} moving); publishing MarkerArray on /viz/markers, telemetry on /viz/status_text')

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

    def _build_wall_marker(self, wall):
        # Capsule (padded line segment) obstacle -- see
        # research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md. RViz LINE_STRIP
        # doesn't round the caps the way the solver's exact capsule distance
        # does; this is a documented visualization-only simplification, it
        # doesn't affect the constraint math itself.
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id = 'walls', hash(wall.id) & 0x7FFFFFFF
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.pose = Pose(orientation=Quaternion(w=1.0))
        m.points = [_plot_point(wall.x0, wall.y0, 0.0), _plot_point(wall.x1, wall.y1, 0.0)]
        m.scale = Vector3(x=2.0 * float(wall.radius), y=0.0, z=0.0)
        m.color = _color(0.9, 0.2, 0.2, 0.5)
        return m

    def _build_ellipse_marker(self, ellipse):
        # Elliptical obstacle -- see research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md.
        # RViz CYLINDER supports independent x/y scale, so a/b map directly (unlike
        # the circle marker, which sets equal x/y); orientation reuses _yaw_quat
        # exactly as its own docstring anticipates for any world-frame angle.
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id = 'ellipses', hash(ellipse.id) & 0x7FFFFFFF
        m.type, m.action = Marker.CYLINDER, Marker.ADD
        m.pose = Pose(position=_plot_point(ellipse.x, ellipse.y, 0.0), orientation=_yaw_quat(ellipse.theta))
        m.scale = Vector3(x=2.0 * float(ellipse.a), y=2.0 * float(ellipse.b), z=0.5)
        m.color = _color(0.6, 0.3, 0.9, 0.5)
        return m

    def _build_predicted_path_marker(self, path_id, xs, ys):
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id = 'predicted_paths', hash(path_id) & 0x7FFFFFFF
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.scale = Vector3(x=0.2, y=0.0, z=0.0)
        m.color = _color(1.0, 0.85, 0.1, 0.85)  # amber, distinct from the wall/obstacle red
        m.points = [_plot_point(x, y) for x, y in zip(xs, ys)]
        return m

    def _live_xy(self, obj_id, default_x, default_y):
        pos = self._live_pos.get(obj_id)
        return (pos[0], pos[1]) if pos is not None else (default_x, default_y)

    def _live_xyp(self, obj_id, default_x, default_y, default_psi):
        pos = self._live_pos.get(obj_id)
        if pos is None:
            return default_x, default_y, default_psi
        x, y, psi = pos
        return x, y, (psi if psi is not None else default_psi)

    def _build_obstacle_ship_markers(self):
        """Obstacle ships (see this feature's plan): rendered with the same
        _build_ellipse_marker() helper as a real ellipse, but tracked in a
        SEPARATE list/method so they can never leak into
        _ellipse_markers/_ellipses_xyabtheta (the HUD's nearest-obstacle
        readout) -- an obstacle ship isn't a real, avoidable obstacle. Live
        (x, y, psi) comes from self._live_pos (populated by
        _on_predicted_paths's "obstacle_ship_0" entry); a spec with no live
        entry yet (obstacle_ship_node hasn't published /obstacle_ship/state
        yet) just renders at its scenario-authored initial pose.

        Nudged UP in z (0.5 -> 0.9) after the whole point of this feature is
        "let it pass through any other object" -- a moving ship WILL end up
        spatially overlapping a real obstacle/wall/ellipse (all rendered as
        flat, semi-transparent CYLINDERs at the SAME z=0.5), and two coplanar
        transparent surfaces at an identical height is a textbook GPU
        z-fighting setup: the renderer picks a winner per pixel per frame
        near-arbitrarily, which looks exactly like flickering/flashing as the
        ship moves through/near one. Separating the z bands (never touching,
        since real obstacles' own CYLINDER height is 0.5 centered at z=0.5,
        i.e. spanning z in [0.25, 0.75]) removes the coplanar overlap
        entirely -- a rendering-only change, doesn't touch collision/avoidance
        (there is none) or any position/orientation data."""
        markers = []
        for spec in self._obstacle_ship_specs:
            x, y, psi = self._live_xyp(spec.id, spec.x, spec.y, spec.psi0)
            e = Ellipse(id=spec.id, x=x, y=y, a=spec.a, b=spec.b, theta=psi, vx=0.0, vy=0.0)
            m = self._build_ellipse_marker(e)
            m.pose.position.z = 0.9
            markers.append(m)
        return markers

    def _rebuild_obstacle_markers(self):
        """Rebuilds obstacle/wall/ellipse markers (and the HUD's nearest-
        obstacle xyr/xyabtheta caches) from the latest scenario-authored
        msgs, substituting each one's LIVE position from self._live_pos
        (populated by _on_predicted_paths) when it has one -- i.e. any with
        nonzero vx/vy. A stationary one (vx=vy=0.0, the common case) has no
        entry in self._live_pos and renders at its own x/y unchanged, exactly
        as before vx/vy existed. Reuses _build_obstacle_marker/_build_wall_
        marker/_build_ellipse_marker UNCHANGED by passing translated copies
        of the msg objects -- for a wall, translated relative to its own
        midpoint (both endpoints move by the same delta, since a moving wall
        translates rigidly). Called whenever either the static obstacle/wall/
        ellipse topics OR /map/predicted_paths tick -- does NOT publish
        itself, callers call self._publish_all() after."""
        obstacle_markers, obstacles_xyr = [], []
        for o in self._obstacles_msgs:
            x, y = self._live_xy(o.id, o.x, o.y)
            o2 = Obstacle(id=o.id, x=x, y=y, radius=o.radius, vx=o.vx, vy=o.vy)
            obstacle_markers.append(self._build_obstacle_marker(o2))
            obstacles_xyr.append((x, y, o.radius))
        self._obstacle_markers = obstacle_markers
        self._obstacles_xyr = obstacles_xyr

        wall_markers, walls_xyr = [], []
        for w in self._walls_msgs:
            mx0, my0 = (w.x0 + w.x1) / 2.0, (w.y0 + w.y1) / 2.0
            lx, ly = self._live_xy(w.id, mx0, my0)
            dx, dy = lx - mx0, ly - my0
            w2 = WallObstacle(id=w.id, x0=w.x0 + dx, y0=w.y0 + dy, x1=w.x1 + dx, y1=w.y1 + dy,
                               radius=w.radius, vx=w.vx, vy=w.vy)
            wall_markers.append(self._build_wall_marker(w2))
            walls_xyr.append((w2.x0, w2.y0, w2.x1, w2.y1, w2.radius))
        self._wall_markers = wall_markers
        self._walls_xyr = walls_xyr

        ellipse_markers, ellipses_xyabtheta = [], []
        for e in self._ellipses_msgs:
            x, y = self._live_xy(e.id, e.x, e.y)
            e2 = Ellipse(id=e.id, x=x, y=y, a=e.a, b=e.b, theta=e.theta, vx=e.vx, vy=e.vy)
            ellipse_markers.append(self._build_ellipse_marker(e2))
            ellipses_xyabtheta.append((x, y, e.a, e.b, e.theta))
        self._ellipse_markers = ellipse_markers
        self._ellipses_xyabtheta = ellipses_xyabtheta

        self._obstacle_ship_markers = self._build_obstacle_ship_markers()

    # ---- live callbacks --------------------------------------------------
    def _on_mmg_state(self, msg: VesselState):
        self._last_vessel = (msg.x, msg.y, msg.psi)

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
        self.status_overlay_pub.publish(_overlay_text(
            self._build_status_text(msg), OverlayText.LEFT, OverlayText.BOTTOM,
            horizontal_distance=10, vertical_distance=10, width=340, height=260))
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
        self._obstacles_msgs = list(msg.obstacles)
        self._rebuild_obstacle_markers()
        self._publish_all()

    def _on_walls(self, msg: WallObstacleArray):
        self._walls_msgs = list(msg.walls)
        self._rebuild_obstacle_markers()
        self._publish_all()

    def _on_ellipses(self, msg: EllipseArray):
        self._ellipses_msgs = list(msg.ellipses)
        self._rebuild_obstacle_markers()
        self._publish_all()

    def _on_predicted_paths(self, msg: PredictedPathArray):
        # Ticks every /mmg/state step (see map_node.py's _publish_predicted_paths) --
        # only obstacles/walls/ellipses with nonzero velocity appear here at all
        # (map_node skips stationary ones entirely). Each path's first point (k=0)
        # is that obstacle's LIVE current position (a wall's own midpoint) -- see
        # nmpc/moving_obstacle.py. Rebuilding self._live_pos wholesale each tick is
        # correct here since the SET of moving ids never changes mid-run (velocities
        # are fixed at scenario-authoring time).
        self._live_pos = {p.id: (p.x[0], p.y[0], (p.psi[0] if p.psi else None)) for p in msg.paths if p.x}
        self._predicted_path_markers = [self._build_predicted_path_marker(p.id, p.x, p.y)
                                         for p in msg.paths if p.x]
        self._rebuild_obstacle_markers()
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

    def _on_current_state(self, msg: CurrentState):
        if not msg.enabled or self._last_vessel is None:
            self._current_marker = None
        else:
            x, y, _ = self._last_vessel
            self._current_marker = _arrow_marker(
                'current', 0, x, y, msg.heading, msg.speed * _CURRENT_ARROW_GAIN,
                _color(0.2, 0.8, 0.9, 0.9))
        self._current_state_msg = msg
        self._publish_env_overlay()
        self._publish_all()

    def _on_ukf_current(self, msg: CurrentState):
        # Same arrow convention as the actual-current one (_on_current_state),
        # a distinct color so both can be told apart at a glance -- drawn from
        # the same vessel position since they're both "current AT the ship,
        # now", just measured (msg.enabled) vs UKF-estimated.
        if self._last_vessel is None:
            self._ukf_current_marker = None
        else:
            x, y, _ = self._last_vessel
            self._ukf_current_marker = _arrow_marker(
                'ukf_current', 0, x, y, msg.heading, msg.speed * _CURRENT_ARROW_GAIN,
                _color(0.7, 0.3, 1.0, 0.85))
        self._ukf_current_msg = msg
        self._publish_env_overlay()
        self._publish_all()

    def _on_ukf_state(self, msg: VesselState):
        self._ukf_state_msg = msg

    def _on_wave_state(self, msg: WaveState):
        if not msg.enabled or self._last_vessel is None:
            self._wave_marker = None
        else:
            x, y, psi = self._last_vessel
            # body-frame (fx, fy) -> earth frame, for display purposes only
            # (same rotation casadi_mmg.py's R_mat applies to velocity).
            earth_fx = msg.fx * math.cos(psi) - msg.fy * math.sin(psi)
            earth_fy = msg.fx * math.sin(psi) + msg.fy * math.cos(psi)
            heading = math.atan2(earth_fy, earth_fx)
            magnitude = math.hypot(earth_fx, earth_fy)
            self._wave_marker = _arrow_marker(
                'wave', 0, x, y, heading, magnitude * _WAVE_ARROW_GAIN, _color(1.0, 0.4, 0.7, 0.9))
        self._wave_state_msg = msg
        self._publish_env_overlay()
        self._publish_all()

    def _publish_env_overlay(self):
        # top-right numeric readout of everything CurrentState/WaveState carry --
        # companion to the bottom-left status_overlay, kept plain-text/no graphics
        # since the actual compass/wave-scatter icons now live in hud_node.py's window.
        c, w = self._current_state_msg, self._wave_state_msg
        if c is None or w is None:
            return

        if c.enabled:
            # "| <predicted>" appended straight after each actual value, from
            # /ukf/estimated_current -- same lines, no new rows, so actual vs
            # UKF-estimated current can be compared at a glance without a
            # second overlay. Omitted (falls back to actual-only) until the
            # first UKF message arrives.
            u = self._ukf_current_msg
            if u is not None:
                current_text = (
                    f"Speed   : {c.speed:7.4f} | {u.speed:7.4f} m/s\n"
                    f"Heading : {math.degrees(c.heading):7.2f} | {math.degrees(u.heading):7.2f} deg\n"
                    f"Vx, Vy  : {c.vx:7.4f}, {c.vy:7.4f} | {u.vx:7.4f}, {u.vy:7.4f} m/s"
                )
            else:
                current_text = (
                    f"Speed   : {c.speed:7.4f} m/s\n"
                    f"Heading : {math.degrees(c.heading):7.2f} deg\n"
                    f"Vx, Vy  : {c.vx:7.4f}, {c.vy:7.4f} m/s"
                )
        else:
            current_text = "off"

        if w.enabled:
            wave_text = (
                f"Fx      : {w.fx: .3e} N\n"
                f"Fy      : {w.fy: .3e} N\n"
                f"Fn      : {w.fn: .3e} N·m\n"
                f"Hs      : {w.hs:7.4f} m\n"
                f"Tp      : {w.tp:7.2f} s"
            )
        else:
            wave_text = "off"

        text = f"=== CURRENT ===\n{current_text}\n\n=== WAVE ===\n{wave_text}"
        self.env_overlay_pub.publish(_overlay_text(
            text, OverlayText.RIGHT, OverlayText.TOP,
            horizontal_distance=10, vertical_distance=10, width=380, height=230))

    def _build_status_text(self, msg: VesselState) -> str:
        # mirrors mpc_visualization/visualizer.py's info_text block (SHIP STATUS /
        # CONTROLS / TARGETS), just as a plain string instead of a matplotlib overlay.
        ship_pos = (msg.x, msg.y)
        goal_dist = math.hypot(ship_pos[0] - self._goal[0], ship_pos[1] - self._goal[1])
        target = self._active_waypoint if self._active_waypoint is not None else self._goal
        active_wp_dist = math.hypot(ship_pos[0] - target[0], ship_pos[1] - target[1])
        nearest_obstacle_dist = min(
            [math.hypot(ship_pos[0] - ox, ship_pos[1] - oy) - orad for ox, oy, orad in self._obstacles_xyr] +
            [_point_to_segment_dist(ship_pos[0], ship_pos[1], x0, y0, x1, y1) - wrad
             for x0, y0, x1, y1, wrad in self._walls_xyr] +
            [_point_ellipse_distance(ship_pos[0], ship_pos[1], xc, yc, a, b, theta)
             for xc, yc, a, b, theta in self._ellipses_xyabtheta],
            default=float('inf'))

        # "| <predicted>" appended straight after X/Y's actual value, from
        # /ukf/estimated_state -- same lines, no new columns, matching the
        # env_overlay's actual-vs-UKF-current convention above. Omitted
        # (falls back to actual-only) until the first UKF message arrives.
        s = self._ukf_state_msg
        x_line = f"X         : {msg.x:6.2f} | {s.x:6.2f} m\n" if s is not None else f"X         : {msg.x:6.2f} m\n"
        y_line = f"Y         : {msg.y:6.2f} | {s.y:6.2f} m\n" if s is not None else f"Y         : {msg.y:6.2f} m\n"

        return (
            f"=== SHIP STATUS ===\n"
            f"{x_line}"
            f"{y_line}"
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
        markers = (list(self._path_markers) + list(self._obstacle_markers) + list(self._wall_markers) +
                   list(self._ellipse_markers) + list(self._obstacle_ship_markers) +
                   list(self._predicted_path_markers))
        for m in (self._active_wp_marker, self._trail_marker, self._prediction_marker, self._ship_marker,
                  self._current_marker, self._ukf_current_marker, self._wave_marker):
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
