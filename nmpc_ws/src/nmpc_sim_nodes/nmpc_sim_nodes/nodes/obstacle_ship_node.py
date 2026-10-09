"""obstacle_ship_node: true-physics plant for a second, independently
keyboard-controlled ship, used as a moving obstacle the NMPC is currently
inert to (see this feature's plan / nmpc/README.md's "Planned: Nomoto-driven
moving obstacles" note). Uses the EXACT SAME casadi_mmg_solver MMG dynamics as
mmg_node.py's ownship -- not a separate/simplified model -- so it moves like a
real ship, with its own current/wave disturbance instances built from the
SAME sim_params.yaml mmg_node section (same seeds) as the ownship's, so both
experience the same sea state, each correctly resolved against its own
heading (see env_model/config.py's load_current_config()/load_wave_config()).

Deliberately does NOT: call /nmpc/solve, call /ukf/estimate, run any sensor
noise model, or appear on /map/obstacles|walls|ellipses (see map_node.py) --
none of that is this node's job. Its actuator command comes from whatever
publishes /obstacle_ship/cmd (obstacle_ship_teleop_node.py today; real
hardware or an autonomous controller later, with zero change needed here).

Never ticks (stays idle) if no obstacle ship was enabled in the loaded
scenario -- /map/obstacle_ship_initial_state simply never arrives in that
case, so it's always safe to include this node in bringup.launch.py.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.nomoto_obstacle import step as nomoto_step  # noqa: E402
from nmpc_interfaces.msg import ControlCommand, SimStatus, VesselState  # noqa: E402

_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class ObstacleShipNode(Node):
    def __init__(self):
        super().__init__('obstacle_ship_node')

        self.declare_parameter('dt', 0.1)
        self.declare_parameter('obstacle_ship_nomoto_k', 0.15)
        self.declare_parameter('obstacle_ship_nomoto_t', 3.0)

        self.dt = float(self.get_parameter('dt').value)
        self.K = float(self.get_parameter('obstacle_ship_nomoto_k').value)
        self.T = float(self.get_parameter('obstacle_ship_nomoto_t').value)

        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.u = 0.0
        self.r = 0.0
        self.delta = 0.0
        self._cmd_delta = 0.0
        self._cmd_u = 0.0
        self._have_initial_state = False
        self._running = False

        self.state_pub = self.create_publisher(VesselState, '/obstacle_ship/state', 10)

        self.create_subscription(SimStatus, '/map/sim_status', self._on_sim_status, _LATCHED_QOS)
        self.create_subscription(VesselState, '/map/obstacle_ship_initial_state', self._on_initial_state,
                                  _LATCHED_QOS)
        self.create_subscription(ControlCommand, '/obstacle_ship/cmd', self._on_cmd, 10)

        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f'obstacle_ship_node up, dt={self.dt}s, Nomoto K={self.K}, T={self.T}s (no env disturbance). '
            'Waiting for /map/obstacle_ship_initial_state...')

    # ------------------------------------------------------------------
    def _on_initial_state(self, msg: VesselState):
        if self._have_initial_state:
            return
        self.x = float(msg.x)
        self.y = float(msg.y)
        self.psi = float(msg.psi)
        self.u = float(msg.u)
        self.r = float(msg.r)
        self.delta = float(msg.delta)
        self._cmd_delta = self.delta
        self._cmd_u = self.u
        self._have_initial_state = True
        self.get_logger().info(
            f'seeded obstacle ship initial state: pos=({self.x:.2f}, {self.y:.2f}), '
            f'psi={self.psi:.3f} rad, u={self.u:.3f} m/s')

    def _on_sim_status(self, msg: SimStatus):
        self._running = (msg.status == SimStatus.RUNNING)

    def _on_cmd(self, msg: ControlCommand):
        if not self._have_initial_state:
            self.get_logger().warn(
                'Received /obstacle_ship/cmd, but no obstacle has velocity in this scenario! '
                'Give an obstacle a velocity in scenario_editor to make it active.',
                throttle_duration_sec=5.0)
            return
        self._cmd_delta = float(msg.delta)
        if msg.n >= 0.0:
            self._cmd_u = float(msg.n)

    def _state_msg(self) -> VesselState:
        msg = VesselState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.u = float(self.u)
        msg.v = 0.0
        msg.r = float(self.r)
        msg.x = float(self.x)
        msg.y = float(self.y)
        msg.psi = float(self.psi)
        msg.delta = float(self.delta)
        msg.n = float(self.u)
        return msg

    def _tick(self):
        if not self._have_initial_state or not self._running:
            return

        self.delta = self._cmd_delta
        self.u = self._cmd_u

        self.x, self.y, self.psi, self.r = nomoto_step(
            self.x, self.y, self.psi, self.u, self.r, self.delta,
            self.K, self.T, self.dt)

        self.state_pub.publish(self._state_msg())


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleShipNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
