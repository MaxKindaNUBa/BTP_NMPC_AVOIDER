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

import casadi as ca  # noqa: E402
from casadi_mmg_solver.casadi_mmg import make_casadi_integrator  # noqa: E402

from nmpc_interfaces.msg import ControlCommand, SimStatus, VesselState  # noqa: E402

from env_model.config import load_current_config, load_wave_config  # noqa: E402
from env_model.current_model import CurrentModel  # noqa: E402
from env_model.wave_model import WaveModel  # noqa: E402

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
        self.dt = float(self.get_parameter('dt').value)

        self.plant_step = make_casadi_integrator(self.dt, method='rk4', sym_type=ca.SX, with_env=True)

        # Same sim_params.yaml mmg_node.current_*/wave_* section (and seeds) the
        # ownship's own mmg_node.py reads -- own model INSTANCES (not shared
        # objects, not consumed via /env/current_state|wave_state, which are
        # already resolved for the ownship's own heading/state) so this ship's
        # disturbance is correctly evaluated at ITS OWN state while still being
        # "the same sea state" by construction (same config, same seed).
        sim_params = _pkg_paths.load_sim_params()['mmg_node']['ros__parameters']
        self._current_enabled = bool(sim_params['current_enabled'])
        self._wave_enabled = bool(sim_params['wave_enabled'])
        self.current_model = CurrentModel(load_current_config()) if self._current_enabled else None
        self.wave_model = WaveModel(load_wave_config(), self.dt) if self._wave_enabled else None

        self.mmg_state = None   # np.ndarray[6] = [u, v, r, x, y, psi], None until seeded
        self.delta = 0.0
        self.n = 0.0
        self._cmd_delta = 0.0
        self._cmd_n = 0.0
        self._have_initial_state = False
        self._running = False

        self.state_pub = self.create_publisher(VesselState, '/obstacle_ship/state', 10)

        self.create_subscription(SimStatus, '/map/sim_status', self._on_sim_status, _LATCHED_QOS)
        self.create_subscription(VesselState, '/map/obstacle_ship_initial_state', self._on_initial_state,
                                  _LATCHED_QOS)
        self.create_subscription(ControlCommand, '/obstacle_ship/cmd', self._on_cmd, 10)

        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f'obstacle_ship_node up, dt={self.dt}s, current_enabled={self._current_enabled}, '
            f'wave_enabled={self._wave_enabled}, waiting for /map/obstacle_ship_initial_state '
            f'(no-op if this scenario has no enabled obstacle ship)...')

    # ------------------------------------------------------------------
    def _on_initial_state(self, msg: VesselState):
        if self._have_initial_state:
            return
        self.mmg_state = np.array([msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi], dtype=float)
        self.delta, self.n = float(msg.delta), float(msg.n)
        self._cmd_delta, self._cmd_n = self.delta, self.n
        self._have_initial_state = True
        self.get_logger().info(f'seeded obstacle ship initial state: {self.mmg_state.tolist()}')

    def _on_sim_status(self, msg: SimStatus):
        self._running = (msg.status == SimStatus.RUNNING)

    def _on_cmd(self, msg: ControlCommand):
        # Absolute targets, applied verbatim -- same "no clamping at the plant"
        # contract mmg_node.py has for the NMPC's own response; clamping is the
        # commander's job (obstacle_ship_teleop_node.py today).
        self._cmd_delta, self._cmd_n = float(msg.delta), float(msg.n)

    def _state_msg(self) -> VesselState:
        msg = VesselState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi = [float(v) for v in self.mmg_state]
        msg.delta = float(self.delta)
        msg.n = float(self.n)
        return msg

    def _tick(self):
        if not self._have_initial_state or not self._running:
            return

        self.delta, self.n = self._cmd_delta, self._cmd_n

        if self.current_model is not None:
            vx, vy = self.current_model.step(self.dt)
        else:
            vx, vy = 0.0, 0.0
        if self.wave_model is not None:
            fx, fy, fn = self.wave_model.force(float(self.mmg_state[5]))
        else:
            fx, fy, fn = 0.0, 0.0, 0.0

        next_state, _ = self.plant_step(ca.DM(self.mmg_state), ca.DM([self.delta, self.n]),
                                         ca.DM([vx, vy]), ca.DM([fx, fy, fn]))
        self.mmg_state = np.array(next_state).flatten()

        self.state_pub.publish(self._state_msg())


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleShipNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
