"""obstacle_ship_teleop_node: raw WASD terminal control for obstacle_ship_node
(see this feature's plan / nmpc/README.md's "Planned: Nomoto-driven moving
obstacles" note). Deliberately the ONLY place keyboard-input handling lives --
obstacle_ship_node itself knows nothing about keys, only /obstacle_ship/cmd
(ControlCommand) -- so this node can be deleted outright once hardware trials
replace manual driving, with zero change needed anywhere else.

Controls (run this in its own terminal, with focus on that terminal):
  a / d  -- nudge rudder angle left/right (rate-limited by DELTA_DOT_MAX,
            clamped to DELTA_MIN/MAX)
  w / s  -- nudge propeller speed up/down (rate-limited by RPS_DOT_MAX,
            clamped to RPS_MIN/MAX -- s can go negative/astern)
  no key -- HOLD the last commanded rudder/rps (no auto-centering)
  q      -- quit

Raw terminal input (termios/tty cbreak mode + non-blocking select()) rather
than any ROS/robotics keyboard-teleop package: no such dependency exists
anywhere in this repo already, and the standard teleop_twist_keyboard's Twist
output doesn't map onto rudder-angle/rps anyway. A terminal has no real
key-release event, only auto-repeat while a key is physically held -- which is
exactly what "hold last value when nothing is read this tick" already gives
us, with no extra bookkeeping needed.
"""
import math
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.params import DEFAULT_CONFIG  # noqa: E402
from nmpc_interfaces.msg import ControlCommand, VesselState  # noqa: E402


class ObstacleShipTeleopNode(Node):
    def __init__(self):
        super().__init__('obstacle_ship_teleop_node')

        self.declare_parameter('dt', 0.1)
        self.dt = float(self.get_parameter('dt').value)

        self.delta = 0.0
        self.u = 0.0
        self._synced_initial_speed = False

        self._old_term_settings = None
        if sys.stdin.isatty():
            self._old_term_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            # cbreak only turns off line-buffering -- ECHO stays on by default, so
            # every keystroke would otherwise print itself and mangle the live
            # rudder/speed readout below. Turn it off explicitly (restored in
            # destroy_node() along with everything else via the saved settings).
            attrs = termios.tcgetattr(sys.stdin)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, attrs)
        else:
            self.get_logger().warn('stdin is not a TTY -- WASD input will never be read; '
                                    'run this node in an interactive terminal with `ros2 run`')

        self.cmd_pub = self.create_publisher(ControlCommand, '/obstacle_ship/cmd', 10)
        self._obstacle_ship_active = False
        self.create_subscription(VesselState, '/obstacle_ship/state', self._on_ship_state, 10)
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            'obstacle_ship_teleop_node up -- a/d: rudder, w/s: speed (m/s), q: quit '
            '(no key = hold last value, no auto-centering)')

    def _on_ship_state(self, msg: VesselState):
        self._obstacle_ship_active = True
        if not self._synced_initial_speed:
            self.u = float(msg.u)
            self.delta = float(msg.delta)
            self._synced_initial_speed = True

    # ------------------------------------------------------------------
    def _read_keys(self) -> set:
        keys = set()
        if self._old_term_settings is None:
            return keys
        while select.select([sys.stdin], [], [], 0.0)[0]:
            keys.add(sys.stdin.read(1).lower())
        return keys

    def _tick(self):
        keys = self._read_keys()
        cfg = DEFAULT_CONFIG
        SPEED_RATE = 1.0  # m/s^2 change rate when w/s is held
        SPEED_MAX = 5.0   # m/s
        SPEED_MIN = 0.0   # m/s

        if 'q' in keys:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self.get_logger().info('q pressed -- shutting down teleop')
            rclpy.shutdown()  # causes main()'s spin() to return; destroy_node() runs once, in its finally
            return

        if 'a' in keys:
            self.delta = max(self.delta - cfg.DELTA_DOT_MAX * self.dt, cfg.DELTA_MIN)
        if 'd' in keys:
            self.delta = min(self.delta + cfg.DELTA_DOT_MAX * self.dt, cfg.DELTA_MAX)
        if 'w' in keys:
            self.u = min(self.u + SPEED_RATE * self.dt, SPEED_MAX)
        if 's' in keys:
            self.u = max(self.u - SPEED_RATE * self.dt, SPEED_MIN)
        # no key held -> delta/u simply unchanged (hold-last-value control feel)

        msg = ControlCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.delta = float(self.delta)
        msg.n = float(self.u)
        self.cmd_pub.publish(msg)

        # Live single-line readout, overwritten in place each tick (\r, no newline)
        keys_label = ''.join(sorted(k for k in keys if k in ('a', 'd', 'w', 's'))) or '-'
        status_label = "ACTIVE" if self._obstacle_ship_active else "WAITING FOR MOVING OBSTACLE..."
        sys.stdout.write(
            f"\rrudder: {math.degrees(self.delta):+6.1f} deg (lim ±{math.degrees(cfg.DELTA_MAX):.0f})  "
            f"speed: {self.u:5.2f} m/s (lim [0.0, {SPEED_MAX:.1f}])  keys: {keys_label:4s}  [{status_label}]")
        sys.stdout.flush()

    def destroy_node(self):
        if self._old_term_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term_settings)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleShipTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
