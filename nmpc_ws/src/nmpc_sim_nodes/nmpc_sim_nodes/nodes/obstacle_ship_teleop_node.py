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
from nmpc_interfaces.msg import ControlCommand  # noqa: E402


class ObstacleShipTeleopNode(Node):
    def __init__(self):
        super().__init__('obstacle_ship_teleop_node')

        self.declare_parameter('dt', 0.1)
        self.dt = float(self.get_parameter('dt').value)

        self.delta = 0.0
        # NOT 0.0 -- matches map_node's own obstacle-ship initial-state seed
        # (N_TRIM, mirroring the ownship's own convention): idling at n=0
        # with u=v=r=0 sits exactly on the U->0 MMG singularity documented in
        # nmpc/README.md item 7, which showed up as violent, fast yaw
        # oscillation once wave forcing was enabled (nothing here has the
        # NMPC's own solver-side U_REF_MIN floor to protect against it). If
        # this node starts before map_node's seed message arrives, its own
        # first published command would otherwise immediately zero it back
        # out the moment teleop comes up.
        self.n = DEFAULT_CONFIG.N_TRIM

        self._old_term_settings = None
        if sys.stdin.isatty():
            self._old_term_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            # cbreak only turns off line-buffering -- ECHO stays on by default, so
            # every keystroke would otherwise print itself and mangle the live
            # rudder/rps readout below. Turn it off explicitly (restored in
            # destroy_node() along with everything else via the saved settings).
            attrs = termios.tcgetattr(sys.stdin)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, attrs)
        else:
            self.get_logger().warn('stdin is not a TTY -- WASD input will never be read; '
                                    'run this node in an interactive terminal with `ros2 run`')

        self.cmd_pub = self.create_publisher(ControlCommand, '/obstacle_ship/cmd', 10)
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            'obstacle_ship_teleop_node up -- a/d: rudder, w/s: propeller, q: quit '
            '(no key = hold last value, no auto-centering)')

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
            self.n = min(self.n + cfg.RPS_DOT_MAX * self.dt, cfg.RPS_MAX)
        if 's' in keys:
            self.n = max(self.n - cfg.RPS_DOT_MAX * self.dt, cfg.RPS_MIN)
        # no key held -> delta/n simply unchanged (hold-last-value control feel)

        msg = ControlCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.delta = float(self.delta)
        msg.n = float(self.n)
        self.cmd_pub.publish(msg)

        # Live single-line readout, overwritten in place each tick (\r, no
        # newline) -- this is what's actually being commanded on
        # /obstacle_ship/cmd right now, for sanity-checking "control feels off"
        # against the printed numbers rather than guessing from feel alone.
        keys_label = ''.join(sorted(k for k in keys if k in ('a', 'd', 'w', 's'))) or '-'
        sys.stdout.write(
            f"\rrudder: {math.degrees(self.delta):+6.1f} deg (lim ±{math.degrees(cfg.DELTA_MAX):.0f})  "
            f"rps: {self.n:+6.2f} (lim ±{cfg.RPS_MAX:.1f})  keys: {keys_label:4s}")
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
