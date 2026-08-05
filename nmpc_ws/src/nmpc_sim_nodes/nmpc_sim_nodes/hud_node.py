"""hud_node: standalone matplotlib companion window showing the current compass,
a wave-force scatter, and the NMPC control-horizon graph -- the same data
rviz_node.py used to draw as a MarkerArray HUD directly on the 3D view. Broken
out into its own window so it doesn't draw on top of the map; run it alongside
rviz_node.py/rviz2 (see launch/rviz_hud.launch.py) or viz_node.py's own dashboard.
"""
import threading

import numpy as np
import rclpy
from rclpy.node import Node

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc.config import IDX_DELTA, IDX_N  # noqa: E402

from mpc_bridge import CurrentReading, MPCBridge, WaveReading  # noqa: E402
from hud_visualizer import HUDVisualizer  # noqa: E402

from nmpc_interfaces.msg import CurrentState, PredictionHorizon, WaveState  # noqa: E402


class HUDNode(Node):
    def __init__(self):
        super().__init__('hud_node')

        self.bridge = MPCBridge()
        self.visualizer = HUDVisualizer(self.bridge)

        self.create_subscription(CurrentState, '/env/current_state', self._on_current_state, 10)
        self.create_subscription(WaveState, '/env/wave_state', self._on_wave_state, 10)
        self.create_subscription(PredictionHorizon, '/nmpc/prediction_horizon', self._on_prediction_horizon, 10)

        self.get_logger().info('hud_node up: current compass + wave-force scatter + control-horizon companion window')

    def _on_current_state(self, msg: CurrentState):
        self.bridge.update_current(CurrentReading(
            vx=msg.vx, vy=msg.vy, speed=msg.speed, heading=msg.heading, enabled=msg.enabled))

    def _on_wave_state(self, msg: WaveState):
        # raw body-frame (fx, fy), no earth-frame rotation -- the wave widget is a
        # force-space scatter (not a map-aligned compass), so it just needs WaveState's
        # own components as published, unlike rviz_node.py's ship-anchored wave arrow.
        self.bridge.update_wave(WaveReading(fx=msg.fx, fy=msg.fy, fn=msg.fn, enabled=msg.enabled))

    def _on_prediction_horizon(self, msg: PredictionHorizon):
        n_states, horizon_len = msg.n_states, msg.horizon_len
        xi_traj = np.array(msg.xi_traj, dtype=float).reshape(n_states, horizon_len + 1, order='C')
        # VALUE columns [delta, n] (not the u_opt rate control), same as viz_node.py.
        control_horizon = xi_traj[[IDX_DELTA, IDX_N], :-1].T
        self.bridge.update_control_horizon(control_horizon)


def _spin_in_background(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass  # normal on rclpy.shutdown() racing this daemon thread


def main(args=None):
    rclpy.init(args=args)
    node = HUDNode()

    spin_thread = threading.Thread(target=_spin_in_background, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # blocks in THIS (main) thread until the window is closed; matplotlib
        # requires the main thread, same constraint viz_node.py has.
        node.visualizer.start()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
