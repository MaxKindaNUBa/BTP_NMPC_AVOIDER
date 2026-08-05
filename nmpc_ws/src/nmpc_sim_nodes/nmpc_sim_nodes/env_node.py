"""env_node: implements WAVE_CURRENT_DISTURBANCE_PLAN.md. Sits in mmg_node's
tick loop, called once per tick after /nmpc/solve and before integration.
Serves a synchronous /env/disturbance service (mmg_node is the client) and
also republishes the same result on /env/current_state and /env/wave_state --
passive topics for rviz_node/viz_node/logging -- exactly the same dual
pattern sensor_node.py already uses for /sensor/measure + /sensor/measured_state.

Toggles: current_enabled and wave_enabled are independent. Each false means
that component's contribution is identically zero (its model simply isn't
constructed, response fields hardcoded to zero) -- same "disabled means no
partial mode" philosophy sensor_node already uses for its own enabled flag.
"""
import math

import rclpy
from rclpy.node import Node

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc_interfaces.msg import CurrentState, WaveState  # noqa: E402
from nmpc_interfaces.srv import GetEnvDisturbance  # noqa: E402

from env_model.config import CurrentConfig, WaveConfig  # noqa: E402
from env_model.current_model import CurrentModel  # noqa: E402
from env_model.wave_model import WaveModel  # noqa: E402


class EnvNode(Node):
    def __init__(self):
        super().__init__('env_node')

        self.declare_parameter('dt', 0.1)

        self.declare_parameter('current_enabled', False)
        self.declare_parameter('current_mean_speed', 0.0)
        self.declare_parameter('current_mean_heading', 0.0)
        self.declare_parameter('current_time_constant', 600.0)
        self.declare_parameter('current_sigma', 0.01)
        self.declare_parameter('current_seed', 7)

        self.declare_parameter('wave_enabled', False)
        self.declare_parameter('wave_hs', 0.05)
        self.declare_parameter('wave_tp', 1.2)
        self.declare_parameter('wave_gamma', 3.3)
        self.declare_parameter('wave_mean_heading', 0.0)
        self.declare_parameter('wave_num_components', 30)
        self.declare_parameter('wave_seed', 11)

        self.dt = float(self.get_parameter('dt').value)

        self.current_enabled = bool(self.get_parameter('current_enabled').value)
        self.current_model = None
        if self.current_enabled:
            speed = float(self.get_parameter('current_mean_speed').value)
            heading = float(self.get_parameter('current_mean_heading').value)
            current_config = CurrentConfig(
                mean_vx=speed * math.cos(heading),
                mean_vy=speed * math.sin(heading),
                time_constant_s=float(self.get_parameter('current_time_constant').value),
                sigma=float(self.get_parameter('current_sigma').value),
                seed=int(self.get_parameter('current_seed').value),
            )
            self.current_model = CurrentModel(current_config)

        self.wave_enabled = bool(self.get_parameter('wave_enabled').value)
        self.wave_model = None
        if self.wave_enabled:
            wave_config = WaveConfig(
                hs=float(self.get_parameter('wave_hs').value),
                tp=float(self.get_parameter('wave_tp').value),
                gamma=float(self.get_parameter('wave_gamma').value),
                mean_heading=float(self.get_parameter('wave_mean_heading').value),
                num_components=int(self.get_parameter('wave_num_components').value),
                seed=int(self.get_parameter('wave_seed').value),
            )
            self.wave_model = WaveModel(wave_config, self.dt)

        self.current_pub = self.create_publisher(CurrentState, '/env/current_state', 10)
        self.wave_pub = self.create_publisher(WaveState, '/env/wave_state', 10)
        self.create_service(GetEnvDisturbance, '/env/disturbance', self._on_disturbance)

        self.get_logger().info(
            f'env_node up: current_enabled={self.current_enabled}, wave_enabled={self.wave_enabled}, dt={self.dt}')

    def _on_disturbance(self, request: GetEnvDisturbance.Request,
                         response: GetEnvDisturbance.Response) -> GetEnvDisturbance.Response:
        stamp = self.get_clock().now().to_msg()
        frame_id = request.state.header.frame_id or 'map'

        current = CurrentState()
        current.header.stamp = stamp
        current.header.frame_id = frame_id
        current.enabled = self.current_enabled
        if self.current_enabled:
            vx, vy = self.current_model.step(self.dt)
            current.vx, current.vy = float(vx), float(vy)
            current.speed = float(math.hypot(vx, vy))
            current.heading = float(math.atan2(vy, vx))
        else:
            current.vx = current.vy = current.speed = current.heading = 0.0

        wave = WaveState()
        wave.header.stamp = stamp
        wave.header.frame_id = frame_id
        wave.enabled = self.wave_enabled
        if self.wave_enabled:
            fx, fy, fn = self.wave_model.force(request.state.psi)
            wave.hs = float(self.wave_model.config.hs)
            wave.tp = float(self.wave_model.config.tp)
            wave.mean_heading = float(self.wave_model.config.mean_heading)
            wave.fx, wave.fy, wave.fn = float(fx), float(fy), float(fn)
        else:
            wave.hs = wave.tp = wave.mean_heading = wave.fx = wave.fy = wave.fn = 0.0

        self.current_pub.publish(current)
        self.wave_pub.publish(wave)

        response.current = current
        response.wave = wave
        return response


def main(args=None):
    rclpy.init(args=args)
    node = EnvNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
