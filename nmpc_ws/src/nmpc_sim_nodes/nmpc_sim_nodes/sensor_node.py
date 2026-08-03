"""sensor_node: implements SENSOR_NOISE_MODEL.md. Sits strictly between the
true plant state and nmpc_node's solve() call, exactly as anticipated by
ROS2_CONVERSION_PLAN.md section 8 ("a filter on the /mmg/state -> /nmpc/solve
path"). Serves a synchronous /sensor/measure service (nmpc_node is the
client, called once per tick before solver.solve()) and also republishes the
same result on /sensor/measured_state, a passive topic for rviz/diagnostics.

Toggle: the `enabled` parameter is the single on/off switch. When false this
node is a pure pass-through -- the response is identical to the request,
i.e. behavior is exactly what it was before this node existed. There is no
partial/"exclude some terms" mode; disabled means no noise at all.
"""
import rclpy
from rclpy.node import Node

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from nmpc_interfaces.msg import VesselState  # noqa: E402
from nmpc_interfaces.srv import MeasureState  # noqa: E402

from sensor_model.config import PRESETS  # noqa: E402
from sensor_model.sensor_model import SensorModel  # noqa: E402


def _vessel_state_to_tuple(msg: VesselState):
    return [msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi], float(msg.delta), float(msg.n)


class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')

        self.declare_parameter('enabled', False)
        self.declare_parameter('preset', 'light')
        self.declare_parameter('seed', 42)
        self.declare_parameter('dt', 0.1)

        self.enabled = bool(self.get_parameter('enabled').value)
        preset_name = str(self.get_parameter('preset').value)
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown sensor_model preset {preset_name!r}; expected one of {list(PRESETS)}")
        seed = int(self.get_parameter('seed').value)
        dt = float(self.get_parameter('dt').value)

        self.model = SensorModel(PRESETS[preset_name], dt, seed)

        self.measured_pub = self.create_publisher(VesselState, '/sensor/measured_state', 10)
        self.create_service(MeasureState, '/sensor/measure', self._on_measure)

        self.get_logger().info(
            f'sensor_node up: enabled={self.enabled}, preset={preset_name!r}, seed={seed}, dt={dt}')

    def _on_measure(self, request: MeasureState.Request, response: MeasureState.Response) -> MeasureState.Response:
        true_msg = request.true_state

        if not self.enabled:
            response.measured_state = true_msg
        else:
            mmg_state, delta, n = _vessel_state_to_tuple(true_msg)
            mmg_state_meas, delta_meas, n_meas = self.model.measure(mmg_state, delta, n)

            meas = VesselState()
            meas.header.stamp = self.get_clock().now().to_msg()
            meas.header.frame_id = true_msg.header.frame_id
            meas.u, meas.v, meas.r, meas.x, meas.y, meas.psi = [float(v) for v in mmg_state_meas]
            meas.delta = float(delta_meas)
            meas.n = float(n_meas)
            response.measured_state = meas

        self.measured_pub.publish(response.measured_state)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SensorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
