"""mmg_node: owns the plant integrator and the master 1/dt clock (section 2.7 /
2.3 of ROS2_CONVERSION_PLAN.md). Each tick: call /nmpc/solve and block, apply
the returned (delta, n), integrate one RK4 step, publish the new state.
"""
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from . import _pkg_paths

_pkg_paths.ensure_on_path()

import casadi as ca  # noqa: E402
from casadi_mmg_solver.casadi_mmg import make_casadi_integrator  # noqa: E402

from nmpc_interfaces.msg import SimStatus, VesselState  # noqa: E402
from nmpc_interfaces.srv import GetEnvDisturbance, ResetSim, SolveNMPC  # noqa: E402

_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)


def _vessel_state_to_tuple(msg: VesselState):
    return np.array([msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi], dtype=float), float(msg.delta), float(msg.n)


class MmgNode(Node):
    def __init__(self):
        super().__init__('mmg_node')

        self.declare_parameter('dt', 0.1)
        self.dt = float(self.get_parameter('dt').value)

        self.plant_step = make_casadi_integrator(self.dt, method='rk4', sym_type=ca.SX, with_env=True)

        self.mmg_state = None          # np.ndarray[6], seeded from /map/initial_state
        self.delta = None
        self.n = None
        self._default_initial_state = None  # cached for /mmg/reset use_default=True
        self._have_initial_state = False
        self._running = False          # gated by /map/sim_status; no ticking until RUNNING seen

        # Two distinct callback groups so the timer thread's busy-wait for the
        # /nmpc/solve response doesn't deadlock against the response callback
        # itself -- rclpy serializes callbacks *within* one group, so both
        # must live in different groups AND the executor must be multi-threaded.
        self._timer_cbg = MutuallyExclusiveCallbackGroup()
        self._client_cbg = ReentrantCallbackGroup()
        self._misc_cbg = ReentrantCallbackGroup()

        self.state_pub = self.create_publisher(VesselState, '/mmg/state', 10)

        self.solve_client = self.create_client(SolveNMPC, '/nmpc/solve', callback_group=self._client_cbg)
        self.env_client = self.create_client(GetEnvDisturbance, '/env/disturbance', callback_group=self._client_cbg)

        # transient-local: mmg_node's tick loop is gated on seeing RUNNING at
        # least once, so a late subscription must still get map_node's last
        # published status instead of only future ones (see map_node.py, which
        # publishes this latched for the same reason).
        self.create_subscription(SimStatus, '/map/sim_status', self._on_sim_status, _LATCHED_QOS,
                                  callback_group=self._misc_cbg)
        self.create_subscription(VesselState, '/map/initial_state', self._on_initial_state, _LATCHED_QOS,
                                  callback_group=self._misc_cbg)

        self.create_service(ResetSim, '/mmg/reset', self._on_reset, callback_group=self._misc_cbg)

        self.timer = self.create_timer(self.dt, self._tick, callback_group=self._timer_cbg)

        self.get_logger().info(f'mmg_node up, dt={self.dt}s, waiting for /map/initial_state...')

    # ------------------------------------------------------------------
    def _on_initial_state(self, msg: VesselState):
        if self._have_initial_state:
            return
        self.mmg_state, self.delta, self.n = _vessel_state_to_tuple(msg)
        self._default_initial_state = (self.mmg_state.copy(), self.delta, self.n)
        self._have_initial_state = True
        self.get_logger().info(f'seeded initial state: {self.mmg_state.tolist()}, delta={self.delta}, n={self.n}')

    def _on_sim_status(self, msg: SimStatus):
        self._running = (msg.status == SimStatus.RUNNING)

    def _on_reset(self, request: ResetSim.Request, response: ResetSim.Response):
        if request.use_default:
            if self._default_initial_state is None:
                response.success = False
                return response
            state, delta, n = self._default_initial_state
            self.mmg_state = state.copy()
            self.delta, self.n = delta, n
        else:
            self.mmg_state, self.delta, self.n = _vessel_state_to_tuple(request.new_initial_state)
        response.success = True
        return response

    # ------------------------------------------------------------------
    def _state_msg(self, mmg_state, delta, n) -> VesselState:
        msg = VesselState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi = [float(v) for v in mmg_state]
        msg.delta = float(delta)
        msg.n = float(n)
        return msg

    def _tick(self):
        if not self._have_initial_state or not self._running:
            return

        request = SolveNMPC.Request()
        request.state = self._state_msg(self.mmg_state, self.delta, self.n)

        if not self.solve_client.service_is_ready():
            self.get_logger().warn('/nmpc/solve not available yet, skipping tick', throttle_duration_sec=2.0)
            return

        future = self.solve_client.call_async(request)
        # deliberately not rclpy.spin_until_future_complete(): this callback
        # already runs inside the node's own executor, so that would re-enter
        # the executor and deadlock. Busy-wait instead; the response callback
        # runs concurrently on another executor thread (see callback groups above).
        while rclpy.ok() and not future.done():
            time.sleep(0.0005)
        if not rclpy.ok():
            return
        if future.exception() is not None:
            self.get_logger().error(f'/nmpc/solve call failed: {future.exception()}')
            return

        response = future.result()
        delta, n = response.command.delta, response.command.n

        current, wave_force = (0.0, 0.0), (0.0, 0.0, 0.0)
        if not self.env_client.service_is_ready():
            self.get_logger().warn('/env/disturbance not available yet, using zero disturbance this tick',
                                    throttle_duration_sec=2.0)
        else:
            env_request = GetEnvDisturbance.Request()
            env_request.state = request.state
            env_future = self.env_client.call_async(env_request)
            # same busy-wait pattern as /nmpc/solve above, and for the same
            # reason (this callback runs inside the node's own executor).
            while rclpy.ok() and not env_future.done():
                time.sleep(0.0005)
            if not rclpy.ok():
                return
            if env_future.exception() is not None:
                self.get_logger().error(f'/env/disturbance call failed: {env_future.exception()}')
            else:
                env_response = env_future.result()
                current = (env_response.current.vx, env_response.current.vy)
                wave_force = (env_response.wave.fx, env_response.wave.fy, env_response.wave.fn)

        state_ca = ca.DM(self.mmg_state)
        control_ca = ca.DM([delta, n])
        next_state, _ = self.plant_step(state_ca, control_ca, ca.DM(list(current)), ca.DM(list(wave_force)))
        self.mmg_state = np.array(next_state).flatten()
        self.delta, self.n = delta, n

        self.state_pub.publish(self._state_msg(self.mmg_state, self.delta, self.n))


def main(args=None):
    rclpy.init(args=args)
    node = MmgNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
