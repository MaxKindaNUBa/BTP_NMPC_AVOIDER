"""mmg_node: owns the plant integrator and the master 1/dt clock (section 2.7 /
2.3 of ROS2_CONVERSION_PLAN.md). Each tick: sample the current/wave
disturbance, synthesize the true IMU/GPS/gyro sensor measurement, call
/ukf/estimate (always, for observability), populate /nmpc/solve's request
from either the true state or the UKF's estimate depending on use_ukf, block
on the response, apply the returned (delta, n), integrate one true-state RK4
step, publish the new true state.

Sensor noise (formerly sensor_node) and current/wave disturbance (formerly
env_node) live directly in this node now -- no separate ROS2 nodes, no
/sensor/measure or /env/disturbance RPCs. Both remain independently
toggleable via parameters (sensor_enabled, current_enabled, wave_enabled),
same semantics as before: disabled means pass-through/zero, not a partial
mode.

use_ukf governs ONLY what /nmpc/solve's request.state is populated from:
true state when False (bypasses sensor noise AND the UKF entirely), the
UKF's estimate when True -- independent of sensor_enabled, which governs
only whether /sensor/measured_state itself carries injected noise. The UKF
service is called every tick regardless of use_ukf, so /ukf/estimated_state
and the logged UKF columns stay populated for observability either way.
Unlike the pre-UKF version of this node, nmpc_node's docstring claim ("no
path to the plant's true state at all") is now only true when use_ukf=True --
see nmpc_node.py's own docstring, which is not updated to reflect this
(explicitly out of scope; a known, flagged piece of documentation debt).
"""
import math
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
from casadi_mmg_solver.casadi_mmg import make_casadi_accel_function, make_casadi_integrator  # noqa: E402

from nmpc_interfaces.msg import CurrentState, SensorMeasurement, SimStatus, VesselState, WaveState  # noqa: E402
from nmpc_interfaces.srv import EstimateState, ResetSim, SolveNMPC  # noqa: E402

from env_model.config import CurrentConfig, WaveConfig  # noqa: E402
from env_model.current_model import CurrentModel  # noqa: E402
from env_model.wave_model import WaveModel  # noqa: E402

from sensor_model.config import PRESETS  # noqa: E402
from sensor_model.sensor_model import SensorModel  # noqa: E402

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

        # Selects /nmpc/solve's *state* source only (UKF estimate vs. true
        # state). Current feedforward to the NMPC is independent of this flag
        # and always active -- see _tick(), which sources it from the UKF's
        # estimated_current when use_ukf is True/healthy, otherwise straight
        # from the true environment current sampled this tick.
        self.declare_parameter('use_ukf', False)
        self.use_ukf = bool(self.get_parameter('use_ukf').value)

        self.plant_step = make_casadi_integrator(self.dt, method='rk4', sym_type=ca.SX, with_env=True)
        self.accel_fn = make_casadi_accel_function(sym_type=ca.SX)

        self._init_sensor_model()
        self._init_env_models()

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
        self.measured_pub = self.create_publisher(SensorMeasurement, '/sensor/measured_state', 10)
        self.current_pub = self.create_publisher(CurrentState, '/env/current_state', 10)
        self.wave_pub = self.create_publisher(WaveState, '/env/wave_state', 10)

        self.solve_client = self.create_client(SolveNMPC, '/nmpc/solve', callback_group=self._client_cbg)
        self.ukf_client = self.create_client(EstimateState, '/ukf/estimate', callback_group=self._client_cbg)
        self.ukf_reset_client = self.create_client(ResetSim, '/ukf/reset', callback_group=self._misc_cbg)

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

        self.get_logger().info(
            f'mmg_node up, dt={self.dt}s, sensor_enabled={self._sensor_enabled}, use_ukf={self.use_ukf}, '
            f'current_enabled={self.current_model is not None}, wave_enabled={self.wave_model is not None}, '
            f'waiting for /map/initial_state...')

    # ------------------------------------------------------------------
    def _init_sensor_model(self):
        self.declare_parameter('sensor_enabled', False)
        self.declare_parameter('sensor_preset', 'light')
        self.declare_parameter('sensor_seed', 42)

        self._sensor_enabled = bool(self.get_parameter('sensor_enabled').value)
        preset_name = str(self.get_parameter('sensor_preset').value)
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown sensor_model preset {preset_name!r}; expected one of {list(PRESETS)}")
        sensor_seed = int(self.get_parameter('sensor_seed').value)

        # Always constructed (matching the old sensor_node.py's own pattern),
        # gated only at use time in _measure() -- disabled means pass-through,
        # not "no model at all".
        self.sensor_model = SensorModel(PRESETS[preset_name], self.dt, sensor_seed)

    def _init_env_models(self):
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

        self._reset_ukf()

        response.success = True
        return response

    def _reset_ukf(self):
        """Forwards the just-resolved (mmg_state, delta, n) to /ukf/reset so
        the filter's covariance/estimate never carries stale state across a
        scenario reset. Runs on _misc_cbg (this handler's own group, not the
        tick's), so a short blocking wait here is safe."""
        if not self.ukf_reset_client.service_is_ready():
            self.get_logger().warn('/ukf/reset not available yet, skipping UKF reset')
            return
        req = ResetSim.Request()
        req.new_initial_state = self._state_msg(self.mmg_state, self.delta, self.n)
        req.use_default = False
        self._call_sync(self.ukf_reset_client, req)

    # ------------------------------------------------------------------
    def _state_msg(self, mmg_state, delta, n) -> VesselState:
        msg = VesselState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi = [float(v) for v in mmg_state]
        msg.delta = float(delta)
        msg.n = float(n)
        return msg

    def _measure_sensor(self, true_msg: VesselState, true_accel) -> SensorMeasurement:
        """In-process replacement for the old sensor_node's /sensor/measure
        service: true_msg + true (ax,ay) in, a (possibly corrupted)
        SensorMeasurement out -- GPS/compass/gyro/IMU-accel/actuator-encoders,
        explicitly no u,v (reconstructing those is ukf_node's job). Also
        publishes /sensor/measured_state, same as sensor_node did."""
        ax_true, ay_true = true_accel
        measured = SensorMeasurement()
        measured.header.stamp = self.get_clock().now().to_msg()
        measured.header.frame_id = true_msg.header.frame_id

        if not self._sensor_enabled:
            measured.x, measured.y, measured.psi, measured.r = true_msg.x, true_msg.y, true_msg.psi, true_msg.r
            measured.ax, measured.ay = float(ax_true), float(ay_true)
            measured.delta, measured.n = true_msg.delta, true_msg.n
        else:
            mmg_state = [true_msg.u, true_msg.v, true_msg.r, true_msg.x, true_msg.y, true_msg.psi]
            r_m, x_m, y_m, psi_m, ax_m, ay_m, delta_m, n_m = self.sensor_model.measure(
                mmg_state, true_msg.delta, true_msg.n, accel_true=(ax_true, ay_true))
            measured.x, measured.y, measured.psi, measured.r = float(x_m), float(y_m), float(psi_m), float(r_m)
            measured.ax, measured.ay = float(ax_m), float(ay_m)
            measured.delta, measured.n = float(delta_m), float(n_m)

        self.measured_pub.publish(measured)
        return measured

    def _sample_disturbance(self, psi: float, stamp, frame_id: str):
        """In-process replacement for the old env_node's /env/disturbance
        service: samples current/wave off the true heading, publishes
        /env/current_state and /env/wave_state (same as env_node did), and
        returns (current, wave_force) tuples for the plant integrator."""
        current_msg = CurrentState()
        current_msg.header.stamp = stamp
        current_msg.header.frame_id = frame_id
        current_msg.enabled = self.current_enabled
        if self.current_enabled:
            vx, vy = self.current_model.step(self.dt)
            current_msg.vx, current_msg.vy = float(vx), float(vy)
            current_msg.speed = float(math.hypot(vx, vy))
            current_msg.heading = float(math.atan2(vy, vx))
        else:
            current_msg.vx = current_msg.vy = current_msg.speed = current_msg.heading = 0.0

        wave_msg = WaveState()
        wave_msg.header.stamp = stamp
        wave_msg.header.frame_id = frame_id
        wave_msg.enabled = self.wave_enabled
        if self.wave_enabled:
            fx, fy, fn = self.wave_model.force(psi)
            wave_msg.hs = float(self.wave_model.config.hs)
            wave_msg.tp = float(self.wave_model.config.tp)
            wave_msg.mean_heading = float(self.wave_model.config.mean_heading)
            wave_msg.fx, wave_msg.fy, wave_msg.fn = float(fx), float(fy), float(fn)
        else:
            wave_msg.hs = wave_msg.tp = wave_msg.mean_heading = 0.0
            wave_msg.fx = wave_msg.fy = wave_msg.fn = 0.0

        self.current_pub.publish(current_msg)
        self.wave_pub.publish(wave_msg)

        return current_msg, (wave_msg.fx, wave_msg.fy, wave_msg.fn)

    def _call_sync(self, client, request):
        """Shared busy-wait-on-async-future pattern: deliberately not
        rclpy.spin_until_future_complete(), since this callback already runs
        inside the node's own executor and that would re-enter it and
        deadlock. The response callback runs concurrently on another executor
        thread (see the callback groups set up in __init__). Returns the
        response, or None on failure (exception or shutdown) -- caller decides
        how to handle a None."""
        future = client.call_async(request)
        while rclpy.ok() and not future.done():
            time.sleep(0.0005)
        if not rclpy.ok():
            return None
        if future.exception() is not None:
            self.get_logger().error(f'{client.service_name} call failed: {future.exception()}')
            return None
        return future.result()

    def _tick(self):
        if not self._have_initial_state or not self._running:
            return

        true_msg = self._state_msg(self.mmg_state, self.delta, self.n)

        current_msg, wave_force = self._sample_disturbance(
            float(self.mmg_state[5]), true_msg.header.stamp, true_msg.header.frame_id)
        current = (current_msg.vx, current_msg.vy)

        true_accel_dm = self.accel_fn(ca.DM(self.mmg_state), ca.DM([self.delta, self.n]),
                                       ca.DM(list(current)), ca.DM(list(wave_force)))
        true_accel = (float(true_accel_dm[0]), float(true_accel_dm[1]))

        sensor_msg = self._measure_sensor(true_msg, true_accel)

        if not self.solve_client.service_is_ready() or not self.ukf_client.service_is_ready():
            self.get_logger().warn('/nmpc/solve or /ukf/estimate not available yet, skipping tick',
                                    throttle_duration_sec=2.0)
            return

        ukf_req = EstimateState.Request()
        ukf_req.measurement = sensor_msg
        ukf_response = self._call_sync(self.ukf_client, ukf_req)
        ukf_ok = ukf_response is not None and ukf_response.success
        if ukf_response is not None and not ukf_response.success:
            self.get_logger().warn(f'/ukf/estimate reported failure: {ukf_response.return_status}',
                                    throttle_duration_sec=2.0)
        # A transient UKF failure (or use_ukf=False) falls back to the true
        # state for this tick's NMPC request rather than skipping the tick
        # entirely -- a deliberate, revisitable safety choice; the
        # alternative is to skip the tick the way a not-ready /nmpc/solve is
        # handled above.

        request = SolveNMPC.Request()
        request.state = ukf_response.estimated_state if (self.use_ukf and ukf_ok) else true_msg
        # Current feedforward is always on, independent of use_ukf (which only
        # picks the *state* source above): UKF's estimate when enabled/healthy,
        # otherwise the true current already sampled this tick.
        request.current = ukf_response.estimated_current if (self.use_ukf and ukf_ok) else current_msg

        response = self._call_sync(self.solve_client, request)
        if response is None:
            return

        delta, n = response.command.delta, response.command.n

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
