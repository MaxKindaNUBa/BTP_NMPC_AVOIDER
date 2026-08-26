"""ukf_node: Unscented Kalman Filter state estimator, serving /ukf/estimate --
called synchronously by mmg_node once per tick, mirroring nmpc_node's own
/nmpc/solve service pattern (mmg_node's tick is a tight, synchronous,
single-threaded-per-tick sequence, so a bare pub/sub filter risks its
estimate not being ready within the same tick mmg_node needs it).

State: [u, v, r, x, y, psi, vcx, vcy, ax_bias, ay_bias, pos_bias_x, pos_bias_y]
-- see ukf/ukf_core.py for the full process/measurement model. Always runs
once seeded from /map/initial_state,
regardless of mmg_node's own use_ukf toggle, so /ukf/estimated_state and
/ukf/estimated_current stay populated for logging/RViz/HUD observability even
when the NMPC isn't consuming the estimate.
"""
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

import math  # noqa: E402
import numpy as np  # noqa: E402
import casadi as ca  # noqa: E402
from casadi_mmg_solver.casadi_mmg import make_casadi_accel_function, make_casadi_integrator  # noqa: E402

from env_model.config import load_current_config  # noqa: E402

from nmpc_interfaces.msg import CurrentState, VesselState  # noqa: E402
from nmpc_interfaces.srv import EstimateState, ResetSim  # noqa: E402

from sensor_model.config import PRESETS  # noqa: E402

import dataclasses  # noqa: E402

from ukf.config import (  # noqa: E402
    UKFConfig, accel_bias_decay, bias_noise_diag, current_noise_diag, default_r_diag, pos_bias_decay,
)
from ukf.ukf_core import UnscentedKalmanFilter  # noqa: E402

_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class UkfNode(Node):
    def __init__(self):
        super().__init__('ukf_node')

        self.declare_parameter('dt', 0.1)  # MUST match mmg_node.dt
        self.declare_parameter('sensor_preset', 'light')  # MUST match mmg_node.sensor_preset (sizes default R)
        self.declare_parameter('alpha', 1e-3)
        self.declare_parameter('beta', 2.0)
        self.declare_parameter('kappa', 0.0)
        self.declare_parameter('q_diag', list(UKFConfig().q_diag))
        self.declare_parameter('p0_diag', list(UKFConfig().p0_diag))
        self.declare_parameter('adapt_r', UKFConfig().adapt_r)
        self.declare_parameter('adapt_q', UKFConfig().adapt_q)
        self.declare_parameter('forgetting_factor', UKFConfig().forgetting_factor)
        # See ukf/config.py's load_ukf_config() comment: q_diag[8:12]
        # (ax_bias/ay_bias/pos_bias_x/pos_bias_y) is always the LIVE physical
        # value times this scale, not a hand-copied absolute literal -- these
        # two knobs are what a tune_ukf.py run's accel_bias/pos_bias
        # group_scale should actually be set to now.
        self.declare_parameter('accel_bias_q_scale', 1.0)
        self.declare_parameter('pos_bias_q_scale', 1.0)

        dt = float(self.get_parameter('dt').value)
        preset_name = str(self.get_parameter('sensor_preset').value)
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown sensor_model preset {preset_name!r}; expected one of {list(PRESETS)}")
        default_r = default_r_diag(PRESETS[preset_name])
        self.declare_parameter('r_diag', default_r.tolist())

        config = UKFConfig(
            alpha=float(self.get_parameter('alpha').value),
            beta=float(self.get_parameter('beta').value),
            kappa=float(self.get_parameter('kappa').value),
            q_diag=tuple(float(v) for v in self.get_parameter('q_diag').value),
            p0_diag=tuple(float(v) for v in self.get_parameter('p0_diag').value),
            adapt_r=bool(self.get_parameter('adapt_r').value),
            adapt_q=bool(self.get_parameter('adapt_q').value),
            forgetting_factor=float(self.get_parameter('forgetting_factor').value),
        )
        r_diag = np.array([float(v) for v in self.get_parameter('r_diag').value], dtype=float)

        # Seed vcx/vcy at the configured expected current (0,0 if none
        # configured) rather than always 0 -- p0_diag's vcx/vcy entries are
        # now tight (see ukf/config.py's own comment: the OU process's
        # stationary variance around ITS mean, not "totally unknown"), so a
        # 0 seed would make the filter start confidently wrong instead of
        # confidently right, and its now-small Q would then keep it wrong for
        # a long time (measured in test_ukf.py: this exact pairing -- 0 seed
        # + tight p0 -- made vcx RMSE ~3x WORSE, not better, before this
        # fix). See ukf/config.py's p0_diag comment for the full derivation.
        current_cfg = load_current_config()
        self._vcx0, self._vcy0 = current_cfg.mean_vx, current_cfg.mean_vy

        # Override the vcx/vcy (indices 6,7) q_diag/p0_diag entries with a
        # LIVE derivation from current_cfg's actual sigma/time_constant_s,
        # rather than trusting whatever numeric literal this yaml's
        # q_diag/p0_diag happen to hold for those two slots -- see
        # current_noise_diag()'s docstring: a stale hand-copied literal here
        # is exactly what made the UKF unable to track a faster-varying
        # current after only current_sigma/current_time_constant were
        # changed for a test.
        q_cur, p0_cur = current_noise_diag(current_cfg.sigma, current_cfg.time_constant_s, dt=dt)
        q_diag_list = list(config.q_diag)
        p0_diag_list = list(config.p0_diag)
        q_diag_list[6] = q_diag_list[7] = q_cur
        p0_diag_list[6] = p0_diag_list[7] = p0_cur

        # Same override for ax_bias/ay_bias (8,9) and pos_bias_x/pos_bias_y
        # (10,11), from the ACTIVE sensor preset's own bias_sigma/tau, times
        # accel_bias_q_scale/pos_bias_q_scale -- see ukf/config.py's
        # load_ukf_config() comment for the full account of why q_diag[8:12]
        # needs an explicit scale knob (unlike current's q_diag[6:7]).
        sensor_cfg = PRESETS[preset_name]
        accel_bias_q_scale = float(self.get_parameter('accel_bias_q_scale').value)
        pos_bias_q_scale = float(self.get_parameter('pos_bias_q_scale').value)
        q_ax, p0_ax = bias_noise_diag(sensor_cfg.accel_bias_sigma, sensor_cfg.accel_bias_tau, dt)
        q_pos, p0_pos = bias_noise_diag(sensor_cfg.pos_bias_sigma, sensor_cfg.pos_bias_tau, dt)
        q_diag_list[8] = q_diag_list[9] = q_ax * accel_bias_q_scale
        p0_diag_list[8] = p0_diag_list[9] = p0_ax
        q_diag_list[10] = q_diag_list[11] = q_pos * pos_bias_q_scale
        p0_diag_list[10] = p0_diag_list[11] = p0_pos

        config = dataclasses.replace(config, q_diag=tuple(q_diag_list), p0_diag=tuple(p0_diag_list))

        predict_integrator = make_casadi_integrator(dt, method='rk4', sym_type=ca.SX, with_env=True)
        accel_fn = make_casadi_accel_function(sym_type=ca.SX)
        ax_bias_decay = accel_bias_decay(sensor_cfg, dt)
        gps_bias_decay = pos_bias_decay(sensor_cfg, dt)
        self.ukf = UnscentedKalmanFilter(config, predict_integrator, accel_fn, r_diag,
                                          ax_bias_decay, gps_bias_decay)

        self._have_initial_state = False

        self._service_cbg = MutuallyExclusiveCallbackGroup()
        self._misc_cbg = MutuallyExclusiveCallbackGroup()

        self.estimated_state_pub = self.create_publisher(VesselState, '/ukf/estimated_state', 10)
        self.estimated_current_pub = self.create_publisher(CurrentState, '/ukf/estimated_current', 10)

        self.create_subscription(VesselState, '/map/initial_state', self._on_initial_state, _LATCHED_QOS,
                                  callback_group=self._misc_cbg)

        self.create_service(EstimateState, '/ukf/estimate', self._on_estimate, callback_group=self._service_cbg)
        self.create_service(ResetSim, '/ukf/reset', self._on_reset, callback_group=self._misc_cbg)

        self.get_logger().info(f'ukf_node up, dt={dt}s, sensor_preset={preset_name}, waiting for /map/initial_state...')

    # ------------------------------------------------------------------
    def _seed(self, u, v, r, x, y, psi):
        x0 = np.array([u, v, r, x, y, psi, self._vcx0, self._vcy0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.ukf.reset(x0)
        self._have_initial_state = True

    def _on_initial_state(self, msg: VesselState):
        if self._have_initial_state:
            return
        self._seed(msg.u, msg.v, msg.r, msg.x, msg.y, msg.psi)
        self.get_logger().info(f'seeded UKF state: {self.ukf.x.tolist()}')

    def _on_reset(self, request: ResetSim.Request, response: ResetSim.Response):
        if request.use_default and not self._have_initial_state:
            response.success = False
            return response
        s = request.new_initial_state
        self._seed(s.u, s.v, s.r, s.x, s.y, s.psi)
        self.get_logger().info(f'UKF reset: {self.ukf.x.tolist()}')
        response.success = True
        return response

    # ------------------------------------------------------------------
    def _on_estimate(self, request: EstimateState.Request, response: EstimateState.Response):
        if not self._have_initial_state:
            response.success = False
            response.return_status = "NOT_SEEDED"
            return response

        m = request.measurement
        z_meas = [m.x, m.y, m.psi, m.r, m.ax, m.ay]
        control = [m.delta, m.n]

        x_hat, _, success, status = self.ukf.step(control, z_meas)
        if not success:
            self.get_logger().warn(f'UKF update failed: {status}', throttle_duration_sec=2.0)

        estimated_state = VesselState()
        estimated_state.header.stamp = self.get_clock().now().to_msg()
        estimated_state.header.frame_id = 'map'
        estimated_state.u, estimated_state.v, estimated_state.r = float(x_hat[0]), float(x_hat[1]), float(x_hat[2])
        estimated_state.x, estimated_state.y, estimated_state.psi = float(x_hat[3]), float(x_hat[4]), float(x_hat[5])
        estimated_state.delta = float(m.delta)
        estimated_state.n = float(m.n)

        vcx, vcy = float(x_hat[6]), float(x_hat[7])
        estimated_current = CurrentState()
        estimated_current.header.stamp = estimated_state.header.stamp
        estimated_current.header.frame_id = 'map'
        estimated_current.vx, estimated_current.vy = vcx, vcy
        estimated_current.speed = float(math.hypot(vcx, vcy))
        estimated_current.heading = float(math.atan2(vcy, vcx))
        estimated_current.enabled = True

        self.estimated_state_pub.publish(estimated_state)
        self.estimated_current_pub.publish(estimated_current)

        response.estimated_state = estimated_state
        response.estimated_current = estimated_current
        response.success = success
        response.return_status = status
        return response


def main(args=None):
    rclpy.init(args=args)
    node = UkfNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
