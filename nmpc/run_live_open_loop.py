"""
Live visualization of the NMPC open-loop closed-loop rollout (Phase 2 scenarios),
using mpc_visualization/ instead of saving static plots to nmpc/results/.

Drives mpc_bridge.MPCBridge every solve step (ship state + prediction/control
horizon), same way run_demo.py's mock loop does, but with the real NMPC solvers
instead of a dead-reckoning mock. run_demo.py itself is untouched — just used
as the reference for how the bridge/visualizer expect to be fed.

Run: python nmpc/run_live_open_loop.py --solver acados --test 1
"""
import os
import sys
import time
import argparse
import threading
import numpy as np
import casadi as ca

# repo root, so `nmpc` / `casadi_mmg_solver` resolve as packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# mpc_visualization/ itself, since mpc_bridge.py / visualizer.py use flat
# imports ("from mpc_bridge import MPCBridge"), not package-relative ones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mpc_visualization")))

from casadi_mmg_solver.casadi_mmg import make_casadi_integrator
from nmpc.config import DEFAULT_CONFIG, IDX_U, IDX_V, IDX_R, IDX_X, IDX_Y, IDX_PSI, IDX_DELTA, IDX_N
from nmpc.path_following import compute_path_angle, select_active_waypoint

from mpc_bridge import MPCBridge  # noqa: E402
from visualizer import MPCVisualizer  # noqa: E402

# order the bridge expects ship_state/predicted_trajectory rows in: [u, v, r, x, y, psi]
_MMG_ROWS = [IDX_U, IDX_V, IDX_R, IDX_X, IDX_Y, IDX_PSI]

SCENARIOS = {
    1: dict(waypoints=[(0.0, 0.0), (0.0, +30.0)], mmg_init=[0.78, 0.1, 0.0, 0.0, 0.0, 0.0], sim_time=450.0),
    2: dict(waypoints=[(0.0, 0.0), (0.0, -30.0)], mmg_init=[0.78, 1.5, 0.0, 0.0, 0.0, 0.5], sim_time=350.0),
    3: dict(waypoints=[(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0), (0.0, 0.0)], mmg_init=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], sim_time=1550.0),
}


def run_scenario_live(nmpc_solver, bridge: MPCBridge, waypoints, mmg_init, sim_time,
                       config=DEFAULT_CONFIG, delta_init=0.0, n_init=None,
                       speed_factor=3.0, label=""):
    """Same closed-loop rollout as test_nmpc_open_loop.run_scenario, but pushes
    every step into the bridge instead of logging arrays for a static plot."""
    if n_init is None:
        n_init = config.N_TRIM

    plant_step = make_casadi_integrator(config.dt, method="rk4", sym_type=ca.SX)

    bridge.set_start_goal(tuple(waypoints[0]), tuple(waypoints[-1]))
    bridge._reference_path = np.array(waypoints, dtype=float)  # show the full multi-leg path, not just start/goal
    bridge.set_obstacles([])  # Phase 2 has no obstacles — always zero/empty

    mmg_state = np.array(mmg_init, dtype=float)
    delta, n = float(delta_init), float(n_init)
    target_idx = 1

    n_steps = int(sim_time / config.dt)
    print(f"--- Live: {label}, {n_steps} steps, dt={config.dt}s, T={sim_time}s (speed x{speed_factor}) ---")

    last_wall = time.time()
    for step in range(n_steps):
        if not getattr(bridge, "_visualizer_active", False):
            print("Visualizer closed — stopping simulation.")
            return

        prev_wp = waypoints[target_idx - 1]
        target_wp = waypoints[target_idx]
        chi_p = compute_path_angle(prev_wp, target_wp)
        x_d, y_d = target_wp
        bridge.set_active_waypoint(target_wp)  # which leg-target the NMPC is steering toward, for the visualizer

        result = nmpc_solver.solve(mmg_state.tolist(), delta, n, chi_p, x_d, y_d, obstacles=[])
        delta, n = result["delta"], result["n"]

        state_ca = ca.DM(mmg_state)
        control_ca = ca.DM([delta, n])
        next_state, _ = plant_step(state_ca, control_ca)
        mmg_state = np.array(next_state).flatten()

        t = step * config.dt
        xi_traj = result["xi_traj"]  # (11, N+1), augmented state prediction

        predicted_trajectory = xi_traj[_MMG_ROWS, :].T          # (N+1, 6) -> [u,v,r,x,y,psi]
        control_horizon = xi_traj[[IDX_DELTA, IDX_N], :-1].T    # (N, 2)   -> actual [delta, rps] over horizon

        bridge.update_ship_state(mmg_state, t)
        bridge.update_prediction(predicted_trajectory, control_horizon)
        bridge.set_obstacles([])  # keep re-asserting: no obstacles in these Phase 2 scenarios

        target_idx = select_active_waypoint(mmg_state[3], mmg_state[4], waypoints, target_idx, config.WP_RADIUS)

        if not result["success"]:
            print(f"  [step {step}] solver FAILED: {result['return_status']}")

        # pace playback to roughly speed_factor x real time (solve latency counts against the budget)
        wall_elapsed = time.time() - last_wall
        sleep_needed = (config.dt / speed_factor) - wall_elapsed
        if sleep_needed > 0:
            time.sleep(sleep_needed)
        last_wall = time.time()

    print("--- Live scenario finished ---")


def main():
    parser = argparse.ArgumentParser(description="Live-visualized NMPC open-loop rollout")
    parser.add_argument("--solver", choices=["acados", "casadi"], default="acados")
    parser.add_argument("--test", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--speed", type=float, default=3.0, help="playback speed multiplier vs real time")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.solver == "acados":
        from nmpc.nmpc_acados import AcadosNMPC
        nmpc_solver = AcadosNMPC(config)
    else:
        from nmpc.nmpc_casadi import CasadiNMPC
        nmpc_solver = CasadiNMPC(config)

    scenario = SCENARIOS[args.test]
    bridge = MPCBridge(start=tuple(scenario["waypoints"][0]), goal=tuple(scenario["waypoints"][-1]))
    bridge._visualizer_active = True

    sim_thread = threading.Thread(
        target=run_scenario_live,
        kwargs=dict(nmpc_solver=nmpc_solver, bridge=bridge, waypoints=scenario["waypoints"],
                    mmg_init=scenario["mmg_init"], sim_time=scenario["sim_time"], config=config,
                    speed_factor=args.speed, label=f"Test {args.test} ({args.solver})"),
        daemon=True,
    )
    sim_thread.start()

    visualizer = MPCVisualizer(bridge, update_interval_ms=50, run_name=f"scenario_{args.test}_{args.solver}")
    # visualizer.py's default axis limits assume run_demo.py's mock scenario (Y,X in
    # [-10,70]/[-10,50]); these NMPC test waypoints run Y negative, so rescale to fit
    # (with generous padding — a diverging run can drift well past the waypoints).
    wp = np.array(scenario["waypoints"])
    y_pad, x_pad = 40.0, 60.0
    visualizer.ax_map.set_xlim(wp[:, 1].min() - y_pad, wp[:, 1].max() + y_pad)
    visualizer.ax_map.set_ylim(wp[:, 0].min() - x_pad, wp[:, 0].max() + x_pad)

    try:
        visualizer.start()
    finally:
        bridge._visualizer_active = False
        sim_thread.join(timeout=1.0)
        print("Live visualization closed.")


if __name__ == "__main__":
    main()
