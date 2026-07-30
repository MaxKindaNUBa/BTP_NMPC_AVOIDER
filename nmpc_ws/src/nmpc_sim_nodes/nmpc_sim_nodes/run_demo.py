import time
import threading
import numpy as np

from . import _pkg_paths

_pkg_paths.ensure_on_path()

from mpc_bridge import MPCBridge, Obstacle  # noqa: E402
from visualizer import MPCVisualizer  # noqa: E402

def run_mock_simulation(bridge: MPCBridge, sim_speed_factor: float = 2.0):
    """
    Background simulation loop generating a simple circular motion (mock ship)
    and populating predicted trajectory & control horizons for visualization.
    """
    print(f"Mock Visualizer Demo started (Speed factor: {sim_speed_factor}x)")

    dt_control = 0.5
    t = 0.0

    # Start at origin
    x, y = 0.0, 0.0
    psi = 0.0
    u, v = 0.8, 0.0
    r = 0.05 # Constant turning rate (rad/s)

    # Command inputs
    delta_cmd = np.deg2rad(35.0)
    rps_cmd = 18.2

    last_wall_time = time.time()

    while t < 120.0:
        if not hasattr(bridge, '_visualizer_active') or not bridge._visualizer_active:
            break

        # Kinematic update for mock turning motion
        psi += r * dt_control
        x += (u * np.cos(psi) - v * np.sin(psi)) * dt_control
        y += (u * np.sin(psi) + v * np.cos(psi)) * dt_control

        current_state = np.array([u, v, r, x, y, psi])

        # Generate mock predicted trajectory (dead-reckoning over N=100 steps)
        N_horizon = 100
        C_horizon = 20
        dt_pred = 0.5
        pred_x = np.zeros((N_horizon + 1, 6))
        pred_x[0] = current_state
        for k in range(N_horizon):
            u_k, v_k, r_k, x_k, y_k, psi_k = pred_x[k]
            psi_next = psi_k + r_k * dt_pred
            x_next = x_k + (u_k * np.cos(psi_k) - v_k * np.sin(psi_k)) * dt_pred
            y_next = y_k + (u_k * np.sin(psi_k) + v_k * np.cos(psi_k)) * dt_pred
            pred_x[k+1] = np.array([u_k, v_k, r_k, x_next, y_next, psi_next])

        # Generate mock control horizon with some variation to show it on the plot
        pred_u = np.zeros((C_horizon, 2))
        for k in range(C_horizon):
            pred_u[k, 0] = delta_cmd * np.cos(0.12 * k) # Decaying/oscillating planned rudder angle
            pred_u[k, 1] = rps_cmd + 2.0 * np.sin(0.3 * k)  # Oscillating propeller speed

        # Update bridge
        bridge.update_ship_state(current_state, t)
        bridge.update_prediction(pred_x, pred_u)

        # Control rate pacing
        sim_elapsed = dt_control
        wall_elapsed = time.time() - last_wall_time
        sleep_needed = (sim_elapsed / sim_speed_factor) - wall_elapsed
        if sleep_needed > 0:
            time.sleep(sleep_needed)
        last_wall_time = time.time()
        t += dt_control

    print("Mock simulation loop ended.")

def main():
    # Setup mock scenario markers and obstacles
    start = (0.0, 0.0)
    goal = (45.0, 25.0)
    obstacles = [
        Obstacle("obs_1", 10.0, 10.0, 4.0),
        Obstacle("obs_2", 25.0, 22.0, 5.0),
        Obstacle("obs_3", 38.0, 12.0, 4.5)
    ]

    # Initialize shared bridge
    bridge = MPCBridge(start=start, goal=goal)
    bridge.set_obstacles(obstacles)
    bridge._visualizer_active = True

    # Start the simulation thread
    sim_thread = threading.Thread(
        target=run_mock_simulation,
        args=(bridge, 2.0), # Run at 2x real-time speed
        daemon=True
    )
    sim_thread.start()

    # Initialize and start visualizer in the main thread
    visualizer = MPCVisualizer(bridge, update_interval_ms=50, run_name="mock_demo")

    try:
        visualizer.start()
    finally:
        bridge._visualizer_active = False
        sim_thread.join(timeout=1.0)
        print("Visualization Demo Closed.")

if __name__ == "__main__":
    main()
