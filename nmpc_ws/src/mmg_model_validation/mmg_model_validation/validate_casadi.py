import os
import ctypes

# Set ACADOS_SOURCE_DIR programmatically
os.environ["ACADOS_SOURCE_DIR"] = "/home/chandran/acados"

# Pre-load acados shared libraries to bypass LD_LIBRARY_PATH requirement on Linux
acados_lib_dir = "/home/chandran/acados/lib"
if os.path.exists(acados_lib_dir):
    try:
        # Load in topological dependency order
        mode = ctypes.RTLD_GLOBAL
        ctypes.CDLL(os.path.join(acados_lib_dir, "libqdldl.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libosqp.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libqpOASES_e.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libblasfeo.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libhpipm.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libacados.so"), mode=mode)
    except Exception as e:
        print(f"Warning: programmatically loading acados libraries failed: {e}")

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Import original NumPy implementation
from .Preliminary_func import activateMMG

# Import CasADi implementation -- casadi_mmg_solver now lives inside nmpc_sim_nodes
# (see ROS2_CONVERSION_PLAN.md's "true move" of the legacy Python codebases),
# so this is a cross-package import; mmg_model_validation's package.xml
# declares a runtime dependency on nmpc_sim_nodes for exactly this.
from nmpc_sim_nodes.casadi_mmg_solver.casadi_mmg import make_acados_integrator

OUTPUT_DIR = os.path.expanduser("~/nmpc_sim_logs")

def run_validation(sim_time=200.0, dt=0.05, smooth=True):
    time_steps = np.arange(0, sim_time, dt)
    # Initial state [u, v, r, x, y, psi]
    state_np = np.array([0.78, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_casadi = ca.DM(state_np)

    # Constant control: rudder 35 deg, propeller 18.2 rps (same as previous tests)
    rudder_angle = np.deg2rad(35.0)
    control = ca.DM([rudder_angle, 18.2])

    # Prepare integrator
    integrator = make_acados_integrator(dt, smooth=smooth)

    # History containers
    hist_np = []
    hist_casadi = []

    for _ in time_steps:
        # NumPy model step
        state_np, _ = activateMMG(state_np, 0, False, 0, rudder_angle, h=dt)
        hist_np.append(state_np.copy())

        # CasADi model step
        state_casadi, _ = integrator(state_casadi, control)
        hist_casadi.append(np.array(state_casadi).flatten())

    # Convert histories to arrays
    hist_np = np.vstack(hist_np)
    hist_casadi = np.vstack(hist_casadi)

    # Compute absolute errors over the trajectory
    errors = np.abs(hist_np - hist_casadi)
    max_err = errors.max(axis=0)
    state_names = ['u', 'v', 'r', 'x', 'y', 'psi']
    print('Maximum absolute error per state (NumPy vs CasADi):')
    for name, err in zip(state_names, max_err):
        print(f'{name:>3}: {err:e}')

    # Plot trajectories for visual check
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_np[:, 4], hist_np[:, 3], 'b-', label='NumPy Trajectory')
    plt.plot(hist_casadi[:, 4], hist_casadi[:, 3], 'r--', label='CasADi Trajectory')
    plt.title('Ship Trajectory (Turning Circle)')
    plt.xlabel('Y (m)')
    plt.ylabel('X (m)')
    plt.legend()
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.plot(time_steps, hist_np[:, 2], 'b-', label='NumPy r')
    plt.plot(time_steps, hist_casadi[:, 2], 'r--', label='CasADi r')
    plt.title('Yaw Rate vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('r (rad/s)')
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'validation_comparison.png')
    plt.savefig(out_path)
    print(f'Plot saved to {out_path}')


def main():
    print("--- RUNNING VALIDATION WITH SMOOTH=TRUE ---")
    run_validation(smooth=True)
    print("\n--- RUNNING VALIDATION WITH SMOOTH=FALSE ---")
    run_validation(smooth=False)


if __name__ == '__main__':
    main()

