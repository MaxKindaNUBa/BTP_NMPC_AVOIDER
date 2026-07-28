"""Ad-hoc diagnostic: Q_DIAG[IDX_U]=0.0 (old) vs current weight, in simulation.
Uses CasadiNMPC (fast to rebuild per-config vs acados' C recompile).
Same waypoints/initial condition as test1_straight_aligned in test_nmpc.py.

Run: python nmpc/compare_qu.py
"""
import os
import sys
import dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow running this file directly (python nmpc/compare_qu.py) by putting
# the repo root on sys.path, so `nmpc` resolves as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nmpc.config import DEFAULT_CONFIG, IDX_U
from nmpc.nmpc_casadi import CasadiNMPC
from nmpc.test_nmpc import run_scenario

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

waypoints = [(0.0, 0.0), (0.0, -30.0)]
mmg_init = [0.78, 0.0, 0.0, 0.0, 0.0, 0.0]
SIM_TIME = 150.0  # shorter than test1's 450s; the u-collapse trend is visible well before that

q_current = DEFAULT_CONFIG.Q_DIAG
q_zero_u = tuple(0.0 if i == IDX_U else v for i, v in enumerate(q_current))

cfg_zero = dataclasses.replace(DEFAULT_CONFIG, Q_DIAG=q_zero_u)
cfg_curr = dataclasses.replace(DEFAULT_CONFIG, Q_DIAG=q_current)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Q[u]=0 case:   Q_DIAG={cfg_zero.Q_DIAG}")
    print(f"Q[u]=1.0 case: Q_DIAG={cfg_curr.Q_DIAG}")

    solver_zero = CasadiNMPC(cfg_zero)
    log_zero = run_scenario(solver_zero, waypoints, mmg_init, sim_time=SIM_TIME, config=cfg_zero, label="Q[u]=0")

    solver_curr = CasadiNMPC(cfg_curr)
    log_curr = run_scenario(solver_curr, waypoints, mmg_init, sim_time=SIM_TIME, config=cfg_curr, label="Q[u]=1.0")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(log_zero["t"], log_zero["u"], "r-", label="u, Q[u]=0.0 (old)")
    ax.plot(log_curr["t"], log_curr["u"], "b-", label="u, Q[u]=1.0 (current)")
    ax.axhline(DEFAULT_CONFIG.U_REF, color="grey", linestyle="--", label="U_REF")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("u (m/s)")
    ax.set_title("Q[u] weight comparison: surge speed")
    ax.grid(True, linestyle="--", alpha=0.5); ax.legend()
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "qu_comparison_speed.png")
    fig.savefig(out_path)
    print(f"saved {out_path}")

    mask = log_zero["t"] > 15.0
    print(f"Q[u]=0.0   : u at t_end={log_zero['u'][-1]:.4f}  "
          f"max|u-U_REF| after t=15s = {np.max(np.abs(log_zero['u'][mask]-DEFAULT_CONFIG.U_REF)):.4f}")
    mask2 = log_curr["t"] > 15.0
    print(f"Q[u]=1.0   : u at t_end={log_curr['u'][-1]:.4f}  "
          f"max|u-U_REF| after t=15s = {np.max(np.abs(log_curr['u'][mask2]-DEFAULT_CONFIG.U_REF)):.4f}")


if __name__ == "__main__":
    main()
