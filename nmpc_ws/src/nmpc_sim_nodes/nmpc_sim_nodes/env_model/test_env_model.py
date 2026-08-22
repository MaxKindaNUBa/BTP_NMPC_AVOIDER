"""Standalone comparison harness for env_model (WAVE_CURRENT_DISTURBANCE_PLAN.md
section 3, unit-level test). Drives CurrentModel/WaveModel directly, no ROS
graph, no plant integrator -- a sanity check on parameter choices (Tc, sigma,
hs, tp, gamma) before wiring into the live sim. See test_closed_loop_env.py
(top-level, not in this dir) for the full-pipeline counterpart.

Run: ros2 run nmpc_sim_nodes test_env_model
     ros2 run nmpc_sim_nodes test_env_model --sim-time 600 --seed 7
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import _pkg_paths

_pkg_paths.ensure_on_path()

from env_model.config import CurrentConfig, WaveConfig, load_current_config, load_wave_config  # noqa: E402
from env_model.current_model import CurrentModel  # noqa: E402
from env_model.wave_model import WaveModel  # noqa: E402
from nmpc.params import DEFAULT_CONFIG  # noqa: E402

RESULTS_DIR = os.path.expanduser("~/nmpc_sim_logs/test_env_model_results")


def run_current(config: CurrentConfig, dt: float, sim_time: float) -> dict:
    model = CurrentModel(config)
    n_steps = int(sim_time / dt)
    t = np.arange(n_steps) * dt
    vx = np.zeros(n_steps)
    vy = np.zeros(n_steps)
    for i in range(n_steps):
        vx[i], vy[i] = model.step(dt)
    speed = np.hypot(vx, vy)
    heading = np.arctan2(vy, vx)
    return {"t": t, "vx": vx, "vy": vy, "speed": speed, "heading": heading}


def run_wave(config: WaveConfig, dt: float, sim_time: float, psi_schedule) -> dict:
    model = WaveModel(config, dt)
    n_steps = int(sim_time / dt)
    t = np.arange(n_steps) * dt
    fx = np.zeros(n_steps)
    fy = np.zeros(n_steps)
    fn = np.zeros(n_steps)
    for i in range(n_steps):
        fx[i], fy[i], fn[i] = model.force(psi_schedule(t[i]))
    return {"t": t, "fx": fx, "fy": fy, "fn": fn, "omega": model.omega, "S": model.S}


def plot_current(log: dict, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(log["t"], log["vx"], label="vx (m/s)")
    axes[0].plot(log["t"], log["vy"], label="vy (m/s)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Current velocity (m/s)")
    axes[0].set_title("CurrentModel: earth-frame components vs time")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(log["vy"], log["vx"], linewidth=0.8, alpha=0.8)
    axes[1].scatter([log["vy"][0]], [log["vx"][0]], color="green", marker="P", s=80, label="start")
    axes[1].set_xlabel("vy (m/s)")
    axes[1].set_ylabel("vx (m/s)")
    axes[1].set_title("Current vector wandering (OU process)")
    axes[1].axis("equal")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_wave(log: dict, out_path: str):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(log["omega"], log["S"], marker="o", markersize=3)
    axes[0].set_xlabel("omega (rad/s)")
    axes[0].set_ylabel("S(omega) (m^2 s)")
    axes[0].set_title("JONSWAP spectral density (discretized components)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(log["t"], log["fx"], label="fx (N)")
    axes[1].plot(log["t"], log["fy"], label="fy (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Drift force (N)")
    axes[1].set_title("WaveModel: body-frame surge/sway drift force vs time")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    axes[2].plot(log["t"], log["fn"], color="tab:purple")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Drift moment (N*m)")
    axes[2].set_title("WaveModel: yaw drift moment vs time")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def print_summary(log_current: dict, log_wave: dict):
    print("\n--- CurrentModel summary ---")
    print(f"  vx: mean={np.mean(log_current['vx']):.4f}  std={np.std(log_current['vx']):.4f}  "
          f"min={np.min(log_current['vx']):.4f}  max={np.max(log_current['vx']):.4f}")
    print(f"  vy: mean={np.mean(log_current['vy']):.4f}  std={np.std(log_current['vy']):.4f}  "
          f"min={np.min(log_current['vy']):.4f}  max={np.max(log_current['vy']):.4f}")
    print(f"  speed: mean={np.mean(log_current['speed']):.4f}  max={np.max(log_current['speed']):.4f}")

    print("\n--- WaveModel summary ---")
    print(f"  fx: mean={np.mean(log_wave['fx']):.4f}  std={np.std(log_wave['fx']):.4f}  "
          f"min={np.min(log_wave['fx']):.4f}  max={np.max(log_wave['fx']):.4f}")
    print(f"  fy: mean={np.mean(log_wave['fy']):.4f}  std={np.std(log_wave['fy']):.4f}  "
          f"min={np.min(log_wave['fy']):.4f}  max={np.max(log_wave['fy']):.4f}")
    print(f"  fn: mean={np.mean(log_wave['fn']):.4f}  std={np.std(log_wave['fn']):.4f}  "
          f"min={np.min(log_wave['fn']):.4f}  max={np.max(log_wave['fn']):.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-time", type=float, default=300.0, help="duration in seconds (default: 300)")
    parser.add_argument("--dt", type=float, default=DEFAULT_CONFIG.dt,
                         help="step (default: sim_params.yaml's nmpc_node.dt)")
    parser.add_argument("--seed", type=int, default=None,
                         help="override both CurrentModel and WaveModel (=seed+1) RNG seeds; "
                              "default: use sim_params.yaml's own current_seed/wave_seed")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="where to write plots")
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)

    # Both configs (mean/Tc/sigma/hs/tp/gamma/...) come straight out of
    # sim_params.yaml's mmg_node section -- no separate copy of those numbers
    # kept in this file, so this harness can never silently drift from what a
    # live mmg_node actually runs with. Only the seeds are ever overridden
    # here, and only if --seed was passed.
    current_config = load_current_config(seed=args.seed)
    wave_config = load_wave_config(seed=(args.seed + 1) if args.seed is not None else None)

    print(f"--- Running CurrentModel: sim_time={args.sim_time}s, dt={args.dt}s ---")
    log_current = run_current(current_config, args.dt, args.sim_time)

    print(f"--- Running WaveModel: sim_time={args.sim_time}s, dt={args.dt}s (vessel heading fixed at 0) ---")
    log_wave = run_wave(wave_config, args.dt, args.sim_time, psi_schedule=lambda t: 0.0)

    current_path = os.path.join(args.results_dir, "current_model.png")
    wave_path = os.path.join(args.results_dir, "wave_model.png")

    plot_current(log_current, current_path)
    plot_wave(log_wave, wave_path)

    print_summary(log_current, log_wave)
    print(f"\n  wrote {current_path}")
    print(f"  wrote {wave_path}")
    print(f"\nAll output under: {args.results_dir}")


if __name__ == "__main__":
    main()
