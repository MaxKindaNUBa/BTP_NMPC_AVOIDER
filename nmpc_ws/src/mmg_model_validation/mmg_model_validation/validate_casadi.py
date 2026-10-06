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

# Import CasADi implementation from nmpc_sim_nodes
from nmpc_sim_nodes.casadi_mmg_solver.casadi_mmg import make_acados_integrator

OUTPUT_DIR = "/mnt/0BF1C240574D9C37/BTP_NMPC_AVOIDER/nmpc_sim_logs/mmg_validation"
DEFAULT_RUDDER_RATE_DEG_S = 30.0  # [deg/s] maximum steering gear rate (DELTA_DOT_MAX)
LPP = 2.902  # [m] length between perpendiculars for KVLCC2 model


# =============================================================================
# STEADY APPROACH UTILITY (IMO STANDARDS)
# =============================================================================

def run_straight_to_steady_state(integrator, dt, rps=18.2, accel_tol=1e-3, hold_time=3.0, max_time=180.0):
    """Runs delta=0, rps=`rps` open-loop from rest until u/v/r have all
    stopped changing (finite-difference accel below `accel_tol` for a
    sustained `hold_time` seconds) -- the IMO steady "approach speed"
    condition a maneuvering test must start from. Returns the state
    history (including the t=0 rest sample) and the time at which steady
    state was declared (== the last sample's timestamp)."""
    hold_steps = max(1, int(round(hold_time / dt)))
    control = ca.DM([0.0, rps])
    state_casadi = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    hist = [np.array(state_casadi).flatten()]
    prev_uvr = hist[0][:3]
    consecutive = 0
    n_steps = int(round(max_time / dt))

    for step in range(n_steps):
        state_casadi, _ = integrator(state_casadi, control)
        cur = np.array(state_casadi).flatten()
        hist.append(cur)

        accel = np.abs(cur[:3] - prev_uvr) / dt
        prev_uvr = cur[:3]

        if np.all(accel < accel_tol):
            consecutive += 1
            if consecutive >= hold_steps:
                break
        else:
            consecutive = 0
    else:
        print(f'Warning: straight-run approach did not reach steady state within {max_time}s '
              f'(accel_tol={accel_tol}); using the un-converged endpoint as the approach speed.')

    hist = np.vstack(hist)
    straight_time = (len(hist) - 1) * dt
    return hist, straight_time


# =============================================================================
# TURNING CIRCLE MANEUVER (IMO MSC.137(76))
# =============================================================================

def simulate_turning_circle(rudder_deg, sim_time, dt, integrator, rps=18.2,
                            rudder_rate_deg_s=DEFAULT_RUDDER_RATE_DEG_S):
    """IMO-style turning circle: run straight (delta=0) until the ship
    reaches a steady approach speed, then ramp the rudder at `rudder_rate_deg_s`
    to `rudder_deg` and hold it for `sim_time` seconds."""
    hist_straight, straight_time = run_straight_to_steady_state(integrator, dt, rps=rps)
    state_casadi = ca.DM(hist_straight[-1])

    rudder_target = np.deg2rad(rudder_deg)
    rate_rad_s = np.deg2rad(rudder_rate_deg_s) if rudder_rate_deg_s is not None else None
    max_delta_step = rate_rad_s * dt if rate_rad_s is not None else None

    n_turn_steps = int(round(sim_time / dt))
    hist_turn = []
    delta_turn_deg = []
    cur_delta = 0.0

    for _ in range(n_turn_steps):
        if max_delta_step is not None:
            diff = rudder_target - cur_delta
            if abs(diff) <= max_delta_step:
                cur_delta = rudder_target
            else:
                cur_delta += np.sign(diff) * max_delta_step
        else:
            cur_delta = rudder_target

        control = ca.DM([cur_delta, rps])
        state_casadi, _ = integrator(state_casadi, control)
        hist_turn.append(np.array(state_casadi).flatten())
        delta_turn_deg.append(np.rad2deg(cur_delta))

    hist_casadi = np.vstack([hist_straight] + ([np.vstack(hist_turn)] if hist_turn else []))
    time_steps = np.arange(len(hist_casadi)) * dt
    delta_deg_hist = np.concatenate([np.zeros(len(hist_straight)), delta_turn_deg])

    return time_steps, hist_casadi, delta_deg_hist, straight_time


def compute_turning_params(hist_casadi, straight_time, dt, Lpp=LPP):
    """Standard IMO turning-circle maneuver parameters, measured from the
    rudder-execute point (end of the straight approach run):
      - Advance: distance traveled along the original heading until the
        heading has changed by 90 deg.
      - Transfer: perpendicular offset from the original heading line at
        that same 90 deg point.
      - Tactical diameter: perpendicular offset from the original heading
        line at the 180 deg heading-change point.
      - Steady turning diameter/radius: from the steady-state speed and yaw
        rate once the transient has died out (last 10% of the run).
      - Advance ratio: Advance / Lpp (IMO criteria are expressed in ship
        lengths, e.g. Advance < 4.5L, Tactical diameter < 5L).
    Returns a dict; entries are None/NaN if the run never turns that far.
    """
    idx0 = int(round(straight_time / dt))
    x0, y0, psi0 = hist_casadi[idx0, 3], hist_casadi[idx0, 4], hist_casadi[idx0, 5]
    psi = hist_casadi[:, 5]
    dpsi = np.unwrap(psi - psi0)

    def find_idx(deg_target):
        target = np.deg2rad(deg_target)
        cond = np.abs(dpsi[idx0:]) >= target
        return idx0 + int(np.argmax(cond)) if np.any(cond) else None

    idx90 = find_idx(90.0)
    idx180 = find_idx(180.0)

    def to_local(x, y):
        dx, dy = x - x0, y - y0
        fwd = dx * np.cos(psi0) + dy * np.sin(psi0)
        rgt = -dx * np.sin(psi0) + dy * np.cos(psi0)
        return fwd, rgt

    advance = transfer = tactical_diameter = None
    point90 = point180 = None
    if idx90 is not None:
        point90 = (hist_casadi[idx90, 3], hist_casadi[idx90, 4])
        advance, transfer = to_local(*point90)
        transfer = abs(transfer)
    if idx180 is not None:
        point180 = (hist_casadi[idx180, 3], hist_casadi[idx180, 4])
        _, rgt180 = to_local(*point180)
        tactical_diameter = abs(rgt180)

    n_tail = max(1, int(0.1 * (len(hist_casadi) - idx0)))
    u_ss = hist_casadi[-n_tail:, 0].mean()
    v_ss = hist_casadi[-n_tail:, 1].mean()
    r_ss = hist_casadi[-n_tail:, 2].mean()
    U_ss = np.hypot(u_ss, v_ss)
    steady_radius = U_ss / abs(r_ss) if abs(r_ss) > 1e-9 else np.nan
    steady_diameter = 2.0 * steady_radius
    advance_ratio = advance / Lpp if advance is not None else np.nan

    return dict(x0=x0, y0=y0, psi0=psi0, idx0=idx0, point90=point90, point180=point180,
                advance=advance, transfer=transfer, tactical_diameter=tactical_diameter,
                steady_radius=steady_radius, steady_diameter=steady_diameter,
                advance_ratio=advance_ratio)


def plot_turning_circle(fig, row_idx, n_rows, time_steps, hist_casadi, delta_deg_hist, rudder_deg, straight_time, dt,
                        rudder_rate_deg_s=DEFAULT_RUDDER_RATE_DEG_S):
    """Plots standard IMO/ITTC turning-circle maneuver with coordinate origin (0, 0)
    and t=0 set at Rudder Execute. Shows both Midship track and Stern track (illustrating
    the initial counter-drift/stern kick)."""
    params = compute_turning_params(hist_casadi, straight_time, dt)
    idx0 = params['idx0']
    x0, y0, psi0 = params['x0'], params['y0'], params['psi0']

    # Coordinate transformation relative to Rudder Execute
    dx = hist_casadi[:, 3] - x0
    dy = hist_casadi[:, 4] - y0
    fwd = dx * np.cos(psi0) + dy * np.sin(psi0)
    rgt = -dx * np.sin(psi0) + dy * np.cos(psi0)
    psi = hist_casadi[:, 5] - psi0

    # Stern path relative to execute course line:
    fwd_stern = fwd - 0.5 * LPP * np.cos(psi)
    rgt_stern = rgt - 0.5 * LPP * np.sin(psi)

    t_rel = time_steps - straight_time
    tail_mask = (t_rel >= -10.0)
    idx_tail_start = np.where(tail_mask)[0][0]

    # --- Trajectory Subplot ---
    ax_traj = fig.add_subplot(n_rows, 2, 2 * row_idx + 1)

    # Approach tail
    ax_traj.plot(rgt[idx_tail_start:idx0 + 1], fwd[idx_tail_start:idx0 + 1], 'k--',
                 linewidth=1.5, label='Steady Approach Run')

    # Turning circle trajectories: Midship and Stern
    ax_traj.plot(rgt[idx0:], fwd[idx0:], 'r-', linewidth=2.0, label='Midship Track (x = 0)')
    ax_traj.plot(rgt_stern[idx0:], fwd_stern[idx0:], 'b--', linewidth=1.2, alpha=0.85,
                 label='Stern Track (x = -Lpp/2, kick)')

    # Rudder Execute marker at origin (0, 0)
    ax_traj.plot(0, 0, 'ko', markersize=7, label='Rudder Execute (t=0)')

    # Original heading reference line, extended through advance
    ref_len = (params['advance'] * 1.15) if params['advance'] is not None else 4.5 * LPP
    ax_traj.plot([0, 0], [0, ref_len], 'k:', linewidth=1.2, label='Original Course Line')

    # 90 deg heading change (Advance & Transfer)
    if params['point90'] is not None:
        p90_x, p90_y = params['point90']
        p90_fwd = (p90_x - x0) * np.cos(psi0) + (p90_y - y0) * np.sin(psi0)
        p90_rgt = -(p90_x - x0) * np.sin(psi0) + (p90_y - y0) * np.cos(psi0)
        ax_traj.plot(p90_rgt, p90_fwd, 'b^', markersize=8, label='90° Heading Change')
        ax_traj.plot([0, p90_rgt], [p90_fwd, p90_fwd], 'b:', linewidth=1.0)
        ax_traj.plot([0, 0], [0, p90_fwd], 'b-', linewidth=2.0, alpha=0.45,
                     label=f"Advance: {params['advance']:.2f} m")

    # 180 deg heading change (Tactical Diameter)
    if params['point180'] is not None:
        p180_x, p180_y = params['point180']
        p180_fwd = (p180_x - x0) * np.cos(psi0) + (p180_y - y0) * np.sin(psi0)
        p180_rgt = -(p180_x - x0) * np.sin(psi0) + (p180_y - y0) * np.cos(psi0)
        ax_traj.plot(p180_rgt, p180_fwd, 'ms', markersize=8, label='180° Heading Change')
        ax_traj.plot([0, p180_rgt], [p180_fwd, p180_fwd], 'm:', linewidth=1.0)

    def fmt(v, unit=''):
        return f'{v:.2f}{unit}' if v is not None and np.isfinite(v) else 'n/a'

    advance_pass = "PASS (< 4.5L)" if params['advance_ratio'] < 4.5 else "FAIL"
    tac_ratio = params['tactical_diameter'] / LPP if params['tactical_diameter'] is not None else np.nan
    tactical_pass = "PASS (< 5.0L)" if tac_ratio < 5.0 else "FAIL"

    transfer_ratio = params['transfer'] / LPP if params['transfer'] is not None else np.nan
    radius_ratio = params['steady_radius'] / LPP if params['steady_radius'] is not None else np.nan
    diameter_ratio = params['steady_diameter'] / LPP if params['steady_diameter'] is not None else np.nan

    # Peak stern kick (outswing to opposite side) in first 15 seconds
    n_kick = min(len(rgt_stern) - idx0, int(round(15.0 / dt)))
    if rudder_deg > 0:
        stern_kick = np.min(rgt_stern[idx0:idx0 + n_kick])  # negative = to port
    else:
        stern_kick = np.max(rgt_stern[idx0:idx0 + n_kick])  # positive = to stbd
    kick_cm = abs(stern_kick) * 100.0
    kick_pct = (abs(stern_kick) / LPP) * 100.0

    summary = (
        f"Approach Speed u₀: {hist_casadi[idx0, 0]:.3f} m/s\n"
        f"Rudder Rate Limit: {rudder_rate_deg_s:.0f}°/s\n"
        f"Advance: {fmt(params['advance'], ' m')} ({fmt(params['advance_ratio'])} Lpp) [{advance_pass}]\n"
        f"Transfer: {fmt(params['transfer'], ' m')} ({fmt(transfer_ratio)} Lpp)\n"
        f"Tactical Diameter: {fmt(params['tactical_diameter'], ' m')} ({fmt(tac_ratio)} Lpp) [{tactical_pass}]\n"
        f"Steady Turning Radius: {fmt(params['steady_radius'], ' m')} ({fmt(radius_ratio)} Lpp)\n"
        f"Steady Turning Diameter: {fmt(params['steady_diameter'], ' m')} ({fmt(diameter_ratio)} Lpp)\n"
        f"Peak Stern Kick (Outswing): {kick_cm:.1f} cm ({kick_pct:.1f}% Lpp)"
    )
    ax_traj.text(0.03, 0.03, summary, transform=ax_traj.transAxes, fontsize=8,
                 va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax_traj.set_title(f'Ship Trajectory (Turning Circle $\\delta={rudder_deg:+.0f}^\\circ$, Rate Lim={rudder_rate_deg_s:.0f}°/s)')
    ax_traj.set_xlabel('Y [Cross-track offset] (m)')
    ax_traj.set_ylabel('X [Along original heading] (m)')
    ax_traj.axis('equal')
    ax_traj.grid(True, linestyle='--', alpha=0.5)

    # --- Velocity Subplot ---
    ax_vel = fig.add_subplot(n_rows, 2, 2 * row_idx + 2)
    mask_plot = (t_rel >= -10.0)
    ax_vel.plot(t_rel[mask_plot], hist_casadi[mask_plot, 0], 'b-', linewidth=1.5, label='u (surge, m/s)')
    ax_vel.plot(t_rel[mask_plot], hist_casadi[mask_plot, 1], 'g-', linewidth=1.5, label='v (sway, m/s)')
    ax_vel.plot(t_rel[mask_plot], hist_casadi[mask_plot, 2], 'r-', linewidth=1.5, label='r (yaw rate, rad/s)')
    ax_vel.axvline(0, color='k', linestyle=':', label='Rudder Execute (t=0)')
    ax_vel.set_title(f'Velocity States vs Time ($\\delta={rudder_deg:+.0f}^\\circ$ ramp at {rudder_rate_deg_s:.0f}°/s)')
    ax_vel.set_xlabel('Time from Rudder Execute (s)')
    ax_vel.set_ylabel('u, v (m/s) / r (rad/s)')
    ax_vel.grid(True, linestyle='--', alpha=0.5)

    ax_delta = ax_vel.twinx()
    ax_delta.plot(t_rel[mask_plot], delta_deg_hist[mask_plot], 'k--', linewidth=1.2, label='$\\delta$ (rudder, deg)')
    ax_delta.set_ylabel('$\\delta$ (rudder, deg)')
    ax_delta.set_ylim(-45, 45)

    lines_vel, labels_vel = ax_vel.get_legend_handles_labels()
    lines_delta, labels_delta = ax_delta.get_legend_handles_labels()
    ax_vel.legend(lines_vel + lines_delta, labels_vel + labels_delta, fontsize=8, loc='center right')


def run_turning_circle_validation(sim_time=200.0, dt=0.05, filename='turning_circle.png',
                                  rudder_rate_deg_s=DEFAULT_RUDDER_RATE_DEG_S):
    """Runs standard IMO 35 deg Turning Circle maneuvers (+35 deg Stbd, -35 deg Port)
    with smooth MMG formulation and saves turning_circle.png."""
    integrator = make_acados_integrator(dt, smooth=True)

    rudder_angles_deg = [35.0, -35.0]
    fig = plt.figure(figsize=(13.5, 11))
    for row_idx, rudder_deg in enumerate(rudder_angles_deg):
        time_steps, hist_casadi, delta_deg_hist, straight_time = simulate_turning_circle(
            rudder_deg, sim_time, dt, integrator, rudder_rate_deg_s=rudder_rate_deg_s)
        print(f'Turning Circle delta={rudder_deg:+.0f} deg: reached steady approach speed after {straight_time:.2f}s')
        plot_turning_circle(fig, row_idx, len(rudder_angles_deg), time_steps, hist_casadi,
                            delta_deg_hist, rudder_deg, straight_time, dt,
                            rudder_rate_deg_s=rudder_rate_deg_s)

    traj_legend_handles = [
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.5),
        plt.Line2D([0], [0], color='r', linestyle='-', linewidth=2.0),
        plt.Line2D([0], [0], color='b', linestyle='--', linewidth=1.2),
        plt.Line2D([0], [0], marker='o', color='k', linestyle='', markersize=7),
        plt.Line2D([0], [0], color='k', linestyle=':', linewidth=1.2),
        plt.Line2D([0], [0], marker='^', color='b', linestyle='', markersize=8),
        plt.Line2D([0], [0], color='b', linestyle='-', linewidth=2.0, alpha=0.5),
        plt.Line2D([0], [0], marker='s', color='m', linestyle='', markersize=8),
    ]
    traj_legend_labels = [
        'Steady Approach Run',
        'Midship Track ($x = 0$)',
        'Stern Track ($x = -L_{pp}/2$, kick)',
        'Rudder Execute ($t = 0$)',
        'Original Course Line',
        '90° Heading Change',
        'Advance Projection ($x_{90}$)',
        '180° Heading Change',
    ]

    fig.legend(traj_legend_handles, traj_legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=4, fontsize=8.5, frameon=True,
               title='Turning Circle Maneuver Elements (Common Legend for Left Column)',
               title_fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Turning Circle plot saved to {out_path}')


# =============================================================================
# ZIG-ZAG MANEUVER (TWO FULL OSCILLATIONS: 4 OVERSHOOTS, SIMMAN / IMO MSC.137(76))
# =============================================================================

def simulate_zigzag(target_angle_deg=20.0, check_angle_deg=20.0, sim_time=135.0, dt=0.05,
                    integrator=None, rps=18.2, rudder_rate_deg_s=DEFAULT_RUDDER_RATE_DEG_S,
                    initial_dir=1.0):
    """Simulates a 20°/20° Zig-Zag test for two full oscillations (two full sine waves):
      Wave 1:
        1. 1st execute (t=0): rudder to +initial_dir * target_angle_deg.
        2. 2nd execute (t2): heading reaches +initial_dir * check_angle_deg, rudder reversed.
        3. 1st overshoot (alpha1): heading peaks at peak1 before swinging back.
        4. 3rd execute (t3): heading reaches -initial_dir * check_angle_deg, rudder reversed.
        5. 2nd overshoot (alpha2): heading reaches opposite peak2 before turning back.
        6. Complete Cycle 1: heading crosses 0 deg after 2nd overshoot.
      Wave 2:
        7. 4th execute (t4): heading reaches +initial_dir * check_angle_deg, rudder reversed.
        8. 3rd overshoot (alpha3): heading peaks at peak3 before swinging back.
        9. 5th execute (t5): heading reaches -initial_dir * check_angle_deg, rudder reversed.
       10. 4th overshoot (alpha4): heading reaches opposite peak4 before turning back.
       11. Complete Cycle 2: heading crosses 0 deg after 4th overshoot (two full oscillations).
    """
    hist_straight, straight_time = run_straight_to_steady_state(integrator, dt, rps=rps)
    state = ca.DM(hist_straight[-1])
    idx0 = len(hist_straight) - 1

    rate_rad_s = np.deg2rad(rudder_rate_deg_s) if rudder_rate_deg_s is not None else None
    max_d_delta = rate_rad_s * dt if rate_rad_s is not None else None

    delta_cmd_deg = initial_dir * target_angle_deg
    cur_delta_rad = 0.0
    phase = 1

    events = {
        'execute_1': {'t': 0.0, 'psi_deg': 0.0, 'state': np.array(state).flatten()}
    }

    t_hist = []
    delta_hist = []
    hist_turn = []

    peak1_psi = 0.0
    peak2_psi = 0.0
    peak3_psi = 0.0
    peak4_psi = 0.0
    t_peak1 = 0.0
    t_peak2 = 0.0
    t_peak3 = 0.0
    t_peak4 = 0.0
    state_peak1 = None
    state_peak2 = None
    state_peak3 = None
    state_peak4 = None

    psi0 = float(state[5])
    n_steps = int(round(sim_time / dt))

    for step in range(n_steps):
        t = step * dt
        cur_state = np.array(state).flatten()
        psi_rel_deg = np.rad2deg(np.unwrap([cur_state[5] - psi0])[0])

        hist_turn.append(cur_state)
        delta_hist.append(np.rad2deg(cur_delta_rad))
        t_hist.append(t)

        if phase == 1:
            if initial_dir * psi_rel_deg >= check_angle_deg:
                phase = 2
                delta_cmd_deg = -initial_dir * target_angle_deg
                events['execute_2'] = {'t': t, 'psi_deg': psi_rel_deg, 'state': cur_state}
                peak1_psi = psi_rel_deg
                t_peak1 = t
                state_peak1 = cur_state
        elif phase == 2:
            if 'overshoot_1' not in events:
                if initial_dir * psi_rel_deg >= initial_dir * peak1_psi:
                    peak1_psi = psi_rel_deg
                    t_peak1 = t
                    state_peak1 = cur_state
                else:
                    alpha1 = abs(peak1_psi) - check_angle_deg
                    events['overshoot_1'] = {'t': t_peak1, 'psi_deg': peak1_psi, 'alpha_deg': alpha1,
                                            'state': state_peak1, 't_ov': t_peak1 - events['execute_2']['t']}
            if initial_dir * psi_rel_deg <= -check_angle_deg:
                phase = 3
                delta_cmd_deg = initial_dir * target_angle_deg
                events['execute_3'] = {'t': t, 'psi_deg': psi_rel_deg, 'state': cur_state}
                peak2_psi = psi_rel_deg
                t_peak2 = t
                state_peak2 = cur_state
        elif phase == 3:
            if 'overshoot_2' not in events:
                if initial_dir * psi_rel_deg <= initial_dir * peak2_psi:
                    peak2_psi = psi_rel_deg
                    t_peak2 = t
                    state_peak2 = cur_state
                else:
                    alpha2 = abs(peak2_psi) - check_angle_deg
                    events['overshoot_2'] = {'t': t_peak2, 'psi_deg': peak2_psi, 'alpha_deg': alpha2,
                                            'state': state_peak2, 't_ov': t_peak2 - events['execute_3']['t']}
            if 'overshoot_2' in events and (initial_dir * psi_rel_deg >= 0.0) and 'complete_1' not in events:
                events['complete_1'] = {'t': t, 'psi_deg': psi_rel_deg, 'state': cur_state}

            if 'overshoot_2' in events and (initial_dir * psi_rel_deg >= check_angle_deg):
                phase = 4
                delta_cmd_deg = -initial_dir * target_angle_deg
                events['execute_4'] = {'t': t, 'psi_deg': psi_rel_deg, 'state': cur_state}
                peak3_psi = psi_rel_deg
                t_peak3 = t
                state_peak3 = cur_state
        elif phase == 4:
            if 'overshoot_3' not in events:
                if initial_dir * psi_rel_deg >= initial_dir * peak3_psi:
                    peak3_psi = psi_rel_deg
                    t_peak3 = t
                    state_peak3 = cur_state
                else:
                    alpha3 = abs(peak3_psi) - check_angle_deg
                    events['overshoot_3'] = {'t': t_peak3, 'psi_deg': peak3_psi, 'alpha_deg': alpha3,
                                            'state': state_peak3, 't_ov': t_peak3 - events['execute_4']['t']}
            if 'overshoot_3' in events and (initial_dir * psi_rel_deg <= -check_angle_deg):
                phase = 5
                delta_cmd_deg = initial_dir * target_angle_deg
                events['execute_5'] = {'t': t, 'psi_deg': psi_rel_deg, 'state': cur_state}
                peak4_psi = psi_rel_deg
                t_peak4 = t
                state_peak4 = cur_state
        elif phase == 5:
            if 'overshoot_4' not in events:
                if initial_dir * psi_rel_deg <= initial_dir * peak4_psi:
                    peak4_psi = psi_rel_deg
                    t_peak4 = t
                    state_peak4 = cur_state
                else:
                    alpha4 = abs(peak4_psi) - check_angle_deg
                    events['overshoot_4'] = {'t': t_peak4, 'psi_deg': peak4_psi, 'alpha_deg': alpha4,
                                            'state': state_peak4, 't_ov': t_peak4 - events['execute_5']['t']}
            if 'overshoot_4' in events and (initial_dir * psi_rel_deg >= 0.0):
                if 'complete_2' not in events:
                    events['complete_2'] = {'t': t, 'psi_deg': psi_rel_deg, 'state': cur_state}
                    events['complete'] = events['complete_2']
                if t >= events['complete_2']['t'] + 6.0:
                    break

        target_rad = np.deg2rad(delta_cmd_deg)
        if max_d_delta is not None:
            diff = target_rad - cur_delta_rad
            if abs(diff) <= max_d_delta:
                cur_delta_rad = target_rad
            else:
                cur_delta_rad += np.sign(diff) * max_d_delta
        else:
            cur_delta_rad = target_rad

        control = ca.DM([cur_delta_rad, rps])
        state, _ = integrator(state, control)

    hist_casadi = np.vstack([hist_straight] + ([np.vstack(hist_turn)] if hist_turn else []))
    time_steps = np.arange(len(hist_casadi)) * dt
    delta_deg_hist = np.concatenate([np.zeros(len(hist_straight)), delta_hist])

    return time_steps, hist_casadi, delta_deg_hist, straight_time, events


def plot_zigzag(fig, row_idx, n_rows, time_steps, hist_casadi, delta_deg_hist, straight_time, dt, events,
                target_angle_deg=20.0, check_angle_deg=20.0, initial_dir=1.0,
                rudder_rate_deg_s=DEFAULT_RUDDER_RATE_DEG_S):
    """Plots one row (trajectory on left, time histories on right) for a 2-oscillation 20°/20° zig-zag."""
    idx0 = int(round(straight_time / dt))
    x0, y0, psi0 = hist_casadi[idx0, 3], hist_casadi[idx0, 4], hist_casadi[idx0, 5]

    dx = hist_casadi[:, 3] - x0
    dy = hist_casadi[:, 4] - y0
    fwd = dx * np.cos(psi0) + dy * np.sin(psi0)
    rgt = -dx * np.sin(psi0) + dy * np.cos(psi0)
    psi = hist_casadi[:, 5] - psi0
    psi_deg = np.rad2deg(np.unwrap(psi))

    fwd_stern = fwd - 0.5 * LPP * np.cos(psi)
    rgt_stern = rgt - 0.5 * LPP * np.sin(psi)

    t_rel = time_steps - straight_time
    tail_mask = (t_rel >= -8.0)
    idx_tail_start = np.where(tail_mask)[0][0]

    dir_str = "Starboard First (+20°)" if initial_dir > 0 else "Port First (-20°)"

    # --- Subplot 1: Trajectory ---
    ax_traj = fig.add_subplot(n_rows, 2, 2 * row_idx + 1)
    ax_traj.plot(rgt[idx_tail_start:idx0 + 1], fwd[idx_tail_start:idx0 + 1], 'k--',
                 linewidth=1.5, label='Steady Approach Run')
    ax_traj.plot(rgt[idx0:], fwd[idx0:], 'r-', linewidth=2.0, label='Midship Track (x = 0)')
    ax_traj.plot(rgt_stern[idx0:], fwd_stern[idx0:], 'b--', linewidth=1.2, alpha=0.85,
                 label='Stern Track (x = -Lpp/2)')
    ax_traj.plot(0, 0, 'ko', markersize=7, label='1st Execute (t = 0)')

    # Course line projection
    max_fwd = np.max(fwd)
    ax_traj.plot([0, 0], [0, max_fwd * 1.05], 'k:', linewidth=1.2, label='Original Course Line')

    def to_traj_coord(st):
        d_x = st[3] - x0
        d_y = st[4] - y0
        f = d_x * np.cos(psi0) + d_y * np.sin(psi0)
        r = -d_x * np.sin(psi0) + d_y * np.cos(psi0)
        return r, f

    # Event markers on trajectory
    if 'execute_2' in events:
        r2, f2 = to_traj_coord(events['execute_2']['state'])
        ax_traj.plot(r2, f2, 'b^', markersize=8)
    if 'overshoot_1' in events:
        ro1, fo1 = to_traj_coord(events['overshoot_1']['state'])
        ax_traj.plot(ro1, fo1, 'ms', markersize=8)
    if 'execute_3' in events:
        r3, f3 = to_traj_coord(events['execute_3']['state'])
        ax_traj.plot(r3, f3, 'bv', markersize=8)
    if 'overshoot_2' in events:
        ro2, fo2 = to_traj_coord(events['overshoot_2']['state'])
        ax_traj.plot(ro2, fo2, 'cs', markersize=8)
    if 'execute_4' in events:
        r4, f4 = to_traj_coord(events['execute_4']['state'])
        ax_traj.plot(r4, f4, 'b^', markersize=8)
    if 'overshoot_3' in events:
        ro3, fo3 = to_traj_coord(events['overshoot_3']['state'])
        ax_traj.plot(ro3, fo3, 'ms', markersize=8)
    if 'execute_5' in events:
        r5, f5 = to_traj_coord(events['execute_5']['state'])
        ax_traj.plot(r5, f5, 'bv', markersize=8)
    if 'overshoot_4' in events:
        ro4, fo4 = to_traj_coord(events['overshoot_4']['state'])
        ax_traj.plot(ro4, fo4, 'cs', markersize=8)
    if 'complete_2' in events:
        rc, fc = to_traj_coord(events['complete_2']['state'])
        ax_traj.plot(rc, fc, 'g*', markersize=9)

    alpha1 = events.get('overshoot_1', {}).get('alpha_deg', np.nan)
    t_ov1 = events.get('overshoot_1', {}).get('t_ov', np.nan)
    alpha2 = events.get('overshoot_2', {}).get('alpha_deg', np.nan)
    t_ov2 = events.get('overshoot_2', {}).get('t_ov', np.nan)
    alpha3 = events.get('overshoot_3', {}).get('alpha_deg', np.nan)
    t_ov3 = events.get('overshoot_3', {}).get('t_ov', np.nan)
    alpha4 = events.get('overshoot_4', {}).get('alpha_deg', np.nan)
    t_ov4 = events.get('overshoot_4', {}).get('t_ov', np.nan)

    t_reach = events.get('execute_2', {}).get('t', np.nan)
    t_cycle1 = events.get('complete_1', {}).get('t', np.nan)
    t_cycle2_abs = events.get('complete_2', {}).get('t', np.nan)
    t_cycle2 = t_cycle2_abs - t_cycle1 if np.isfinite(t_cycle2_abs) and np.isfinite(t_cycle1) else np.nan

    imo_alpha1_pass = "PASS (≤ 25°)" if alpha1 <= 25.0 else "FAIL"

    def fmt(v, u=''):
        return f'{v:.2f}{u}' if np.isfinite(v) else 'n/a'

    u0 = hist_casadi[idx0, 0]
    summary = (
        f"Approach Speed u₀: {u0:.3f} m/s | Rate Lim: {rudder_rate_deg_s:.0f}°/s\n"
        f"1st Reach (t₂): {fmt(t_reach, ' s')}\n"
        f"1st Overshoot (α₁): {fmt(alpha1, '°')} [{imo_alpha1_pass}] (Δt={fmt(t_ov1, ' s')})\n"
        f"2nd Overshoot (α₂): {fmt(alpha2, '°')} (Δt={fmt(t_ov2, ' s')})\n"
        f"3rd Overshoot (α₃): {fmt(alpha3, '°')} (Δt={fmt(t_ov3, ' s')})\n"
        f"4th Overshoot (α₄): {fmt(alpha4, '°')} (Δt={fmt(t_ov4, ' s')})\n"
        f"Cycle 1 Period (T_c1): {fmt(t_cycle1, ' s')} | Cycle 2: {fmt(t_cycle2, ' s')}"
    )
    ax_traj.text(0.03, 0.03, summary, transform=ax_traj.transAxes, fontsize=7.8,
                 va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax_traj.set_title(f'Ship Trajectory (20°/20° Zig-Zag [2 Waves], {dir_str})')
    ax_traj.set_xlabel('Y [Cross-track offset] (m)')
    ax_traj.set_ylabel('X [Along original heading] (m)')
    ax_traj.axis('equal')
    ax_traj.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot 2: Time History ---
    ax_time = fig.add_subplot(n_rows, 2, 2 * row_idx + 2)
    mask = (t_rel >= -5.0)

    ax_time.plot(t_rel[mask], psi_deg[mask], 'b-', linewidth=1.8, label='ψ (heading, deg)')
    ax_time.plot(t_rel[mask], delta_deg_hist[mask], 'k--', linewidth=1.3, label='δ (rudder, deg)')

    # Check bands
    ax_time.axhline(+check_angle_deg, color='gray', linestyle=':', alpha=0.7)
    ax_time.axhline(-check_angle_deg, color='gray', linestyle=':', alpha=0.7)
    ax_time.axhline(0.0, color='k', linestyle='-', linewidth=0.6, alpha=0.5)

    # Mark key events on heading trace
    if 'execute_2' in events:
        e = events['execute_2']
        ax_time.plot(e['t'], e['psi_deg'], 'b^', markersize=8)
    if 'overshoot_1' in events:
        e = events['overshoot_1']
        ax_time.plot(e['t'], e['psi_deg'], 'ms', markersize=8)
        ax_time.annotate(f'α₁ = {e["alpha_deg"]:.1f}°', (e['t'], e['psi_deg']),
                         xytext=(6, 5 if e['psi_deg'] > 0 else -15), textcoords='offset points',
                         fontsize=8, fontweight='bold', color='m')
    if 'execute_3' in events:
        e = events['execute_3']
        ax_time.plot(e['t'], e['psi_deg'], 'bv', markersize=8)
    if 'overshoot_2' in events:
        e = events['overshoot_2']
        ax_time.plot(e['t'], e['psi_deg'], 'cs', markersize=8)
        ax_time.annotate(f'α₂ = {e["alpha_deg"]:.1f}°', (e['t'], e['psi_deg']),
                         xytext=(6, -15 if e['psi_deg'] < 0 else 5), textcoords='offset points',
                         fontsize=8, fontweight='bold', color='teal')
    if 'execute_4' in events:
        e = events['execute_4']
        ax_time.plot(e['t'], e['psi_deg'], 'b^', markersize=8)
    if 'overshoot_3' in events:
        e = events['overshoot_3']
        ax_time.plot(e['t'], e['psi_deg'], 'ms', markersize=8)
        ax_time.annotate(f'α₃ = {e["alpha_deg"]:.1f}°', (e['t'], e['psi_deg']),
                         xytext=(6, 5 if e['psi_deg'] > 0 else -15), textcoords='offset points',
                         fontsize=8, fontweight='bold', color='m')
    if 'execute_5' in events:
        e = events['execute_5']
        ax_time.plot(e['t'], e['psi_deg'], 'bv', markersize=8)
    if 'overshoot_4' in events:
        e = events['overshoot_4']
        ax_time.plot(e['t'], e['psi_deg'], 'cs', markersize=8)
        ax_time.annotate(f'α₄ = {e["alpha_deg"]:.1f}°', (e['t'], e['psi_deg']),
                         xytext=(6, -15 if e['psi_deg'] < 0 else 5), textcoords='offset points',
                         fontsize=8, fontweight='bold', color='teal')
    if 'complete_1' in events:
        e = events['complete_1']
        ax_time.plot(e['t'], e['psi_deg'], 'g.', markersize=6)
    if 'complete_2' in events:
        e = events['complete_2']
        ax_time.plot(e['t'], e['psi_deg'], 'g*', markersize=9)

    ax_time.set_xlabel('Time from Rudder Execute (s)')
    ax_time.set_ylabel('Heading ψ / Rudder δ (deg)')
    ax_time.set_ylim(-45, 45)
    ax_time.grid(True, linestyle='--', alpha=0.5)

    # Secondary axis: velocity states
    ax_vel = ax_time.twinx()
    ax_vel.plot(t_rel[mask], hist_casadi[mask, 0], 'g-', linewidth=1.2, alpha=0.85, label='u (surge, m/s)')
    ax_vel.plot(t_rel[mask], np.rad2deg(hist_casadi[mask, 2]), 'r:', linewidth=1.2, alpha=0.85, label='r (yaw rate, deg/s)')
    ax_vel.set_ylabel('Surge u (m/s) / Yaw rate r (deg/s)')

    lines_main, labels_main = ax_time.get_legend_handles_labels()
    lines_sec, labels_sec = ax_vel.get_legend_handles_labels()
    ax_time.legend(lines_main + lines_sec, labels_main + labels_sec, fontsize=8, loc='upper right', ncol=2)
    ax_time.set_title(f'Heading, Rudder & Velocity vs Time (20°/20° Zig-Zag [2 Waves], {dir_str})')


def run_zigzag_validation(target_angle_deg=20.0, check_angle_deg=20.0, sim_time=135.0, dt=0.05,
                          filename='zigzag.png', rudder_rate_deg_s=DEFAULT_RUDDER_RATE_DEG_S):
    """Runs the 20°/20° Zig-Zag maneuver test for two full oscillations (Starboard & Port)
    with smooth MMG formulation and saves zigzag.png."""
    integrator = make_acados_integrator(dt, smooth=True)

    initial_dirs = [1.0, -1.0]  # Starboard first, then Port first
    fig = plt.figure(figsize=(14.0, 11.5))

    for row_idx, init_dir in enumerate(initial_dirs):
        dir_name = "Starboard" if init_dir > 0 else "Port"
        time_steps, hist_casadi, delta_deg_hist, straight_time, events = simulate_zigzag(
            target_angle_deg=target_angle_deg, check_angle_deg=check_angle_deg,
            sim_time=sim_time, dt=dt, integrator=integrator,
            rudder_rate_deg_s=rudder_rate_deg_s, initial_dir=init_dir)

        alpha1 = events.get('overshoot_1', {}).get('alpha_deg', np.nan)
        alpha2 = events.get('overshoot_2', {}).get('alpha_deg', np.nan)
        alpha3 = events.get('overshoot_3', {}).get('alpha_deg', np.nan)
        alpha4 = events.get('overshoot_4', {}).get('alpha_deg', np.nan)
        print(f'20°/20° Zig-Zag ({dir_name}-first, 2 Oscillations): α₁={alpha1:.2f}°, α₂={alpha2:.2f}°, α₃={alpha3:.2f}°, α₄={alpha4:.2f}° (approach {straight_time:.2f}s)')

        plot_zigzag(fig, row_idx, len(initial_dirs), time_steps, hist_casadi, delta_deg_hist,
                    straight_time, dt, events, target_angle_deg=target_angle_deg,
                    check_angle_deg=check_angle_deg, initial_dir=init_dir,
                    rudder_rate_deg_s=rudder_rate_deg_s)

    traj_legend_handles = [
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.5),
        plt.Line2D([0], [0], color='r', linestyle='-', linewidth=2.0),
        plt.Line2D([0], [0], color='b', linestyle='--', linewidth=1.2),
        plt.Line2D([0], [0], color='k', linestyle=':', linewidth=1.2),
        plt.Line2D([0], [0], marker='o', color='k', linestyle='', markersize=7),
        plt.Line2D([0], [0], marker='^', color='b', linestyle='', markersize=8),
        plt.Line2D([0], [0], marker='v', color='b', linestyle='', markersize=8),
        plt.Line2D([0], [0], marker='s', color='m', linestyle='', markersize=8),
        plt.Line2D([0], [0], marker='s', color='c', linestyle='', markersize=8),
        plt.Line2D([0], [0], marker='*', color='g', linestyle='', markersize=9),
    ]
    traj_legend_labels = [
        'Steady Approach Run',
        'Midship Track ($x = 0$)',
        'Stern Track ($x = -L_{pp}/2$)',
        'Original Course Line',
        '1st Execute ($t = 0$)',
        r'Check (+20°) Executes ($t_2, t_4$)',
        r'Check (-20°) Executes ($t_3, t_5$)',
        r'Positive Peaks ($\alpha_1, \alpha_3$)',
        r'Negative Peaks ($\alpha_2, \alpha_4$)',
        r'2nd Cycle Complete ($\psi = 0^\circ$)',
    ]

    fig.legend(traj_legend_handles, traj_legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=5, fontsize=8.2, frameon=True,
               title='Zig-Zag Maneuver Elements (Common Legend for Left Column)',
               title_fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Zig-Zag plot saved to {out_path}')


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    print("==============================================================")
    print("  RUNNING MMG MODEL VALIDATION SUITE (SMOOTH DYNAMICS)")
    print("==============================================================")
    print("\n--- 1. Turning Circle Maneuver Validation (±35° Rudder) ---")
    run_turning_circle_validation(filename='turning_circle.png')

    print("\n--- 2. Zig-Zag Maneuver Validation (20°/20° SIMMAN Benchmark) ---")
    run_zigzag_validation(filename='zigzag.png')
    print("\nValidation suite complete. Plots generated in:")
    print(f"  {OUTPUT_DIR}/turning_circle.png")
    print(f"  {OUTPUT_DIR}/zigzag.png")


if __name__ == '__main__':
    main()
