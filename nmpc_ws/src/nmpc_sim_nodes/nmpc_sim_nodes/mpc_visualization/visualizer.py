import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from mpc_bridge import MPCBridge
import csv
import json
import os
from datetime import datetime

class MPCVisualizer:
    def __init__(self, bridge: MPCBridge, update_interval_ms: int = 50, run_name: str = "mpc_run", log_dir: str = None, enable_logging: bool = True):
        self.bridge = bridge
        self.update_interval = update_interval_ms
        self.is_running = True
        self.run_name = run_name
        self.enable_logging = enable_logging
        self.last_logged_timestamp = None
        self.csv_file = None
        self.csv_writer = None

        if log_dir is None:
            # Default to mpc_visualization/logs relative to this file
            self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        else:
            self.log_dir = log_dir

        if self.enable_logging:
            os.makedirs(self.log_dir, exist_ok=True)
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.run_name}_{now_str}.csv"
            self.log_filepath = os.path.join(self.log_dir, filename)

            try:
                self.csv_file = open(self.log_filepath, mode='w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file)
                # Write CSV Header
                self.csv_writer.writerow([
                    "Date_Time",
                    "Run_Name",
                    "Timestamp_s",
                    "u_m_s",
                    "v_m_s",
                    "r_rad_s",
                    "x_m",
                    "y_m",
                    "psi_rad",
                    "psi_deg",
                    "start_x",
                    "start_y",
                    "goal_x",
                    "goal_y",
                    "predicted_trajectory",
                    "control_horizon",
                    "obstacles"
                ])
                self.csv_file.flush()
                print(f"[MPCVisualizer] Live logging initialized. CSV file created at: {self.log_filepath}")
            except Exception as e:
                print(f"[MPCVisualizer] Failed to initialize CSV logging: {e}")
                self.enable_logging = False

        # Setup Figure and Subplots
        # 1-row, 2-column layout: Left for the Map, Right for the Control Horizon
        self.fig, (self.ax_map, self.ax_ctrl) = plt.subplots(1, 2, figsize=(14, 7))

        # Setup Map axis
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, which='both', linestyle='--', alpha=0.5)
        self.ax_map.set_xlim([-10, 70])
        self.ax_map.set_ylim([-10, 50])
        self.ax_map.set_xlabel('Y Coordinate (meters) - East/Lateral', fontsize=11)
        self.ax_map.set_ylabel('X Coordinate (meters) - North/Longitudinal', fontsize=11)
        self.ax_map.set_title('Ship Trajectory & State Prediction Overlay', fontsize=13, fontweight='bold')

        # Setup Control Horizon axis
        self.ax_ctrl.grid(True, linestyle='--', alpha=0.5)
        self.ax_ctrl.set_title('Control Horizon Inputs (Planned)', fontsize=13, fontweight='bold')
        self.ax_ctrl.set_xlabel('Horizon Step', fontsize=11)
        self.ax_ctrl_right = self.ax_ctrl.twinx() # Secondary y-axis for propeller speed

        # Ship geometry parameters (meters)
        self.L_a = 2.5  # Nose length
        self.L_b = 1.2  # Stern length
        self.W_s = 0.8  # Semi-width

        # Map Plot Elements
        self.ship_patch = None
        self.pred_line, = self.ax_map.plot([], [], 'g--', linewidth=2, label='NMPC Spatial Prediction')
        self.ref_line, = self.ax_map.plot([], [], 'k:', alpha=0.4, label='Reference Path')
        self.start_marker, = self.ax_map.plot([], [], 'go', markersize=9, label='Start')
        self.goal_marker, = self.ax_map.plot([], [], 'ro', markersize=11, marker='*', label='Goal')
        self.active_wp_marker, = self.ax_map.plot([], [], marker='D', color='orange', markersize=10,
                                                   markeredgecolor='darkorange', linestyle='None',
                                                   zorder=6, label='Active Waypoint')
        self.trail_x = []
        self.trail_y = []
        self.trail_line, = self.ax_map.plot([], [], 'b-', linewidth=1.5, alpha=0.6, label='Trajectory History')

        # Telemetry overlay text on map
        self.info_text = self.ax_map.text(0.02, 0.98, "", transform=self.ax_map.transAxes,
                                         fontsize=9, fontfamily='monospace',
                                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # Current/wave compass HUD, top-right corner of the map panel: a ring with an
        # arrow pointing in the earth-frame direction, magnitude printed underneath.
        # Screen axes match the main plot's (East=x, North=y) convention throughout.
        self.ax_current = self.ax_map.inset_axes([0.80, 0.80, 0.17, 0.17])
        self.ax_wave = self.ax_map.inset_axes([0.80, 0.58, 0.17, 0.17])
        self.current_arrow, self.current_label = self._init_compass(self.ax_current, 'CURRENT', '#1f9fd6')
        self.wave_arrow, self.wave_label = self._init_compass(self.ax_wave, 'WAVE', '#e6659a')
        # 2nd arrow on the SAME current ring (no new inset/title/label -- just
        # another needle) for the UKF-predicted current, matching rviz_node.py's/
        # hud_visualizer.py's "| predicted" convention (appended to current_label
        # in _update_compass instead of a separate text object).
        self.ukf_current_arrow = self._add_arrow(self.ax_current, '#b34dff')

        # Horizon Plot Elements
        self.line_rudder, = self.ax_ctrl.plot([], [], 'g-s', markersize=4, label='Rudder δ (deg)')
        self.line_rps, = self.ax_ctrl_right.plot([], [], 'r-o', markersize=4, label='Propeller n_p (rps)')

        self.ax_ctrl.set_ylabel('Rudder Angle (deg)', color='g')
        self.ax_ctrl_right.set_ylabel('Propeller Speed (rps)', color='r')

        # Handle close window event to stop simulation
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        self.obstacle_patches = []
        self.fig.tight_layout()

    def on_close(self, event):
        self.is_running = False
        if self.csv_file:
            try:
                self.csv_file.close()
                print(f"[MPCVisualizer] Log file closed cleanly: {self.log_filepath}")
            except Exception as e:
                print(f"[MPCVisualizer] Error closing log file: {e}")
        print("Visualizer window closed. Exiting sim...")

    def __del__(self):
        if hasattr(self, 'csv_file') and self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass


    def _init_compass(self, ax, title, color):
        """Builds a static ring + centered arrow/label pair on a small inset axes;
        returns (arrow_annotation, label_text) for _update_compass() to drive per-frame."""
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.3, 1.15)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=7, pad=2)
        ax.patch.set_alpha(0.85)
        ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor='gray', linewidth=1.0))
        arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle='-|>', color=color, linewidth=2))
        arrow.set_visible(False)
        label = ax.text(0, -1.3, '', fontsize=7, ha='center', va='top')
        return arrow, label

    def _add_arrow(self, ax, color):
        """A second needle on an ALREADY-initialized compass ring (_init_compass
        was already called for this ax) -- no new ring/title/label, just another
        arrow annotation, for overlaying a UKF-predicted reading on the same
        compass instead of adding a whole new inset."""
        arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle='-|>', color=color, linewidth=2.0))
        arrow.set_visible(False)
        return arrow

    def _update_compass(self, arrow, label, screen_dx, screen_dy, magnitude, unit,
                         ukf_arrow=None, ukf_screen_dx=None, ukf_screen_dy=None, ukf_magnitude=None):
        """Points the arrow toward (screen_dx, screen_dy) at fixed ring radius (direction
        only -- magnitude is not arrow length, since current/wave magnitudes are tiny and
        unbounded) and prints the magnitude below. ukf_arrow/ukf_* (optional): a 2nd
        needle plus "| <predicted>" appended straight after the actual magnitude in the
        SAME label, from /ukf/estimated_current -- no new panel/label object, matching
        rviz_node.py's/hud_visualizer.py's convention."""
        if magnitude > 1e-6:
            arrow.xy = (0.85 * screen_dx / magnitude, 0.85 * screen_dy / magnitude)
            arrow.set_visible(True)
        else:
            arrow.set_visible(False)

        if ukf_arrow is not None and ukf_magnitude is not None:
            if ukf_magnitude > 1e-6:
                ukf_arrow.xy = (0.85 * ukf_screen_dx / ukf_magnitude, 0.85 * ukf_screen_dy / ukf_magnitude)
                ukf_arrow.set_visible(True)
            else:
                ukf_arrow.set_visible(False)
            label.set_text(f'{magnitude:.2f} | {ukf_magnitude:.2f} {unit}')
        else:
            if ukf_arrow is not None:
                ukf_arrow.set_visible(False)
            label.set_text(f'{magnitude:.2f} {unit}')

    def draw_ship_polygon(self, x, y, psi):
        """
        Creates a triangle/wedge polygon aligned with heading psi.
        x is North (vertical on plot), y is East (horizontal on plot).
        """
        x1 = x + self.L_a * np.cos(psi)
        y1 = y + self.L_a * np.sin(psi)

        x2 = x - self.L_b * np.cos(psi) + self.W_s * np.sin(psi)
        y2 = y - self.L_b * np.sin(psi) - self.W_s * np.cos(psi)

        x3 = x - self.L_b * np.cos(psi) - self.W_s * np.sin(psi)
        y3 = y - self.L_b * np.sin(psi) + self.W_s * np.cos(psi)

        return np.array([[y1, x1], [y2, x2], [y3, x3]])

    def init_plot(self):
        self.pred_line.set_data([], [])
        self.ref_line.set_data([], [])
        self.trail_line.set_data([], [])
        self.line_rudder.set_data([], [])
        self.line_rps.set_data([], [])
        self.active_wp_marker.set_data([], [])
        return (self.pred_line, self.ref_line, self.trail_line, self.info_text,
                self.line_rudder, self.line_rps, self.active_wp_marker,
                self.current_arrow, self.current_label, self.ukf_current_arrow, self.wave_arrow, self.wave_label)

    def update_plot(self, frame):
        if not self.is_running:
            return (self.pred_line, self.ref_line, self.trail_line, self.info_text,
                    self.line_rudder, self.line_rps, self.active_wp_marker,
                    self.current_arrow, self.current_label, self.ukf_current_arrow, self.wave_arrow, self.wave_label)

        snap = self.bridge.snapshot()
        u, v, r, x, y, psi = snap.ship_state

        # Current/wave compass HUD: direction only (arrow), magnitude as text.
        # Screen (East, North) == data (y, x), same convention as the ship/trail plot.
        cur = snap.current
        ukf_cur = snap.ukf_current
        self._update_compass(
            self.current_arrow, self.current_label,
            cur.vy, cur.vx, cur.speed if cur.enabled else 0.0, 'm/s',
            ukf_arrow=self.ukf_current_arrow,
            ukf_screen_dx=ukf_cur.vy if ukf_cur is not None else None,
            ukf_screen_dy=ukf_cur.vx if ukf_cur is not None else None,
            ukf_magnitude=ukf_cur.speed if ukf_cur is not None else None)
        wav = snap.wave
        wave_mag = float(np.hypot(wav.fx, wav.fy)) if wav.enabled else 0.0
        self._update_compass(self.wave_arrow, self.wave_label, wav.fy, wav.fx, wave_mag, 'N')

        # Update map history
        self.trail_x.append(x)
        self.trail_y.append(y)
        self.trail_line.set_data(self.trail_y, self.trail_x)

        # Draw ship
        vertices = self.draw_ship_polygon(x, y, psi)
        if self.ship_patch is not None:
            self.ship_patch.remove()
        self.ship_patch = Polygon(vertices, facecolor='royalblue', edgecolor='darkblue', zorder=5)
        self.ax_map.add_patch(self.ship_patch)

        # Update predicted trajectory overlay on the map
        if len(snap.predicted_trajectory) > 1:
            pred_y = snap.predicted_trajectory[:, 4]  # Y coordinate
            pred_x = snap.predicted_trajectory[:, 3]  # X coordinate
            self.pred_line.set_data(pred_y, pred_x)
        else:
            self.pred_line.set_data([], [])

        # Reference path
        if len(snap.reference_path) > 0:
            ref_y = snap.reference_path[:, 1]
            ref_x = snap.reference_path[:, 0]
            self.ref_line.set_data(ref_y, ref_x)

        self.start_marker.set_data([snap.start[1]], [snap.start[0]])
        self.goal_marker.set_data([snap.goal[1]], [snap.goal[0]])

        # Active waypoint: the intermediate leg target the NMPC is currently
        # steering toward, distinct from the overall start/goal on multi-leg paths.
        if snap.active_waypoint is not None:
            self.active_wp_marker.set_data([snap.active_waypoint[1]], [snap.active_waypoint[0]])
        else:
            self.active_wp_marker.set_data([], [])

        # Redraw obstacles
        for p in self.obstacle_patches:
            p.remove()
        self.obstacle_patches = []

        nearest_obstacle_dist = np.inf
        ship_pos = np.array([x, y])
        for obs in snap.obstacles:
            circle = Circle((obs.y, obs.x), obs.radius, color='red', alpha=0.35, ec='darkred', zorder=3)
            self.ax_map.add_patch(circle)
            self.obstacle_patches.append(circle)

            dist = np.linalg.norm(ship_pos - np.array([obs.x, obs.y])) - obs.radius
            if dist < nearest_obstacle_dist:
                nearest_obstacle_dist = dist

        goal_dist = np.linalg.norm(ship_pos - np.array(snap.goal))
        if snap.active_waypoint is not None:
            active_wp_dist = np.linalg.norm(ship_pos - np.array(snap.active_waypoint))
        else:
            active_wp_dist = goal_dist

        # Get active control input to show in telemetry
        if len(snap.control_horizon) > 0:
            delta_curr = snap.control_horizon[0, 0]
            rps_curr = snap.control_horizon[0, 1]
        else:
            delta_curr, rps_curr = 0.0, 0.0

        # Live CSV Logging
        if self.enable_logging and self.csv_writer:
            if self.last_logged_timestamp is None or snap.timestamp != self.last_logged_timestamp:
                self.last_logged_timestamp = snap.timestamp

                wall_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                try:
                    pred_traj_list = snap.predicted_trajectory.tolist()
                except Exception:
                    pred_traj_list = []

                try:
                    ctrl_horiz_list = snap.control_horizon.tolist()
                except Exception:
                    ctrl_horiz_list = []

                obstacles_list = [
                    {
                        "id": obs.id,
                        "x": float(obs.x),
                        "y": float(obs.y),
                        "radius": float(obs.radius),
                        "static": bool(obs.static)
                    }
                    for obs in snap.obstacles
                ]

                row = [
                    wall_time_str,
                    self.run_name,
                    f"{snap.timestamp:.3f}",
                    f"{u:.6f}",
                    f"{v:.6f}",
                    f"{r:.6f}",
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{psi:.6f}",
                    f"{np.rad2deg(psi):.3f}",
                    f"{snap.start[0]:.6f}",
                    f"{snap.start[1]:.6f}",
                    f"{snap.goal[0]:.6f}",
                    f"{snap.goal[1]:.6f}",
                    json.dumps(pred_traj_list),
                    json.dumps(ctrl_horiz_list),
                    json.dumps(obstacles_list)
                ]

                try:
                    self.csv_writer.writerow(row)
                    self.csv_file.flush()
                except Exception as e:
                    print(f"[MPCVisualizer] Error writing to CSV: {e}")

        # "| <predicted>" appended straight after X/Y's actual value, from
        # /ukf/estimated_state -- same lines, no new rows, matching the compass
        # HUD's/rviz_node.py's actual-vs-UKF convention. Omitted (falls back to
        # actual-only) until the first UKF message arrives.
        ukf_pos = snap.ukf_position
        if ukf_pos is not None:
            x_line = f"X         : {x:6.2f} | {ukf_pos[0]:6.2f} m\n"
            y_line = f"Y         : {y:6.2f} | {ukf_pos[1]:6.2f} m\n"
        else:
            x_line = f"X         : {x:6.2f} m\n"
            y_line = f"Y         : {y:6.2f} m\n"

        # Update telemetry overlay
        text = (
            f"=== SHIP STATUS ===\n"
            f"{x_line}"
            f"{y_line}"
            f"u (Surge) : {u:6.3f} m/s\n"
            f"v (Sway)  : {v:6.3f} m/s\n"
            f"r (Yaw Rt): {r:6.3f} rad/s\n"
            f"Heading   : {np.rad2deg(psi):6.1f} deg\n\n"
            f"=== CONTROLS ===\n"
            f"Rudder    : {np.rad2deg(delta_curr):6.1f} deg\n"
            f"Propeller : {rps_curr:6.1f} rps\n\n"
            f"=== TARGETS ===\n"
            f"Active WP : {active_wp_dist:6.2f} m\n"
            f"Goal Dist : {goal_dist:6.2f} m\n"
            f"Obs Dist  : {nearest_obstacle_dist:6.2f} m"
        )
        self.info_text.set_text(text)

        # Update Control Horizon plot
        steps_ctrl = np.arange(len(snap.control_horizon))
        if len(steps_ctrl) > 0:
            rudders_deg = np.rad2deg(snap.control_horizon[:, 0])
            rpss = snap.control_horizon[:, 1]

            self.line_rudder.set_data(steps_ctrl, rudders_deg)
            self.line_rps.set_data(steps_ctrl, rpss)

            # Rescale axis bounds dynamically
            self.ax_ctrl.set_xlim([0, max(1, len(steps_ctrl) - 1)])
            self.ax_ctrl.set_ylim([min(rudders_deg) - 2.0, max(rudders_deg) + 2.0])
            self.ax_ctrl_right.set_ylim([min(rpss) - 1.0, max(rpss) + 1.0])

        if frame == 0:
            # 'lower right': 'upper right' is now the current/wave compass HUD's corner.
            self.ax_map.legend(loc='lower right', fontsize=8)
            lines_ctrl = [self.line_rudder, self.line_rps]
            self.ax_ctrl.legend(lines_ctrl, [l.get_label() for l in lines_ctrl], loc='upper right', fontsize=8)

        return (self.pred_line, self.ref_line, self.trail_line, self.info_text,
                self.line_rudder, self.line_rps, self.active_wp_marker,
                self.current_arrow, self.current_label, self.ukf_current_arrow, self.wave_arrow, self.wave_label)

    def start(self):
        self.anim = FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot,
                                  interval=self.update_interval, blit=True)
        plt.show()
