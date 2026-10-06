"""
Live GUI for building NMPC scenarios: click to place a start point, an end
point, and any number of intermediate waypoints in between; drag to "paint"
circular obstacles at a coordinate with a chosen radius, click-click to
place a wall (a padded line segment / capsule -- for harbor walls, quays,
breakwaters), or drag to place an elliptical obstacle (e.g. another vessel --
drag direction/length set orientation/semi-major axis, a fixed semi-minor
axis comes from a textbox; see research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md).

Any placed circle/wall/ellipse can be given a constant velocity: switch to
Velocity mode, click the obstacle to select it (its current speed/heading
populate the sidebar boxes), type a new Speed (m/s) / Heading (deg, 0=N,
+90=E), and press "Apply Velocity". Velocity is an ATTRIBUTE of the existing
obstacle -- there is no separate "moving obstacle" shape -- so a wall
translates rigidly (both endpoints share the same vx/vy) and an ellipse's own
orientation (theta) stays independent of its heading of travel (drifting
sideways is a valid, common case, e.g. current-set debris). See
research_papers/COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md and
nmpc/moving_obstacle.py; this is prediction/visualization-only, NEVER fed
into the NMPC solver.

ObstacleShip mode places a different kind of object entirely: an ellipse-
shaped "obstacle ship," driven live by obstacle_ship_node's own MMG
simulation (+ a separate WASD teleop node) rather than by scenario-authored
velocity -- drag to place/orient (same gesture as Ellipse mode: direction ->
initial heading psi0, length -> semi-major axis a), or plain-click an
existing one to select it, then "Toggle Ship Enabled" to include/exclude it
from the next run without deleting it. Also NEVER fed into the NMPC solver,
and never published on the same topics as a real obstacle/wall/ellipse --
see this feature's plan / nmpc/README.md's "Planned: Nomoto-driven moving
obstacles" note.

Saves everything to scenario.json in the same dict shape as one entry of
SCENARIOS in nmpc/run_live.py (waypoints / mmg_init / sim_time), plus
"obstacles", "walls", and "ellipses" keys that map_node/nmpc_node consume
directly -- each row now carries trailing (vx, vy) fields (default 0.0, 0.0
= stationary); _pad_row below keeps loading scenario files authored before
those fields existed. A separate "obstacle_ships" key (rows: [xc, yc, a, b,
psi0, enabled]) holds ObstacleShip mode's placements -- map_node reads it,
but never republishes it on /map/obstacles|walls|ellipses (nmpc_node never
sees it at all).

Deliberately zero-dependency on the rest of the repo, same as
mpc_visualization/: no CasADi, no Acados, no nmpc/ imports. Standalone tool.

Run: python scenario_maker/scenario_editor.py [--out PATH] [--load]

Axis convention matches mpc_visualization/visualizer.py: X is
North/Longitudinal (vertical on screen), Y is East/Lateral (horizontal on
screen). All points are stored/saved as (x, y) tuples in that order, same as
SCENARIOS' waypoints lists.
"""
import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, TextBox
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MODES = ["Start", "Waypoint", "Goal", "Obstacle", "Wall", "Ellipse", "Velocity", "ObstacleShip", "Remove"]


def _find_default_scenario_path() -> str:
    """Finds src/nmpc_sim_nodes/params/scenario.json by walking upward looking
    for the workspace root containing src/nmpc_sim_nodes. Avoids hardcoded
    relative depths which break when installed without --symlink-install.
    """
    curr = os.path.dirname(os.path.abspath(__file__))
    while curr and curr != os.path.dirname(curr):
        candidate_ws = os.path.join(curr, "src", "nmpc_sim_nodes", "params", "scenario.json")
        if os.path.exists(candidate_ws):
            return os.path.normpath(candidate_ws)
        candidate_repo = os.path.join(curr, "nmpc_ws", "src", "nmpc_sim_nodes", "params", "scenario.json")
        if os.path.exists(candidate_repo):
            return os.path.normpath(candidate_repo)
        curr = os.path.dirname(curr)

    try:
        from ament_index_python.packages import get_package_share_directory
        share_path = os.path.join(get_package_share_directory("nmpc_sim_nodes"), "params", "scenario.json")
        if os.path.exists(share_path):
            return os.path.normpath(share_path)
    except Exception:
        pass

    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "nmpc_sim_nodes", "params", "scenario.json"))


DEFAULT_OUT = _find_default_scenario_path()

_VELOCITY_EPS = 1e-6  # below this speed [m/s], an obstacle/wall/ellipse draws as stationary


def _pad_row(row, n: int):
    """Backward-compat: scenario rows authored before vx/vy existed are
    shorter (3-tuple obstacle, 5-tuple wall/ellipse) -- pad with trailing 0.0
    (stationary) so every row this editor works with is always the current,
    full-length shape. A no-op for rows already at length n. Duplicated (not
    imported) from map_node.py's identical helper -- this file is
    deliberately zero-dependency on the rest of the repo, see module
    docstring."""
    row = list(row)
    while len(row) < n:
        row.append(0.0)
    return row


def _point_to_segment_dist(px, py, x0, y0, x1, y1):
    """Plain-numpy twin of nmpc/path_following.py's capsule_distance_casadi
    (minus radius padding, minus CasADi symbolics) -- used only for the
    editor's nearest-item hit-testing, not the solver."""
    ex, ey = x1 - x0, y1 - y0
    denom = ex * ex + ey * ey
    t = 0.0 if denom < 1e-9 else max(0.0, min(1.0, ((px - x0) * ex + (py - y0) * ey) / denom))
    cx, cy = x0 + t * ex, y0 + t * ey
    return float(np.hypot(px - cx, py - cy))


def _capsule_polygon_plot_xy(x0, y0, x1, y1, r):
    """Rectangle body of a capsule footprint, in PLOT coords (East, North) --
    the two rounded end caps are drawn separately as Circle patches at the
    segment's endpoints (see _redraw). Points are in storage (x, y) =
    (North, East); this returns (plotX, plotY) = (East, North) vertices."""
    ex, ey = x1 - x0, y1 - y0
    length = float(np.hypot(ex, ey))
    if length < 1e-9:
        return []
    perp_x, perp_y = -ey / length, ex / length  # unit normal, in storage (x, y) coords
    corners_storage = [
        (x0 + perp_x * r, y0 + perp_y * r),
        (x1 + perp_x * r, y1 + perp_y * r),
        (x1 - perp_x * r, y1 - perp_y * r),
        (x0 - perp_x * r, y0 - perp_y * r),
    ]
    return [(cy, cx) for cx, cy in corners_storage]  # (x, y) -> plot (East, North)


class ScenarioEditor:
    def __init__(self, out_path=DEFAULT_OUT, xlim=(-30, 60), ylim=(-20, 60),
                 sim_time=800.0, u_init=0.1, psi_init=0.0, default_obstacle_radius=2.0,
                 default_wall_radius=1.5, default_ellipse_b=1.5):
        self.out_path = out_path
        self.sim_time = sim_time
        self.u_init = u_init
        self.psi_init = psi_init  # initial heading, DEGREES (compass-style: 0=North, +90=East,
        # matching compute_path_angle's atan2(dEast, dNorth) convention) -- converted to/from
        # radians only at save/load time, since mmg_init's psi slot is radians like everywhere
        # else in nmpc/.
        self.default_obstacle_radius = default_obstacle_radius
        self.default_wall_radius = default_wall_radius
        self.default_ellipse_b = default_ellipse_b  # fixed semi-minor axis (m); semi-major axis
        # and orientation instead come from the placement drag itself (see _on_release)

        # --- scenario data (all points stored as (x, y) = (North, East)) ---
        self.start = None
        self.goal = None
        self.waypoints = []      # intermediate points only, in click order
        self.obstacles = []      # [x, y, radius, vx, vy]
        self.walls = []          # [x0, y0, x1, y1, radius, vx, vy] capsule (wall) obstacles --
        # see research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md
        self.ellipses = []       # [xc, yc, a, b, theta, vx, vy] elliptical obstacles, e.g. other
        # vessels -- theta in RADIANS, world frame (same convention as every other physics-facing
        # angle in nmpc/, e.g. mmg_init's psi -- no degrees conversion needed since theta is set
        # purely by the drag gesture, never typed into a textbox). Same design doc as walls above.
        # vx/vy (earth-frame m/s, default 0.0 = stationary) on all three: set via Velocity mode
        # (select, then Speed/Heading boxes + Apply Velocity), independent of the placement
        # gesture -- see module docstring and research_papers/COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md.
        self.obstacle_ships = []  # [xc, yc, a, b, psi0, enabled] -- an ellipse-shaped, keyboard/
        # hardware-driven "obstacle ship": placed/oriented exactly like Ellipse mode (drag
        # direction -> psi0, drag length -> a, b from ellipseb_box), but NEVER given vx/vy --
        # its live motion comes from obstacle_ship_node's own MMG simulation, not scenario-
        # authored velocity. `enabled` (0.0/1.0) gates whether map_node spawns it for a run;
        # see this feature's plan / nmpc/README.md's "Planned: Nomoto-driven moving obstacles".

        self.mode = "start"
        self._history = []       # stack of no-arg undo callables
        self._obs_drag_start = None
        self._obs_preview = None
        self._wall_start = None  # first-click endpoint, pending the second click
        self._wall_preview = None
        self._ell_drag_start = None
        self._ell_preview = None
        self._ship_drag_start = None
        self._ship_preview = None
        self._selected = None    # (kind, idx) for Velocity mode -- kind in {"obstacle","wall","ellipse"}
        self._selected_ship = None  # index into self.obstacle_ships, for ObstacleShip mode's Toggle Enabled
        self._dynamic_artists = []

        self._build_figure(xlim, ylim)
        self._redraw()

    # ------------------------------------------------------------------
    # figure / widget layout
    # ------------------------------------------------------------------
    def _build_figure(self, xlim, ylim):
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.canvas.manager.set_window_title("NMPC Scenario Editor")

        self.ax_map = self.fig.add_axes([0.06, 0.08, 0.60, 0.86])
        self.ax_map.set_aspect("equal")
        self.ax_map.grid(True, which="both", linestyle="--", alpha=0.5)
        self.ax_map.set_xlim(xlim)
        self.ax_map.set_ylim(ylim)
        self.ax_map.set_xlabel("Y Coordinate (meters) - East/Lateral", fontsize=11)
        self.ax_map.set_ylabel("X Coordinate (meters) - North/Longitudinal", fontsize=11)
        self.ax_map.set_title(
            "Scenario Editor — click to place, drag to size; Velocity mode: click to select, "
            "type Speed/Heading, Apply",
            fontsize=11, fontweight="bold")

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Start"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=13, label="Goal"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="orange", markersize=8, label="Waypoint"),
            Line2D([0], [0], color="black", linestyle=":", label="Path"),
            Patch(facecolor="red", alpha=0.3, edgecolor="darkred", label="Obstacle"),
            Patch(facecolor="firebrick", alpha=0.35, edgecolor="darkred", label="Wall"),
            Patch(facecolor="mediumpurple", alpha=0.3, edgecolor="indigo", label="Ellipse"),
            Line2D([0], [0], color="darkorange", lw=2, label="Velocity (any obstacle)"),
            Patch(facecolor="steelblue", alpha=0.4, edgecolor="navy", label="Obstacle Ship"),
        ]
        self.ax_map.legend(handles=legend_handles, loc="upper right", fontsize=8)

        self.status_text = self.ax_map.text(
            0.02, 0.98, "", transform=self.ax_map.transAxes, fontsize=9,
            fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), zorder=10,
        )
        self.msg_text = self.fig.text(0.06, 0.02, "", fontsize=9, color="darkblue")

        # ---- sidebar widgets ----
        self.fig.text(0.71, 0.97, "Mode", fontsize=10, fontweight="bold")
        ax_mode = self.fig.add_axes([0.70, 0.75, 0.27, 0.19])
        ax_mode.set_frame_on(True)
        self.radio_mode = RadioButtons(ax_mode, MODES, active=0)
        self.radio_mode.on_clicked(self._on_mode_change)

        # Compact label+textbox rows (row_step=0.052, tighter than the original 0.07 to fit
        # the two new Velocity fields below without the sidebar overflowing the figure).
        self.simtime_box = self._add_field(0.735, "Sim Time (s)", self.sim_time, self._on_simtime_submit)
        self.uinit_box = self._add_field(0.683, "Initial Surge u (m/s)", self.u_init, self._on_uinit_submit)
        self.psiinit_box = self._add_field(0.631, "Initial Heading (deg, 0=N/+90=E)", self.psi_init,
                                            self._on_psiinit_submit, fontsize=8)
        self.obsr_box = self._add_field(0.579, "Default Obstacle R (m)", self.default_obstacle_radius,
                                         self._on_obsr_submit)
        self.wallr_box = self._add_field(0.527, "Wall R (m) -- click, click", self.default_wall_radius,
                                          self._on_wallr_submit)
        self.ellipseb_box = self._add_field(0.475, "Ellipse Semi-Minor b (m) -- drag", self.default_ellipse_b,
                                             self._on_ellipseb_submit, fontsize=8)

        self.vel_status_text = self.fig.text(0.71, 0.423, "Selected: none", fontsize=8, color="darkorange")
        self.velspeed_box = self._add_field(0.388, "Velocity Speed (m/s)", 0.0, None, fontsize=8)
        self.velheading_box = self._add_field(0.336, "Velocity Heading (deg, 0=N/+90=E)", 0.0, None, fontsize=8)

        ax_applyvel = self.fig.add_axes([0.70, 0.244, 0.27, 0.036])
        self.btn_apply_vel = Button(ax_applyvel, "Apply Velocity")
        self.btn_apply_vel.on_clicked(self._on_apply_velocity)

        # ObstacleShip mode's select-then-apply flow (select nearest ship by
        # clicking it, same convention as Velocity mode's _select_for_velocity)
        # -- flips the selected ship's enabled flag, doesn't create/remove one.
        ax_toggleship = self.fig.add_axes([0.70, 0.200, 0.27, 0.036])
        self.btn_toggle_ship = Button(ax_toggleship, "Toggle Ship Enabled")
        self.btn_toggle_ship.on_clicked(self._on_toggle_ship_enabled)

        ax_save = self.fig.add_axes([0.70, 0.156, 0.27, 0.036])
        self.btn_save = Button(ax_save, "Save scenario.json")
        self.btn_save.on_clicked(self._on_save)

        ax_load = self.fig.add_axes([0.70, 0.112, 0.27, 0.036])
        self.btn_load = Button(ax_load, "Load")
        self.btn_load.on_clicked(self._on_load)

        ax_undo = self.fig.add_axes([0.70, 0.068, 0.27, 0.036])
        self.btn_undo = Button(ax_undo, "Undo")
        self.btn_undo.on_clicked(self._on_undo)

        ax_clear = self.fig.add_axes([0.70, 0.024, 0.27, 0.036])
        self.btn_clear = Button(ax_clear, "Clear All")
        self.btn_clear.on_clicked(self._on_clear)

        self.fig.text(0.70, 0.008, f"Saves to:\n{self.out_path}", fontsize=6, color="dimgray")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        # Poll (rather than monkey-patch toolbar.pan/zoom) for the nav
        # toolbar's Pan/Zoom tool turning on: most backends (Tkinter, Qt)
        # bind the toolbar button's click callback to the bound method
        # object AT WIDGET-CONSTRUCTION TIME, so patching toolbar.pan/.zoom
        # afterward would silently never fire -- polling toolbar.mode
        # directly sidesteps that entirely, backend-agnostic.
        self._last_toolbar_mode = ""
        self._toolbar_poll_timer = self.fig.canvas.new_timer(interval=150)
        self._toolbar_poll_timer.add_callback(self._poll_toolbar_mode)
        self._toolbar_poll_timer.start()

    def _add_field(self, y, label_text, initial, on_submit, fontsize=9):
        """One label+TextBox row at label y-coordinate y (box sits just below
        it) -- see _build_figure's row_step comment for the layout budget
        this was fit against."""
        self.fig.text(0.71, y, label_text, fontsize=fontsize)
        ax = self.fig.add_axes([0.70, y - 0.04, 0.27, 0.032])
        box = TextBox(ax, "", initial=str(initial))
        if on_submit is not None:
            box.on_submit(on_submit)
        return box

    # ------------------------------------------------------------------
    # mouse handlers
    # ------------------------------------------------------------------
    def _event_point(self, event):
        """matplotlib gives (Y, X) in data coords (plot x=East, plot y=North);
        scenario points are stored as (X, Y) = (North, East)."""
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return None
        return (float(event.ydata), float(event.xdata))

    def _get_toolbar_mode(self) -> str:
        manager = getattr(self.fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None) if manager is not None else None
        return str(getattr(toolbar, "mode", "")) if toolbar is not None else ""

    def _poll_toolbar_mode(self):
        mode = self._get_toolbar_mode()
        if mode != self._last_toolbar_mode:
            self._last_toolbar_mode = mode
            if mode:  # Pan or Zoom just became active -- suspend placement
                self._deselect_mode_radio()

    def _deselect_mode_radio(self):
        """Visually clears the Mode radio buttons (no dot shown) without
        firing on_clicked -- matplotlib's RadioButtons has no public "clear
        selection" method, so this reaches into its internal PathCollection
        the same way RadioButtons.set_active() itself does, just leaving
        every button transparent instead of activating one. Guarded so a
        future matplotlib version at worst loses the visual cue, not the
        actual placement-blocking (see _on_press's own toolbar-mode check,
        which is what really matters)."""
        self.mode = None
        try:
            from matplotlib import colors as mcolors
            facecolors = self.radio_mode._buttons.get_facecolor()
            facecolors[:] = mcolors.to_rgba("none")
            self.radio_mode._buttons.set_facecolor(facecolors)
            self.radio_mode.value_selected = None
        except AttributeError:
            pass
        self._flash("Pan/Zoom active -- placement paused (pick a Mode to resume)")
        self._update_status_text()
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if self._get_toolbar_mode():
            return  # Pan/Zoom is active -- let the toolbar handle the click, don't also place
        pt = self._event_point(event)
        if pt is None or event.button != 1:
            return

        if self.mode == "obstacle":
            self._obs_drag_start = pt
            self._obs_preview = Circle((pt[1], pt[0]), 0.0, facecolor="red", alpha=0.25,
                                        edgecolor="darkred", linestyle="--", zorder=3)
            self.ax_map.add_patch(self._obs_preview)
            self.fig.canvas.draw_idle()
            return

        if self.mode == "ellipse":
            # single press-drag-release, like Obstacle mode -- but the drag
            # vector's DIRECTION becomes theta and its LENGTH becomes the
            # semi-major axis a, both in one motion (a drag naturally carries
            # both). Semi-minor axis b is fixed, from ellipseb_box.
            self._ell_drag_start = pt
            self._ell_preview = Ellipse((pt[1], pt[0]), width=0.0, height=2.0 * self.default_ellipse_b,
                                         angle=0.0, facecolor="mediumpurple", alpha=0.25,
                                         edgecolor="indigo", linestyle="--", zorder=3)
            self.ax_map.add_patch(self._ell_preview)
            self.fig.canvas.draw_idle()
            return

        if self.mode == "obstacleship":
            # Same press-drag-release gesture as Ellipse mode (direction ->
            # psi0, length -> a) -- but a SHORT drag/plain click near an
            # EXISTING obstacle ship selects it instead of placing a new
            # default-sized one (see _on_release), so this one mode covers
            # both "place new" and "select for Toggle Enabled" without a
            # separate mode, mirroring Velocity mode's own select-by-click
            # convention for the latter.
            self._ship_drag_start = pt
            self._ship_preview = Ellipse((pt[1], pt[0]), width=0.0, height=2.0 * self.default_ellipse_b,
                                          angle=0.0, facecolor="steelblue", alpha=0.3,
                                          edgecolor="navy", linestyle="--", zorder=3)
            self.ax_map.add_patch(self._ship_preview)
            self.fig.canvas.draw_idle()
            return

        if self.mode == "wall":
            # two-click placement (not press-drag-release like obstacles): a
            # wall has no natural "drag distance" to reuse for its radius, so
            # the first click sets p0 and the second sets p1; radius comes
            # from wallr_box instead.
            if self._wall_start is None:
                self._wall_start = pt
                self._wall_preview, = self.ax_map.plot(
                    [pt[1], pt[1]], [pt[0], pt[0]], "--", color="darkred", linewidth=1.5, zorder=3)
                self.fig.canvas.draw_idle()
            else:
                sx, sy = self._wall_start
                if self._wall_preview is not None:
                    self._wall_preview.remove()
                    self._wall_preview = None
                length = float(np.hypot(pt[0] - sx, pt[1] - sy))
                if length < 0.3:
                    self._flash("Wall too short (use Obstacle mode for circles)")
                else:
                    self.walls.append([sx, sy, pt[0], pt[1], self.default_wall_radius, 0.0, 0.0])
                    self._push_undo(lambda: self.walls.pop() if self.walls else None)
                self._wall_start = None
                self._redraw()
            return

        if self.mode == "velocity":
            self._select_for_velocity(pt)
            return

        if self.mode == "remove":
            self._remove_nearest(pt)
        elif self.mode == "start":
            old = self.start
            self.start = pt
            self._push_undo(lambda: setattr(self, "start", old))
        elif self.mode == "goal":
            old = self.goal
            self.goal = pt
            self._push_undo(lambda: setattr(self, "goal", old))
        elif self.mode == "waypoint":
            self.waypoints.append(pt)
            self._push_undo(lambda: self.waypoints.pop() if self.waypoints else None)

        self._redraw()

    def _on_motion(self, event):
        if self._wall_start is not None:
            pt = self._event_point(event)
            if pt is not None and self._wall_preview is not None:
                sx, sy = self._wall_start
                self._wall_preview.set_data([sy, pt[1]], [sx, pt[0]])
                self.fig.canvas.draw_idle()
            return

        if self._ell_drag_start is not None:
            pt = self._event_point(event)
            if pt is not None and self._ell_preview is not None:
                sx, sy = self._ell_drag_start
                dx_storage, dy_storage = pt[0] - sx, pt[1] - sy
                a = float(np.hypot(dx_storage, dy_storage))
                theta = float(np.arctan2(dy_storage, dx_storage))  # bearing: 0=North, +90=East
                plot_angle_deg = float(np.degrees(np.pi / 2.0 - theta))  # axis-swap compensation,
                # same plot_yaw = pi/2 - theta convention rviz_node.py's _yaw_quat documents
                self._ell_preview.set_center((sy, sx))
                self._ell_preview.width = 2.0 * a
                self._ell_preview.angle = plot_angle_deg
                self.fig.canvas.draw_idle()
            return

        if self._ship_drag_start is not None:
            pt = self._event_point(event)
            if pt is not None and self._ship_preview is not None:
                sx, sy = self._ship_drag_start
                dx_storage, dy_storage = pt[0] - sx, pt[1] - sy
                a = float(np.hypot(dx_storage, dy_storage))
                theta = float(np.arctan2(dy_storage, dx_storage))
                plot_angle_deg = float(np.degrees(np.pi / 2.0 - theta))
                self._ship_preview.set_center((sy, sx))
                self._ship_preview.width = 2.0 * a
                self._ship_preview.angle = plot_angle_deg
                self.fig.canvas.draw_idle()
            return

        if self._obs_drag_start is None:
            return
        pt = self._event_point(event)
        if pt is None:
            return
        sx, sy = self._obs_drag_start
        r = float(np.hypot(pt[0] - sx, pt[1] - sy))
        self._obs_preview.set_radius(r)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._ship_drag_start is not None:
            pt = self._event_point(event)
            sx, sy = self._ship_drag_start
            if self._ship_preview is not None:
                self._ship_preview.remove()
                self._ship_preview = None
            if pt is None:
                self._ship_drag_start = None
                self.fig.canvas.draw_idle()
                return
            dx_storage, dy_storage = pt[0] - sx, pt[1] - sy
            a = float(np.hypot(dx_storage, dy_storage))
            if a < 0.3:
                # plain click, no drag -- select the nearest EXISTING ship
                # (for Toggle Enabled) if one is close by, same tolerance
                # convention as _select_for_velocity/_remove_nearest; only
                # place a new default-sized one if nothing is nearby.
                if self._select_nearest_ship(pt):
                    self._ship_drag_start = None
                    self._redraw()
                    return
                a, psi0 = self.default_obstacle_radius, 0.0
            else:
                psi0 = float(np.arctan2(dy_storage, dx_storage))
            self.obstacle_ships.append([sx, sy, a, self.default_ellipse_b, psi0, 1.0])
            self._push_undo(lambda: self.obstacle_ships.pop() if self.obstacle_ships else None)
            self._ship_drag_start = None
            self._redraw()
            return

        if self._ell_drag_start is not None:
            pt = self._event_point(event)
            sx, sy = self._ell_drag_start
            if self._ell_preview is not None:
                self._ell_preview.remove()
                self._ell_preview = None
            if pt is None:
                self._ell_drag_start = None
                self.fig.canvas.draw_idle()
                return
            dx_storage, dy_storage = pt[0] - sx, pt[1] - sy
            a = float(np.hypot(dx_storage, dy_storage))
            if a < 0.3:
                # plain click, no drag -> default size/orientation, same fallback
                # convention as Obstacle mode's own "< 0.3" no-drag case
                a, theta = self.default_obstacle_radius, 0.0
            else:
                theta = float(np.arctan2(dy_storage, dx_storage))
            self.ellipses.append([sx, sy, a, self.default_ellipse_b, theta, 0.0, 0.0])
            self._push_undo(lambda: self.ellipses.pop() if self.ellipses else None)
            self._ell_drag_start = None
            self._redraw()
            return

        if self._obs_drag_start is None:
            return
        pt = self._event_point(event)
        sx, sy = self._obs_drag_start

        if self._obs_preview is not None:
            self._obs_preview.remove()
            self._obs_preview = None

        if pt is None:
            self._obs_drag_start = None
            self.fig.canvas.draw_idle()
            return

        r = float(np.hypot(pt[0] - sx, pt[1] - sy))
        if r < 0.3:
            r = self.default_obstacle_radius  # plain click, no drag -> default size

        self.obstacles.append([sx, sy, r, 0.0, 0.0])
        self._push_undo(lambda: self.obstacles.pop() if self.obstacles else None)
        self._obs_drag_start = None
        self._redraw()

    def _select_for_velocity(self, pt):
        """Velocity mode's click handler: finds the nearest obstacle/wall/
        ellipse to pt (same nearest-item + tolerance pattern as
        _remove_nearest, restricted to just these three kinds since Start/
        Goal/Waypoint have no velocity concept), selects it, and populates
        the Speed/Heading boxes with its CURRENT velocity so re-selecting an
        already-moving item shows what it's currently set to."""
        x, y = pt
        candidates = []
        for i, (ox, oy, _r, _vx, _vy) in enumerate(self.obstacles):
            candidates.append((np.hypot(ox - x, oy - y), "obstacle", i))
        for i, (x0, y0, x1, y1, _r, _vx, _vy) in enumerate(self.walls):
            candidates.append((_point_to_segment_dist(x, y, x0, y0, x1, y1), "wall", i))
        for i, (xc, yc, a, b, _theta, _vx, _vy) in enumerate(self.ellipses):
            candidates.append((max(0.0, np.hypot(xc - x, yc - y) - max(a, b)), "ellipse", i))

        if not candidates:
            self._flash("No obstacles/walls/ellipses to select -- place one first")
            return

        candidates.sort(key=lambda c: c[0])
        dist, kind, idx = candidates[0]
        xlim = self.ax_map.get_xlim()
        tol = abs(xlim[1] - xlim[0]) / 25.0
        if dist > tol:
            self._flash("Click closer to an obstacle/wall/ellipse to select it")
            return

        self._selected = (kind, idx)
        lists = {"obstacle": self.obstacles, "wall": self.walls, "ellipse": self.ellipses}
        vx, vy = lists[kind][idx][-2], lists[kind][idx][-1]
        speed = float(np.hypot(vx, vy))
        heading = float(np.degrees(np.arctan2(vy, vx))) if speed > _VELOCITY_EPS else 0.0
        self.velspeed_box.set_val(str(round(speed, 3)))
        self.velheading_box.set_val(str(round(heading, 2)))
        self._flash(f"Selected {kind} #{idx} -- edit Speed/Heading, then Apply Velocity")
        self._redraw()

    def _select_nearest_ship(self, pt) -> bool:
        """ObstacleShip mode's plain-click selection: True (and sets
        self._selected_ship) if an existing obstacle ship is within the same
        tolerance _select_for_velocity/_remove_nearest use, else False (so
        the caller falls back to placing a new one)."""
        if not self.obstacle_ships:
            return False
        x, y = pt
        dists = [np.hypot(xc - x, yc - y) for xc, yc, _a, _b, _psi0, _en in self.obstacle_ships]
        idx = int(np.argmin(dists))
        xlim = self.ax_map.get_xlim()
        tol = abs(xlim[1] - xlim[0]) / 25.0
        if dists[idx] > tol:
            return False
        self._selected_ship = idx
        self._flash(f"Selected obstacle ship #{idx} -- click Toggle Ship Enabled to flip it")
        return True

    def _on_toggle_ship_enabled(self, event):
        if self._selected_ship is None or self._selected_ship >= len(self.obstacle_ships):
            self._flash("Click an obstacle ship in ObstacleShip mode first")
            return
        idx = self._selected_ship
        row = self.obstacle_ships[idx]
        old_enabled = row[5]
        row[5] = 0.0 if old_enabled else 1.0
        self._push_undo(lambda i=idx, v=old_enabled: self.obstacle_ships[i].__setitem__(5, v))
        self._flash(f"Obstacle ship #{idx} enabled={bool(row[5])}")
        self._redraw()

    def _remove_nearest(self, pt):
        x, y = pt
        candidates = []
        if self.start is not None:
            candidates.append((np.hypot(self.start[0] - x, self.start[1] - y), "start", None))
        if self.goal is not None:
            candidates.append((np.hypot(self.goal[0] - x, self.goal[1] - y), "goal", None))
        for i, (wx, wy) in enumerate(self.waypoints):
            candidates.append((np.hypot(wx - x, wy - y), "waypoint", i))
        for i, (ox, oy, _r, _vx, _vy) in enumerate(self.obstacles):
            candidates.append((np.hypot(ox - x, oy - y), "obstacle", i))
        for i, (x0, y0, x1, y1, _r, _vx, _vy) in enumerate(self.walls):
            candidates.append((_point_to_segment_dist(x, y, x0, y0, x1, y1), "wall", i))
        for i, (xc, yc, a, b, _theta, _vx, _vy) in enumerate(self.ellipses):
            # rough center-distance-minus-max(a,b) proxy -- sufficient for UI
            # hit-testing, exactness doesn't matter here (unlike the solver's
            # gradient-normalized ellipse_distance_casadi)
            candidates.append((max(0.0, np.hypot(xc - x, yc - y) - max(a, b)), "ellipse", i))
        for i, (xc, yc, a, b, _psi0, _en) in enumerate(self.obstacle_ships):
            candidates.append((max(0.0, np.hypot(xc - x, yc - y) - max(a, b)), "obstacle_ship", i))

        if not candidates:
            self._flash("Nothing to remove")
            return

        candidates.sort(key=lambda c: c[0])
        dist, kind, idx = candidates[0]
        xlim = self.ax_map.get_xlim()
        tol = abs(xlim[1] - xlim[0]) / 25.0
        if dist > tol:
            self._flash("Click closer to an item to remove it")
            return

        if kind == "start":
            old = self.start
            self.start = None
            self._push_undo(lambda: setattr(self, "start", old))
        elif kind == "goal":
            old = self.goal
            self.goal = None
            self._push_undo(lambda: setattr(self, "goal", old))
        elif kind == "waypoint":
            old = self.waypoints.pop(idx)
            self._push_undo(lambda i=idx, v=old: self.waypoints.insert(i, v))
        elif kind == "obstacle":
            old = self.obstacles.pop(idx)
            self._push_undo(lambda i=idx, v=old: self.obstacles.insert(i, v))
        elif kind == "wall":
            old = self.walls.pop(idx)
            self._push_undo(lambda i=idx, v=old: self.walls.insert(i, v))
        elif kind == "ellipse":
            old = self.ellipses.pop(idx)
            self._push_undo(lambda i=idx, v=old: self.ellipses.insert(i, v))
        elif kind == "obstacle_ship":
            old = self.obstacle_ships.pop(idx)
            self._push_undo(lambda i=idx, v=old: self.obstacle_ships.insert(i, v))
        if self._selected == (kind, idx):
            self._selected = None  # the selected item was just removed
        if kind == "obstacle_ship" and self._selected_ship == idx:
            self._selected_ship = None
        self._flash(f"Removed {kind}")

    # ------------------------------------------------------------------
    # widget callbacks
    # ------------------------------------------------------------------
    def _on_mode_change(self, label):
        self._deactivate_toolbar_tools()
        self.mode = label.lower()
        self._selected = None  # avoid a stale selection surviving a mode switch
        self._selected_ship = None
        self._update_status_text()
        self._redraw()

    def _deactivate_toolbar_tools(self):
        """Turn off the nav toolbar's Pan/Zoom tool whenever a Mode button
        is picked -- otherwise clicks in ax_map would keep panning/zooming
        instead of placing points (see _on_press's toolbar-mode guard), and
        the toolbar button would stay visually "pressed" despite placement
        being active again. toolbar.pan()/.zoom() are TOGGLES, so only call
        the one matching the currently-active mode."""
        manager = getattr(self.fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None) if manager is not None else None
        if toolbar is None:
            return
        mode = str(getattr(toolbar, "mode", "")).lower()
        if "pan" in mode and hasattr(toolbar, "pan"):
            toolbar.pan()
        elif "zoom" in mode and hasattr(toolbar, "zoom"):
            toolbar.zoom()

    def _on_simtime_submit(self, text):
        try:
            self.sim_time = float(text)
        except ValueError:
            self._flash("Sim time must be a number")
        self._update_status_text()
        self.fig.canvas.draw_idle()

    def _on_uinit_submit(self, text):
        try:
            self.u_init = float(text)
        except ValueError:
            self._flash("Initial surge must be a number")
        self._update_status_text()
        self.fig.canvas.draw_idle()

    def _on_psiinit_submit(self, text):
        try:
            self.psi_init = float(text)
        except ValueError:
            self._flash("Initial heading must be a number (degrees)")
        self._redraw()

    def _on_obsr_submit(self, text):
        try:
            self.default_obstacle_radius = float(text)
        except ValueError:
            self._flash("Obstacle radius must be a number")

    def _on_wallr_submit(self, text):
        try:
            self.default_wall_radius = float(text)
        except ValueError:
            self._flash("Wall radius must be a number")

    def _on_ellipseb_submit(self, text):
        try:
            self.default_ellipse_b = float(text)
        except ValueError:
            self._flash("Ellipse semi-minor axis must be a number")

    def _on_apply_velocity(self, event):
        if self._selected is None:
            self._flash("Click an obstacle/wall/ellipse in Velocity mode first")
            return
        try:
            speed = float(self.velspeed_box.text)
            heading_deg = float(self.velheading_box.text)
        except ValueError:
            self._flash("Speed/heading must be numbers")
            return

        heading_rad = np.deg2rad(heading_deg)
        # (x, y) = (North, East) storage convention -- same compass bearing
        # (0=North, +90=East) as psi_init/compute_path_angle throughout nmpc/.
        vx, vy = speed * np.cos(heading_rad), speed * np.sin(heading_rad)
        kind, idx = self._selected
        lists = {"obstacle": self.obstacles, "wall": self.walls, "ellipse": self.ellipses}
        row = lists[kind][idx]
        old = list(row)
        row[-2], row[-1] = float(vx), float(vy)
        self._push_undo(lambda k=kind, i=idx, v=old: lists[k].__setitem__(i, v))
        self._flash(f"Set {kind} #{idx} velocity: speed={speed:.2f} m/s, heading={heading_deg:.1f} deg")
        self._redraw()

    def _on_save(self, event):
        if self.start is None or self.goal is None:
            self._flash("Set both a Start and a Goal before saving!")
            self._redraw()
            return

        full_path = [self.start] + list(self.waypoints) + [self.goal]
        data = {
            "waypoints": [[round(p[0], 4), round(p[1], 4)] for p in full_path],
            "mmg_init": [
                round(self.u_init, 4), 0.0, 0.0,
                round(self.start[0], 4), round(self.start[1], 4),
                round(float(np.deg2rad(self.psi_init)), 6),
            ],
            "sim_time": self.sim_time,
            "obstacles": [[round(v, 4) for v in o] for o in self.obstacles],
            "walls": [[round(v, 4) for v in w] for w in self.walls],
            "ellipses": [[round(v, 4) for v in e] for e in self.ellipses],
            "obstacle_ships": [[round(v, 4) for v in s[:5]] + [bool(s[5])] for s in self.obstacle_ships],
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(data, f, indent=2)
        n_moving = sum(1 for o in self.obstacles if np.hypot(o[3], o[4]) > _VELOCITY_EPS) + \
            sum(1 for w in self.walls if np.hypot(w[5], w[6]) > _VELOCITY_EPS) + \
            sum(1 for e in self.ellipses if np.hypot(e[5], e[6]) > _VELOCITY_EPS)
        n_ships_enabled = sum(1 for s in self.obstacle_ships if s[5])
        self._flash(f"Saved {len(full_path)} waypoints, {len(self.obstacles)} obstacles, "
                    f"{len(self.walls)} walls, {len(self.ellipses)} ellipses ({n_moving} moving), "
                    f"{len(self.obstacle_ships)} obstacle ship(s) ({n_ships_enabled} enabled) -> {self.out_path}")

    def _on_load(self, event):
        if not os.path.exists(self.out_path):
            self._flash(f"No file at {self.out_path}")
            return
        with open(self.out_path) as f:
            data = json.load(f)

        wps = [tuple(p) for p in data.get("waypoints", [])]
        if len(wps) >= 2:
            self.start = wps[0]
            self.goal = wps[-1]
            self.waypoints = wps[1:-1]
        elif len(wps) == 1:
            self.start, self.goal, self.waypoints = wps[0], None, []

        self.obstacles = [_pad_row(o, 5) for o in data.get("obstacles", [])]
        self.walls = [_pad_row(w, 7) for w in data.get("walls", [])]
        self.ellipses = [_pad_row(e, 7) for e in data.get("ellipses", [])]
        self.obstacle_ships = [_pad_row(s, 6) for s in data.get("obstacle_ships", [])]
        self.sim_time = float(data.get("sim_time", self.sim_time))
        mmg_init = data.get("mmg_init")
        if mmg_init:
            self.u_init = float(mmg_init[0])
            if len(mmg_init) >= 6:
                self.psi_init = float(np.rad2deg(mmg_init[5]))

        self.simtime_box.set_val(str(self.sim_time))
        self.uinit_box.set_val(str(self.u_init))
        self.psiinit_box.set_val(str(round(self.psi_init, 4)))
        self._selected = None
        self._selected_ship = None
        self._history = []
        self._redraw()
        self._flash(f"Loaded {self.out_path}")

    def _on_undo(self, event):
        if not self._history:
            self._flash("Nothing to undo")
            return
        action = self._history.pop()
        action()
        self._redraw()

    def _on_clear(self, event):
        self.start, self.goal = None, None
        self.waypoints, self.obstacles, self.walls, self.ellipses = [], [], [], []
        self.obstacle_ships = []
        self._selected = None
        self._selected_ship = None
        self._history = []
        self._redraw()
        self._flash("Cleared")

    def _push_undo(self, fn):
        self._history.append(fn)

    def _flash(self, text):
        print(f"[scenario_editor] {text}")
        self.msg_text.set_text(text)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def _fmt_pt(self, p):
        return f"({p[0]:.1f}, {p[1]:.1f})" if p is not None else "not set"

    def _update_status_text(self):
        mode_label = self.mode.capitalize() if self.mode is not None else "(none -- Pan/Zoom active)"
        selected_label = f"{self._selected[0]} #{self._selected[1]}" if self._selected is not None else "none"
        text = (
            f"Mode: {mode_label}\n"
            f"Start : {self._fmt_pt(self.start)}\n"
            f"Goal  : {self._fmt_pt(self.goal)}\n"
            f"Waypoints: {len(self.waypoints)}\n"
            f"Obstacles: {len(self.obstacles)}\n"
            f"Walls    : {len(self.walls)}\n"
            f"Ellipses : {len(self.ellipses)}\n"
            f"Obstacle Ships: {len(self.obstacle_ships)} "
            f"({sum(1 for s in self.obstacle_ships if s[5])} enabled)\n"
            f"Selected : {selected_label}\n"
            f"Selected Ship: {self._selected_ship if self._selected_ship is not None else 'none'}\n"
            f"Sim Time : {self.sim_time:.1f} s\n"
            f"u_init   : {self.u_init:.2f} m/s\n"
            f"psi_init : {self.psi_init:.1f} deg"
        )
        self.status_text.set_text(text)
        self.vel_status_text.set_text(f"Selected: {selected_label}")

    def _draw_velocity_arrow(self, ox, oy, vx, vy):
        """Velocity arrow from an obstacle/wall/ellipse's reference point (ox, oy)
        -- storage (x,y)=(North,East) -> plot (East,North), same swap
        _plot_point/_yaw_quat use elsewhere in this repo (rviz_node.py)."""
        arrow = self.ax_map.annotate(
            "", xy=(oy + vy, ox + vx), xytext=(oy, ox),
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=2), zorder=6)
        self._dynamic_artists.append(arrow)

    def _redraw(self):
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists = []

        path_pts = ([self.start] if self.start else []) + self.waypoints + ([self.goal] if self.goal else [])
        if len(path_pts) >= 2:
            xs = [p[1] for p in path_pts]  # East -> plot x
            ys = [p[0] for p in path_pts]  # North -> plot y
            line, = self.ax_map.plot(xs, ys, "k:", alpha=0.5, linewidth=1.5, zorder=2)
            self._dynamic_artists.append(line)

        if self.start is not None:
            m, = self.ax_map.plot([self.start[1]], [self.start[0]], "o", color="green", markersize=10, zorder=5)
            self._dynamic_artists.append(m)
            # Initial heading arrow: psi_init is a compass bearing (0=North,
            # +90=East, matching compute_path_angle's atan2(dEast, dNorth)
            # convention), so its (North, East) direction is
            # (cos(psi), sin(psi)) -- plotted as (East, North) = (sin, cos).
            psi_rad = np.deg2rad(self.psi_init)
            arrow_len = 3.0
            dx_plot, dy_plot = arrow_len * np.sin(psi_rad), arrow_len * np.cos(psi_rad)
            arrow = self.ax_map.annotate(
                "", xy=(self.start[1] + dx_plot, self.start[0] + dy_plot),
                xytext=(self.start[1], self.start[0]),
                arrowprops=dict(arrowstyle="->", color="green", lw=2), zorder=6)
            self._dynamic_artists.append(arrow)
        if self.goal is not None:
            m, = self.ax_map.plot([self.goal[1]], [self.goal[0]], marker="*", color="red", markersize=15, zorder=5)
            self._dynamic_artists.append(m)
        for i, (wx, wy) in enumerate(self.waypoints):
            m, = self.ax_map.plot([wy], [wx], marker="D", color="orange", markersize=8, zorder=5)
            self._dynamic_artists.append(m)
            t = self.ax_map.text(wy, wx, f"  {i + 1}", fontsize=8, color="darkorange", zorder=6)
            self._dynamic_artists.append(t)

        for i, (ox, oy, orad, ovx, ovy) in enumerate(self.obstacles):
            selected = self._selected == ("obstacle", i)
            c = Circle((oy, ox), orad, facecolor="red", alpha=0.3,
                       edgecolor="yellow" if selected else "darkred",
                       linewidth=2.5 if selected else 1.0, zorder=3)
            self.ax_map.add_patch(c)
            self._dynamic_artists.append(c)
            speed = float(np.hypot(ovx, ovy))
            if speed > _VELOCITY_EPS:
                self._draw_velocity_arrow(ox, oy, ovx, ovy)
            label = f"{orad:.1f}m" + (f"\n{speed:.2f}m/s" if speed > _VELOCITY_EPS else "")
            t = self.ax_map.text(oy, ox, label, fontsize=7, ha="center", va="center", zorder=4)
            self._dynamic_artists.append(t)

        for i, (x0, y0, x1, y1, wrad, wvx, wvy) in enumerate(self.walls):
            # exact capsule footprint: rectangle body + two round end caps
            # (matches the solver's true point-to-capsule distance, unlike a
            # linewidth-hack line) -- see
            # research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md
            selected = self._selected == ("wall", i)
            edge_color = "yellow" if selected else "darkred"
            edge_width = 2.5 if selected else 1.0
            poly_xy = _capsule_polygon_plot_xy(x0, y0, x1, y1, wrad)
            if poly_xy:
                p = Polygon(poly_xy, facecolor="firebrick", alpha=0.35, edgecolor=edge_color,
                            linewidth=edge_width, zorder=3)
                self.ax_map.add_patch(p)
                self._dynamic_artists.append(p)
            for (cx, cy) in ((x0, y0), (x1, y1)):
                cap = Circle((cy, cx), wrad, facecolor="firebrick", alpha=0.35, edgecolor=edge_color,
                              linewidth=edge_width if selected else 0, zorder=3)
                self.ax_map.add_patch(cap)
                self._dynamic_artists.append(cap)
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            speed = float(np.hypot(wvx, wvy))
            if speed > _VELOCITY_EPS:
                self._draw_velocity_arrow(mx, my, wvx, wvy)  # whole wall translates rigidly from its midpoint
            label = f"{wrad:.1f}m" + (f"\n{speed:.2f}m/s" if speed > _VELOCITY_EPS else "")
            t = self.ax_map.text(my, mx, label, fontsize=7, ha="center", va="center", zorder=4)
            self._dynamic_artists.append(t)

        for i, (xc, yc, a, b, theta, evx, evy) in enumerate(self.ellipses):
            # matplotlib has a NATIVE ellipse patch (unlike the wall capsule,
            # which needed a custom polygon builder) -- angle needs the same
            # plot_yaw = pi/2 - theta axis-swap compensation rviz_node.py's
            # _yaw_quat documents. theta stays independent of evx/evy (see
            # module docstring) -- setting a velocity never rotates this ellipse.
            selected = self._selected == ("ellipse", i)
            plot_angle_deg = float(np.degrees(np.pi / 2.0 - theta))
            e = Ellipse((yc, xc), width=2.0 * a, height=2.0 * b, angle=plot_angle_deg,
                        facecolor="mediumpurple", alpha=0.3,
                        edgecolor="yellow" if selected else "indigo",
                        linewidth=2.5 if selected else 1.0, zorder=3)
            self.ax_map.add_patch(e)
            self._dynamic_artists.append(e)
            speed = float(np.hypot(evx, evy))
            if speed > _VELOCITY_EPS:
                self._draw_velocity_arrow(xc, yc, evx, evy)
            label = f"{a:.1f}x{b:.1f}m" + (f"\n{speed:.2f}m/s" if speed > _VELOCITY_EPS else "")
            t = self.ax_map.text(yc, xc, label, fontsize=7, ha="center", va="center", zorder=4)
            self._dynamic_artists.append(t)

        for i, (xc, yc, a, b, psi0, enabled) in enumerate(self.obstacle_ships):
            # Bold outline + heading arrow (psi0, same compass convention as
            # the initial-heading arrow above), dimmed when disabled -- NEVER
            # gets a Velocity-mode arrow (obstacle ships have no vx/vy; their
            # motion comes from obstacle_ship_node's own MMG simulation, not
            # scenario-authored velocity).
            selected = self._selected_ship == i
            alpha = 0.35 if enabled else 0.12
            plot_angle_deg = float(np.degrees(np.pi / 2.0 - psi0))
            e = Ellipse((yc, xc), width=2.0 * a, height=2.0 * b, angle=plot_angle_deg,
                        facecolor="steelblue", alpha=alpha,
                        edgecolor="yellow" if selected else "navy",
                        linewidth=2.5 if selected else 1.5, zorder=3)
            self.ax_map.add_patch(e)
            self._dynamic_artists.append(e)
            arrow_len = max(a, 1.0)
            dx_plot, dy_plot = arrow_len * np.sin(psi0), arrow_len * np.cos(psi0)
            arrow = self.ax_map.annotate(
                "", xy=(yc + dx_plot, xc + dy_plot), xytext=(yc, xc),
                arrowprops=dict(arrowstyle="->", color="navy" if enabled else "gray", lw=2), zorder=6)
            self._dynamic_artists.append(arrow)
            label = f"SHIP {i}\n{a:.1f}x{b:.1f}m" + ("" if enabled else "\n(disabled)")
            t = self.ax_map.text(yc, xc, label, fontsize=7, ha="center", va="center", zorder=4)
            self._dynamic_artists.append(t)

        self._update_status_text()
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Live scenario editor for NMPC waypoints/obstacles")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output path for scenario.json")
    parser.add_argument("--load", action="store_true", help="load --out on startup if it already exists")
    parser.add_argument("--sim-time", type=float, default=800.0)
    parser.add_argument("--u-init", type=float, default=0.1)
    parser.add_argument("--psi-init", type=float, default=0.0, help="initial heading in degrees (0=North, +90=East)")
    args = parser.parse_args()

    editor = ScenarioEditor(out_path=args.out, sim_time=args.sim_time, u_init=args.u_init, psi_init=args.psi_init)
    if args.load:
        editor._on_load(None)
    editor.show()


if __name__ == "__main__":
    main()
