"""
Live GUI for building NMPC scenarios: click to place a start point, an end
point, and any number of intermediate waypoints in between; drag to "paint"
circular obstacles at a coordinate with a chosen radius, click-click to
place a wall (a padded line segment / capsule -- for harbor walls, quays,
breakwaters), or drag to place an elliptical obstacle (e.g. another vessel --
drag direction/length set orientation/semi-major axis, a fixed semi-minor
axis comes from a textbox; see research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md).
Saves everything to scenario.json in the same dict shape as one entry of
SCENARIOS in nmpc/run_live.py (waypoints / mmg_init / sim_time), plus
"obstacles", "walls", and "ellipses" keys that map_node/nmpc_node consume
directly.

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

MODES = ["Start", "Waypoint", "Goal", "Obstacle", "Wall", "Ellipse", "Remove"]
DEFAULT_OUT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "nmpc_sim_nodes", "params", "scenario.json"))


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
                 sim_time=600.0, u_init=0.1, psi_init=0.0, default_obstacle_radius=2.0,
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
        self.obstacles = []      # [x, y, radius]
        self.walls = []          # [x0, y0, x1, y1, radius] capsule (wall) obstacles --
        # see research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md
        self.ellipses = []       # [xc, yc, a, b, theta] elliptical obstacles, e.g. other vessels --
        # theta in RADIANS, world frame (same convention as every other physics-facing angle in
        # nmpc/, e.g. mmg_init's psi -- no degrees conversion needed since theta is set purely by
        # the drag gesture, never typed into a textbox). Same design doc as walls above.

        self.mode = "start"
        self._history = []       # stack of no-arg undo callables
        self._obs_drag_start = None
        self._obs_preview = None
        self._wall_start = None  # first-click endpoint, pending the second click
        self._wall_preview = None
        self._ell_drag_start = None
        self._ell_preview = None
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
            "Scenario Editor — click to place, drag to size obstacles/ellipses, click-click for a wall",
            fontsize=12, fontweight="bold")

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Start"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=13, label="Goal"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="orange", markersize=8, label="Waypoint"),
            Line2D([0], [0], color="black", linestyle=":", label="Path"),
            Patch(facecolor="red", alpha=0.3, edgecolor="darkred", label="Obstacle"),
            Patch(facecolor="firebrick", alpha=0.35, edgecolor="darkred", label="Wall"),
            Patch(facecolor="mediumpurple", alpha=0.3, edgecolor="indigo", label="Ellipse"),
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

        self.fig.text(0.71, 0.715, "Sim Time (s)", fontsize=9)
        ax_simtime = self.fig.add_axes([0.70, 0.665, 0.27, 0.04])
        self.simtime_box = TextBox(ax_simtime, "", initial=str(self.sim_time))
        self.simtime_box.on_submit(self._on_simtime_submit)

        self.fig.text(0.71, 0.645, "Initial Surge u (m/s)", fontsize=9)
        ax_uinit = self.fig.add_axes([0.70, 0.595, 0.27, 0.04])
        self.uinit_box = TextBox(ax_uinit, "", initial=str(self.u_init))
        self.uinit_box.on_submit(self._on_uinit_submit)

        self.fig.text(0.71, 0.575, "Initial Heading (deg, 0=N/+90=E)", fontsize=8)
        ax_psiinit = self.fig.add_axes([0.70, 0.525, 0.27, 0.04])
        self.psiinit_box = TextBox(ax_psiinit, "", initial=str(self.psi_init))
        self.psiinit_box.on_submit(self._on_psiinit_submit)

        self.fig.text(0.71, 0.505, "Default Obstacle R (m)", fontsize=9)
        ax_obsr = self.fig.add_axes([0.70, 0.455, 0.27, 0.04])
        self.obsr_box = TextBox(ax_obsr, "", initial=str(self.default_obstacle_radius))
        self.obsr_box.on_submit(self._on_obsr_submit)

        self.fig.text(0.71, 0.435, "Wall R (m) -- click, click", fontsize=9)
        ax_wallr = self.fig.add_axes([0.70, 0.385, 0.27, 0.04])
        self.wallr_box = TextBox(ax_wallr, "", initial=str(self.default_wall_radius))
        self.wallr_box.on_submit(self._on_wallr_submit)

        self.fig.text(0.71, 0.365, "Ellipse Semi-Minor b (m) -- drag", fontsize=8)
        ax_ellb = self.fig.add_axes([0.70, 0.315, 0.27, 0.04])
        self.ellipseb_box = TextBox(ax_ellb, "", initial=str(self.default_ellipse_b))
        self.ellipseb_box.on_submit(self._on_ellipseb_submit)

        ax_save = self.fig.add_axes([0.70, 0.245, 0.27, 0.05])
        self.btn_save = Button(ax_save, "Save scenario.json")
        self.btn_save.on_clicked(self._on_save)

        ax_load = self.fig.add_axes([0.70, 0.18, 0.27, 0.05])
        self.btn_load = Button(ax_load, "Load")
        self.btn_load.on_clicked(self._on_load)

        ax_undo = self.fig.add_axes([0.70, 0.115, 0.27, 0.05])
        self.btn_undo = Button(ax_undo, "Undo")
        self.btn_undo.on_clicked(self._on_undo)

        ax_clear = self.fig.add_axes([0.70, 0.05, 0.27, 0.05])
        self.btn_clear = Button(ax_clear, "Clear All")
        self.btn_clear.on_clicked(self._on_clear)

        self.fig.text(0.70, 0.012, f"Saves to:\n{self.out_path}", fontsize=7, color="dimgray")

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
                    self.walls.append([sx, sy, pt[0], pt[1], self.default_wall_radius])
                    self._push_undo(lambda: self.walls.pop() if self.walls else None)
                self._wall_start = None
                self._redraw()
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
            self.ellipses.append([sx, sy, a, self.default_ellipse_b, theta])
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

        self.obstacles.append([sx, sy, r])
        self._push_undo(lambda: self.obstacles.pop() if self.obstacles else None)
        self._obs_drag_start = None
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
        for i, (ox, oy, _r) in enumerate(self.obstacles):
            candidates.append((np.hypot(ox - x, oy - y), "obstacle", i))
        for i, (x0, y0, x1, y1, _r) in enumerate(self.walls):
            candidates.append((_point_to_segment_dist(x, y, x0, y0, x1, y1), "wall", i))
        for i, (xc, yc, a, b, _theta) in enumerate(self.ellipses):
            # rough center-distance-minus-max(a,b) proxy -- sufficient for UI
            # hit-testing, exactness doesn't matter here (unlike the solver's
            # gradient-normalized ellipse_distance_casadi)
            candidates.append((max(0.0, np.hypot(xc - x, yc - y) - max(a, b)), "ellipse", i))

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
        self._flash(f"Removed {kind}")

    # ------------------------------------------------------------------
    # widget callbacks
    # ------------------------------------------------------------------
    def _on_mode_change(self, label):
        self.mode = label.lower()
        self._update_status_text()
        self.fig.canvas.draw_idle()

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
            "obstacles": [[round(o[0], 4), round(o[1], 4), round(o[2], 4)] for o in self.obstacles],
            "walls": [[round(v, 4) for v in w] for w in self.walls],
            "ellipses": [[round(v, 4) for v in e] for e in self.ellipses],
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(data, f, indent=2)
        self._flash(f"Saved {len(full_path)} waypoints, {len(self.obstacles)} obstacles, "
                    f"{len(self.walls)} walls, {len(self.ellipses)} ellipses -> {self.out_path}")

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

        self.obstacles = [list(o) for o in data.get("obstacles", [])]
        self.walls = [list(w) for w in data.get("walls", [])]
        self.ellipses = [list(e) for e in data.get("ellipses", [])]
        self.sim_time = float(data.get("sim_time", self.sim_time))
        mmg_init = data.get("mmg_init")
        if mmg_init:
            self.u_init = float(mmg_init[0])
            if len(mmg_init) >= 6:
                self.psi_init = float(np.rad2deg(mmg_init[5]))

        self.simtime_box.set_val(str(self.sim_time))
        self.uinit_box.set_val(str(self.u_init))
        self.psiinit_box.set_val(str(round(self.psi_init, 4)))
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
        text = (
            f"Mode: {mode_label}\n"
            f"Start : {self._fmt_pt(self.start)}\n"
            f"Goal  : {self._fmt_pt(self.goal)}\n"
            f"Waypoints: {len(self.waypoints)}\n"
            f"Obstacles: {len(self.obstacles)}\n"
            f"Walls    : {len(self.walls)}\n"
            f"Ellipses : {len(self.ellipses)}\n"
            f"Sim Time : {self.sim_time:.1f} s\n"
            f"u_init   : {self.u_init:.2f} m/s\n"
            f"psi_init : {self.psi_init:.1f} deg"
        )
        self.status_text.set_text(text)

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
        for (ox, oy, orad) in self.obstacles:
            c = Circle((oy, ox), orad, facecolor="red", alpha=0.3, edgecolor="darkred", zorder=3)
            self.ax_map.add_patch(c)
            self._dynamic_artists.append(c)
            t = self.ax_map.text(oy, ox, f"{orad:.1f}m", fontsize=7, ha="center", va="center", zorder=4)
            self._dynamic_artists.append(t)
        for (x0, y0, x1, y1, wrad) in self.walls:
            # exact capsule footprint: rectangle body + two round end caps
            # (matches the solver's true point-to-capsule distance, unlike a
            # linewidth-hack line) -- see
            # research_papers/NON_CIRCULAR_OBSTACLE_PRIMITIVES.md
            poly_xy = _capsule_polygon_plot_xy(x0, y0, x1, y1, wrad)
            if poly_xy:
                p = Polygon(poly_xy, facecolor="firebrick", alpha=0.35, edgecolor="darkred", zorder=3)
                self.ax_map.add_patch(p)
                self._dynamic_artists.append(p)
            for (cx, cy) in ((x0, y0), (x1, y1)):
                cap = Circle((cy, cx), wrad, facecolor="firebrick", alpha=0.35, edgecolor="darkred",
                              linewidth=0, zorder=3)
                self.ax_map.add_patch(cap)
                self._dynamic_artists.append(cap)
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            t = self.ax_map.text(my, mx, f"{wrad:.1f}m", fontsize=7, ha="center", va="center", zorder=4)
            self._dynamic_artists.append(t)
        for (xc, yc, a, b, theta) in self.ellipses:
            # matplotlib has a NATIVE ellipse patch (unlike the wall capsule,
            # which needed a custom polygon builder) -- angle needs the same
            # plot_yaw = pi/2 - theta axis-swap compensation rviz_node.py's
            # _yaw_quat documents.
            plot_angle_deg = float(np.degrees(np.pi / 2.0 - theta))
            e = Ellipse((yc, xc), width=2.0 * a, height=2.0 * b, angle=plot_angle_deg,
                        facecolor="mediumpurple", alpha=0.3, edgecolor="indigo", zorder=3)
            self.ax_map.add_patch(e)
            self._dynamic_artists.append(e)
            t = self.ax_map.text(yc, xc, f"{a:.1f}x{b:.1f}m", fontsize=7, ha="center", va="center", zorder=4)
            self._dynamic_artists.append(t)

        self._update_status_text()
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Live scenario editor for NMPC waypoints/obstacles")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output path for scenario.json")
    parser.add_argument("--load", action="store_true", help="load --out on startup if it already exists")
    parser.add_argument("--sim-time", type=float, default=600.0)
    parser.add_argument("--u-init", type=float, default=0.1)
    parser.add_argument("--psi-init", type=float, default=0.0, help="initial heading in degrees (0=North, +90=East)")
    args = parser.parse_args()

    editor = ScenarioEditor(out_path=args.out, sim_time=args.sim_time, u_init=args.u_init, psi_init=args.psi_init)
    if args.load:
        editor._on_load(None)
    editor.show()


if __name__ == "__main__":
    main()
