"""
Live GUI for building NMPC scenarios: click to place a start point, an end
point, and any number of intermediate waypoints in between; drag to "paint"
circular obstacles at a coordinate with a chosen radius. Saves everything to
scenario.json in the same dict shape as one entry of SCENARIOS in
nmpc/run_live.py (waypoints / mmg_init / sim_time), plus an "obstacles" key
that nmpc/ doesn't consume yet.

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
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MODES = ["Start", "Waypoint", "Goal", "Obstacle", "Remove"]
DEFAULT_OUT = os.path.expanduser("~/nmpc_scenarios/scenario.json")


class ScenarioEditor:
    def __init__(self, out_path=DEFAULT_OUT, xlim=(-30, 60), ylim=(-20, 60),
                 sim_time=600.0, u_init=0.1, default_obstacle_radius=2.0):
        self.out_path = out_path
        self.sim_time = sim_time
        self.u_init = u_init
        self.default_obstacle_radius = default_obstacle_radius

        # --- scenario data (all points stored as (x, y) = (North, East)) ---
        self.start = None
        self.goal = None
        self.waypoints = []      # intermediate points only, in click order
        self.obstacles = []      # [x, y, radius]

        self.mode = "start"
        self._history = []       # stack of no-arg undo callables
        self._obs_drag_start = None
        self._obs_preview = None
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
        self.ax_map.set_title("Scenario Editor — click to place, drag to size obstacles",
                               fontsize=12, fontweight="bold")

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Start"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=13, label="Goal"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="orange", markersize=8, label="Waypoint"),
            Line2D([0], [0], color="black", linestyle=":", label="Path"),
            Patch(facecolor="red", alpha=0.3, edgecolor="darkred", label="Obstacle"),
        ]
        self.ax_map.legend(handles=legend_handles, loc="upper right", fontsize=8)

        self.status_text = self.ax_map.text(
            0.02, 0.98, "", transform=self.ax_map.transAxes, fontsize=9,
            fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), zorder=10,
        )
        self.msg_text = self.fig.text(0.06, 0.02, "", fontsize=9, color="darkblue")

        # ---- sidebar widgets ----
        self.fig.text(0.71, 0.955, "Mode", fontsize=10, fontweight="bold")
        ax_mode = self.fig.add_axes([0.70, 0.72, 0.27, 0.22])
        ax_mode.set_frame_on(True)
        self.radio_mode = RadioButtons(ax_mode, MODES, active=0)
        self.radio_mode.on_clicked(self._on_mode_change)

        self.fig.text(0.71, 0.685, "Sim Time (s)", fontsize=9)
        ax_simtime = self.fig.add_axes([0.70, 0.635, 0.27, 0.045])
        self.simtime_box = TextBox(ax_simtime, "", initial=str(self.sim_time))
        self.simtime_box.on_submit(self._on_simtime_submit)

        self.fig.text(0.71, 0.60, "Initial Surge u (m/s)", fontsize=9)
        ax_uinit = self.fig.add_axes([0.70, 0.55, 0.27, 0.045])
        self.uinit_box = TextBox(ax_uinit, "", initial=str(self.u_init))
        self.uinit_box.on_submit(self._on_uinit_submit)

        self.fig.text(0.71, 0.515, "Default Obstacle R (m)", fontsize=9)
        ax_obsr = self.fig.add_axes([0.70, 0.465, 0.27, 0.045])
        self.obsr_box = TextBox(ax_obsr, "", initial=str(self.default_obstacle_radius))
        self.obsr_box.on_submit(self._on_obsr_submit)

        ax_save = self.fig.add_axes([0.70, 0.36, 0.27, 0.055])
        self.btn_save = Button(ax_save, "Save scenario.json")
        self.btn_save.on_clicked(self._on_save)

        ax_load = self.fig.add_axes([0.70, 0.29, 0.27, 0.055])
        self.btn_load = Button(ax_load, "Load")
        self.btn_load.on_clicked(self._on_load)

        ax_undo = self.fig.add_axes([0.70, 0.22, 0.27, 0.055])
        self.btn_undo = Button(ax_undo, "Undo")
        self.btn_undo.on_clicked(self._on_undo)

        ax_clear = self.fig.add_axes([0.70, 0.15, 0.27, 0.055])
        self.btn_clear = Button(ax_clear, "Clear All")
        self.btn_clear.on_clicked(self._on_clear)

        self.fig.text(0.70, 0.10, f"Saves to:\n{self.out_path}", fontsize=7, color="dimgray")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------------
    # mouse handlers
    # ------------------------------------------------------------------
    def _event_point(self, event):
        """matplotlib gives (Y, X) in data coords (plot x=East, plot y=North);
        scenario points are stored as (X, Y) = (North, East)."""
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return None
        return (float(event.ydata), float(event.xdata))

    def _on_press(self, event):
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

    def _on_obsr_submit(self, text):
        try:
            self.default_obstacle_radius = float(text)
        except ValueError:
            self._flash("Obstacle radius must be a number")

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
                round(self.start[0], 4), round(self.start[1], 4), 0.0,
            ],
            "sim_time": self.sim_time,
            "obstacles": [[round(o[0], 4), round(o[1], 4), round(o[2], 4)] for o in self.obstacles],
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(data, f, indent=2)
        self._flash(f"Saved {len(full_path)} waypoints, {len(self.obstacles)} obstacles -> {self.out_path}")

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
        self.sim_time = float(data.get("sim_time", self.sim_time))
        mmg_init = data.get("mmg_init")
        if mmg_init:
            self.u_init = float(mmg_init[0])

        self.simtime_box.set_val(str(self.sim_time))
        self.uinit_box.set_val(str(self.u_init))
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
        self.waypoints, self.obstacles = [], []
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
        text = (
            f"Mode: {self.mode.capitalize()}\n"
            f"Start : {self._fmt_pt(self.start)}\n"
            f"Goal  : {self._fmt_pt(self.goal)}\n"
            f"Waypoints: {len(self.waypoints)}\n"
            f"Obstacles: {len(self.obstacles)}\n"
            f"Sim Time : {self.sim_time:.1f} s\n"
            f"u_init   : {self.u_init:.2f} m/s"
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
    args = parser.parse_args()

    editor = ScenarioEditor(out_path=args.out, sim_time=args.sim_time, u_init=args.u_init)
    if args.load:
        editor._on_load(None)
    editor.show()


if __name__ == "__main__":
    main()
