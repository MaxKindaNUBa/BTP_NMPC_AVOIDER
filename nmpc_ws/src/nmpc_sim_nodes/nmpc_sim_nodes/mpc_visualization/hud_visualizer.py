import collections

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

_WAVE_TRAIL_LEN = 25  # how many recent wave-force samples the fading scatter keeps


class HUDVisualizer:
    """Standalone companion window: current compass, wave-force scatter, and the
    control-horizon graph -- the same data MPCVisualizer draws inset over its own
    map, broken out here so it can run alongside RViz2 (or headless) without
    drawing on top of a 3D/2D view.
    """

    def __init__(self, bridge, update_interval_ms=100):
        self.bridge = bridge
        self.update_interval = update_interval_ms
        self.is_running = True

        self.fig = plt.figure(figsize=(13.5, 7.0), constrained_layout=True)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.3, 2.2], height_ratios=[1.0, 1.7],
                                   hspace=0.3, wspace=0.2)
        self.ax_current = self.fig.add_subplot(gs[0, 0])
        self.ax_wave = self.fig.add_subplot(gs[1, 0])
        self.ax_ctrl = self.fig.add_subplot(gs[:, 1])
        self.fig.canvas.manager.set_window_title('NMPC HUD: Current / Wave / Control Horizon')

        self.current_arrow = self._init_compass(self.ax_current, '#1f9fd6')
        # 2nd arrow on the SAME ring for the UKF-predicted current -- no new
        # panel, matching rviz_node.py's/visualizer.py's "| predicted" convention.
        self.ukf_current_arrow = self._init_compass(self.ax_current, '#b34dff')
        self._init_wave_scatter(self.ax_wave)
        self._wave_trail = collections.deque(maxlen=_WAVE_TRAIL_LEN)

        self.ax_ctrl.grid(True, linestyle='--', alpha=0.5)
        self.ax_ctrl.set_title('Control Horizon Inputs (Planned)', fontsize=12, fontweight='bold')
        self.ax_ctrl.set_xlabel('Horizon Step')
        self.ax_ctrl_right = self.ax_ctrl.twinx()
        self.line_rudder, = self.ax_ctrl.plot([], [], 'g-s', markersize=4, label='Rudder δ (deg)')
        self.line_rps, = self.ax_ctrl_right.plot([], [], 'r-o', markersize=4, label='Propeller n_p (rps)')
        self.ax_ctrl.set_ylabel('Rudder Angle (deg)', color='g')
        self.ax_ctrl_right.set_ylabel('Propeller Speed (rps)', color='r')

        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def on_close(self, event):
        self.is_running = False

    # ---- current compass -------------------------------------------------
    def _init_compass(self, ax, color):
        # xlim/ylim exactly match the ring's radius -- no reserved margin for a
        # below-ring label, since the reading now lives in the title (matplotlib
        # spaces that automatically) instead of hand-placed data-space text that
        # either wastes room or crowds the border depending on how much margin.
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor='gray', linewidth=1.2))
        arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle='-|>', color=color, linewidth=2.5))
        arrow.set_visible(False)
        return arrow

    def _update_compass(self, ax, arrow, title_prefix, screen_dx, screen_dy, magnitude, unit,
                         ukf_arrow=None, ukf_screen_dx=None, ukf_screen_dy=None, ukf_magnitude=None):
        if magnitude > 1e-6:
            arrow.xy = (0.85 * screen_dx / magnitude, 0.85 * screen_dy / magnitude)
            arrow.set_visible(True)
        else:
            arrow.set_visible(False)

        # "| <predicted>" straight after the actual reading in the SAME title,
        # from /ukf/estimated_current -- no new panel, matching visualizer.py's/
        # rviz_node.py's convention. Omitted (falls back to actual-only) until
        # the first UKF message arrives (ukf_arrow is None for the wave compass,
        # which has no UKF equivalent).
        if ukf_arrow is not None:
            if ukf_magnitude is not None and ukf_magnitude > 1e-6:
                ukf_arrow.xy = (0.85 * ukf_screen_dx / ukf_magnitude, 0.85 * ukf_screen_dy / ukf_magnitude)
                ukf_arrow.set_visible(True)
            else:
                ukf_arrow.set_visible(False)
            if ukf_magnitude is not None:
                title = f'{title_prefix}: {magnitude:.2f} | {ukf_magnitude:.2f} {unit}'
            else:
                title = f'{title_prefix}: {magnitude:.2f} {unit}'
        else:
            title = f'{title_prefix}: {magnitude:.2f} {unit}' if magnitude > 1e-6 else f'{title_prefix}: off'
        ax.set_title(title, fontsize=10, fontweight='bold')

    # ---- wave force scatter ------------------------------------------------
    def _init_wave_scatter(self, ax):
        # no aspect='equal' here: it forces a square data box, which -- combined
        # with a wider-than-tall grid cell -- left a large empty gap above/below
        # the actual plot. Symmetric xlim/ylim already keep it visually balanced.
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('Fx -- surge (N)', fontsize=8)
        ax.set_ylabel('Fy -- sway (N)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color='gray', linewidth=0.6)
        ax.axvline(0, color='gray', linewidth=0.6)
        self.wave_scatter = ax.scatter([], [], s=22)

    def _update_wave_scatter(self, ax, fx, fy):
        # body-frame fx/fy, straight off WaveState -- this is a force-space
        # scatter (not a map-aligned compass), so no earth-frame rotation needed.
        self._wave_trail.append((fx, fy))
        pts = np.array(self._wave_trail)
        n = len(pts)

        # fading trail: oldest points nearly transparent, newest fully opaque
        alphas = np.linspace(0.15, 1.0, n)
        colors = np.tile([0.90, 0.26, 0.60, 1.0], (n, 1))
        colors[:, 3] = alphas
        self.wave_scatter.set_offsets(pts)
        self.wave_scatter.set_facecolor(colors)

        # no large fixed floor here: at small wave_hs (scaled-model units), forces
        # are routinely ~1e-4 N -- a 0.5 N floor would swamp that to an invisible
        # dot at the origin, which is exactly the "stuck at 0,0" bug this fixes.
        span = max(1e-4, float(np.abs(pts).max()) * 1.3)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        magnitude = float(np.hypot(fx, fy))
        ax.set_title(f'WAVE: {magnitude:.2f} N' if magnitude >= 0.01 else f'WAVE: {magnitude:.2e} N',
                     fontsize=10, fontweight='bold')

    # ------------------------------------------------------------------
    def init_plot(self):
        self.line_rudder.set_data([], [])
        self.line_rps.set_data([], [])
        return (self.line_rudder, self.line_rps, self.current_arrow, self.ukf_current_arrow, self.wave_scatter)

    def update_plot(self, frame):
        if not self.is_running:
            return (self.line_rudder, self.line_rps, self.current_arrow, self.ukf_current_arrow, self.wave_scatter)

        snap = self.bridge.snapshot()

        # screen (East, North) == data (y, x), same convention as visualizer.py's map
        cur = snap.current
        ukf_cur = snap.ukf_current
        self._update_compass(
            self.ax_current, self.current_arrow, 'CURRENT',
            cur.vy, cur.vx, cur.speed if cur.enabled else 0.0, 'm/s',
            ukf_arrow=self.ukf_current_arrow,
            ukf_screen_dx=ukf_cur.vy if ukf_cur is not None else None,
            ukf_screen_dy=ukf_cur.vx if ukf_cur is not None else None,
            ukf_magnitude=ukf_cur.speed if ukf_cur is not None else None)

        wav = snap.wave
        fx, fy = (wav.fx, wav.fy) if wav.enabled else (0.0, 0.0)
        self._update_wave_scatter(self.ax_wave, fx, fy)

        steps_ctrl = np.arange(len(snap.control_horizon))
        if len(steps_ctrl) > 0:
            rudders_deg = np.rad2deg(snap.control_horizon[:, 0])
            rpss = snap.control_horizon[:, 1]
            self.line_rudder.set_data(steps_ctrl, rudders_deg)
            self.line_rps.set_data(steps_ctrl, rpss)
            self.ax_ctrl.set_xlim([0, max(1, len(steps_ctrl) - 1)])
            self.ax_ctrl.set_ylim([min(rudders_deg) - 2.0, max(rudders_deg) + 2.0])
            self.ax_ctrl_right.set_ylim([min(rpss) - 1.0, max(rpss) + 1.0])

        # recomputed every frame (cheap enough now that blit=False forces a full
        # redraw anyway) since 'best' has to dodge wherever the curves currently are --
        # a fixed corner like 'upper right' gets covered whenever a curve peaks there.
        lines_ctrl = [self.line_rudder, self.line_rps]
        self.ax_ctrl.legend(lines_ctrl, [line.get_label() for line in lines_ctrl],
                             loc='best', fontsize=8)

        return (self.line_rudder, self.line_rps, self.current_arrow, self.ukf_current_arrow, self.wave_scatter)

    def start(self):
        # blit=False: both the control-horizon panel's and the wave scatter's
        # set_xlim/set_ylim change every frame, and with blitting on, only the
        # returned artists get redrawn -- the axes' own tick labels/titles are
        # cached from the first frame and never refresh otherwise.
        self.anim = FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot,
                                  interval=self.update_interval, blit=False)
        plt.show()
