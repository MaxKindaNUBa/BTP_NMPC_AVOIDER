# mpc_visualization

Shared infrastructure for `hud_node.py`'s standalone matplotlib companion
window (current compass, wave-force scatter, NMPC control-horizon graph),
run alongside `rviz_node`/RViz2 via `nmpc_sim_nodes/launch/rviz_hud.launch.py`.

The full two-panel map+control-horizon dashboard (`visualizer.py`,
`MPCVisualizer`) and its `viz_node`/`run_demo` entry points were removed once
`rviz_node`/RViz2 (a real 3D scene, obstacles/walls/ellipses rendered as
actual shapes, etc.) fully superseded it as the primary way to watch a run —
`hud_visualizer.py` below is what's left of this package's own rendering
code, everything else here is the shared data-bridge it (and the removed
dashboard) were built on.

Deliberately zero-dependency on the rest of the repo: no CasADi, no Acados,
no `nmpc/` imports. It only knows about plain NumPy arrays passed through
`MPCBridge`.

## Files

- **`mpc_bridge.py`** — `MPCBridge`, a thread-safe shared-state object
  connecting a simulation/solver thread to the visualizer thread:
  `update_ship_state(state, t)`, `update_prediction(predicted_trajectory,
  control_horizon)`, `set_obstacles(...)`, `set_start_goal(...)`, and
  `set_active_waypoint(...)` for marking which intermediate leg-target a
  multi-waypoint path is currently steering toward (distinct from the
  overall start/goal), plus `update_current(...)`/`update_wave(...)` (the
  true/measured environment reading) and `update_ukf_current(...)`/
  `update_ukf_position(...)` (the UKF's own estimate of the same, for the
  "actual vs predicted" overlay `hud_visualizer.py` draws). `snapshot()`
  returns an immutable copy for the render thread to read without racing
  the writer.
- **`hud_visualizer.py`** — `HUDVisualizer`, a standalone companion window
  (current compass, wave-force scatter, control-horizon graph only — no map,
  no `SHIP STATUS` text) meant to run alongside RViz2 instead of drawing on
  top of it. Draws a **2nd needle** for `ukf_node`'s predicted current on the
  same compass ring, with the reading folded into the panel's title as
  `actual | predicted` — the same convention `rviz_node.py`'s overlays use.

## Dependencies

`numpy`, `matplotlib` only.
