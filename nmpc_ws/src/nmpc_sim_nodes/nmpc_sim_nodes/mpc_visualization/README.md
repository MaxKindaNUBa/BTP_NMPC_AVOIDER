# mpc_visualization

A standalone live-visualization dashboard for the NMPC ship — a matplotlib
`FuncAnimation` showing the vessel's traversed path, its currently-planned
prediction horizon, and its planned control horizon, all updating in real
time. Built and validated with **mock/dead-reckoned data** first, before any
real NMPC solver existed, specifically so the visualization pipeline itself
could be debugged in isolation.

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
  "actual vs predicted" overlays both visualizers below draw). `snapshot()`
  returns an immutable copy for the render thread to read without racing
  the writer.
- **`visualizer.py`** — `MPCVisualizer`, the full two-panel dashboard: a map
  (ship polygon, traversed-path trail, reference path, predicted-trajectory
  overlay, obstacles, start/goal/active-waypoint markers, a live `SHIP
  STATUS`/`CONTROLS`/`TARGETS` telemetry text box, and inset current/wave
  compass rings) on the left, and a dual-axis control-horizon plot (planned
  rudder angle vs. planned propeller speed over the horizon) on the right.
  The current compass draws a **2nd needle** for `ukf_node`'s predicted
  current on the same ring (not a separate inset), with its magnitude label
  showing `actual | predicted`; `SHIP STATUS`'s `X`/`Y` lines do the same.
  Also writes a timestamped CSV log of every state update to
  `mpc_visualization/logs/` (full state, predicted trajectory and control
  horizon serialized as JSON per row) so a run can be replayed/analyzed
  after the fact without re-running the simulation.
- **`hud_visualizer.py`** — `HUDVisualizer`, a standalone companion window
  (current compass, wave-force scatter, control-horizon graph only — no map,
  no `SHIP STATUS` text) meant to run alongside RViz2 instead of drawing on
  top of it. Same current-compass "2nd needle, `actual | predicted`" overlay
  as `visualizer.py`'s, just with the reading folded into the panel's title
  instead of a separate label object.
- **`run_demo.py`** — a mock-data demo: a background thread drives a fake
  ship in a slow circle with a synthetic oscillating control horizon, purely
  to exercise the visualizer. Untouched by later NMPC integration work —
  it's the reference implementation for how a driver script is expected to
  feed the bridge.
- **`logs/`** — CSV output directory (gitignored; see the root README for
  why the accumulated debug logs aren't tracked).

## Running it

```bash
cd mpc_visualization
python run_demo.py
```

Opens the dashboard window with the mock ship circling and dummy
prediction/control horizons animating, so you can confirm map scaling, ship
heading rendering, and both plots update correctly before wiring in a real
solver. Close the window to stop the background thread cleanly.

The real NMPC solvers are wired into this same dashboard by
`nmpc_sim_nodes/viz_node.py` (the full `visualizer.py` dashboard) and
`nmpc_sim_nodes/hud_node.py` (the standalone `hud_visualizer.py` companion
window, meant to run alongside `rviz_node`/RViz2 — see
`nmpc_sim_nodes/launch/rviz_hud.launch.py`), both of which import
`MPCBridge`/the relevant visualizer class directly instead of duplicating
any of this code. Run via `ros2 run nmpc_sim_nodes viz_node` /
`ros2 launch nmpc_sim_nodes rviz_hud.launch.py` — see the top-level README's
"Getting started" section, not a bare `python` invocation (these are ROS2
node entry points, not standalone scripts, once wired to a live solver).

## Dependencies

`numpy`, `matplotlib` only.
