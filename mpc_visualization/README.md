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
  overall start/goal). `snapshot()` returns an immutable copy for the
  render thread to read without racing the writer.
- **`visualizer.py`** — `MPCVisualizer`. Two-panel layout: a map (ship
  polygon, traversed-path trail, reference path, predicted-trajectory
  overlay, obstacles, start/goal/active-waypoint markers, a live telemetry
  text box) on the left, and a dual-axis control-horizon plot (planned
  rudder angle vs. planned propeller speed over the horizon) on the right.
  Also writes a timestamped CSV log of every state update to
  `mpc_visualization/logs/` (full state, predicted trajectory and control
  horizon serialized as JSON per row) so a run can be replayed/analyzed
  after the fact without re-running the simulation.
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
`nmpc/run_live_open_loop.py`, which imports `MPCBridge`/`MPCVisualizer`
directly (see `nmpc/README.md`) instead of duplicating any of this code.

## Dependencies

`numpy`, `matplotlib` only.
