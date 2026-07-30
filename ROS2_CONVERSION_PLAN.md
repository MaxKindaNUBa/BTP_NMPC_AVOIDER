# ROS2 (Jazzy, Ubuntu 24.04) Conversion Plan — NMPC Ship Obstacle-Avoidance Simulator

## How to use this document

This is a specification for converting the current single-process Python simulation
(`nmpc/run_live.py` and its dependencies) into three ROS2 nodes. It is written to be
handed to a code assistant with no prior context on this repo, so it over-specifies
rather than under-specifies: exact function signatures, exact field lists, exact file
paths, exact line references into the current code. Where a design decision was made
(rather than being forced by the existing code), it is called out explicitly as a
**DECISION** block with its rationale, so it can be revisited independently of the rest
of the plan.

**Scope note (important):** sensor-noise modeling and current/wave disturbance forces
are a planned *future* addition to this simulator, but are explicitly **out of scope**
for this conversion. Do not implement them here. However, the architecture below is
chosen so that adding them later doesn't require re-wiring the node graph — see
"Forward-compatibility notes" at the end. Treat any mention of noise/waves in this doc
as context only, never as a task.

---

## 1. Current architecture (what exists today, for context)

The simulation is currently one Python process, `nmpc/run_live.py`, running a
synchronous loop at `dt = 0.1s` (10 Hz, `nmpc/config.py:54`). Per iteration:

1. **Reference selection** (`nmpc/path_following.py`): `select_active_waypoint()`
   (lines 100-136) decides which waypoint leg is currently active, given the ship's
   live `(x, y)` and the full waypoint list; `compute_path_angle()` (line 35) computes
   the path heading `chi_p` for that leg.
2. **NMPC solve**: `nmpc_solver.solve(mmg_state, delta, n, chi_p, x_d, y_d, obstacles=...)`
   — implemented in `nmpc/nmpc_acados.py:193` (acados/SQP-RTI backend) and
   `nmpc/nmpc_casadi.py:210` (CasADi/IPOPT debug backend, same signature/return shape).
   Inputs:
   - `mmg_state`: 6-vector `[u, v, r, x, y, psi]` (surge, sway, yaw rate, position, heading)
   - `delta, n`: the *previous* actuator command (rudder angle [rad], propeller speed [rps])
     — required because the NMPC's internal augmented state includes actuator position
     as state, not just the 6-dim MMG state (see `build_xi_full()` in `path_following.py:164`)
   - `chi_p, x_d, y_d`: path heading and active waypoint target
   - `obstacles`: list of `(x, y, radius)` tuples
   Returns a dict:
   ```
   {
     "u_opt":        (2, N) ndarray   — optimal [delta_dot, n_dot] rate trajectory
     "xi_traj":      (11, N+1) ndarray — predicted augmented-state trajectory
     "delta":        float            — next-step rudder angle command
     "n":            float            — next-step propeller speed command
     "solve_time":   float            — wall-clock solver time [s]
     "success":      bool
     "return_status": int/str          — solver status code
   }
   ```
3. **Plant integration**: `make_casadi_integrator()` in `casadi_mmg_solver/casadi_mmg.py:207`
   builds an RK4 stepper over `MMG_Time_Derivative_casadi()` (line 26 — the ship's
   equations of motion, hull + propeller + rudder hydrodynamic forces). `run_live.py`
   calls this once per iteration (`plant_step(state_ca, control_ca)`, lines 92-95 of
   the current file) to advance the *true* 6-vector `mmg_state` using the accepted
   `(delta, n)`.
4. **Visualization**: `mpc_visualization/mpc_bridge.py` (`MPCBridge`) and
   `mpc_visualization/visualizer.py` (`MPCVisualizer`) are fed directly via Python
   method calls (`bridge.update_ship_state()`, `bridge.update_prediction()`,
   `bridge.set_obstacles()`, `bridge.set_active_waypoint()`) from inside the same loop.
5. **Termination**: controlled by `config.SIM_TIME_MODE` (`"infinite" | "fixed" | "endpoint"`,
   `nmpc/config.py:85-86`) — evaluated at the bottom of the loop in `run_live.py`.

Scenario data (waypoints, obstacles, initial state, sim duration) is loaded once at
startup from a JSON file (default `scenario_maker/scenario.json`), authored by the
separate GUI tool `scenario_maker/scenario_editor.py`. Format (see the file for the
full example):
```json
{
  "waypoints": [[x0, y0], [x1, y1], ...],
  "mmg_init":  [u, v, r, x, y, psi],
  "sim_time":  600.0,
  "obstacles": [[x, y, radius], ...]
}
```

All tunable parameters live in one dataclass, `nmpc.config.NMPCConfig`
(`nmpc/config.py:48-149`), instantiated once as `DEFAULT_CONFIG` and imported
everywhere else in the package. Nothing is hardcoded outside this file except the
MMG hydrodynamic coefficients themselves, which are literal constants inside
`MMG_Time_Derivative_casadi()`.

---

## 2. Target ROS2 architecture

Three nodes: `map_node`, `nmpc_node`, `mmg_node`. Two new packages: an interface-only
package for custom messages/services, and a Python package hosting the three node
executables.

### 2.1 Node responsibilities, at a glance

| Node | Owns | Role |
|---|---|---|
| `map_node` | scenario data, active-waypoint bookkeeping, run-termination logic, the live visualizer | "environment + display" — everything that isn't physics or optimization |
| `nmpc_node` | the OCP solve (acados or CasADi backend) | pure optimizer: state+reference+obstacles in, control+prediction out |
| `mmg_node` | the plant integrator, the master clock | pure physics: state+control in, next state out; drives the loop's timing |

### 2.2 Interaction pattern (DECISION)

**DECISION:** the tight per-tick coupling between `mmg_node` and `nmpc_node` is
implemented as a **ROS2 service** (`mmg_node` is the client, `nmpc_node` is the
server), not a pair of topics. Everything else is a topic.

*Rationale:* the current code is a strict, blocking call sequence — `solve()` then
`plant_step()`, always in that order, always using the freshest state. A service
call reproduces that exactly (the client blocks until it gets a response), whereas
two independent topics with independent publish timers would introduce staleness
races that don't exist in the current script. This is not a behavior change, just
the same synchronous relationship expressed with ROS2 primitives. One-shot
setup/configuration handshakes (fetch the static scenario, reset a run) are also
services, since "ask once, get an answer" isn't a stream either.

### 2.3 Clock ownership (DECISION)

**DECISION:** `mmg_node` owns a `1/dt` (10 Hz) timer and is the only node with a
periodic timer driving the control loop. Each tick it: calls `/nmpc/solve` and
blocks → applies the returned `(delta, n)` → integrates one RK4 step → publishes
the new state. `nmpc_node` and `map_node` are otherwise purely reactive (service
handlers / topic subscription callbacks).

*Rationale:* in the current script, the plant integration step is what advances
"simulated time" by exactly `dt`; making the plant node the timekeeper is the
closest ROS2 analogue and keeps `dt` consistency a property of one node's timer
period rather than something that has to be independently reasoned about.

### 2.4 Waypoint-switching and termination ownership (DECISION)

**DECISION:** `select_active_waypoint()` and the `SIM_TIME_MODE` termination check
both move from `run_live.py`'s loop into `map_node`.

*Rationale:* both are "which part of the environment/reference-path are we at,"
not controller internals. Consequence: **`map_node` must subscribe to `/mmg/state`**
(it needs live position to decide leg-switching and goal-reached), in addition to
publishing. If you'd rather `nmpc_node` own waypoint-switching (arguably cleaner
since it's the one consuming the result directly), that is a self-contained change
confined to sections 2.5 (map) and 2.6 (nmpc) below — nothing else in this plan
depends on which node owns it, as long as *something* publishes `ActiveReference`
before `nmpc_node`'s solve callback needs it.

### 2.5 `map_node` — full specification

**Publishes:**

| Topic | Type | QoS | Notes |
|---|---|---|---|
| `/map/obstacles` | `nmpc_interfaces/ObstacleArray` | transient-local, depth 1 | latched; republish only if obstacles ever change at runtime |
| `/map/initial_state` | `nmpc_interfaces/VesselState` | transient-local, depth 1 | one-shot at startup |
| `/map/active_reference` | `nmpc_interfaces/ActiveReference` | reliable, depth 1 (keep-last) | republished every time waypoint-switch logic runs (~10 Hz, i.e. once per received `/mmg/state`) |
| `/map/sim_status` | `nmpc_interfaces/SimStatus` | reliable, depth 1 | `RUNNING` / `GOAL_REACHED` / `TIMEOUT`; `mmg_node` subscribes to this to know when to stop ticking |

**Subscribes:**

| Topic | Type | Used for |
|---|---|---|
| `/mmg/state` | `nmpc_interfaces/VesselState` | drives `select_active_waypoint()` re-evaluation, feeds the live visualizer, feeds the `SIM_TIME_MODE` termination check |
| `/nmpc/prediction_horizon` | `nmpc_interfaces/PredictionHorizon` | feeds the visualizer's predicted-trajectory overlay (replaces the direct `bridge.update_prediction()` call in current `run_live.py`) |
| `/nmpc/control_command` | `nmpc_interfaces/ControlCommand` | feeds the visualizer's control-horizon panel |

**Serves:**

| Service | Type | Notes |
|---|---|---|
| `/map/get_scenario` | `nmpc_interfaces/GetScenario` | one-shot full scenario handoff for late-joining nodes/tools; response built directly from the parsed scenario JSON |

**Hosts internally (no ROS interface of its own):** the existing
`mpc_visualization/mpc_bridge.py` (`MPCBridge`) and `mpc_visualization/visualizer.py`
(`MPCVisualizer`) run inside this node's process. Replace the direct method-call
wiring currently in `run_live.py` (`bridge.update_ship_state(...)`,
`bridge.update_prediction(...)`, `bridge.set_obstacles(...)`,
`bridge.set_active_waypoint(...)`) with calls made from the topic-subscription
callbacks listed above.

**Parameters (loaded from a scenario file + shared YAML):**
- `scenario_path` (string) — path to the scenario JSON (default:
  `scenario_maker/scenario.json`'s format, unchanged)
- `wp_radius` ← `NMPCConfig.WP_RADIUS` (`nmpc/config.py:78`)
- `sim_time_mode` ← `NMPCConfig.SIM_TIME_MODE` (`nmpc/config.py:85`)
- `sim_time_fixed` ← `NMPCConfig.SIM_TIME_FIXED` (`nmpc/config.py:86`)
- `max_obstacles` ← `NMPCConfig.MAX_OBSTACLES` (`nmpc/config.py:100`)

**Logic to port verbatim (no behavior change):**
- `select_active_waypoint()`, `compute_path_angle()` — both from
  `nmpc/path_following.py`, called with data now arriving via topic instead of
  local variables.
- The `SIM_TIME_MODE` branch logic currently in `run_live.py`'s
  `run_scenario_live()` (the `"infinite"` / `"fixed"` / `"endpoint"` handling,
  including the endpoint distance check against `waypoints[last_idx]`).

### 2.6 `nmpc_node` — full specification

**Serves:**

| Service | Type | Notes |
|---|---|---|
| `/nmpc/solve` | `nmpc_interfaces/SolveNMPC` | request carries the *current* vessel state (including previous `delta, n` — see `VesselState` shape below, which already has these fields); response carries the control command + full prediction. Internally: pulls the latest cached `/map/active_reference` and `/map/obstacles` values (received via the subscriptions below) to fill in `chi_p, x_d, y_d, obstacles` before calling the existing `solve()` method on whichever backend (`AcadosNMPC` from `nmpc_acados.py` or `CasadiNMPC` from `nmpc_casadi.py`) is configured. |

**Publishes (broadcast copies of the same solve result, for passive consumers — not the authoritative path, which is the service response above):**

| Topic | Type |
|---|---|
| `/nmpc/control_command` | `nmpc_interfaces/ControlCommand` |
| `/nmpc/prediction_horizon` | `nmpc_interfaces/PredictionHorizon` |
| `/nmpc/solver_status` | diagnostic message: `success` (bool), `return_status` (string), `solve_time` (float64) |

**Subscribes:**

| Topic | Type | Used for |
|---|---|---|
| `/map/active_reference` | `nmpc_interfaces/ActiveReference` | cached, consumed by the next `/nmpc/solve` call |
| `/map/obstacles` | `nmpc_interfaces/ObstacleArray` | cached, consumed by the next `/nmpc/solve` call |

**Parameters** (the entire current `NMPCConfig` dataclass, `nmpc/config.py:48-149`,
becomes ROS2 parameters declared in this node and loaded from the shared YAML):
`N`, `dt`, `LPP`, `R_ASV_FACTOR`, `DELTA_MIN/MAX`, `DELTA_DOT_MIN/MAX`,
`RPS_MIN/MAX`, `RPS_DOT_MIN/MAX`, `U_REF`, `DELTA_TRIM`, `N_TRIM`, `BRAKE_DISTANCE`,
`U_REF_MIN`, `EPS`, `SIGMA`, `W_SLACK`, `OBSTACLE_START_K`, `MAX_OBSTACLES`,
`Q_DIAG`, `R_DIAG`, `QE_SCALE`, `IPOPT_*` (if CasADi backend selected),
`ACADOS_*` (if acados backend selected). Add one new parameter not in the current
dataclass: `solver_backend` (`"acados"` or `"casadi"`) — replaces today's
`--solver` CLI flag on `run_live.py`.

**Logic to port verbatim:** the bodies of `AcadosNMPC.solve()`
(`nmpc/nmpc_acados.py:193-246`) and `CasadiNMPC.solve()`
(`nmpc/nmpc_casadi.py:210-...`, same shape) become the internals of the
`/nmpc/solve` service callback. `state_augmentation.py` and the solve-time helpers
in `path_following.py` (`get_reference_state()`, `compute_effective_u_ref()`,
`build_xi_full()`, `pad_obstacles()`) are unchanged, just called from inside the
service handler instead of a plain function.

**Known packaging gotcha:** `nmpc_acados.py:156` sets
`ocp.code_export_directory = config.ACADOS_CODE_EXPORT_DIR`, and line 187-188
resolves `config.ACADOS_JSON_FILE` relative to `os.path.dirname(__file__)` before
triggering C code-gen + compilation on first solver construction. Under `ros2 run`,
the process working directory convention differs from running the script directly —
resolve these paths from the installed package's share directory (e.g. via
`ament_index_python.packages.get_package_share_directory`) or from an absolute
parameter, not a path relative to the working directory, or first-solve code
generation will fail to find/write its output.

### 2.7 `mmg_node` — full specification

**Publishes:**

| Topic | Type | Notes |
|---|---|---|
| `/mmg/state` | `nmpc_interfaces/VesselState` | the single source of ground truth, once per tick |

**Calls (client):**

| Service | Type | Notes |
|---|---|---|
| `/nmpc/solve` | `nmpc_interfaces/SolveNMPC` | called once per timer tick; blocks for the response |

**Serves:**

| Service | Type | Notes |
|---|---|---|
| `/mmg/reset` | `nmpc_interfaces/ResetSim` | reinitializes state without relaunching nodes |

**Subscribes:**

| Topic | Type | Used for |
|---|---|---|
| `/map/sim_status` | `nmpc_interfaces/SimStatus` | stops the timer's `/nmpc/solve` calls once status is not `RUNNING` |

**Parameters:**
- `dt` ← `NMPCConfig.dt` (`nmpc/config.py:54`) — **must be numerically identical to
  `nmpc_node`'s `dt` parameter.** Put this value once in the shared YAML params
  file and have both nodes' launch entries reference the same key, rather than two
  independently-set values that can drift apart.

**Logic to port verbatim:** `make_casadi_integrator()` and
`MMG_Time_Derivative_casadi()`, both in `casadi_mmg_solver/casadi_mmg.py`, are
unchanged — the timer callback replaces the current `plant_step(state_ca,
control_ca)` call at `run_live.py:92-95`, using the `(delta, n)` returned by the
`/nmpc/solve` service response instead of the local variable.

---

## 3. Interface package: `nmpc_interfaces` (ament_cmake, NOT ament_python)

**Important:** ROS2 cannot generate custom `.msg`/`.srv` types from a pure-Python
(`ament_python`) package — that requires `rosidl_generate_interfaces` in an
`ament_cmake` package, even though every node in this project is plain Python. This
package contains only interface definitions, no node code.

### 3.1 Messages

**`VesselState.msg`**
```
std_msgs/Header header
float64 u
float64 v
float64 r
float64 x
float64 y
float64 psi
float64 delta
float64 n
```
(8 fields: the 6-dim MMG state `[u,v,r,x,y,psi]` plus the 2 actuator states
`delta, n` — the NMPC needs the actuator state too, since its internal augmented
state includes actuator position, see `build_xi_full()` in `path_following.py:164`.)

**`ControlCommand.msg`**
```
std_msgs/Header header
float64 delta
float64 n
```

**`PredictionHorizon.msg`**
```
std_msgs/Header header
int32 n_states        # = 11 (STATE_DIM, nmpc/config.py:22)
int32 n_controls       # = 2  (CONTROL_DIM, nmpc/config.py:23)
int32 horizon_len      # = N, nmpc/config.py:53
float64[] xi_traj       # flattened (n_states, horizon_len+1), row-major
float64[] control_horizon  # flattened (horizon_len, n_controls), row-major
```
(ROS2 message arrays are 1-D; flatten/reshape at the publish/subscribe boundary.
`xi_traj` corresponds to `result["xi_traj"]` from `solve()`; `control_horizon`
corresponds to `result["u_opt"]` transposed, matching what `run_live.py` currently
extracts as `predicted_trajectory`/`control_horizon` for the bridge.)

**`Obstacle.msg`**
```
string id
float64 x
float64 y
float64 radius
```

**`ObstacleArray.msg`**
```
std_msgs/Header header
Obstacle[] obstacles
```

**`ActiveReference.msg`**
```
std_msgs/Header header
float64 chi_p
float64 x_d
float64 y_d
int32 target_idx
```

**`SimStatus.msg`**
```
std_msgs/Header header
uint8 RUNNING = 0
uint8 GOAL_REACHED = 1
uint8 TIMEOUT = 2
uint8 status
float64 sim_time
```

*Alternative worth considering instead of a custom waypoints message:* ROS2's
standard `nav_msgs/msg/Path` could carry the full waypoint list instead of a custom
type, at the cost of being a 3D-pose/quaternion shape for what's really flat 2D
data — but it gets free RViz2 visualization if that's ever wanted alongside/instead
of the existing matplotlib visualizer. Not required by this plan either way, since
`GetScenario.srv` below returns waypoints as parallel float arrays.

### 3.2 Services

**`SolveNMPC.srv`**
```
VesselState state
---
ControlCommand command
PredictionHorizon horizon
bool success
string return_status
float64 solve_time
```

**`GetScenario.srv`**
```
---
VesselState initial_state
float64[] waypoints_x
float64[] waypoints_y
Obstacle[] obstacles
float64 sim_time_fixed
```

**`ResetSim.srv`**
```
VesselState new_initial_state
bool use_default
---
bool success
```

---

## 4. Package/workspace layout

```
ros2_ws/src/
├── nmpc_interfaces/            (ament_cmake)
│   ├── msg/
│   │   ├── VesselState.msg
│   │   ├── ControlCommand.msg
│   │   ├── PredictionHorizon.msg
│   │   ├── Obstacle.msg
│   │   ├── ObstacleArray.msg
│   │   ├── ActiveReference.msg
│   │   └── SimStatus.msg
│   ├── srv/
│   │   ├── SolveNMPC.srv
│   │   ├── GetScenario.srv
│   │   └── ResetSim.srv
│   ├── CMakeLists.txt
│   └── package.xml
└── nmpc_sim_nodes/              (ament_python)
    ├── nmpc_sim_nodes/
    │   ├── map_node.py
    │   ├── nmpc_node.py
    │   └── mmg_node.py
    ├── params/
    │   └── sim_params.yaml      # single shared source for dt, N, weights, bounds, etc.
    ├── launch/
    │   └── bringup.launch.py
    ├── setup.py                  # three console_scripts entry points
    └── package.xml
```

One combined Python package for all three nodes (matching the current single-repo
style) rather than three separate packages. Split later if independent
versioning/reuse of one node becomes necessary — nothing in this plan depends on
which packaging choice is made, as long as all three end up as independently
launchable/runnable nodes.

Existing directories `nmpc/`, `casadi_mmg_solver/`, `mpc_visualization/`,
`scenario_maker/` are **not deleted** — their logic is imported into (or copied
into, if import paths become inconvenient inside the ROS2 package) the three node
files as described in section 5 below. `scenario_maker/scenario_editor.py` remains
an untouched, standalone offline authoring tool; it does not become a ROS2 node.

---

## 5. File-by-file migration map

| Existing file | Destination | What changes |
|---|---|---|
| `nmpc/nmpc_acados.py` | `nmpc_node.py` (or imported by it) | `AcadosNMPC.solve()` body becomes the `/nmpc/solve` service callback body. Fix the code-export path issue noted in section 2.6. |
| `nmpc/nmpc_casadi.py` | `nmpc_node.py` (or imported by it) | same shape, becomes the alternate backend when `solver_backend` param = `"casadi"` |
| `nmpc/config.py` | `nmpc_node.py` params + `params/sim_params.yaml` | dataclass fields become ROS2 parameter declarations; `DEFAULT_CONFIG`'s values become the YAML defaults |
| `nmpc/state_augmentation.py` | unchanged, imported by `nmpc_node.py` | no changes |
| `nmpc/path_following.py` — `get_reference_state`, `compute_effective_u_ref`, `build_xi_full`, `pad_obstacles`, `wrap_to_pi`, `wrap180_casadi` | unchanged, imported by `nmpc_node.py` | no changes — still solve-time helpers |
| `nmpc/path_following.py` — `select_active_waypoint`, `compute_path_angle`, `compute_cross_track_error` | moved to `map_node.py` | per DECISION in section 2.4 |
| `casadi_mmg_solver/casadi_mmg.py` | unchanged, imported by `mmg_node.py` | no changes |
| `mpc_visualization/mpc_bridge.py`, `mpc_visualization/visualizer.py` | unchanged, imported by `map_node.py` | wiring changes from direct method calls to topic-callback-triggered calls; internals unchanged |
| `scenario_maker/scenario.json` (format) | read by `map_node.py` at startup | unchanged format, now read via a `scenario_path` ROS2 parameter instead of a CLI `--scenario` argument |
| `scenario_maker/scenario_editor.py` | unchanged | stays a standalone offline tool, not part of the ROS2 graph |
| `nmpc/run_live.py` | **retired** | logic redistributed: main loop → `mmg_node`'s timer + service call chain; `SIM_TIME_MODE` termination check → `map_node`'s `/map/sim_status` publisher; `--speed` playback pacing → dropped, or reworked as a `sim_rate` parameter on `mmg_node`'s timer if slowed-down-from-realtime playback is still wanted (the matplotlib visualizer's own redraw cadence already paces the *display* independently of the physics timer) |

---

## 6. Parameter reference (current `NMPCConfig` field → owning node → param name)

All names below are proposed 1:1 with the current dataclass field names
(lower-cased, ROS2 parameter convention), so cross-referencing `nmpc/config.py`
against the shared YAML stays trivial.

| `NMPCConfig` field | Current value | Owning node(s) |
|---|---|---|
| `N` | 200 | `nmpc_node` |
| `dt` | 0.1 | `nmpc_node` **and** `mmg_node` — must match |
| `LPP` | 2.902 | `nmpc_node` |
| `R_ASV_FACTOR` | 0.7 | `nmpc_node` |
| `DELTA_MIN/MAX`, `DELTA_DOT_MIN/MAX` | ±45°, ±30°/s | `nmpc_node` |
| `RPS_MIN/MAX`, `RPS_DOT_MIN/MAX` | ±18.2, ±5.0 | `nmpc_node` |
| `U_REF`, `DELTA_TRIM`, `N_TRIM` | 0.78, 0.0, 10.0 | `nmpc_node` |
| `WP_RADIUS` | 2.0 | `map_node` |
| `SIM_TIME_MODE`, `SIM_TIME_FIXED` | `"endpoint"`, 300.0 | `map_node` |
| `BRAKE_DISTANCE`, `U_REF_MIN` | 8.0, 0.05 | `nmpc_node` |
| `EPS` | 1e-6 | `nmpc_node` |
| `SIGMA`, `W_SLACK`, `OBSTACLE_START_K`, `MAX_OBSTACLES` | 0.2, 50.0, 1, 5 | `nmpc_node` (`MAX_OBSTACLES` also needed by `map_node` to size `ObstacleArray` handling) |
| `Q_DIAG`, `R_DIAG`, `QE_SCALE` | see file | `nmpc_node` |
| `IPOPT_*` | see file | `nmpc_node` (CasADi backend only) |
| `ACADOS_*` | see file | `nmpc_node` (acados backend only) |

New parameters not derived from `NMPCConfig`:
- `nmpc_node`: `solver_backend` (`"acados" | "casadi"`)
- `map_node`: `scenario_path` (string)
- `mmg_node`: none beyond `dt`

---

## 7. Action plan (execution order)

1. Scaffold `ros2_ws/src/nmpc_interfaces` and `ros2_ws/src/nmpc_sim_nodes` per the
   layout in section 4. Confirm `colcon build` succeeds with empty/stub content
   before writing any real logic.
2. Write all `.msg`/`.srv` files from section 3. Build, then confirm each type
   with `ros2 interface show nmpc_interfaces/msg/<Name>` and
   `ros2 interface show nmpc_interfaces/srv/<Name>`.
3. Implement `mmg_node.py` first — it has the simplest dependency graph (no
   service serving of its own beyond `/mmg/reset`, just a timer + one service
   client + one publisher). Test standalone against a throwaway `/nmpc/solve`
   server stub that returns a fixed `(delta, n)`, to validate the
   timer → service-call → integrate → publish cycle before real NMPC is involved.
4. Implement `nmpc_node.py` — wrap the existing `solve()` bodies (section 2.6) in
   the `/nmpc/solve` service handler, plus the two caching subscriptions. Test
   with `ros2 service call /nmpc/solve nmpc_interfaces/srv/SolveNMPC "{...}"` by
   hand, feeding a fabricated state, before connecting it to `mmg_node`.
5. Implement `map_node.py` last (most moving parts: scenario JSON loading,
   waypoint-switching port, `sim_status` port, hosting the visualizer). Test its
   publishers and `/map/get_scenario` service in isolation before wiring the live
   plot to the topic callbacks.
6. Write `launch/bringup.launch.py` and `params/sim_params.yaml`; bring all three
   nodes up together. Explicitly verify `dt` is identical between `mmg_node` and
   `nmpc_node`'s resolved parameter values — this is the most likely silent bug
   in this migration.
7. Validate against an existing scenario (`scenario_maker/scenario.json`) and
   compare the resulting trajectory/plot against a known-good run of the current
   `nmpc/run_live.py` on the same scenario, to catch any behavioral drift
   introduced by the restructuring.
8. **(Optional) Add an RViz2 visualization layer.** None of the custom
   `nmpc_interfaces` messages are natively renderable in RViz2 — it only has
   built-in displays for standard types (`Marker`/`MarkerArray`, `Path`,
   `Odometry`, `PoseStamped`, etc.). Rather than changing any interface defined
   in section 3, have `map_node` publish additional "shadow" topics, in
   parallel with the existing custom-type ones, purely for visualization:
   - `/mmg/state` (`VesselState`) → republish as `nav_msgs/Odometry`:
     `pose.pose.position` = `(x, y, 0)`, `pose.pose.orientation` = quaternion
     from yaw `psi` (roll = pitch = 0), `twist.twist.linear.x/y` = `u, v`,
     `twist.twist.angular.z` = `r`. RViz's Odometry display draws the pose
     arrow and can keep a trailing history (free trajectory trace). `delta`/`n`
     have no spatial slot in `Odometry` — omit them from this shadow topic.
   - `/nmpc/prediction_horizon` (`PredictionHorizon`) → unflatten the
     `(x, y, psi)` columns of `xi_traj` into a `nav_msgs/Path`
     (`PoseStamped[]`); RViz's Path display renders it as the predicted-
     trajectory overlay. `control_horizon` (delta/n over time) isn't spatial —
     leave it to `rqt_plot`/PlotJuggler instead.
   - The static waypoint list → publish once (transient-local) as a
     `nav_msgs/Path`, per the alternative already noted in section 3.1.
   - `/map/obstacles` (`ObstacleArray`) → one `visualization_msgs/Marker`
     (`CYLINDER` or `SPHERE`) per obstacle in a `MarkerArray`:
     `pose.position` = `(x, y, 0)`, `scale.x = scale.y = 2 * radius`, thin
     `scale.z`.
   - `/map/active_reference` (`ActiveReference`) → `geometry_msgs/PoseStamped`:
     position = `(x_d, y_d, 0)`, orientation = quaternion from `chi_p`.
     Optionally add a `TEXT_VIEW_FACING` marker for `target_idx`.
   - `ControlCommand`, `SimStatus`, and the solver-status diagnostics have no
     spatial content — they're not a good fit for RViz2 regardless of format;
     keep those on `rqt_plot`/PlotJuggler/logging rather than forcing them
     into a marker.
   - Services (`SolveNMPC`, `GetScenario`, `ResetSim`) have no RViz
     representation at all — RViz only subscribes to topics. If scenario data
     from `GetScenario` needs to be visible in RViz, it must also exist as a
     latched topic in one of the standard types above, not just as a service
     response.

---

## 8. Forward-compatibility notes (context only — not part of this conversion's scope)

Two future additions are anticipated and should not require re-architecting this
node graph when they land:

- **Sensor noise** will sit between `mmg_node`'s true-state output and
  `nmpc_node`'s consumption of it — e.g. as a filter on the `/mmg/state` →
  `/nmpc/solve` path, or a fourth node/topic inserted transparently between them.
  Nothing in this plan should assume `nmpc_node` receives the *unmodified* output
  of `mmg_node` — it should only assume it receives *a* `VesselState` on the
  service request.
- **Current/wave disturbance forces** will be added inside `mmg_node`'s plant
  integration only (mirroring the current `F_Drift` hook already present but
  zeroed in `casadi_mmg.py` line 188) — never inside `nmpc_node`'s internal
  model, so the controller continues to reject them blind, same as a real vessel.

Do not implement either of these now; this section exists only so the node
boundaries chosen above aren't accidentally drawn in a way that would need
undoing later.
