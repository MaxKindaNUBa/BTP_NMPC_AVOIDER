# Action plan: current & wave disturbances as a separate, toggleable piece of the sim

Status: **plan only, no code written yet.** This document is meant to be handed
to a fresh chat/session to execute. It assumes familiarity with the repo's
existing architecture (`README.md`'s "Architecture" and "The vessel dynamics
model (MMG)" sections) but repeats the load-bearing facts inline so it's
self-contained.

## 0. What already exists (context the plan builds on)

- **One shared dynamics function**, `MMG_Time_Derivative_casadi()` in
  `nmpc_ws/src/nmpc_sim_nodes/nmpc_sim_nodes/casadi_mmg_solver/casadi_mmg.py`,
  used by two call sites:
  - `nmpc/state_augmentation.py` — the NMPC's **internal prediction model**
    (imports `MMG_Time_Derivative_casadi` directly).
  - `mmg_node.py` — the **plant** (calls it via `make_casadi_integrator()`'s
    RK4 wrapper).
- `Preliminary_func.py` (the older NumPy "ground truth" model, in the
  `mmg_model_validation` package) already has a **wave-drift term**, driven by
  `Wave_Data/*.mat` (non-dimensional drift coefficient tables `D_X, D_Y,
  D_N(heading, frequency)`), but only evaluated at a single fixed
  regular-wave frequency/amplitude — not ported into `casadi_mmg.py` at all.
- **No current model exists anywhere in the repo today.**
- Existing pattern for "a node that sits in the tick loop and both serves a
  synchronous per-tick service *and* republishes the same result on a passive
  topic for visualization" is `sensor_node.py` (`/sensor/measure` service +
  `/sensor/measured_state` topic). This plan's `env_node` follows that same
  shape deliberately.
- `mmg_node.py`'s `_tick()` already does one synchronous service call per tick
  (`/nmpc/solve`, busy-wait pattern on a separate `ReentrantCallbackGroup` so
  it doesn't deadlock the executor). The new `/env/disturbance` call is a
  second instance of that exact pattern.

## 1. Design decision: one new node, one new module dir, inside the existing package

Not a new ROS package. A new node (`env_node`) and a new Python module dir
(`env_model/`, sibling to `sensor_model/`) inside the existing
`nmpc_sim_nodes` package — this mirrors exactly how sensor noise was added
(`sensor_node.py` + `sensor_model/`), keeps one `colcon build`, one
`sim_params.yaml`, and reuses the existing `_pkg_paths.py` import wiring.
"Separate piece of code" is satisfied by: current and waves are **not**
edited into `casadi_mmg.py`'s existing force terms — they arrive as new
*optional, default-zero* inputs, computed entirely outside that file, in
`env_model/` and `env_node.py`.

## 2. New ROS interfaces (`nmpc_interfaces` package)

Add to `nmpc_ws/src/nmpc_interfaces/msg/`:

**`CurrentState.msg`**
```
std_msgs/Header header
float64 vx        # earth-frame current velocity x-component [m/s]
float64 vy        # earth-frame current velocity y-component [m/s]
float64 speed     # derived: sqrt(vx^2+vy^2) [m/s], for convenience
float64 heading   # derived: atan2(vy,vx) [rad], for convenience
bool enabled       # echoes env_node's current_enabled param
```

**`WaveState.msg`**
```
std_msgs/Header header
float64 hs             # significant wave height [m], scaled-model units
float64 tp             # peak period [s]
float64 mean_heading   # earth-fixed mean wave direction [rad]
float64 fx             # body-frame surge drift force applied this tick [N]
float64 fy             # body-frame sway drift force applied this tick [N]
float64 fn             # body-frame yaw drift moment applied this tick [N*m]
bool enabled            # echoes env_node's wave_enabled param
```

Add to `nmpc_ws/src/nmpc_interfaces/srv/`:

**`GetEnvDisturbance.srv`**
```
VesselState state
---
CurrentState current
WaveState wave
```
(Request carries the vessel's current true state because the wave force
needs `psi` — relative wave heading is `mean_heading - psi` — and future
current models could use `x,y` for spatially-varying fields. Response is the
same pair of messages also published on the passive topics below, exactly
matching `MeasureState.srv` / `/sensor/measured_state`'s dual pattern.)

Register both in `nmpc_ws/src/nmpc_interfaces/CMakeLists.txt`'s
`rosidl_generate_interfaces()` call, alongside the existing entries.

## 3. New module: `env_model/` (mirrors `sensor_model/`'s layout)

`nmpc_ws/src/nmpc_sim_nodes/nmpc_sim_nodes/env_model/`

- **`__init__.py`** — empty, matches `sensor_model/__init__.py`.

- **`config.py`** — two frozen dataclasses:
  - `CurrentConfig(mean_vx, mean_vy, time_constant_s, sigma, seed)`
  - `WaveConfig(hs, tp, gamma, mean_heading, num_components, seed)`

- **`current_model.py`** — `CurrentModel` class:
  - Holds a 2D state `(vx, vy)` initialized to `(mean_vx, mean_vy)`.
  - `step(dt)`: one Euler–Maruyama update of an Ornstein-Uhlenbeck process,
    **mean-reverting toward `(mean_vx, mean_vy)`**, on each Cartesian
    component independently:
    `v[k+1] = v[k] + dt * (-(v[k]-mean)/Tc) + sqrt(dt) * sigma * N(0,1)`.
    Cartesian, not polar, specifically to avoid heading-wraparound math (see
    conversation background — this was deliberately chosen over a
    speed+heading OU pair).
  - Own `np.random.default_rng(seed)` instance — never touches global numpy
    RNG state, matching `sensor_model/noise_primitives.py`'s convention.
  - Returns `(vx, vy)` each call; `env_node` derives `speed`/`heading` for
    the message.

- **`wave_model.py`** — `WaveModel` class:
  - At construction: loads `Wave_Data/*.mat` via the **same** path convention
    `Preliminary_func.py` already uses (`NMPC_WAVE_DATA_DIR` env var,
    defaulting to the repo's `Wave_Data/` dir), builds the same
    `RegularGridInterpolator`s over `(heading, frequency)` for `D_X, D_Y,
    D_N`.
  - Discretizes a **JONSWAP spectrum** from `(hs, tp, gamma)` into
    `num_components` frequency bins `omega_i` with spectral density
    `S(omega_i)` — the Pierson-Moskowitz base spectrum shaped by JONSWAP's
    peak-enhancement factor `gamma` (peakedness), which the simpler
    two-parameter Bretschneider spectrum doesn't have. `gamma` defaults to
    the standard `3.3` (JONSWAP's original North Sea fit) but is exposed as
    its own config value since it materially changes how concentrated the
    energy is around `tp` — relevant here because a narrower spectrum
    produces a more coherent, slower-beating drift force, while a wider one
    averages toward the mean drift faster. Draws `num_components` random
    phases once at construction (seeded RNG, reproducible per run — same
    reasoning as `current_model.py`'s own RNG).
  - `force(psi)`: given current vessel heading, computes relative wave
    heading `mean_heading - psi`, looks up `D_X/D_Y/D_N` at each `omega_i`
    for that heading, and returns the **mean drift force** (`2 * sum(D_i *
    S_i * domega)`) **plus** the **slowly-varying drift force** via Newman's
    approximation (`2 * sum_ij sqrt(S_i S_j) sqrt(D_i D_j) cos((omega_i -
    omega_j) t + phase_i - phase_j)`, `t` tracked internally as an elapsed-
    time counter incremented once per `force()` call at the node's `dt`).
    Applies the same non-dimensionalization/scale factors (`Lambda`,
    `FS_force`, `FS_moment`) `Preliminary_func.py` already uses for this
    hull, reusing that logic rather than re-deriving it.
  - This is the promised upgrade from "one fixed regular wave" (current
    `Preliminary_func.py` behavior) to "a real Hs/Tp sea state" — no new
    hydrodynamic data required, same tables, richer use of them.

- **`test_env_model.py`** (new standalone executable, same spirit as
  `sensor_model/test_sensor_model.py`): drives `CurrentModel`/`WaveModel`
  standalone (no ROS graph needed) for a few hundred seconds, plots the
  current vector wandering and the wave force time series, saves under
  `~/nmpc_sim_logs/test_env_model_results/`. Useful to sanity-check
  parameter choices (`Tc`, `sigma`, `hs`, `tp`, `gamma`) before wiring into
  the live sim. This is a **unit-level** test of `env_model/` in isolation —
  it does not touch the NMPC or the plant integrator. See section 3.1 below
  for the full-pipeline counterpart.

### 3.1 New executable: `test_closed_loop_env` (full-pipeline, headless, accelerated)

Directly analogous to the existing `test_closed_loop_noise.py`
(`nmpc_ws/src/nmpc_sim_nodes/nmpc_sim_nodes/test_closed_loop_noise.py`) —
same structure, same conventions, new file at the same top level (**not**
inside `env_model/`, matching how `test_closed_loop_noise.py` also lives at
the top level rather than inside `sensor_model/`, since it spans multiple
subpackages: `nmpc`, `casadi_mmg_solver`, and now `env_model`):

- **No rclpy, no ROS graph, no `env_node`.** Exactly like
  `test_closed_loop_noise.py` bypasses `sensor_node` and calls `SensorModel`
  directly, this bypasses `env_node` and calls `CurrentModel`/`WaveModel`
  directly inside the loop — one `.step(dt)` / `.force(psi)` call per
  iteration, right next to the existing `plant_step(...)` call.
- **Accelerated clock**: reuses the exact same timing approach already in
  `test_closed_loop_noise.py` — no `create_timer`/wall-clock throttling
  anywhere; the loop runs every tick back-to-back as fast as the CPU
  allows, and the script reports `sim_time / wall_time` as the "effective
  speedup" at the end, same as the noise test already does. This satisfies
  "accelerated clock time" using the pattern the repo already established,
  rather than inventing a second timing convention.
- **Runs over the package's own `scenario.json`** (same
  `get_package_share_directory("nmpc_sim_nodes")/params/scenario.json`
  default, same `--scenario-path` override flag as `test_closed_loop_noise.py`).
- Two runs, same comparison structure as the noise test: (1) both
  `current`/`wave` disabled — must reproduce today's plant behavior exactly
  (this doubles as part of the regression check in section 10); (2)
  `current` and `wave` both enabled, using `env_model`'s config defaults
  (or `sim_params.yaml`'s `env_node` section values, loaded the same way
  `_build_solver()` already loads `DEFAULT_CONFIG`) — `plant_step(...,
  with_env=True)` fed the live `CurrentModel`/`WaveModel` outputs each
  step.
- **Saves the resultant path** the same way: a comparison plot (both runs'
  executed `(x, y)` trajectories over the scenario's waypoints/obstacles,
  plus each run's stop status and speedup) written to
  `~/nmpc_sim_logs/test_closed_loop_env_results/path_comparison.png` —
  exact same `RESULTS_DIR` / `plot_comparison()` pattern as
  `test_closed_loop_noise.py`, new directory name only.
- CLI flags mirroring the noise test: `--scenario-path`, `--seed` (for
  `CurrentModel`/`WaveModel`'s RNGs), `--results-dir`, plus
  `--current-only` / `--wave-only` switches so either disturbance can be
  isolated in run 2 without editing the script (defaults to both enabled,
  matching "the entire mmg model-nmpc stack ... with current and wave" as
  the headline scenario).

## 4. Core dynamics change: `casadi_mmg.py` (additive, backward-compatible)

**`MMG_Time_Derivative_casadi(state, control, current=(0.0, 0.0),
wave_force=(0.0, 0.0, 0.0), smooth=True, scale=1000.0)`**

- New optional kwargs, plain-Python defaults `(0.0, 0.0)` / `(0.0, 0.0, 0.0)`
  — CasADi transparently mixes MX/SX symbols with Python float constants in
  arithmetic, so no wrapping needed for the default (disturbance-free) path.
- **Current**: `vcx, vcy = current` (earth-frame). Convert to body frame using
  the state's own `psi`: `uc = vcx*cos(psi) + vcy*sin(psi)`, `vc =
  -vcx*sin(psi) + vcy*cos(psi)`. Relative velocity `ur = u_val - uc`, `vr = v
  - vc`. **Substitute `ur, vr` in place of `u_val, v`** everywhere they
  currently feed hydrodynamic quantities: `U`/`beta` (hull), `v_ndm`/`r_ndm`
  (hull force polynomial), `beta_P`/`Jp` (propeller inflow), `beta_R`/`uR`
  (rudder inflow). **Do not** touch the mass matrix `M`, `LHS_r`, or the
  kinematics block (`R_mat` / `Vel_Mom`) — those stay in terms of the
  absolute `u_val, v, r` so the vessel's earth-fixed position correctly
  drifts with the current while rigid-body momentum stays physically
  consistent. (See conversation background for why this split, not a full
  Fossen `M_RB`/`M_A` decomposition, was chosen — it's the standard
  MMG-plus-current approximation in the literature and doesn't require
  restructuring the already-validated mass matrix.)
- **Wave**: `wave_force` is added directly into `RHS` before the `ca.solve(M,
  RHS)` call: `RHS = F_hull + F_propellar + F_rudder - LHS_r + F_Drift +
  ca.vertcat(*wave_force)` (replaces the existing always-zero `F_Drift`
  placeholder already sitting in this function).
- **Existing call site `nmpc/state_augmentation.py`'s `MMG_Time_Derivative_casadi(s,
  c)` needs zero changes** — both new args default to no disturbance, so the
  NMPC's internal prediction model is byte-for-byte unaffected unless someone
  later deliberately passes non-default values there.

**`make_casadi_integrator(h, method="rk4", smooth=True, scale=1000.0,
sym_type=ca.MX, with_env=False)`**

- New `with_env: bool = False` kwarg. When `False` (the default — every
  existing call site: `test_nmpc.py`, `test_closed_loop_noise.py`,
  `sensor_model/test_sensor_model.py`, `casadi_mmg.py`'s own `__main__`
  demo), behavior and the exported `ca.Function`'s input signature
  (`state, control -> next_state, r_dot_a`) are **completely unchanged** —
  none of those files need editing.
- When `True` (only `mmg_node.py` will set this), the built `ca.Function`
  gains two more required inputs, `current(2,)` and `wave_force(3,)`,
  threaded into every `K1_.../K4_` call to `MMG_Time_Derivative_casadi`
  **held constant across the RK4 substeps** — same convention already used
  for `control` (`delta, rps`) within one integration step. New signature:
  `(state, control, current, wave_force) -> (next_state, r_dot_a)`.

## 5. New node: `env_node.py`

Structured to closely mirror `sensor_node.py`:

- Params (all declared, defaults below): `dt` (must match `mmg_node.dt`,
  same "MUST match" convention already used across `sim_params.yaml`),
  `current_enabled` (bool, default `false`), `current_mean_speed`,
  `current_mean_heading`, `current_time_constant`, `current_sigma`,
  `current_seed`; `wave_enabled` (bool, default `false`), `wave_hs`,
  `wave_tp`, `wave_mean_heading`, `wave_num_components`, `wave_seed`.
- **`current_enabled` and `wave_enabled` are independent** — exactly the
  "both individually toggleable" requirement. Each `false` means that
  component's contribution is identically zero (`CurrentModel`/`WaveModel`
  simply not constructed, response fields hardcoded to zero), same
  "disabled means fully disabled, no partial mode" philosophy `sensor_node`
  already uses for its own `enabled` flag.
- Builds `CurrentModel`/`WaveModel` from `env_model/` only for whichever
  toggle is on.
- Serves `/env/disturbance` (`GetEnvDisturbance`): on each call, steps
  `CurrentModel` by `dt` (if enabled) and evaluates `WaveModel.force(psi)`
  from `request.state.psi` (if enabled), fills `CurrentState`/`WaveState`,
  **publishes both on `/env/current_state` and `/env/wave_state`** (passive
  topics, for `rviz_node`/`viz_node`/logging — this is what makes the
  disturbance visualizable every tick, independent of whether anything
  else ever calls the service), and returns the response.

## 6. `mmg_node.py` changes

- New client: `self.env_client = self.create_client(GetEnvDisturbance,
  '/env/disturbance', callback_group=self._client_cbg)` — same
  `ReentrantCallbackGroup` as `solve_client`, same busy-wait-on-`future`
  pattern already used for `/nmpc/solve`, same "warn and skip this tick if
  not ready yet" guard.
- Build `self.plant_step = make_casadi_integrator(self.dt, method='rk4',
  sym_type=ca.SX, with_env=True)` in `__init__` (only line in this file
  that changes from what exists today).
- In `_tick()`, after the existing `/nmpc/solve` call and before
  integrating: call `/env/disturbance` synchronously with the vessel's
  current state, extract `(vx, vy)` from the `CurrentState` response and
  `(fx, fy, fn)` from the `WaveState` response, and pass them as the two new
  arguments to `self.plant_step(...)`.

## 7. Overall dataflow with everything enabled (sensor noise + current + wave)

One tick = one firing of `mmg_node`'s `dt` timer — it's still the sole
master clock; `map_node`, `nmpc_node`, `sensor_node`, `env_node` are all
otherwise passive, reacting only to service calls or topic updates.

```
mmg_node._tick()  [owns self.mmg_state = TRUE state, pre-integration]
│
├─ 1. /nmpc/solve  (mmg_node -> nmpc_node, sync, busy-wait)
│      request.state = TRUE state (this tick's self.mmg_state)
│      │
│      └─ inside nmpc_node.solve():
│            ├─ /sensor/measure (nmpc_node -> sensor_node, sync)
│            │      request.true_state = TRUE state
│            │      sensor_node runs GPS/compass/gyro/actuator noise model,
│            │      publishes /sensor/measured_state (passive topic),
│            │      returns response.measured_state = NOISY state
│            │
│            └─ NMPC (acados/CasADi) solves the OCP against the NOISY
│               state only -- the optimizer's internal MMG prediction model
│               is the zero-default MMG_Time_Derivative_casadi(s,c) call in
│               state_augmentation.py, i.e. calm water, no current, no
│               wave, ever. It has no idea current/wave exist.
│
│      mmg_node receives response.command = (delta, n)
│
├─ 2. /env/disturbance  (mmg_node -> env_node, sync, busy-wait)
│      request.state = the SAME pre-integration TRUE state used in step 1
│      │
│      └─ inside env_node:
│            ├─ CurrentModel.step(dt)  -> (vx, vy), OU process advances one step
│            ├─ WaveModel.force(psi)   -> (fx, fy, fn), JONSWAP mean + Newman
│            │                           slow-drift force at this tick's true psi
│            ├─ publishes /env/current_state  (passive topic)
│            ├─ publishes /env/wave_state     (passive topic)
│            └─ returns response.current, response.wave
│
├─ 3. Integration (local, no service call)
│      next_state, _ = plant_step(TRUE state, (delta,n), (vx,vy), (fx,fy,fn))
│      -- this is make_casadi_integrator(..., with_env=True): current enters
│         via the relative-velocity substitution inside the hull/propeller/
│         rudder terms, wave force is added straight into the RHS before the
│         M^-1 solve. Both disturbances are now physically baked into the
│         vessel's actual next-step motion.
│      self.mmg_state = next_state   (this becomes next tick's TRUE state)
│
└─ 4. publishes /mmg/state  (the new TRUE state, disturbance-affected)
```

Independent of this per-tick sequence: `map_node` watches `/mmg/state` for
waypoint-arrival/termination bookkeeping and publishes
`/map/active_reference` (`chi_p, x_d, y_d`), which `nmpc_node` reads before
each solve. `rviz_node`/`viz_node` (whichever is running) passively
subscribe to `/mmg/state`, `/sensor/measured_state`, `/env/current_state`,
`/env/wave_state` — this is what makes both disturbances visualizable every
tick, without being in the control loop at all.

**The one fact that matters most in this whole design**: the NMPC never
sees current or wave, at any point, even indirectly through this tick's
inputs — it only ever solves against the noisy *measured* state. The only
channel by which disturbances influence the controller's behavior is
closed-loop feedback: this tick's current+wave push the true state
somewhere the calm-water model didn't expect, sensor noise gets layered on
top of that already-perturbed state, and next tick's solve starts from
that shifted (and noisy) starting point and corrects toward the path
again. So tracking-error growth under "everything enabled" is a test of
feedback robustness, not something the optimizer is silently cheating on by
having disturbance knowledge baked into its own prediction model.

Also worth being explicit about: **sensor noise and current/wave are
applied at different stages** — sensor noise corrupts what the controller
*observes* (before solving, step 1), current/wave corrupt what the plant
*does* (after the control decision is already made, during integration,
steps 2-3). They compose but don't interact with each other directly; both
simply stack their independent effects onto the same underlying true-state
timeline.

## 8. `params/sim_params.yaml` — new section

```yaml
env_node:
  ros__parameters:
    dt: 0.1   # MUST match mmg_node.dt above (OU/wave-phase recursions stepped once per tick)

    current_enabled: false        # independent toggle
    current_mean_speed: 0.0       # [m/s]
    current_mean_heading: 0.0     # [rad], earth-fixed, same convention as psi
    current_time_constant: 600.0  # [s] Gauss-Markov Tc -- larger == slower wandering
    current_sigma: 0.01           # [m/s per sqrt(s)] driving noise intensity
    current_seed: 7

    wave_enabled: false           # independent toggle
    wave_hs: 0.05                 # [m] significant wave height, scaled-model units
    wave_tp: 1.2                  # [s] peak period
    wave_gamma: 3.3                # JONSWAP peak-enhancement factor (3.3 = standard North Sea fit)
    wave_mean_heading: 0.0        # [rad] earth-fixed mean wave direction
    wave_num_components: 30       # spectral discretization for Newman's approximation
    wave_seed: 11
```
Both default `false` — adding this section changes nothing about current sim
behavior until a scenario/launch override flips one or both on.

## 9. Build-system / packaging changes

- `nmpc_ws/src/nmpc_interfaces/CMakeLists.txt`: add the two new `.msg` and
  one new `.srv` filenames to `rosidl_generate_interfaces()`.
- `nmpc_ws/src/nmpc_sim_nodes/setup.py`: add `'env_node =
  nmpc_sim_nodes.env_node:main'`, `'test_env_model =
  nmpc_sim_nodes.env_model.test_env_model:main'`, and `'test_closed_loop_env =
  nmpc_sim_nodes.test_closed_loop_env:main'` to `console_scripts`.
- `nmpc_ws/src/nmpc_sim_nodes/package.xml`: add `<depend>scipy</depend>` (or
  the rosdep key actually resolved in this environment — check how
  `mmg_model_validation` currently pulls in scipy, since that package
  depends on `nmpc_sim_nodes` rather than declaring scipy itself; this repo
  may be relying on system/venv scipy without a rosdep entry today, in which
  case match whatever that existing precedent is rather than inventing a new
  one).
- `nmpc_ws/src/nmpc_sim_nodes/launch/bringup.launch.py`: add an `env_node`
  entry (same `parameters=[params_file]` pattern as the other four nodes).

## 10. Visualization hookup

`rviz_node.py` already republishes sim topics as
`visualization_msgs/MarkerArray` (that's its entire job). Add two more
subscriptions there, following whatever marker-publishing pattern it already
uses for obstacles:
- `/env/current_state` → an arrow marker at the vessel's position, direction
  = `heading`, length scaled by `speed`.
- `/env/wave_state` → a second arrow (different color) for the body-frame
  `(fx, fy)` force vector, rotated into the earth frame using the vessel's
  own `psi` for display purposes.

`viz_node.py` (the matplotlib dashboard) can likewise add a small
current-vector glyph and a wave-force time-series subplot, subscribing to
the same two topics. Exact implementation left to whoever picks this up —
noted here only so the two visualization entry points aren't missed, since
neither is in `bringup.launch.py` (both are launched separately per
`README.md`'s "Getting started" section).

## 11. Validation plan

1. `test_env_model` (new, section 3) — validate `CurrentModel` wanders and
   mean-reverts sensibly for chosen `Tc`/`sigma`, and `WaveModel.force()`
   magnitude/frequency content looks like a real Hs/Tp/gamma JONSWAP sea
   state, entirely outside ROS.
2. Turning-circle regression: with both toggles `false`, `mmg_node`'s
   `plant_step` output must be numerically identical to today's (same
   pattern `casadi_mmg.py`'s own `__main__` block already uses to
   cross-check CasADi vs acados) — proves the `with_env=True` plumbing adds
   zero disturbance when disabled. `test_closed_loop_env`'s own run 1
   (section 3.1, both disabled) doubles as this check at the full-pipeline
   level.
3. **`test_closed_loop_env`** (new, section 3.1) — the primary deliverable
   for this plan: full acados NMPC + MMG plant integrator, headless, run
   over the existing `scenario.json` at accelerated (unthrottled) clock
   time, disturbance-off vs. disturbance-on, saved path plot under
   `~/nmpc_sim_logs/test_closed_loop_env_results/`. This is the most direct
   answer to "how much does this actually perturb the ship."
4. Once `test_closed_loop_env` looks right headless, repeat live via
   `env_node` + `bringup.launch.py` with `env_node`'s toggles flipped on in
   `sim_params.yaml`, to confirm the ROS wiring (service + topics) matches
   the headless result.

## 12. Documentation follow-up (do this last, once code lands)

- `README.md`'s "Available executables" table: add rows for `env_node`,
  `test_env_model`, and `test_closed_loop_env`, matching the existing
  table's format exactly (per this repo's own convention of keeping that
  table in sync with `setup.py`).
- `README.md`'s architecture diagram and "The vessel dynamics model (MMG)"
  section: note the new `F_env` term and that it's plant-only.
- `Wave_Data/README.md`: update its current claim that wave-drift is "not
  ported into the CasADi/NMPC model chain" — after this work, it partially
  is (plant only, still not the NMPC's internal prediction model, but the
  wording should reflect the new `env_model/wave_model.py` consumer).

## Open decisions left to the implementer / next session

- Whether `current_mean_speed`/`current_mean_heading` should themselves be
  allowed to vary scenario-to-scenario via `scenario.json` rather than only
  `sim_params.yaml` (this plan keeps them in `sim_params.yaml` only, for
  parity with how `sensor_node`'s noise preset is configured).
- Whether to also expose a `/env/reset` service (paralleling `/mmg/reset`)
  to reseed the current/wave RNGs mid-run for repeated test scenarios
  without restarting the whole node graph.
