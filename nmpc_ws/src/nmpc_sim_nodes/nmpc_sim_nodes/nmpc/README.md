# nmpc

The actual NMPC controller: a path-following + (eventually) obstacle-avoidance
Nonlinear Model Predictive Controller built on top of the MMG vessel dynamics
(`casadi_mmg_solver/`), adapted from the two papers in `research_papers/`.
Solved with Acados/SQP-RTI, the only backend nmpc_node uses (a CasADi/IPOPT
debug backend existed early on but was removed once SQP-RTI was validated —
too slow for anything beyond one-off formulation checks).

## Package layout

| File | Role |
|---|---|
| `config.py` | `NMPCConfig` dataclass and its `DEFAULT_CONFIG` instance -- single source of truth for every tunable parameter (horizon, actuator bounds, cost weights, obstacle/slack settings, solver options). Nothing is hardcoded anywhere else. |
| `params.py` | Resolves `sim_params.yaml`'s `nmpc_node`/`map_node` ROS2 parameters into an `NMPCConfig` (`nmpc_node.py`'s own `_declare_and_build_config()` reads the same fields as ROS2 parameters instead, so this and that stay in sync by construction, not by convention). |
| `path_following.py` | Pure NumPy guidance geometry: cross-track error, course angle/error, waypoint switching (`select_active_waypoint`, `compute_path_angle`), the distance-scaled speed-reference ramp (`compute_effective_u_ref`), obstacle padding (`pad_obstacles`). Also holds `wrap180_casadi`, the one CasADi-symbolic helper this file needs for the solvers' cost functions. |
| `state_augmentation.py` | The CasADi symbolic augmented-state ODE (`augmented_dynamics_casadi`) wrapping `casadi_mmg_solver`'s MMG dynamics with the new guidance/actuator-rate rows, plus an RK4 step and a validation harness against the raw MMG model. Current (`vcx, vcy`) is threaded straight through into the MMG sub-block here -- see [Current-awareness](#current-awareness-in-the-ocp-model) below. |
| `nmpc_acados.py` | `AcadosNMPC`, the formulation compiled to a real-time SQP-RTI OCP via acados -- the only solver `nmpc_node` builds (a CasADi/IPOPT debug backend, `nmpc_casadi.py`, existed early on for formulation debugging but was removed once SQP-RTI was validated -- too slow for anything beyond one-off checks). |

Closed-loop test harnesses that exercise this package now live one level up,
in `nmpc_sim_nodes/tests/` (`test_nmpc.py`, `test_closed_loop_noise.py`,
`test_closed_loop_env.py`) -- see the top-level README's executables table --
not inside this directory. `run_live.py`/`compare_qu.py` (an early live-viz
runner and an ad-hoc `Q[u]` diagnostic script) predate the ROS2 restructuring
and no longer exist; `rviz_node`/`hud_node` are their replacements.

## The augmented state

```
xi = [e_y, sin(psi_e), cos(psi_e), r, x, y, psi, u, v, delta, n]   (11,)
u_aug = [delta_dot, n_dot]                                         (2,)
```

Folding path-following error (`e_y`, `sin/cos(psi_e)`) directly into the NMPC's
own state — rather than running a separate outer guidance loop — is the
central idea taken from the reference papers (see `research_papers/README.md`).
`delta`/`n` (rudder angle, propeller speed) are states, not controls: the
actual control inputs are their *rates* (`delta_dot`, `n_dot`), so the
optimizer is penalized for how fast it changes the actuators, not just for
using them, which is what produces smooth (rather than bang-bang) commands.
`u`/`v` are kept as real states rather than algebraically approximated, so the
formulation stays accurate through aggressive maneuvers.

`sin(psi_e)`/`cos(psi_e)` exist specifically so the *guidance* state doesn't
need to differentiate through a ±180° wraparound. The raw heading `psi` is
still tracked directly too (needed for the rotation matrix and for `x,y`
kinematics) — see the "psi wraparound" bug below for what happens if that
row's own residual isn't separately wrapped in the cost.

## Solver formulation

- **Cost**: quadratic tracking (`Q`) + control-rate penalty (`R`) + terminal
  cost (`Qe = QE_SCALE * Q`), summed over the horizon.
- **Dynamics constraint**: RK4-discretized `augmented_dynamics_casadi`,
  enforced via the OCP's own integrator.
- **Obstacle avoidance**: soft, slack-relaxed quadratic distance constraints,
  one per fixed obstacle slot (`config.MAX_OBSTACLES`, unused slots padded
  with a harmless far-away dummy via `pad_obstacles`) — never hard, since a
  hard constraint would make the NLP infeasible the moment an obstacle gets
  close.
- **Actuator bounds**: rudder angle/rate and propeller speed/rate, applied as
  state/control bounds.

## Current-awareness in the OCP model

The OCP's runtime parameter vector is `p = [chi_p, x_d, y_d, vcx, vcy,
obstacle slots...]` (`nmpc_acados.py`'s `_param_vector`/`build_acados_ocp`).
`vcx, vcy` (earth-frame current) feed directly into
`augmented_dynamics_casadi`'s call to `MMG_Time_Derivative_casadi`, so the
solver's own predicted dynamics-continuity constraint reflects the same
current-aware physics `mmg_node`'s plant integrator uses — not still water.
`AcadosNMPC.solve(..., current=(vcx, vcy))` sets this once per call and holds
it fixed across the whole horizon (a frozen-disturbance approximation, since
only a single online estimate exists, not a horizon-length forecast).
`current` defaults to `(0.0, 0.0)` if the caller doesn't pass it.

**Who actually passes a real value**: `nmpc_node.py`'s `/nmpc/solve` handler
reads `request.current` (a `CurrentState` field on `SolveNMPC.Request`) and
forwards it straight through. `mmg_node.py` populates that field from
`ukf_response.estimated_current` whenever `use_ukf=True` (falling back to the
true `/env/current_state` reading otherwise) — so in the live ROS graph this
is wired end-to-end already. Wave has no equivalent parameter anywhere in
this model and is never seen by the OCP at all.

**Why this matters, concretely**: an acados closed-loop rollout with a real
current active in the plant but this parameter left at its `(0,0)` default
(a plant/solver dynamics mismatch, not a tuning issue) fails its QP solver
on almost every tick and can time out never reaching its goal, even at a
fairly modest current speed — confirmed via `nmpc_sim_nodes/tests/test_closed_loop_noise.py`
(see its own module docstring). Wave-only disturbance with no current active
does not cause this; only current does, since only current has a channel
into the OCP's own dynamics that can go missing.

## Bugs found and fixed along the way

These are recorded here because they're the kind of thing that will bite
again if the formulation is ever touched without this context:

1. **`Q[e_y]=0` (paper default) → cross-track error never corrected.**
   The reference paper's own published weight leaves cross-track error
   unpenalized (its formulation relies on heading-alignment alone). Fixed by
   giving it a small non-zero weight (see `config.py`'s `Q_DIAG` comment).
2. **Acados dummy-obstacle overflow.** Padding unused obstacle slots with a
   "very far away" dummy position (`1e6`) put the squared-distance
   constraint value past acados' internal infinity threshold (`1e10`),
   causing spurious QP infeasibility any time fewer than `MAX_OBSTACLES` real
   obstacles were passed in. Fixed by lowering the dummy distance to `1e3` —
   still far enough to be trivially inactive.
3. **Course-angle wraparound.** `Q[psi]=30` (the dominant weight) penalizes
   *raw* `psi` against `chi_p`, but raw `psi` accumulates unboundedly while
   `chi_p` stays wrapped — a long rollout with sustained turning can read an
   otherwise-correct heading as a huge cost. Fixed by wrapping the `(psi -
   chi_p)` residual itself in the cost (`wrap180_casadi`); Acados had to
   switch from `LINEAR_LS` to `NONLINEAR_LS` cost type to express this, since
   a wrapped residual isn't affine in the state.
4. **Sideslip-formula mismatch.** `path_following.py` used `asin(v/U)`;
   `casadi_mmg_solver`'s own internal convention is `atan2(-v, u)` — different
   sign, and `asin` can't distinguish forward from reverse motion. Unified to
   `atan2(-v, u)` everywhere.
5. **Waypoint switching that could get stuck forever.** Pure "within radius
   of the target point" switching fails when a wide turn-in transient (large
   initial heading error) converges onto the path's *line* well past the
   waypoint's along-track position — Euclidean distance to the exact point
   never dips below the radius again. Fixed with an along-track "gate
   crossing" test in `select_active_waypoint`, OR'd with the original radius
   check. See the regression test at the bottom of `path_following.py`.
5b. **Follow-up: the same gate test had no cross-track bound.** Once the
   along-track gate above existed, it accepted *any* lateral offset once the
   ship was past a waypoint's along-track position — fine for a wide turning
   transient (which stays close to the line by construction), but a real bug
   for an obstacle-avoidance detour: this project's obstacles have radii up
   to ~6m, well past `WP_RADIUS`'s default 2m, so a detour swinging that far
   off the line could cross the gate plane nowhere near the actual waypoint
   and get counted as "reached" (observed live: the active-waypoint marker
   jumping straight to the final endpoint while the ship was still visibly
   nowhere near the intermediate one). Fixed by also requiring the
   perpendicular/cross-track offset at the gate-crossing point to be within
   `wp_radius`, leaving the original fix's own case (small cross-track
   offset by construction) unaffected.
6. **No braking near a target.** `Q[x]`/`Q[y]` (position error, meters²)
   dwarfs `Q[u]` (speed error, (m/s)²) at any real distance, so a constant
   speed reference never gets "discovered" as needing to shrink — the ship
   kept cruising at full reference speed until nearly on top of the target,
   then couldn't decelerate fast enough and overshot. Fixed by explicitly
   shrinking the *speed reference itself* with distance
   (`compute_effective_u_ref`, a linear ramp inside `config.BRAKE_DISTANCE`),
   rather than relying on the cost weights to find braking on their own.
7. **Acados QP failures during a low-speed pivot (`U→0` singularity).**
   Under a low-speed pivot maneuver (e.g. correcting heavily near a target
   with the braking ramp active), `u` and `v` can both approach zero
   simultaneously. `casadi_mmg_solver`'s `u_val = ca.fmax(u, 1e-5)` floor
   only guards the propeller/rudder terms — it does **not** prevent the
   resultant relative speed `U = sqrt(ur^2 + vr^2)` itself from collapsing,
   and `r_ndm = r*Lpp/U` (used throughout the hull force terms) blows up for
   any nonzero yaw rate as `U→0`. This made Acados' SQP-RTI QP solver fail
   outright during a live run, and because SQP-RTI's warm-started iterate
   persists across calls, a single poisoned solve could freeze the
   controller permanently rather than recovering on the next step. Fixed by
   giving `IDX_U` a hard lower bound (`config.U_REF_MIN`, deliberately kept
   just above zero) via `idxbx`/`lbx` in `nmpc_acados.py`'s own OCP (not
   touching the read-only MMG model) — the optimizer can no longer drive
   surge speed into the singular region mid-horizon, even though the model's
   own `fmax` floor is just a smoothing device, not something the optimizer
   was ever constrained to respect on its own.
8. **No lookahead across a waypoint switch → rudder saturation + reactive
   (self-defeating) braking on sharp turns.** `chi_p`/`x_d`/`y_d` (and the
   `u_ref` fed through `compute_effective_u_ref`) were pure functions of the
   *single currently active* waypoint/leg, and `nmpc_acados.py`'s `solve()`
   pushed that one fixed reference into every stage of its 20s/`N=200`
   horizon. A sharp turn (this project's own `scenario.json` has legs
   turning up to ~150°) produced an instantaneous course-error step the
   moment `select_active_waypoint`'s gate fired, saturating the rudder;
   `compute_effective_u_ref`'s distance-only ramp then reset to full cruise
   the instant the target switched (now measuring distance to the new,
   far-away target), so deceleration was an emergent side-effect of the QP
   minimizing `Q[x]`/`Q[y]` position error under a stuck rudder rather than
   a deliberate pre-turn brake — confirmed live via
   `nmpc_experiment_20260826_222937`'s telemetry: `cmd_delta_deg` pegged at
   +45° and `true_psi_deg` stalled ~140-160° off target for 20+ seconds
   after crossing a waypoint, while `cmd_n_rps` was pushed hard negative
   *after* the rudder was already saturated (rudder authority scales with
   surge speed, so that reactive braking was actively working against the
   turn it was reacting to). Fixed by giving the solver a genuine preview of
   however many upcoming path segments fall inside the horizon, not just the
   currently active one: `path_following.py`'s `SegmentQueue` (append
   newer segments to the back, `pop_crossed()` retires one the instant its
   endpoint is passed — driven by the same `is_segment_crossed` predicate
   `select_active_waypoint` itself now delegates to) feeds
   `build_horizon_references()`, which walks the queue by predicted
   arclength (`speed_est`, floored/recomputed fresh every `solve()` call)
   and returns per-stage `(chi_p, x_d, y_d, u_ref)` arrays — `nmpc_acados.py`
   pushes these into acados' existing per-stage `p`/`yref` slots (already
   stage-indexed, just previously fed the same value at every `k`) instead
   of one fixed reference. `map_node` owns the live queue (seeded once with
   the whole remaining static path, popped in lockstep with `target_idx`);
   the design is deliberately queue-based (not just "preview the next
   waypoint") so it also handles waypoints spaced closer together than the
   horizon's travel distance (e.g. ~1-2 ship-lengths, `LPP=2.902m` →
   ~3-6m — several such legs fit inside a 20s/`U_REF=0.70m/s` ≈14m horizon),
   and so a future *dynamic* segment producer (e.g. a LIDAR-derived rolling
   window of locally-sensed segments) can drive the same queue via
   `append()`/`pop_crossed()` without changing anything on the
   NMPC-consuming side. `select_active_waypoint()`'s own gate logic is
   unchanged — this only changes what's *previewed* inside a solve's
   horizon, not when the real `target_idx` advances.
8b. **Follow-up: item 8 alone doesn't complete a genuinely sharp turn.**
   Verified only against a mild ~60° turn at first, item 8's design let
   `chi_p_arr` AND `x_d_arr`/`y_d_arr` switch together, per stage, at the
   same predicted arclength boundary. On a real ~117° turn
   (`scenario.json`'s wp0→wp1→wp2) this let part of the horizon reference a
   position target tens of meters away in a very different direction while
   the ship was still only ~5m from the REAL current waypoint — the large
   predicted position error to that far target dominated (and defeated) the
   much smaller cost of slowing down, so the ship never actually closed
   `select_active_waypoint`'s gate: it got within ~3.3-3.8m
   (`WP_RADIUS=2.0m`) and then circled the waypoint indefinitely (`psi`
   climbing monotonically past 600° over ~150s, rudder pinned near ±45°,
   thrust mostly still at max) rather than braking and threading it.
   Reproduced identically with item 9's boost disabled, confirming it was
   an item-8 bug, not an item-9 one. Fixed by decoupling heading preview
   from position targeting in `build_horizon_references`: `chi_p_arr` still
   previews ahead across however many segments fit in the horizon (so the
   solver can start turning early), but `x_d_arr`/`y_d_arr` now stay pinned
   to `survivors[0]`'s endpoint — the single REAL current target — for
   every stage, changing only when `SegmentQueue.pop_crossed()` actually
   pops it between `solve()` calls, never inside a single horizon's
   internal walk. This makes later stages ask for "be *at* the real target,
   already facing the next leg's direction" instead of "rush toward a
   distant point," which removes the rush-instead-of-brake pressure by
   construction. Verified on the same ~117° turn: crosses the gate cleanly
   at `t=89s`, closest approach 0.93m (with item 9's boost) / 1.33m
   (without) — both comfortably inside `WP_RADIUS`, zero solver failures
   across 2600 steps, versus never crossing at all before this fix.
9. **Corner-cutting: the solver passes NEAR a waypoint, not through it.**
   Even with item 8's horizon preview, the optimizer would route inside
   both adjacent lines at a corner rather than actually threading the
   waypoint point — cheap under the existing cost, since cross-track error
   to the LINE (`Q[e_y]=10.0`) dominates raw position error to the target
   POINT (`Q[x]=Q[y]=5.0`), and cutting the corner keeps `e_y` low against
   *both* lines simultaneously without the sharper maneuvering "must reach
   the exact point" would cost. Fixed *without* raising `Q[x]`/`Q[y]`
   everywhere (which would fight cheap line-following, and cost more
   control effort, along the entire leg): `build_horizon_references()` also
   returns `dist_to_corner_arr`, each stage's predicted arclength distance
   to the nearest real waypoint (from the same `cum_dist` breakpoints
   already computed for the segment walk); `AcadosNMPC.__init__` precomputes
   a second `(W, W_e)` pair with `Q[x]`/`Q[y]` scaled by
   `WAYPOINT_PASSAGE_XY_BOOST`, and `solve()` applies it — via acados'
   runtime `cost_set(stage, "W", ...)`, confirmed available with no OCP
   rebuild needed — only to stages where `dist_to_corner_arr <=
   WAYPOINT_PASSAGE_DIST`; every other stage keeps the unmodified default
   weights. The boost (default 4.0×, `Q[x]=Q[y]` 5.0→20.0, comfortably past
   `Q[e_y]=10.0`) and window (default 4.0m, ~2×`WP_RADIUS`) are both
   `sim_params.yaml` tunables. Verified on the same ~117° turn used in item
   8b, *after* that fix (this one alone doesn't help while item 8b's bug is
   still present — the two were confounded in initial testing, 0.98m
   no-boost vs 1.02m boosted on a mild turn, no measurable difference):
   closest approach to the waypoint 0.93m with the boost vs 1.33m without,
   both on top of item 8b's fix — a real, if modest, improvement.

## Running things

```bash
# Self-tests (no solver needed)
python nmpc/path_following.py
python nmpc/state_augmentation.py

# Closed-loop validation harness (plant + solver, no obstacles) -- now lives
# in nmpc_sim_nodes/tests/, not here; see the top-level README's executables
# table for the full list (test_nmpc, test_closed_loop_noise, test_closed_loop_env)
ros2 run nmpc_sim_nodes test_nmpc

# Live-visualized run -- ros2 launch/run, not a standalone script (see
# top-level README's "Getting started")
ros2 launch nmpc_sim_nodes bringup.launch.py
ros2 launch nmpc_sim_nodes rviz_hud.launch.py
```

## Dependencies

`numpy`, `casadi`, `acados_template` (for `nmpc_acados.py` and
`run_live.py`'s default solver), `matplotlib` (test/demo plotting).
