# BTP NMPC Obstacle Avoidance

A Nonlinear Model Predictive Controller (NMPC) for autonomous path-following
and (in-progress) obstacle avoidance on a small rudder-and-propeller surface
vessel, built from a validated 3-DOF MMG maneuvering model up through a
real-time acados-based solver. This is a B.Tech Project (BTP) — the
repository documents the whole path from "here is a MATLAB-derived NumPy
dynamics model" to "here is a working closed-loop NMPC controller," including
every bug that showed up along the way and how it was diagnosed and fixed.

If you only read one other file, read
[`nmpc/README.md`](nmpc_ws/src/nmpc_sim_nodes/nmpc_sim_nodes/nmpc/README.md) —
it documents the actual controller and its full bug history in detail. This
file is the narrative/overview: what the project does, why it's built this
way, where the ideas came from, and how the pieces fit together.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Why it's built this way](#why-its-built-this-way)
3. [Architecture](#architecture)
4. [The vessel dynamics model (MMG)](#the-vessel-dynamics-model-mmg)
5. [From NumPy to a real-time solver: CasADi and acados](#from-numpy-to-a-real-time-solver-casadi-and-acados)
6. [Research foundations](#research-foundations)
7. [The NMPC formulation](#the-nmpc-formulation)
8. [Development timeline and what actually went wrong](#development-timeline-and-what-actually-went-wrong)
9. [Repository layout](#repository-layout)
10. [Available executables](#available-executables)
11. [Getting started](#getting-started)
12. [Current status and open work](#current-status-and-open-work)
13. [A note on how this repository's history was built](#a-note-on-how-this-repositorys-history-was-built)

---

## What this project does

The end goal is a surface vessel (conceptually a WAM-V/VRX-style unmanned
surface vehicle, though the specific hull parameters used throughout are a
small scaled model, `Lpp = 2.902 m`) that can:

1. Follow a sequence of waypoints, correcting for cross-track and heading
   error, at a commanded speed.
2. Slow down and settle near the final waypoint instead of sailing straight
   through it.
3. (Not yet complete — see [Current status](#current-status-and-open-work))
   detect and steer around obstacles using LiDAR-style point/circle
   representations, without ever making the underlying optimization
   problem infeasible.
4. Run all of the above in real time on modest hardware, using a
   receding-horizon Nonlinear Model Predictive Controller.

Everything in this repo up to the current commit implements and validates
(1) and (2) in simulation, with the formulation already structured to support
(3) — the obstacle-avoidance terms exist in both solvers and are exercised in
the live-visualization test harness with an empty obstacle list, but haven't
yet been driven through a scenario with real obstacles.

## Why it's built this way

Three deliberate architectural choices shape everything else in the repo:

**A trusted dynamics model comes first, and stays the single source of
truth for "what does this vessel actually do."** `Preliminary_func.py` is a
NumPy implementation of a published MMG (Maneuvering Modeling Group) 3-DOF
maneuvering model — a standard, hydrodynamically-derived vessel model, not
something ad-hoc. Every other dynamics representation in the repo (the
CasADi symbolic version, the acados-compiled version, the NMPC's own
prediction model) is validated against this one to a tight numerical
tolerance rather than being independently re-derived, so a bug in the
physics can't silently hide behind a bug in the optimizer, or vice versa.

**Optimization-friendly dynamics are a separate concern from
high-fidelity dynamics.** The NumPy model includes wave-drift disturbance
forces (interpolated from lookup tables) and hard `if` branches wherever a
term could divide by zero or flip sign. Gradient-based solvers (IPOPT,
acados' SQP) need everything they touch to be differentiable, and table
interpolation has no place inside an NLP's inner loop. So the CasADi port
deliberately drops the wave-drift term entirely and replaces every
discontinuity with a smooth equivalent (`ca.fmax`, `atan2`, tanh-blended
switches) — see [CasADi and acados](#from-numpy-to-a-real-time-solver-casadi-and-acados)
below. The disturbance-free, smoothed model is what the NMPC predicts with
for *wave*; the full high-fidelity NumPy model (or a real vessel) is what
it's actually controlling. (An optional current/wave disturbance term was
later added back into the CasADi model — see the `F_env` note in
[the MMG section](#the-vessel-dynamics-model-mmg) — as a *plant*-side input
for both; **current specifically is also fed into the NMPC's own OCP model**
as a live runtime parameter, sourced from `ukf_node`'s own current estimate
each solve, so its dynamics-continuity constraint reflects the same
current-aware physics the plant uses, frozen across the horizon since only a
single online estimate exists, not a horizon-length forecast — see
[the NMPC formulation](#the-nmpc-formulation). Wave is not: it stays
plant-only, with no representation in the OCP at all. The table lookups
themselves still happen entirely outside `casadi_mmg.py`, in `env_model/`.)

**Two solvers, same formulation, built in that order on purpose.** Every
NMPC problem in `nmpc/` is implemented twice: once with CasADi + IPOPT
(`nmpc_casadi.py`), once with acados' SQP-RTI (`nmpc_acados.py`). IPOPT is
slow (10s-100s of ms per solve) but every intermediate value is easy to
inspect, so it's what the *formulation itself* (cost, constraints, dynamics
wiring) gets debugged against first. acados compiles the same problem to C
and solves one SQP iteration per call in under a couple of milliseconds —
fast enough for a real control loop — but is much harder to debug directly.
Building CasADi first and cross-validating acados against it (trajectories
matching to within about a centimeter) means acados-specific bugs (like the
obstacle-padding overflow described below) get caught by comparison against
a known-good reference, instead of being debugged blind.

## Architecture

```
                     ┌─────────────────────────┐
                     │   Preliminary_func.py    │   NumPy, high-fidelity,
                     │   (+ Wave_Data/)          │   ground-truth MMG model
                     └────────────┬─────────────┘   (wave drift included)
                                  │ validated against
                                  ▼
                     ┌─────────────────────────┐
                     │ casadi_mmg_solver/        │   CasADi symbolic port,
                     │   casadi_mmg.py            │   smoothed, no wave drift
                     │  (+ optional F_env,        │   in the NMPC's own model;
                     │   plant-only, from         │   + acados SimSolver
                     │   mmg_node/env_model)      │
                     └────────────┬─────────────┘
                                  │ imported by
                                  ▼
                     ┌─────────────────────────┐
                     │        nmpc/               │   augmented-state NMPC:
                     │  config / path_following /  │   path-following +
                     │  state_augmentation /       │   obstacle avoidance,
                     │  nmpc_casadi / nmpc_acados  │   two solvers
                     └────────────┬─────────────┘
                                  │ streams live state/predictions to
                                  ▼
                     ┌─────────────────────────┐
                     │   mpc_visualization/        │   standalone dashboard,
                     │   mpc_bridge / visualizer   │   built + proven with mock
                     └─────────────────────────┘   data before real NMPC existed
```

`research_papers/` (cited by title/DOI, not committed as PDFs — see below)
is the literature the `nmpc/` formulation is adapted from.

## The vessel dynamics model (MMG)

`Preliminary_func.py` implements the **MMG (Maneuvering Modeling Group)**
3-degree-of-freedom maneuvering model — surge (`u`), sway (`v`), and yaw
rate (`r`) — for a hull with principal particulars matching a scaled
KVLCC2-type tanker model (`Lpp = 2.902 m`, `d = 0.189 m`,
`Volume = 0.235 m³`). The state is `[u, v, r, x, y, psi]`; controls are
rudder angle `delta` and propeller speed `rps`.

The governing equation is a standard mass-matrix solve,
`M · [u̇, v̇, ṙ]ᵀ = F_hull + F_propeller + F_rudder − C(v) + F_wave`,
where:

- **`F_hull`** comes from a polynomial expansion of non-dimensional
  hydrodynamic derivatives (`X_vv`, `Y_v`, `N_vvr`, etc.) in non-dimensional
  sway velocity and yaw rate — the classic MMG hull-force parameterization.
- **`F_propeller`** uses a quadratic `K_T`-vs-advance-ratio curve to get
  thrust from propeller rps and inflow velocity.
- **`F_rudder`** models rudder normal force from effective inflow velocity
  and angle, including flow-straightening and hull-rudder interaction
  coefficients (`aH`, `tR`, `gamma_r`).
- **`F_wave`** (only in the NumPy model) interpolates second-order wave-drift
  force/moment coefficients from `Wave_Data/*.mat` lookup tables, as a
  function of relative wave heading and non-dimensional frequency.

`activateMMG()` integrates one step with RK4, stepping `u, v, r` in the body
frame and rotating the resulting position increment into the Earth-fixed
frame once per step.

**`F_env` (current + wave).** `casadi_mmg.py`'s `MMG_Time_Derivative_casadi()`
accepts optional `current` and `wave_force` arguments (both default to zero,
so any call site that doesn't pass them is unaffected). `current`
(earth-frame current velocity) enters through a relative-velocity
substitution feeding the hull/propeller/rudder terms only, not the mass
matrix or kinematics; `wave_force` (a body-frame surge/sway/yaw drift
force/moment, computed outside `casadi_mmg.py` by `env_model/wave_model.py`
using the same `Wave_Data/*.mat` tables and JONSWAP-spectrum discretization)
is added straight into the dynamics' RHS.

`mmg_node.py` (the plant integrator) passes REAL, live values for both —
it samples `env_model`'s `CurrentModel`/`WaveModel` in-process, toggleable
via its own `current_enabled`/`wave_enabled` params. `nmpc/state_augmentation.py`
(the NMPC's own internal prediction model, `augmented_dynamics_casadi`) also
calls this same function, but only ever passes a real value for `current`
— sourced from whatever `ukf_node` currently estimates, threaded through
`mmg_node`'s `/nmpc/solve` request → `nmpc_node`'s `AcadosNMPC.solve(...,
current=...)` → the OCP's own runtime parameter vector — while `wave_force`
is always left at its zero default there, matching the
"optimization-friendly dynamics are a separate concern from high-fidelity
dynamics" argument above for wave specifically, but NOT for current, which
the NMPC's own dynamics-continuity constraint does see.

## From NumPy to a real-time solver: CasADi and acados

`casadi_mmg_solver/casadi_mmg.py` is a from-scratch CasADi (`ca.MX`/`ca.SX`)
rewrite of `MMG_Time_Derivative`, needed because none of the downstream
optimization tooling can differentiate through NumPy. Six things had to
change beyond a mechanical `np.*` → `ca.*` swap:

- Every discontinuous guard (`u<0`, `rps<0`) became a smooth floor via
  `ca.fmax`.
- The rudder flow-straightening coefficient `gamma_r` (which switches
  between two constants depending on the sign of the effective rudder
  inflow angle) became a `tanh`-blended interpolation between the two
  values, so its Jacobian stays continuous through the switch.
- `arctan(-v/u)` became `atan2(-v, u)` everywhere sideslip is computed —
  correct and non-singular at `u≈0`, and correct in reverse (`u<0`), unlike
  a plain `arctan`.
- The wave-drift lookup-table block was **removed entirely** from the
  symbolic model (see [Why it's built this way](#why-its-built-this-way)).
- Both RK4 and single-step Euler integration were wrapped as a single
  `ca.Function` mapping `(state, control) → next_state`, so either
  discretization can be swapped in.
- The exact (`smooth=False`, using `ca.if_else`) and smoothed
  (`smooth=True`, using `ca.fmax`/`tanh`) variants were kept side by side,
  specifically so the smoothing's effect on the physics could be measured
  directly (see the validation results below) rather than assumed safe.

**acados** was then integrated on top (`make_acados_integrator`), code-
generating the same dynamics to C and compiling it to a shared library via
`AcadosSim`/`AcadosSimSolver`, because the pure-CasADi integrator is too
slow in Python for the thousands of repeated steps a closed-loop test run
needs. Acados had to be built from source (v0.4.3, with qpOASES and OSQP QP
backends) since it isn't distributed as a package; getting it running also
required preloading its shared-library dependencies programmatically via
`ctypes.CDLL(..., RTLD_GLOBAL)` in strict topological order
(`qdldl → osqp → qpOASES → blasfeo → hpipm → acados`), since a fresh shell
without `LD_LIBRARY_PATH` pre-set would otherwise fail to import
`acados_template` at all.

**Validation** (`validate_casadi.py`, a 200-second, 35°-rudder/18.2-rps
turning-circle test, the same maneuver used throughout the project as the
standard regression scenario): comparing the exact (`smooth=False`) CasADi
model against the original NumPy model gives errors on the order of
`1e-15`–`1e-17` — floating-point noise, i.e. a bit-identical port. Comparing
the *smoothed* (`smooth=True`, the version actually used everywhere
downstream) version against NumPy gives errors around `1e-8`–`1e-9` in
`u, v, r` and `1e-6`–`1e-7 m` in position — several orders of magnitude
below anything that matters physically, confirming the smoothing needed for
solver-friendliness doesn't meaningfully change the vessel's behavior.

## Research foundations

The NMPC formulation in `nmpc/` is adapted from two Ocean Engineering papers
studying the VTec S-III autonomous surface vehicle (full citations, DOIs,
and a concept-by-concept mapping onto this codebase are in
[`research_papers/README.md`](research_papers/README.md); the PDFs
themselves aren't committed here since they're copyrighted journal
articles):

1. **Gonzalez-Garcia et al. (2022), "Path-following and LiDAR-based obstacle
   avoidance via NMPC for an autonomous surface vehicle,"** *Ocean
   Engineering* 266 — the primary structural source. Contributes the
   augmented-state idea (folding path-following error directly into the
   NMPC's own state), the cross-track-error formula, the
   `sin/cos(course-angle-error)` representation (to avoid a wraparound
   discontinuity in the cost), the soft slack-relaxed obstacle constraint,
   and the acados SQP-RTI real-time solve strategy.
2. **Collado-Gonzalez et al. (2024), "Adaptive sliding mode control with
   nonlinear MPC-based obstacle avoidance using LiDAR for an autonomous
   surface vehicle under disturbances,"** *Ocean Engineering* 311 — a
   related follow-up on the same platform; corroborates the cross-track and
   obstacle-constraint formulas independently, and its explicit statement
   that the path angle is piecewise-constant per leg (not a continuously
   varying parametric curve) directly shaped how waypoint switching is
   structured here.

Both papers control a twin-thruster differential-drive vehicle
(`[T_port, T_stbd]`); this project adapts their guidance/cost/constraint
*structure* onto a completely different actuation model — rudder angle +
single propeller speed, via the MMG dynamics above — which is why the
augmented state here has `delta`/`n` rows instead of two thruster forces.

## The NMPC formulation

Full detail (including every parameter and every bug fixed while getting
here) is in
[`nmpc/README.md`](nmpc_ws/src/nmpc_sim_nodes/nmpc_sim_nodes/nmpc/README.md).
Summary:

**Augmented state** (11-dimensional):
```
xi = [e_y, sin(psi_e), cos(psi_e), r, x, y, psi, u, v, delta, n]
```
`e_y` is signed cross-track distance to the line through the active
waypoint; `psi_e` is course-angle error (represented as its sine/cosine to
stay wraparound-safe in the cost). `delta`/`n` (rudder angle, propeller rps)
are carried as **states**, not controls — the actual control input is their
*rate* (`u_aug = [delta_dot, n_dot]`), so the optimizer is penalized for how
fast it moves the actuators, which is what produces smooth commands instead
of bang-bang ones.

**Cost**: quadratic tracking error (`Q`) + control-rate penalty (`R`) +
terminal cost (`Qe`), summed over a receding horizon (`N=100` steps,
`dt=0.1s` → 10s lookahead in the current tuning).

**Dynamics**: the MMG accelerations/kinematics come directly, unmodified,
from `casadi_mmg_solver.MMG_Time_Derivative_casadi` — the augmentation layer
only adds the new guidance/actuator-rate rows on top, RK4-discretized the
same way in both solvers so their trajectories can be meaningfully compared.

**Current-aware (acados only)**: the OCP's runtime parameter vector is
`p = [chi_p, x_d, y_d, vcx, vcy, obstacle slots...]` — `vcx, vcy` (earth-frame
current) feed straight into the same `MMG_Time_Derivative_casadi` call above,
so the solver's own predicted dynamics account for whatever current
`ukf_node` currently estimates, not still water. `AcadosNMPC.solve(...,
current=(vcx, vcy))` sets it once per call and holds it fixed across the
whole horizon (a frozen-disturbance approximation — a single online estimate,
not a horizon-length forecast). `nmpc_node.py`'s `/nmpc/solve` handler reads
this straight off `SolveNMPC.Request.current`, which `mmg_node.py` populates
from `ukf_response.estimated_current` whenever `use_ukf=True`. This closes a
real, measured gap: an acados closed-loop rollout with current enabled but
this parameter left at its default `(0,0)` fails its QP solver almost every
tick (a genuine plant/solver dynamics mismatch, not a tuning issue) — see
`tests/test_closed_loop_noise.py`'s own module docstring for the reproduction.
Wave has no equivalent parameter and is never seen by the OCP.

**Obstacle avoidance**: soft, slack-relaxed quadratic distance constraints
per obstacle, in a fixed-size slot budget (`config.MAX_OBSTACLES`) padded
with a harmless far-away dummy when fewer real obstacles are present — this
keeps the NLP/OCP structure fixed regardless of the live obstacle count,
which acados in particular requires (fixed sizes at code-generation time).

**Two solvers, one interface**: `CasadiNMPC` and `AcadosNMPC` both expose
`.solve(mmg_state, delta, n, chi_p, x_d, y_d, obstacles=[]) -> dict` with the
same return shape, so either can be dropped into the same test harness or
visualization loop interchangeably.

## Development timeline and what actually went wrong

This project's actual git history didn't exist until this repository was
assembled from the working directory's final state plus its accumulated AI
coding-assistant chat logs (see the [note](#a-note-on-how-this-repositorys-history-was-built)
at the bottom) — but the commit history *does* faithfully walk through the
real sequence of milestones and real bugs, in order, because that sequence
was reconstructed from those logs rather than invented. The short version:

1. **Baseline MMG model** existed first, as the trusted ground truth.
2. **CasADi port**, then **acados integration** on top of it, validated at
   each step against the NumPy baseline.
3. A **standalone visualization dashboard** was built and proven out with
   fake/mock data — deliberately before any real NMPC existed — so the
   rendering pipeline itself wasn't a confound once the real solver arrived.
4. **Literature review** of the two papers above produced a detailed action
   plan for the actual NMPC formulation.
5. The `nmpc/` package was built in dependency order — config, guidance
   geometry, augmented dynamics, then the CasADi solver, then the acados
   solver.
6. **First NMPC test run: total failure.** All four validation
   scenarios failed with multi-meter cross-track error. Root cause: the
   reference paper's own published cross-track weight is `0` (their
   formulation relies on heading-alignment alone), copied verbatim into
   this project's config — giving the optimizer literally zero incentive to
   correct a constant lateral offset. Fixed with a small non-zero weight,
   alongside re-tuning actuator-rate bounds that had produced unrealistic
   bang-bang behavior.
7. **Course-angle wraparound + sideslip-formula bugs**, found by going back
   to how the source papers themselves handle course-angle representation
   rather than guessing at a fix — a raw, never-wrapped `psi` compared
   against a wrapped `chi_p` could register a huge spurious cost error after
   sustained turning, and a sign/formula mismatch in the sideslip
   computation between two files.
8. **Live visualization wired to the real solvers**, plus CSV telemetry
   logging, so runs could be watched and later re-analyzed instead of only
   inspected as static end-of-run plots.
9. **A waypoint-switching bug** where a wide turn-in transient (from a large
   initial heading error) could make the ship converge onto a path *line*
   well past the target waypoint, so pure radius-based switching never
   fired again — fixed with an along-track "gate crossing" test.
10. **No braking near a target** — position-error cost terms so thoroughly
    dominate speed-error terms at any real distance that the optimizer never
    "discovers" deceleration on its own; fixed by explicitly shrinking the
    *speed reference* itself as a function of remaining distance.
11. **A UKF state estimator** (`ukf_node`/`ukf`) was added to reconstruct
    `[u,v,r,x,y,psi]` and estimate earth-frame current `[vcx,vcy]` from a
    simulated noisy GPS/gyro/IMU-accelerometer stream — see
    [Current status](#current-status-and-open-work) — followed by an extended
    tuning pass: several of its `q_diag`/`p0_diag` entries turned out to be
    hand-derived numeric literals for one specific current/sensor-preset
    configuration rather than recomputed live, so changing `current_sigma`/
    `current_time_constant` or `sensor_preset` silently left the filter's
    noise model sized for the *old* configuration (`ukf/config.py`'s
    `current_noise_diag()`/`bias_noise_diag()` now derive these live instead).
    A separate, genuine tuning bug was found the same way for
    `pos_bias_q_scale`: the NEES-search-tuned value left the GPS-bias state
    visibly under-tracking a real, several-meter drift, showing up directly
    as x/y position error in `tests/test_ukf.py`'s trajectory plot —
    re-tuning it (`1.230 → 20.0`) roughly halved x/y RMSE with no measured
    downside. The equivalent search for `accel_bias_q_scale` found a much
    smaller, genuine SNR ceiling instead (pushing it further actively hurts
    `ay_bias` and current tracking) — not every "looks under-tracked" plot is
    the same bug.
12. **The zig-zag test maneuver was replaced with a turning circle**
    (`tests/test_ukf.py`/`tests/tune_ukf.py`) after noticing `vcy` (current's
    y-component) consistently tracked worse than `vcx`: a zig-zag confined to
    a narrow heading band only ever rotates the current vector through that
    same band, so whichever component is misaligned with it stays stuck in
    the accelerometer's lower-SNR sway channel for the entire run. A
    continuous turning circle sweeps every heading, letting both components
    take a turn in the high-SNR surge channel — measured directly against the
    old zig-zag at matched current settings: `vcy` went from *worse* than a
    trivial "never update" baseline to meaningfully better than it.
13. **A follow-up waypoint-switching bug**, a different failure mode of the
    same along-track "gate" test added in item 9: the gate had no bound on
    *lateral* (cross-track) offset, so an obstacle-avoidance detour swinging
    several meters off the path line (this project's obstacles have radii up
    to ~6m, well past `wp_radius=2m`) could cross the gate plane far from the
    actual waypoint and get counted as "reached" — observed live as the
    active-waypoint marker jumping straight to the final endpoint while the
    ship was still visibly nowhere near the intermediate one. Fixed by
    bounding the gate to `wp_radius` laterally too, leaving the original
    wide-turning-transient fix (small cross-track offset by construction)
    intact.

A **known, currently-open issue** (a low-speed singularity in the MMG
model's `U = sqrt(u² + v²)` denominator, which can freeze acados' solver
during a slow pivot near a target) is documented in
[`nmpc/README.md`](nmpc_ws/src/nmpc_sim_nodes/nmpc_sim_nodes/nmpc/README.md)
rather than silently left for someone to rediscover.

## Repository layout

Everything that runs now lives inside a ROS2 (Jazzy) workspace,
`nmpc_ws/`, built with `colcon` and made up of several `ament_python`
packages (the restructuring this followed was originally planned in a
`ROS2_CONVERSION_PLAN.md`, referenced in several code comments by section
number but not itself committed to this repo):

```
nmpc_ws/
  src/
    nmpc_interfaces/            Shared msg/srv definitions for the sim's ROS2 node graph
    nmpc_sim_nodes/              map_node, nmpc_node, ukf_node, mmg_node -- the
                                  simulation graph -- plus viz_node / rviz_node / hud_node
                                  (independently launchable live visualizers) and
                                  run_demo executables. Also contains the actual
                                  controller/model code, physically moved in here
                                  as subpackages:
      nmpc_sim_nodes/nodes/                   The ROS2 node entry points themselves:
                                                map_node, nmpc_node, ukf_node, mmg_node,
                                                viz_node, rviz_node, hud_node, logger_node
                                                (executable names unchanged, only the
                                                module path moved here).
      nmpc_sim_nodes/tests/                    Standalone (no live ROS graph needed)
                                                comparison/validation harnesses:
                                                test_nmpc, test_closed_loop_noise,
                                                test_closed_loop_env, test_ukf,
                                                tune_ukf -- see their own module
                                                docstrings. (test_sensor_model and
                                                test_env_model stay colocated with
                                                sensor_model/ and env_model/ instead,
                                                the standard "unit test next to its
                                                module" convention.)
      nmpc_sim_nodes/nmpc/                    The NMPC controller (both solvers) -- see
                                                its own README, linked above
      nmpc_sim_nodes/casadi_mmg_solver/       CasADi symbolic MMG port + acados SimSolver
      nmpc_sim_nodes/mpc_visualization/       The matplotlib live-visualization dashboard
                                                (viz_node's full map+control-horizon window)
                                                and hud_node's standalone companion window
      nmpc_sim_nodes/sensor_model/            GPS/compass/gyro/IMU-accel/actuator noise
                                                model, run in-process by mmg_node
      nmpc_sim_nodes/env_model/               Toggleable current (OU process) + wave
                                                (JONSWAP + Newman drift) disturbance model,
                                                run in-process by mmg_node
      nmpc_sim_nodes/ukf/                     Unscented Kalman Filter state estimator
                                                (pure math + config), served by ukf_node --
                                                reconstructs [u,v,r,x,y,psi] and estimates
                                                earth-frame current [vcx,vcy] from the
                                                GPS/gyro/IMU sensor stream
    scenario_maker/               GUI for authoring custom track & obstacle scenarios
    mmg_model_validation/         Standalone NumPy vs CasADi vs acados validation harness
                                    (Preliminary_func.py, validate_casadi.py)

Wave_Data/                    Wave-drift force lookup tables (.mat)
research_papers/              Citations for the papers the NMPC is adapted from, plus
                                CURRENT_AWARE_NMPC_PAPERS.md -- a research survey on
                                feeding current into the NMPC's own prediction model
                                (disturbance feedforward/rejection); superseded in part
                                by "The NMPC formulation" section's current-awareness
                                note above, since basic current feedforward is now
                                implemented, not just surveyed
```

Every package/subpackage has its own `README.md` with file-by-file detail;
this top-level file is deliberately the narrative/overview instead of
duplicating that detail.

## Available executables

Every `ros2 run <package> <executable>` currently defined in `nmpc_ws/src/`,
kept in sync with each package's `setup.py` whenever an executable is
added/removed/renamed:

| Package | Executable | What it does |
|---|---|---|
| `nmpc_sim_nodes` | `map_node` | Owns scenario data, active-waypoint bookkeeping, and run-termination logic; part of the core sim graph. |
| `nmpc_sim_nodes` | `nmpc_node` | Pure NMPC optimizer, serving `/nmpc/solve` (acados or CasADi backend). Receives whatever state `mmg_node` populates the request with -- the true state, or `ukf_node`'s estimate, depending on `mmg_node`'s `use_ukf` toggle (see below); this node itself has no opinion on which. |
| `nmpc_sim_nodes` | `ukf_node` | Unscented Kalman Filter state estimator, serving `/ukf/estimate` -- called synchronously by `mmg_node` every tick. Reconstructs `[u,v,r,x,y,psi]` and estimates earth-frame current `[vcx,vcy]` from `mmg_node`'s GPS/gyro/IMU-accel sensor stream (no direct velocity measurement -- see `sensor_model` below). Always runs and publishes `/ukf/estimated_state`/`/ukf/estimated_current` for logging/RViz regardless of `mmg_node`'s `use_ukf` toggle; **the current estimate IS fed into the NMPC's own prediction model** (via `mmg_node`'s `/nmpc/solve` request, when `use_ukf=True` -- see [The NMPC formulation](#the-nmpc-formulation)), though the estimate's own accuracy is independent of that -- most of this project's UKF tuning work concerns accuracy, not this wiring. `q_diag`/`p0_diag` for the current and sensor-bias states are derived live from `sim_params.yaml`'s actual `current_sigma`/`current_time_constant`/`sensor_preset` (`ukf/config.py`'s `current_noise_diag()`/`bias_noise_diag()`), not hardcoded literals, so they can't silently go stale when those change. |
| `nmpc_sim_nodes` | `mmg_node` | Plant integrator and the master `1/dt` clock. Also owns, in-process (no separate nodes): a toggleable GPS/compass/gyro/IMU-accelerometer/actuator noise model (`sensor_enabled`, default off/light preset) applied before every `/ukf/estimate` call, and a toggleable current (Ornstein-Uhlenbeck) + wave (JONSWAP + Newman's-approximation drift force) disturbance model (`current_enabled`/`wave_enabled`) folded into the plant integrator -- current (not wave) is additionally forwarded into `/nmpc/solve`'s request from `ukf_node`'s own estimate, so the NMPC's own dynamics also account for it (see [The NMPC formulation](#the-nmpc-formulation)). `use_ukf` (`mmg_node.py`'s own `declare_parameter` default is `False`; `sim_params.yaml` currently sets it `true`) selects whether `/nmpc/solve`'s request comes from `ukf_node`'s estimate or the true state directly. |
| `nmpc_sim_nodes` | `logger_node` | Synchronous experiment logger; captures metadata, scenario copy, timeseries telemetry CSV, prediction horizons NPZ, and summary JSON, plus (via `ControllerEffortLogger`, fed off `nmpc_node`'s `/nmpc/controller_effort_raw` topic on its own background thread) a per-step Q/R cost breakdown and control-effort diagnostics in `costs_errors.csv`, folded into the same run's `summary.json`. Automatically launched with `bringup.launch.py`. |
| `nmpc_sim_nodes` | `viz_node` | Standalone live matplotlib dashboard; late-joins a running sim via `/map/get_scenario` + topics, independent of the map/nmpc/mmg nodes. Its current compass overlays a 2nd needle for `ukf_node`'s predicted current alongside the true one (`actual \| predicted`, one ring, no new panel), and its `SHIP STATUS` telemetry text shows `X`/`Y` the same way. |
| `nmpc_sim_nodes` | `rviz_node` | Republishes the sim's own topics as `visualization_msgs/MarkerArray` so RViz2 can render the same simulation, plus two `OverlayText` HUDs (bottom-left `SHIP STATUS`, top-right `CURRENT`/`WAVE`). Both current-related lines and `SHIP STATUS`'s `X`/`Y` show `actual \| predicted` (from `/ukf/estimated_current`/`/ukf/estimated_state`) on the same line, no new rows; a 2nd arrow marker (`ukf_current`, distinct color) renders the predicted current at the ship's position alongside the actual-current arrow. Run via `rviz_hud.launch.py` below, not usually standalone. |
| `nmpc_sim_nodes` | `hud_node` | Standalone matplotlib companion window (current compass, wave-force scatter, NMPC control-horizon graph); run alongside `rviz_node`/RViz2 (see `rviz_hud.launch.py` below) or `viz_node`. Current compass shows the same 2nd-needle `actual \| predicted` overlay as `viz_node`'s. |
| `nmpc_sim_nodes` | `run_demo` | Standalone mock-data demo of the visualization dashboard (fake circular-motion ship, no real NMPC or sim node graph). |
| `nmpc_sim_nodes` | `test_nmpc` | Closed-loop NMPC validation harness, no obstacles; produces plots under `~/nmpc_sim_logs/test_nmpc_results/`. |
| `nmpc_sim_nodes` | `test_sensor_model` | Standalone true-vs-measured comparison for `mmg_node`'s sensor noise model (GPS/compass/gyro/IMU-accel -- no u,v); no other node needs to be running. Produces plots/CSVs under `~/nmpc_sim_logs/test_sensor_model_results/`. |
| `nmpc_sim_nodes` | `test_closed_loop_noise` | Standalone headless run of the full pipeline (acados NMPC + MMG plant integrator, WITH current/wave from `sim_params.yaml` in the true plant AND fed into the solver exactly like `mmg_node`/`nmpc_node` do live) on `scenario.json`, twice -- once with no noise (true state straight into NMPC), once with the light noise preset filtered through a `UnscentedKalmanFilter` before reaching NMPC -- at accelerated (unthrottled) speed. Prints true-vs-UKF-estimated position RMSE and saves a path-comparison plot (true, UKF-filtered-true, and UKF-estimated trajectories) under `~/nmpc_sim_logs/test_closed_loop_noise_results/` -- the most direct standalone read on what a live `bringup.launch.py` run's x/y tracking looks like. |
| `nmpc_sim_nodes` | `test_env_model` | Standalone unit-level check of `env_model`'s `CurrentModel`/`WaveModel`, no ROS graph or plant integrator needed. Produces plots under `~/nmpc_sim_logs/test_env_model_results/`. |
| `nmpc_sim_nodes` | `test_closed_loop_env` | Standalone headless run of the full pipeline (acados NMPC + MMG plant integrator) on `scenario.json`, four times -- no disturbance, wave only, current only, and current+wave (each using `env_model`'s default `CurrentModel`/`WaveModel`, with matching seeds so each disturbance's marginal effect is isolated) -- at accelerated (unthrottled) speed. Produces a single 4-path comparison plot under `~/nmpc_sim_logs/test_closed_loop_env_results/`. |
| `nmpc_sim_nodes` | `test_ukf` | Standalone unit-level true-vs-estimated comparison for `ukf.ukf_core.UnscentedKalmanFilter` (state + estimated current) against `sim_params.yaml`'s own current/wave config, no ROS graph needed -- drives a constant-rudder **turning circle** maneuver (not a zig-zag; see item 12 in [the development timeline](#development-timeline-and-what-actually-went-wrong) for why), reads `sim_params.yaml`'s own `q_diag`/`r_diag`/`p0_diag`/`alpha`/`beta`/`kappa` (via `ukf.config.load_ukf_config()`), so this is the harness to rerun after touching any of them. Also prints a current-vs-"never updates from its seed" null-baseline RMSE comparison, since a low absolute current RMSE alone doesn't prove the filter is sensing anything when the true current barely moves. Produces plots (a single combined `state_xy.png` plan-view trajectory instead of separate x/y time series; angle states unwrapped so a multi-revolution turning circle doesn't look broken) and an RMSE summary under `~/nmpc_sim_logs/test_ukf_results/`. |
| `nmpc_sim_nodes` | `tune_ukf` | Automated NEES-consistency Q/R search for the UKF (Nelder-Mead over per-group scale factors; see its own module docstring), reusing `test_ukf`'s turning-circle maneuver/env models. "current" (vcx/vcy) is deliberately excluded from the search -- NEES can't distinguish genuine uncertainty from Q inflated to paper over misattributed error, which previously drove `q_diag[vcx]/[vcy]` to ~7147x its physically-correct value. `accel_bias`/`pos_bias` groups ARE searched, but their result now has to be applied as `sim_params.yaml`'s `accel_bias_q_scale`/`pos_bias_q_scale` (a multiplier on `ukf/config.py`'s live-derived physical value), not as an absolute `q_diag` entry -- **verify with `test_ukf`'s RMSE, not this search's NEES output, before trusting a new scale**; NEES-optimal and RMSE-optimal aren't the same thing, and previous rounds of this search found scales that measurably hurt point accuracy. Prints suggested values for `sim_params.yaml`; does not write them automatically. Produces a NEES comparison plot under `~/nmpc_sim_logs/tune_ukf_results/`. |
| `scenario_maker` | `scenario_editor` | GUI for authoring custom start/waypoints/goal/obstacle scenarios, saved as `scenario.json`. |
| `mmg_model_validation` | `validate_casadi` | Cross-validates NumPy vs CasADi vs acados MMG dynamics on the project's standard turning-circle maneuver. |

Every `ros2 launch <package> <file>` currently defined in `nmpc_ws/src/`:

| Package | Launch file | What it launches |
|---|---|---|
| `nmpc_sim_nodes` | `bringup.launch.py` | The core sim graph: `map_node`, `nmpc_node`, `ukf_node`, `mmg_node`, `logger_node`, all sharing one `params_file` launch argument (defaults to `params/sim_params.yaml`). |
| `nmpc_sim_nodes` | `rviz_hud.launch.py` | `rviz_node` + RViz2 (with this package's `rviz/sim_view.rviz` config) + `hud_node`, together -- the normal way to get both the 3D/2D RViz view (with its `SHIP STATUS`/`CURRENT`/`WAVE` overlays and current arrows) and the standalone control-horizon companion window in one command, instead of `ros2 run`-ing `rviz_node` alone and missing the 2nd window. Clears `GTK_PATH` itself (see `unset GTK_PATH` note below) so it works regardless of which terminal launches it. |

## Getting started

```bash
# Environment: needs casadi, numpy, scipy, matplotlib, acados_template, and ROS2 Jazzy
# (acados itself must be built separately: https://docs.acados.org)

# 1. Build the workspace with --symlink-install (recommended): install/share
#    becomes a symlink chain back to source instead of a copy, so editing a
#    .yaml/.json under params/ (or, for this ament_python package, even a .py
#    source file) takes effect on the next launch with NO rebuild needed.
#    Only needed once -- if you've already got a plain (copy-mode) build/install
#    tree from an earlier `colcon build`, remove it first (`rm -rf build install
#    log` from nmpc_ws/) or colcon will fail trying to replace real directories
#    with symlinks.
cd nmpc_ws && python3 -m colcon build --symlink-install

# 2. Source it (every new terminal, in this order)
source /opt/ros/jazzy/setup.bash
source nmpc_ws/install/setup.bash

# 3. Launch the simulation graph: map_node + nmpc_node + mmg_node (sensor noise +
#    current/wave disturbance run in-process inside mmg_node)
ros2 launch nmpc_sim_nodes bringup.launch.py

# 4. Watch it live, in a separate terminal -- matplotlib dashboard...
ros2 run nmpc_sim_nodes viz_node
# ...or RViz2 + its HUD companion window together (needs `unset GTK_PATH` first
# if RViz2 fails to launch from some terminals -- this launch file does that
# itself already, so it's only relevant if running rviz2 standalone):
ros2 launch nmpc_sim_nodes rviz_hud.launch.py

# 5. Build/edit a custom scenario layout with start, waypoints, goal, and obstacles
ros2 run scenario_maker scenario_editor

# 6. Run the NMPC validation suite (produces ~/nmpc_sim_logs/test_nmpc_results/*.png)
ros2 run nmpc_sim_nodes test_nmpc

# 7. Cross-validate NumPy vs CasADi vs acados MMG dynamics standalone
ros2 run mmg_model_validation validate_casadi

# 8. Compare true vs sensor-noise-corrupted signals standalone (no other node needed;
#    produces plots/CSVs under ~/nmpc_sim_logs/test_sensor_model_results/)
ros2 run nmpc_sim_nodes test_sensor_model

# 9. Run the full closed-loop pipeline headlessly, twice (no noise vs light noise),
#    at accelerated (non-real-time) speed; saves a final path-comparison plot under
#    ~/nmpc_sim_logs/test_closed_loop_noise_results/
ros2 run nmpc_sim_nodes test_closed_loop_noise

# 10. Sanity-check env_model's CurrentModel/WaveModel standalone (no other node needed;
#     produces plots under ~/nmpc_sim_logs/test_env_model_results/)
ros2 run nmpc_sim_nodes test_env_model

# 11. Run the full closed-loop pipeline headlessly, four times (no disturbance,
#     wave only, current only, current+wave), at accelerated (non-real-time)
#     speed; saves a final 4-path comparison plot under
#     ~/nmpc_sim_logs/test_closed_loop_env_results/
ros2 run nmpc_sim_nodes test_closed_loop_env
```

`ACADOS_SOURCE_DIR` is currently hardcoded to `/home/chandran/acados` in a
few places (`nmpc_sim_nodes/casadi_mmg_solver/casadi_mmg.py`,
`nmpc_sim_nodes/nmpc/nmpc_acados.py`,
`mmg_model_validation/validate_casadi.py`) — update this if running on a
different machine.

## Current status and open work

**Working and validated:**
- MMG dynamics model, cross-checked NumPy vs CasADi vs acados.
- Path-following NMPC (both solvers) tracking straight lines, offset
  starts, and multi-waypoint turns, with a distance-scaled braking ramp for
  arrival behavior.
- Live visualization + CSV telemetry logging for any scenario.
- Obstacle avoidance exercised end-to-end against the scenario's obstacles
  (not just an empty list), via `test_closed_loop_noise`'s headless full-pipeline
  rollout.
- A toggleable sensor noise model (`mmg_node`/`sensor_model`), run in-process
  by `mmg_node` between the true plant state and what `ukf_node` estimates
  from — GPS/compass/gyro/IMU-accelerometer/actuator noise (no direct
  surge/sway velocity measurement; reconstructing that is `ukf_node`'s job),
  with standalone comparison harnesses (`test_sensor_model`, `test_closed_loop_noise`)
  for viewing its effect with and without noise.
- A toggleable current + wave disturbance model (`mmg_node`/`env_model`),
  also run in-process by `mmg_node` and applied to the plant integrator — a
  mean-reverting (Ornstein-Uhlenbeck) current velocity plus a JONSWAP-spectrum
  wave drift force/moment — with standalone comparison harnesses
  (`test_env_model`, `test_closed_loop_env`) for viewing its effect with and
  without the disturbance. Current (not wave) is additionally fed into the
  NMPC's own OCP model as a live parameter — see the next bullet and
  [The NMPC formulation](#the-nmpc-formulation).
- An Unscented Kalman Filter state estimator (`ukf_node`/`ukf`), serving
  `/ukf/estimate` and called synchronously by `mmg_node` every tick.
  `mmg_node`'s `use_ukf` toggle selects whether `/nmpc/solve` receives the
  UKF's estimate or the true state directly, instead of always receiving a
  raw unfiltered noisy measurement. State: `[u,v,r,x,y,psi]` reconstructed
  from the GPS/gyro/IMU sensor stream (the accelerometer rows are a
  genuinely nonlinear measurement of the MMG dynamics' own acceleration
  output, evaluated per sigma point — why this needs a UKF, not a linear
  KF), augmented with an earth-frame current estimate `[vcx,vcy]` **and now
  also fed into the NMPC's own OCP dynamics** (`mmg_node` forwards
  `ukf_response.estimated_current` into `/nmpc/solve`'s request, which
  `nmpc_node` passes straight into `AcadosNMPC.solve(..., current=...)` —
  see [The NMPC formulation](#the-nmpc-formulation) for the full wiring),
  plus two Gauss-Markov sensor-bias pairs (`ax_bias`/`ay_bias`,
  `pos_bias_x`/`pos_bias_y`) tracked explicitly since each is the *only*
  channel a specific hidden state is observable through, so an unmodeled
  bias there would otherwise get misattributed almost entirely to the
  current estimate. Standalone comparison harness: `test_ukf` (a turning
  circle maneuver, plus a current-vs-null-baseline RMSE check); automated
  Q/R tuning: `tune_ukf`.
- RViz2/matplotlib HUDs (`rviz_node`, `viz_node`, `hud_node`) all show the
  UKF's predicted current (and, for `rviz_node`/`viz_node`, predicted x/y)
  directly alongside the actual/true values, `actual | predicted`, for at-a-
  glance comparison without needing a separate plotting pass.

**Known open bug:**
- The low-speed MMG singularity described above, which can freeze the
  acados solver during a slow pivot near a target.

**Known open limitation:**
- Wave is never seen by the NMPC's own OCP model (only current is, see
  above) — a real, if smaller, source of plant/solver dynamics mismatch than
  the current-blindness that used to exist. Not yet measured how much this
  matters in practice, since the current fix already closed the dominant gap
  (an acados closed-loop rollout with current enabled but not fed to the
  solver fails its QP almost every tick; the same test with wave-only
  disturbance and no current parameter passed completes cleanly).

**Done since the original action plan:** the controller now runs as a ROS2
(Jazzy) node graph (`nmpc_ws/` — see [Repository layout](#repository-layout)
above), sits behind a simulated noisy sensor layer with a UKF state
estimator rather than consuming the true plant state directly (though still
simulated, not real hardware), and its OCP model now accounts for current
disturbance rather than assuming still water.

**Not yet started** (per the original action plan, roughly in order):
LiDAR-based (rather than hardcoded) obstacle detection and clustering, the
safety fallbacks that would depend on it, feeding wave disturbance (not just
current) into the NMPC's own model, and progressively more realistic
hardware-in-the-loop / real-vessel testing.
