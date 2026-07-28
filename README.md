# BTP NMPC Obstacle Avoidance

A Nonlinear Model Predictive Controller (NMPC) for autonomous path-following
and (in-progress) obstacle avoidance on a small rudder-and-propeller surface
vessel, built from a validated 3-DOF MMG maneuvering model up through a
real-time acados-based solver. This is a B.Tech Project (BTP) — the
repository documents the whole path from "here is a MATLAB-derived NumPy
dynamics model" to "here is a working closed-loop NMPC controller," including
every bug that showed up along the way and how it was diagnosed and fixed.

If you only read one other file, read [`nmpc/README.md`](nmpc/README.md) —
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
10. [Getting started](#getting-started)
11. [Current status and open work](#current-status-and-open-work)
12. [A note on how this repository's history was built](#a-note-on-how-this-repositorys-history-was-built)

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
below. The disturbance-free, smoothed model is what the NMPC predicts with;
the full high-fidelity NumPy model (or a real vessel) is what it's actually
controlling.

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
                     │   casadi_mmg.py            │   smoothed, no wave drift;
                     └────────────┬─────────────┘   + acados SimSolver
                                  │ imported by
                                  ▼
                     ┌─────────────────────────┐
                     │        nmpc/               │   augmented-state NMPC:
                     │  config / path_following /  │   path-following +
                     │  state_augmentation /       │   (planned) obstacle
                     │  nmpc_casadi / nmpc_acados  │   avoidance, two solvers
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
here) is in [`nmpc/README.md`](nmpc/README.md). Summary:

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

A **known, currently-open issue** (a low-speed singularity in the MMG
model's `U = sqrt(u² + v²)` denominator, which can freeze acados' solver
during a slow pivot near a target) is documented in
[`nmpc/README.md`](nmpc/README.md) rather than silently left for someone to
rediscover.

## Repository layout

```
Preliminary_func.py          NumPy MMG dynamics model (ground truth)
Wave_Data/                   Wave-drift force lookup tables (.mat)
validate_casadi.py           NumPy vs CasADi vs acados validation harness
validation_comparison.png    ...its output plot

casadi_mmg_solver/           CasADi symbolic MMG port + acados SimSolver
mpc_visualization/           Standalone live-visualization dashboard
research_papers/             Citations for the papers the NMPC is adapted from
scenario_maker/              GUI for authoring custom track & obstacle scenarios
nmpc/                        The actual NMPC controller (both solvers)

c_generated_code_sim_exact/  Acados-generated C code (gitignored, auto-rebuilt)
c_generated_code_sim_smooth/ Acados-generated C code (gitignored, auto-rebuilt)
```

Every subdirectory has its own `README.md` with file-by-file detail; this
top-level file is deliberately the narrative/overview instead of duplicating
that detail.

## Getting started

```bash
# Environment: needs casadi, numpy, scipy, matplotlib, and acados_template
# (acados itself must be built separately: https://docs.acados.org)

# 1. Sanity-check the baseline dynamics model + wave data load correctly
python Preliminary_func.py    # (turning-circle test is commented out at the bottom)

# 2. Validate the CasADi/acados port against it
python validate_casadi.py

# 3. Run the CasADi MMG turning-circle demo standalone
python casadi_mmg_solver/casadi_mmg.py

# 4. Exercise the visualization dashboard with mock data
python mpc_visualization/run_demo.py

# 5. Build/edit a custom scenario layout with start, waypoints, goal, and obstacles
python scenario_maker/scenario_editor.py

# 6. Run the NMPC validation suite (produces nmpc/results/*.png)
python -m nmpc.test_nmpc

# 7. Watch an NMPC scenario live (including the scenario loaded from the maker)
python nmpc/run_live.py --solver acados --scenario scenario_maker/scenario.json
```

`ACADOS_SOURCE_DIR` is currently hardcoded to `/home/chandran/acados` in a
few places (`casadi_mmg_solver/casadi_mmg.py`, `nmpc/nmpc_acados.py`) —
update this if running on a different machine.

## Current status and open work

**Working and validated:**
- MMG dynamics model, cross-checked NumPy vs CasADi vs acados.
- Path-following NMPC (both solvers) tracking straight lines, offset
  starts, and multi-waypoint turns, with a distance-scaled braking ramp for
  arrival behavior.
- Live visualization + CSV telemetry logging for any scenario.

**Formulated but not yet exercised with real obstacles:**
- The soft obstacle-avoidance constraints exist in both solvers'
  formulations and are tested with an empty obstacle list in every current
  scenario, but haven't been run through an actual obstacle-avoidance test
  case yet (the original action plan's Phase 3).

**Known open bug:**
- The low-speed MMG singularity described above, which can freeze the
  acados solver during a slow pivot near a target.

**Not yet started** (per the original action plan, roughly in order):
LiDAR-based (rather than hardcoded) obstacle detection and clustering, state
estimation integration (EKF fusion of IMU/GPS), a ROS2 node wrapping the
controller with safety fallbacks, and progressively more realistic
hardware-in-the-loop / real-vessel testing.

## A note on how this repository's history was built

This project's actual development happened across several AI coding-
assistant sessions before any of it was under version control — the working
directory simply accumulated files over roughly three weeks, then `git init`
was run with nothing yet committed. The commit history in this repository
was reconstructed afterward from that working directory's final state plus
the accumulated chat history from those sessions (used only to understand
*what changed, when, and why* — the raw chat logs themselves aren't part of
this repository). Where a file went through multiple real, distinct stages
of development (for example `casadi_mmg_solver/casadi_mmg.py` gaining acados
support after its initial CasADi-only port, or the several rounds of bug
fixes to `nmpc/path_following.py`), those stages are represented as separate
commits with real incremental diffs between them, in their actual
chronological order — this history is a faithful reconstruction of how the
project was actually built, not an arbitrary re-packaging of a single final
snapshot.
