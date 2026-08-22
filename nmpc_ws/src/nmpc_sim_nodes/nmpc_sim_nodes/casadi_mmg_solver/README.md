# casadi_mmg_solver

A self-contained CasADi/Acados port of the MMG vessel dynamics model, used
as the prediction model for every NMPC solver in `nmpc/`, and as a
standalone fast simulator for the closed-loop plant in every test/demo.

## Files

- **`casadi_mmg.py`** — the module. Nothing in here depends on `nmpc/`;
  it's imported *by* `nmpc/` (via `MMG_Time_Derivative_casadi` and
  `make_casadi_integrator`), not the other way around.

## What's inside

- **`MMG_Time_Derivative_casadi(state, control, current=(0.0, 0.0), wave_force=(0.0, 0.0, 0.0), smooth=True, scale=1000.0)`**
  — the symbolic (CasADi `ca.MX`/`ca.SX`) equivalent of
  `Preliminary_func.py`'s `MMG_Time_Derivative`. Same hydrodynamic
  derivatives, same non-dimensionalization, same force breakdown
  (hull/propeller/rudder) — but every discontinuous branch is replaced with
  something differentiable:
  - `u`, `rps` floors via `ca.fmax` instead of an `if`.
  - The rudder flow-straightening coefficient `gamma_r` (0.396 vs 0.64,
    depending on the sign of the effective rudder inflow angle) is
    tanh-blended between its two values instead of switching with an `if`.
  - `arctan(-v/u)` → `atan2(-v, u)`, correct and singularity-free at `u≈0`
    and in reverse (`u<0`).
  - The wave-drift lookup-table block is removed entirely (kept only in
    `Preliminary_func.py` — see `Wave_Data/README.md`). Table interpolation
    doesn't belong in an NLP's inner loop.
  - `smooth=False` switches back to exact `ca.if_else` branching, useful for
    confirming the smoothed version doesn't change the physics (see
    `validate_casadi.py`).

  `current` (earth-frame `(vcx, vcy)`, default zero) and `wave_force`
  (body-frame `(fx, fy, fn)` surge/sway/yaw drift force/moment, default
  zero) are optional plant-only disturbance inputs, both defaulting to
  exactly the pre-existing zero-disturbance behavior so every other call
  site (including the NMPC's own prediction model) is unaffected unless it
  opts in. `current` is rotated into the body frame via the state's own
  `psi` and substituted into the *relative* velocity (`ur = u_val - uc`,
  `vr = v - vc`) used by the hull/propeller/rudder force terms only — the
  mass matrix, `LHS_r`, and the kinematics block that integrates position
  still use the state's own (absolute) `u`, `v`, `r` untouched. `wave_force`
  is added straight into the dynamics' RHS before the `M⁻¹` solve. Only
  `mmg_node.py` ever samples non-default values for these (its in-process
  `env_model` current/wave models) — see the `F_env` note in the root
  README's MMG section.

- **`make_casadi_integrator(h, method="rk4"|"euler", smooth=True, scale=1000.0, sym_type=ca.MX, with_env=False)`**
  — wraps the above into a single `ca.Function`. With `with_env=False`
  (the default, and every call site except `mmg_node.py`) it maps
  `(state, control) -> (next_state, r_dot_a)`, `current`/`wave_force` fixed
  at zero. With `with_env=True` it takes two extra symbolic inputs,
  `(state, control, current, wave_force) -> (next_state, r_dot_a)`, held
  constant across the RK4 substeps the same way `control` already is within
  one integration step. The RK4 variant does the same "integrate `u,v,r` in
  body frame, rotate the position increment once per step" trick as
  `Preliminary_func.py`'s `activateMMG`, so the two stay numerically close
  (see the validation harness).

- **`make_acados_integrator(h, smooth=True, scale=1000.0)`** — builds an
  Acados `AcadosSimSolver`: the same dynamics, but code-generated to C and
  compiled to a shared library (`libacados_sim_solver_vessel_mmg_acados_*.so`),
  for near-native execution speed instead of paying Python/CasADi
  interpreter overhead on every step. This is what `nmpc/run_live.py`
  and the closed-loop test harnesses actually step the "real" plant with.
  Generated C code lands in `c_generated_code_sim_exact/` or
  `c_generated_code_sim_smooth/` (both gitignored — they're rebuilt
  automatically the first time this function runs).

  The module preloads Acados' shared library dependencies itself, in
  topological order (`qdldl → osqp → qpOASES → blasfeo → hpipm → acados`)
  via `ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)`, so importing this file
  works without the caller having `LD_LIBRARY_PATH` pre-set in their shell.
  `ACADOS_SOURCE_DIR` is hardcoded to `/home/chandran/acados` — update this
  if running on a different machine.

## Running it standalone

```bash
python casadi_mmg_solver/casadi_mmg.py
```

Runs a 200s, 35°-rudder/18.2-rps turning-circle test with **both** the pure
CasADi RK4 integrator and the Acados SimSolver from a cold start, prints the
max trajectory divergence between the two, and saves an overlay plot to
`casadi_turning_circle_test.png`. Typical divergence is on the order of
2-3 cm in position over the full 200s run (Acados applies its coordinate
rotation per RK4 stage; the hand-rolled integrator applies it once per
step) — negligible relative to the ~1.45m collision radius used downstream.

## Dependencies

`casadi`, `numpy`, `matplotlib` (only for the `__main__` demo),
`acados_template` (only for `make_acados_integrator`).
