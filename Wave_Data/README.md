# Wave_Data

Second-order wave-drift force lookup tables used by the original NumPy MMG
model (`Preliminary_func.py`) to add wave-disturbance forces on top of the
calm-water hydrodynamics.

## Files

| File | MATLAB variable | Meaning |
|---|---|---|
| `DX_SALV_REV1.mat` | `DX` | Non-dimensional surge wave-drift force coefficient, indexed by (relative wave heading, non-dimensional wave frequency) |
| `DY_SALV_REV1.mat` | `DY` | Non-dimensional sway wave-drift force coefficient, same indexing |
| `DN_SALV_REV1.mat` | `DN` | Non-dimensional yaw wave-drift moment coefficient, same indexing |

These are classic "SALV" (added-resistance/drift) coefficient tables, of the
kind produced by seakeeping/potential-flow codes for a hull at a fixed set of
heading angles (every 15°, spanning -180°...+180°) and non-dimensional wave
frequencies. `Preliminary_func.py` loads all three with `scipy.io.loadmat`,
then builds a `scipy.interpolate.RegularGridInterpolator` over
`(heading, frequency)` to evaluate the drift force/moment at the ship's
actual relative wave heading and encounter frequency at each simulation step.

## Usage

Consumed by two things now:

- `Preliminary_func.py`'s `MMG_Time_Derivative(..., w_flag=True)` path (the
  original single-fixed-frequency regular-wave drift term).
- `nmpc_sim_nodes/env_model/wave_model.py`'s `WaveModel` (see
  `WAVE_CURRENT_DISTURBANCE_PLAN.md`), which loads these same three tables
  and builds the same `(heading, frequency)` `RegularGridInterpolator`s, but
  discretizes a full JONSWAP `Hs`/`Tp`/`gamma` sea state into many frequency
  components instead of evaluating at one, and sums a mean drift force with a
  Newman's-approximation slow-drift term.

The wave-drift term is still **not** ported into the CasADi/NMPC *prediction*
model chain (`casadi_mmg_solver/`'s `MMG_Time_Derivative_casadi` as called by
`nmpc/state_augmentation.py`) — table lookups don't belong in an NLP's inner
loop, and the disturbance-free dynamics are what the NMPC prediction model
needs. It's now reintroduced as exactly the "external unmodeled disturbance
term" case this note used to describe as hypothetical: `mmg_node` samples
`WaveModel` in-process (no separate node/service — see the root README's
architecture note) and adds the resulting numeric force straight into the
dynamics' RHS for that tick, via `casadi_mmg.py`'s optional `wave_force`
argument (passed through `make_casadi_integrator(..., with_env=True)`). The
NMPC's own internal model never calls `WaveModel` and never sees this data.

## Regenerating / replacing

These tables are specific to the scaled model hull used in this project (see
`Lpp = 2.902 m` in `Preliminary_func.py`). Swapping in a different vessel
requires new tables generated for that hull's geometry — the loader code
itself is hull-agnostic as long as the `.mat` variable names and grid
structure match.
