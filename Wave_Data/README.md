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

Only consumed by `Preliminary_func.py`'s `MMG_Time_Derivative(..., w_flag=True)`
path. The wave-drift term is deliberately **not** ported into the CasADi/NMPC
model chain (`casadi_mmg_solver/`, `nmpc/`) — table lookups don't belong in an
NLP's inner loop, and the disturbance-free dynamics are what the NMPC
prediction model needs. If wave disturbances are ever reintroduced for the
NMPC (e.g. as an external unmodeled disturbance term rather than an internal
prediction-model term), this is the data they'd come from.

## Regenerating / replacing

These tables are specific to the scaled model hull used in this project (see
`Lpp = 2.902 m` in `Preliminary_func.py`). Swapping in a different vessel
requires new tables generated for that hull's geometry — the loader code
itself is hull-agnostic as long as the `.mat` variable names and grid
structure match.
