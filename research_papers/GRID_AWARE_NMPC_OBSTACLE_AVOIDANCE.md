# Grid-based obstacle avoidance: the NMPC math (theory only, no code changes)

## Context

The project currently represents obstacles as circles `(x, y, radius)`, fed into the
acados NMPC (`nmpc_acados.py`) as a fixed-size set of per-obstacle nonlinear
inequality constraints. This is the start of an integration with a future LIDAR
point-cloud → 2D occupancy-grid predictor, which will instead emit a fixed `N x N`
grid of blocked/free cells, only known within the sensor's range (partial
observability). Assuming that grid is already available (its production is out of
scope here), this document is a **theory-only** design note: how obstacle
avoidance's constraint and cost math in the NMPC changes when the input is a grid
instead of circles, and the design decisions already settled for this project. No
codebase changes are described here — see `nmpc/nmpc_acados.py` and
`nmpc/path_following.py` for the current implementation this compares against.

## 1. Today's math (baseline)

This is this project's own implementation of the soft slack-relaxed circle
constraint from Gonzalez-Garcia et al. 2022 [1] (the primary structural source for
`nmpc/`, per `research_papers/README.md`), corroborated by Collado-Gonzalez et al.
2024 [2]. Per obstacle `i` (up to `MAX_OBSTACLES`, dummy-padded when fewer are present),
`nmpc_acados.py` builds one nonlinear constraint per stage `k`:

```
h_i(x_k, y_k) = (x_k - ox_i)^2 + (y_k - oy_i)^2 - r_c_i^2 ,   r_c_i = radius_i + R_ASV
```

stacked into `con_h_expr = [h_1, ..., h_{MAX_OBSTACLES}]`, each soft-constrained
(`h_i + s_i >= 0`) via slack variables penalized in the cost through `SIGMA` /
`W_SLACK` (`Zl`, `Zu`). Obstacle data enters as **runtime parameters**
`(ox_i, oy_i, r_i)` — the OCP structure (number of constraints, codegen) is fixed at
build time by `MAX_OBSTACLES`; only the parameter *values* change between solves,
pushed identically to every stage in the horizon (obstacles are assumed static over
one horizon — no motion prediction).

Two things worth noting about this baseline, because the grid version changes both:
- The gradient of `h_i` is `2*(x_k - ox_i, y_k - oy_i)` — it scales *with* distance
  from the obstacle center, vanishing exactly at the center. It's not a distance in
  meters, it's a squared-distance-minus-squared-radius.
- Obstacle *count* is baked into the problem's dimensionality (`MAX_OBSTACLES` slots,
  padded with far-away zero-radius dummies when unused).

## 2. Why a raw occupancy grid can't be plugged in directly

A binary grid `O[i,j] ∈ {0,1}` (blocked/free) is piecewise-constant. Used directly as
a constraint, `h(x,y) = 0.5 - O(x,y) >= 0`, its gradient is zero almost everywhere and
undefined at cell boundaries — an SQP solver gets no useful descent direction from it
except exactly at a boundary crossing. This makes the raw grid unusable as-is for a
gradient-based NLP; it must first be turned into a **smooth scalar field**.

## 3. Step 1 — distance transform: grid → continuous distance field

Convert the binary grid into a Euclidean distance-to-nearest-obstacle field:

```
D[i,j] = min over all occupied cells (m,n) of  cell_size * ||(i,j) - (m,n)||
```

computed once per grid update (e.g. via an EDT / fast-marching pass — Sethian's
Fast Marching Method [3] is the classic reference for solving the Eikonal equation
this relies on), *not* inside the solver's per-iteration expression. `D` is defined
on the same discrete grid but is now a smooth, slowly-varying quantity (a true
distance in meters), rather than a step function. For a true Euclidean distance
transform, `D` satisfies the Eikonal property `|∇D| = 1` almost everywhere — its gradient is a unit vector pointing away
from the nearest obstacle, with *uniform* magnitude everywhere. This is a materially
nicer property than the circle constraint's gradient, whose magnitude depended on
distance from the obstacle center: here the solver always gets a consistently-scaled
"move this way to gain 1 meter of clearance" signal, regardless of where the ship is
relative to obstacles.

## 4. Step 2 — interpolation: discrete field → differentiable function of continuous (x,y)

The ship's position at each stage is a continuous decision variable, not a grid
index. `D[i,j]` must be turned into `D(x,y)`, differentiable in `x,y`, via bilinear
(or bspline) interpolation over the 4 (or more) neighboring cells. Symbolically
(CasADi [4], which acados [5] is built on, supports differentiable gridded
interpolants), this becomes a function `D̂(x, y; grid_values)` — differentiable in
`x,y`, and, critically, the `grid_values` themselves can be left as free parameters
rather than baked into the generated code. This is what lets the *same compiled
solver* be re-used solve after solve as the LIDAR predictor's grid updates: the
interpolation *topology* (resolution, extent, cell layout) is fixed at build time;
only the cell *values* — analogous to today's `(ox_i, oy_i, r_i)` — are runtime
parameters. acados's support for setting distinct parameter values per stage (used
here to update the grid, and later for moving obstacles — see the companion doc) is
documented concretely in Frey et al.'s multi-phase-OCP paper [6].

## 5. Step 3 — the new constraint (and how the cost changes)

Replace the `MAX_OBSTACLES` separate `h_i` terms with a **single** constraint per
stage:

```
h(x_k, y_k) = D̂(x_k, y_k) - r_safety ,   r_safety = R_ASV + margin
```

soft-constrained exactly the same way as today (`h + s >= 0`, slack penalized via
`SIGMA`/`W_SLACK`/`Zl`/`Zu`). So structurally, **the cost's penalty mechanism for
constraint violation is unchanged in form** — it's still "penalize the slack that
activates whenever the ship gets closer than `r_safety` to the nearest obstacle."
What changes:

- **One slack term per stage instead of up to `MAX_OBSTACLES`.** Today's `n_obs`
  independent slacks collapse into a single slack, because the distance field
  already encodes the *union* of every obstacle shape (convex or not, however many
  cells) as one number: distance to the nearest occupied cell, whatever it is. This
  also makes the per-stage QP smaller (fewer inequality rows and slacks), which
  should help solve time.
- **Obstacle count is no longer part of the problem's structure.** There's no
  `MAX_OBSTACLES` cap, no dummy-padding logic (`pad_obstacles`) — the grid's
  resolution/extent (fixed at build time) is the only structural parameter; how many
  distinct blobs of occupied cells exist within it is irrelevant to the solver's
  dimensionality. A LIDAR frame with 2 obstacles or 40 costs the same to solve.
- **An optional smooth potential cost term — available in BOTH representations,
  not exclusive to the grid (correction).** `h_i` for a circle is already smooth
  everywhere, not just at its zero-crossing, so the same idea works today:
  `L_obs = Σ_i w * exp(-(sqrt(h_i + r_c_i^2) - r_c_i)/λ)` (the `sqrt` converts
  squared-distance into actual meters of clearance first). What the grid changes is
  narrower than "makes this possible": (a) arbitrary-shaped/concave/unioned occupied
  regions collapse into *one* `D̂(x,y)` term instead of a sum over `MAX_OBSTACLES`
  per-circle exponentials, and (b) a true Euclidean distance transform satisfies
  `|∇D|=1` (Eikonal) everywhere, giving a uniformly-scaled "meters of clearance"
  gradient for free, where the circle's raw `h_i` needs the extra `sqrt` above to
  get the same metric meaning. Either way the term looks like
  ```
  L_obs(x_k, y_k) = w_obs * exp(-(D̂(x_k,y_k) - r_safety) / λ)
  ```
  giving smoother, less "last-moment" avoidance than the current constraint-only
  formulation (today's `Q_DIAG` has no "closer is worse" notion below the hard/soft
  constraint threshold — avoidance only engages once `h_i` crosses zero). This is an
  additive design choice layered on top of section 5's constraint either way, not a
  replacement for it, and not something the grid switch requires.

  **Decision for the initial implementation:** wire `L_obs` in, but set `w_obs = 0`
  to start — the term exists in the formulation but is inert until `λ`/`w_obs` are
  tuned against real grid data (see section 6 on why churn makes this term the
  harder one to tune, and why the constraint in this section, not the potential
  term, is what should carry safety in the meantime).
- **The horizon-static assumption is preserved by default.** Exactly like today
  (the same `(ox,oy,r)` values are pushed to every stage `k`), the same flattened
  grid/distance-field parameter block would be pushed identically to every stage —
  the grid is treated as unchanging over one horizon. A dynamic/forecasted grid (a
  distinct grid per stage) is a much heavier extension (real per-stage obstacle
  prediction) and is explicitly *not* implied by simply switching the obstacle
  *representation* from circles to a grid; call it out separately if it's ever
  wanted.

## 6. Partial observability: the LIDAR only sees a limited range

Everything above implicitly assumed a complete grid. A real LIDAR-derived grid is
only known within the sensor's range/FOV; the rest is genuinely unknown (a third
state, distinct from free/occupied — e.g. `-1` in the `nav_msgs/OccupancyGrid`
convention), and the *set* of known cells keeps changing as the ship moves — cells
flip unknown → known continuously, for reasons that have nothing to do with the
world changing. This has concrete consequences for the math above, and settled
decisions for this project on each:

- **Horizon vs. sensor range: not a concern here (settled).** In general, `N * dt`
  of lookahead could exceed the sensor's range, which would force `D̂`/the
  constraint to go *inactive* beyond the sensed radius rather than guessing
  free/occupied for unseen cells — safety would then only ever hold for what's
  currently visible, corrected next solve (a receding-horizon guarantee, not a
  full-horizon one — inherent to feedback MPC with frequent replanning, here RTI at
  `dt=0.1s`). For this project the sensor range covers the full horizon, so this
  case doesn't need handling; noted here only so the reasoning is on record if the
  sensor or horizon length changes later.
- **The grid churns near its own boundary independent of the real world.** A fringe
  cell's value can swing between consecutive solves purely because visibility
  improved, not because anything in the world moved. A sharp potential term (small
  `λ`, high weight) reacting to that reads as trajectory jitter — the same
  "costmap chatter" familiar from rolling local costmaps in mobile robotics (see
  Nav2's static/obstacle/inflation layered-costmap design [7], and the general
  local-vs-global-costmap split it documents, for the ROS-ecosystem version of
  exactly this static/rolling-window pattern). This is a large part of why
  section 5's `L_obs` starts at `w_obs = 0`: the
  churn/tuning problem below needs to be worked out against real data before that
  term should carry any weight.
  - **Confidence-weight the potential term (decided: adopt).** Fade it to ~0 for
    just-revealed/unknown cells and to full strength only for cells observed for a
    while: `w_obs(x,y) = w_max * conf(x,y)`.
  - **Filter/rate-limit the grid or distance field before it reaches the solver**
    (a low-pass across updates) so one noisy frame can't reshape the cost landscape.
  - Damp the **soft potential term** harder than the **hard/soft constraint** — the
    constraint is the safety backstop and should react fast to genuinely new
    near-ship information; the potential term is a shaping signal further out,
    where jitter is more visible and less safety-critical. This is consistent with
    starting `w_obs = 0`: the hard/soft constraint (section 5) carries safety from
    the start, the potential term is switched on later once tuned.
- **Egocentric rolling grid (decided: adopt), not an accumulating world-frame
  grid.** Circles never forced this choice (an `(x,y,r)` doesn't "scroll out of
  frame"). A ship-centered rolling window keeps the "grid static across one
  horizon" assumption clean, at the cost of zero memory once something leaves
  range — the alternative (accumulating world-frame grid, keeping memory but with
  very different confidence between fringe and core) was considered and not taken.

## 7. A non-math-changing alternative, for contrast

It's possible to consume a grid *without* touching any NMPC math at all: cluster the
occupied cells into connected components and fit a minimum-enclosing (or padded)
circle to each blob, then feed those circles into the existing `(x,y,r)` machinery
unchanged. This is simpler and zero-risk to the solver, but reintroduces the
`MAX_OBSTACLES` cap, discards the uniform-gradient benefit of a true distance field,
and can't represent concave/elongated occupied regions (e.g. a wall or coastline)
faithfully with a single circle. It's the pragmatic fallback if the interpolant-based
approach above turns out to be too heavy for the real-time budget; it is not what
this document is proposing as the primary approach.

## Summary of the actual math change

| | Circles (today) | Grid / distance field |
|---|---|---|
| Constraint count per stage | up to `MAX_OBSTACLES` | 1 |
| Constraint function | `(x-ox)^2+(y-oy)^2-r_c^2` per obstacle | `D̂(x,y) - r_safety`, one interpolated field |
| Gradient magnitude | grows with distance from obstacle center | ≈1 everywhere (Eikonal property) |
| Parameters per solve | `3 * MAX_OBSTACLES` scalars | flattened grid/distance values (fixed count = grid cells) |
| Obstacle count vs. solver structure | capped, padded with dummies | decoupled entirely |
| Cost term for "near but not violating" | possible via `h_i` directly, but unused today | same idea, one field instead of a per-obstacle sum; `w_obs=0` initially |
| Horizon treatment | static across horizon | same, static across horizon (sensor range covers full horizon here) |
| Grid framing | n/a | egocentric rolling window, confidence-weighted potential term |

## References

1. Gonzalez-Garcia, A., Collado-Gonzalez, I., Cuan-Urquizo, R., Sotelo, C., Sotelo,
   D., Castañeda, H. (2022). "Path-following and LiDAR-based obstacle avoidance via
   NMPC for an autonomous surface vehicle." *Ocean Engineering*, 266, 112900.
   https://doi.org/10.1016/j.oceaneng.2022.112900 — already cited in this
   directory's `README.md`; the source of this project's soft slack-relaxed circle
   constraint that section 1 restates as the baseline.
2. Collado-Gonzalez, I., Gonzalez-Garcia, A., Cuan-Urquizo, R., Sotelo, C., Sotelo,
   D., Castañeda, H. (2024). "Adaptive sliding mode control with nonlinear
   MPC-based obstacle avoidance using LiDAR for an autonomous surface vehicle under
   disturbances." *Ocean Engineering*, 311, 118998.
   https://doi.org/10.1016/j.oceaneng.2024.118998 — also cited in `README.md`;
   corroborates the same obstacle-constraint formula independently.
3. Sethian, J.A. (1996). "A fast marching level set method for monotonically
   advancing fronts." *Proceedings of the National Academy of Sciences*, 93(4),
   1591–1595. Classic reference for the Eikonal-equation/fast-marching numerics
   behind section 3's distance transform.
4. Andersson, J.A.E., Gillis, J., Horn, G., Rawlings, J.B., Diehl, M. (2019).
   "CasADi – A software framework for nonlinear optimization and optimal control."
   *Mathematical Programming Computation*, 11(1), 1–36.
   https://doi.org/10.1007/s12532-018-0139-4 — the symbolic framework section 4's
   differentiable gridded interpolant relies on, and that `nmpc_acados.py` is
   already built on.
5. Verschueren, R., Frison, G., Kouzoupis, D., Frey, J., van Duijkeren, N.,
   Zanelli, A., Novoselnik, B., Albin, T., Quirynen, R., Diehl, M. (2022). "acados:
   a modular open-source framework for fast embedded optimal control."
   *Mathematical Programming Computation*, 14, 147–183.
   https://doi.org/10.1007/s12532-021-00208-8 — the solver framework this project
   uses (`nmpc_acados.py`).
6. Frey, J. et al. "Multi-Phase Optimal Control Problems for NMPC with acados."
   arXiv:2408.07382. Confirms acados's native support for distinct parameter
   values per horizon stage, the mechanism section 4 relies on for updating grid
   values between solves (and that the companion moving-obstacle doc relies on for
   per-stage-varying obstacle positions).
7. Nav2 (ROS2 Navigation) documentation: "Costmap 2D"
   (https://docs.nav2.org/configuration/packages/configuring-costmaps.html),
   "Static Layer Parameters"
   (https://docs.nav2.org/configuration/packages/costmap-plugins/static.html),
   "Obstacle Layer Parameters"
   (https://docs.nav2.org/configuration/packages/costmap-plugins/obstacle.html).
   Reference implementation of the layered static/obstacle costmap split and the
   local (rolling, robot-centered) vs. global (static) costmap distinction
   invoked in section 6.
