# Non-circular obstacles: walls as capsules, ellipses via gradient-normalized distance, unified soft-min constraint

## Context

`GRID_AWARE_NMPC_OBSTACLE_AVOIDANCE.md` (theory-only) explored replacing the
circle `(x, y, radius)` obstacle constraint with one derived from a LIDAR-fed
occupancy grid, specifically to accommodate obstacles that aren't circles —
this project's scenarios need long, thin obstacles (harbor walls, quays,
breakwaters) that a circle can only represent by being absurdly conservative.
This document is the implementation record for that same goal, arrived at by
a different, non-grid route. The accompanying `field_viz/` visualizer (a
live occupancy/distance/potential-field overlay in RViz) existed for a
while as a standalone, decoupled tool — nothing in the closed-loop solver
ever read from it — and was later deleted entirely once the analytic
circle/capsule/ellipse primitives below fully covered what it was built to
explore.

## 1. Why the grid was dropped from the solver path

`nmpc_acados.py` solves via acados' SQP-RTI, which needs a smooth,
closed-form, symbolically-differentiable constraint expression rebuilt fresh
into machine code once at OCP-build time — a discretized occupancy/distance
grid is neither smooth nor available as a differentiable expression the
solver's own code-generation can compile against. This isn't a judgment
call specific to this project: acados' own maintainers, asked almost this
exact question on their forum, state that SQP-type methods "require smooth
functions" and that something discrete-by-definition (they cite voxblox) is
"not really suitable" as-is[^1]. Every real acados/CasADi project surveyed
while designing this (a neural-SDF NMPC running acados+CasADi+L4CasADi at
~40-50Hz[^2], a polygonal-SDF real-time MPC[^3]) computes its distance field
analytically, never by querying a live discrete grid from inside the solver.
So the grid stayed exactly what it already was — a passive, decoupled
visualization tool (`field_viz/grid_field.py`, `nodes/field_viz_node.py`) —
while the actual obstacle representation moved to closed-form geometric
primitives instead. Once those primitives (circle/capsule/ellipse below)
fully covered this project's obstacle shapes, the grid tool itself was
removed rather than kept around unused.

## 2. The capsule: the natural generalization of a circle

A **capsule** is a line segment `[p0, p1]` padded by a radius `r`. A circle
is the *degenerate* capsule where `p0 == p1` — so one closed-form,
CasADi-differentiable distance function covers both circles and walls with
no branching:

```
t        = clamp( dot(ship - p0, p1 - p0) / (dot(p1-p0, p1-p0) + eps), 0, 1 )
closest  = p0 + t * (p1 - p0)
d        = ||ship - closest|| - r
```
(`nmpc/path_following.py:capsule_distance_casadi`). `clamp` is `ca.fmin`/
`ca.fmax` — differentiable almost everywhere, the same character as the
project's existing rate-limit bounds. The `+eps` guard (reusing
`config.EPS`) makes the degenerate circle case exact and safe: when
`p0 == p1`, the numerator of `t` is also a dot product against the zero
vector (`0`), so `t` evaluates to `0/eps == 0` regardless of `eps`'s exact
value, giving `closest == p0` — precisely the correct circle-distance
result, with no special-cased branch.

An ellipse doesn't get this same trick: true point-to-ellipse Euclidean
distance has **no closed form** (it requires solving a quartic), so it can't
be embedded directly as a differentiable expression. Its natural
representation is instead an implicit *algebraic* containment test (a
rotated, per-axis-scaled quadratic form, `(dx/a)² + (dy/b)² - 1 >= 0`)
rather than a literal distance — dimensionless, not meters, a genuinely
different shape of problem from the circle/capsule case. §6 resolves this
(gradient-normalized distance) rather than folding ellipses in here.

## 3. Unifying circles and walls into one constraint

Every circle and every wall capsule's `d_i` (up to `MAX_OBSTACLES` circle
slots + `MAX_WALLS` wall slots, dummy-padded exactly like the old
`pad_obstacles` when unused) is aggregated into **one** scalar via a smooth
soft-min, replacing the old one-row-per-circle scheme entirely:

```
D_hat = -1/K * log( sum_i exp(-K * d_i) )
h     = D_hat        (single constraint row, h >= 0, soft-slacked)
```

**Safety property:** since `sum_i exp(-K*d_i) >= max_i exp(-K*d_i)`,
`D_hat <= min_i(d_i)` always — the soft-min constraint is never less
conservative than the true nearest-obstacle distance, only ever equal or
more cautious, regardless of `K`. This collapses cleanly onto the old
per-circle scheme's slack machinery: one row (`idxsh=[0]`), one slack
(`Zl=Zu=[W_SLACK]`, `lsh=[-SIGMA]`), same `SIGMA`/`W_SLACK` values reused
unchanged, since in practice only the single nearest obstacle ever bound the
old per-row constraint anyway.

**Real-world corroboration of this exact shape:** a published acados+CasADi
project using true SDFs formulates its obstacle avoidance the same way —
a *single* hard constraint with slack relaxation[^2], not per-obstacle rows.

## 4. A numerical trap in the soft-min: per-term clipping, not a shared shift

The naive soft-min above underflows in a specific, easy-to-miss way: a
padded/unused obstacle slot sits ~1400m from any real scenario (the
existing `pad_obstacles`/`pad_walls` dummy convention, `dummy_pos=1.0e3`).
With `K=3`, `exp(-K * 1400)` is astronomically below float64's underflow
floor (`exp(x)` underflows to a hard `0.0` once `x` drops below about
`-745`; here it's `-4200`). If **every** slot happens to be an unused dummy
(the plain no-obstacle case), every term in the sum underflows to exactly
`0.0`, the sum is `0.0`, and `log(0.0) = -inf` — poisoning the constraint
Jacobian on **every single QP solve**, confirmed via
`ros2 run nmpc_sim_nodes test_nmpc`: 100% QP failures (`return_status=4`)
on every step of the plain no-obstacle tracking tests, before this fix.

The standard textbook fix — subtract the running minimum before
exponentiating, a numerically-stable log-sum-exp shift (`d_min = min(d_i)`,
`D_hat = d_min - (1/K)*log(sum(exp(-K*(d_i - d_min))))`) — is an exact
algebraic identity, but was actively **wrong** to use here: with `ca.mmin`,
the shift is a single value shared across every term, differentiated
through all of them at once. Every padded dummy slot sits at the *literal
same coordinate*, so their distance functions are identical for every
`(x, y)`, not just numerically tied at one point — a hard `min` over
exactly-identical functions is degenerate *everywhere*, not a rare edge
case. CasADi's automatic differentiation of that shared `mmin` produced a
corrupted gradient severe enough to break ordinary path tracking even with
the QP succeeding and zero real obstacles present (confirmed by running
`test_nmpc` with this version: `e_y` blew out to 25m+ on the plain tracking
tests, despite `return_status=0` on every step).

The fix that actually works: **clip each `d_i` independently** to a fixed
cap before exponentiating (`ca.fmin(d_i, _SOFTMIN_DISTANCE_CAP)`,
`_SOFTMIN_DISTANCE_CAP = 100.0` meters), not a shared data-dependent
minimum. Each slot's clip is completely independent of every other slot —
tied or identical dummy distances are simply tied or identical *clipped*
terms, with no cross-term coupling for CasADi's differentiation to get
wrong. The cap only ever discards precision far past any real safety
margin (verified: `test_capsule_distance.py`'s regression check computes
the exact expected value analytically and matches to `1e-3`), never near
where the constraint could plausibly bind.

(Separately investigated and ruled out as the cause of the `e_y` blowup:
whether it was inherited from unrelated pre-existing work already in the
tree. It wasn't — the same `e_y` blowup, and worse, reproduces at a clean
`git stash` back to the last commit, entirely without any of this session's
changes. That's a real, separate, pre-existing regression in the
"current-aware NMPC" work, out of scope for this document.)

## 4b. A second, more serious bug: `W_SLACK` no longer meant what it used to

After the fixes in §4, a real deployment (a user-drawn scenario via
`scenario_editor.py`'s new Wall mode) still produced a genuine collision:
the ship passed **3.44m through** a real circle obstacle, with the solver
reporting `success=True` on every single step the entire time. Reproduced
deterministically outside ROS (no UKF/sensor noise involved) with the exact
scenario, confirming it wasn't a ROS-plumbing issue.

The cause: `lsh`/`ush` (the "soft bound" fields acados exposes) do **not**
act as a hard cap on how far the constraint can be violated in practice —
only the quadratic slack penalty weight (`W_SLACK`) actually discourages a
violation, by making it costlier than the alternative. The old per-circle
constraint's `h` was in **squared** meters (`h=dist²-r_c²`); the new unified
constraint's `h` is **linear** meters. Same `W_SLACK=50`, completely
different physical meaning: near a large obstacle, `W_SLACK=50` in the old
squared-meter units made even a "generously slack-relaxed" violation tiny
in real terms (~1.4cm at `r_c=7.4m`, from the local linearization
`h≈-2·r_c·Δ`) — the old scheme was never really tested against a case where
this mattered, because its own units happened to mask the gap. In linear
meters, the same numeric weight was nowhere near stiff enough: a real
detour around a large obstacle costs more in tracking-cost terms (accrued
over many horizon stages, `Q[e_y]`/`Q[x]`/`Q[y]`) than eating a large slack
penalty and driving straight through.

Fixed by raising `W_SLACK` to `5000` (`sim_params.yaml`) — found via a
sweep (`2000` is the bare threshold where the obstacle is just barely
avoided; `5000` gives a solid ~0.5m clearance margin with zero solver
failures) and locked in as a permanent regression test,
`test7_real_obstacle_collision_regression` in `tests/test_nmpc.py`, using
the exact scenario from the collision. That test deliberately builds its
own fresh `AcadosNMPC` instance rather than reusing a solver that's already
run through several unrelated scenarios — SQP-RTI's internal iterate
persists across `solve()` calls as a warm start, so chaining this scenario
after very different ones produces a materially different (and misleading)
result than a real deployment, which always starts from one fresh solver.

## 5. What changed, file by file

- **`nmpc_interfaces`**: new `WallObstacle.msg` (`id, x0, y0, x1, y1, radius`)
  / `WallObstacleArray.msg`, a new `/map/walls` topic (latched, mirrors
  `/map/obstacles`), and a `walls` field on `GetScenario.srv`. Circle
  messages (`Obstacle`/`ObstacleArray`) are untouched.
- **`nmpc/path_following.py`**: `pad_walls` (mirrors `pad_obstacles`),
  `capsule_distance_casadi`, `softmin_casadi`.
- **`nmpc/params.py` / `sim_params.yaml`**: new `MAX_WALLS` (default 5) and
  `SOFTMIN_K` (default 3.0) config fields, loaded the same way as
  `MAX_OBSTACLES`.
- **`nmpc/nmpc_acados.py`**: the parameter vector grows to
  `5 + 3*MAX_OBSTACLES + 5*MAX_WALLS`; the old per-circle `h_list` loop is
  replaced by the unified soft-min row described above; `AcadosNMPC.solve()`
  takes a new `walls=` kwarg alongside `obstacles=`.
- **`map_node.py` / `nmpc_node.py`**: load/publish/cache `walls` the same
  way `obstacles` already are.
- **`scenario_editor.py`**: a new "Wall" mode — click sets one endpoint,
  click again sets the other (a wall has no natural drag distance the way a
  circle's radius does); a `wallr_box` sets the padding radius. Saved under
  a new `"walls"` key in `scenario.json`, alongside the untouched
  `"obstacles"` key.
- **`rviz_node.py`**: wall markers (`LINE_STRIP`, `scale.x = 2*radius`) —
  RViz doesn't round the line's end caps the way the solver's exact capsule
  distance does; a documented visualization-only simplification, it doesn't
  affect the constraint math itself.
- **`tests/test_capsule_distance.py`**: headless (no ROS/acados) checks —
  the degenerate-circle case matches the old formula exactly, endpoint
  clamping, perpendicular mid-segment distance, the soft-min safety
  property and its convergence as `K` grows, and the all-dummy-slots
  underflow regression described in §4.
- **`tests/test_nmpc.py`**: `test7_real_obstacle_collision_regression`, an
  acados closed-loop check (its own fresh solver, per §4b) that reproduces
  the exact real collision scenario and asserts the obstacle is actually
  cleared, not just that the solver reports success.

## 6. Ellipses: resolved via gradient-normalized algebraic distance

§2 deferred ellipses (for modeling other ships) because they have no closed-
form Euclidean distance, and their natural representation — an implicit
algebraic containment test, `g(x,y) = (dx/a)² + (dy/b)² - 1 >= 0` (`dx, dy`
being `(x,y)` rotated into the ellipse's own frame) — is dimensionless, not
meters, blocking it from the meters-based soft-min sum without further work.

This is a real, literature-acknowledged problem, not one specific to this
project. Lutz & Meurer[^4], surveying obstacle-avoidance formulations for
exactly this kind of trajectory-optimization problem, note of the plain
ellipsoidal representation: *"the defining function... does not yield a
distance measure that can directly be mapped to the euclidean distance so
that the introduction of a uniform safety distance remains complicated."*

**Resolution: gradient normalization**, the same family of technique as
Rimon-Koditschek navigation-function potentials — a standard approach in the
robotics literature for converting an obstacle's algebraic representation
into an approximate distance:

```
dx = (x-xc)*cos(theta) + (y-yc)*sin(theta)      # world -> ellipse-local frame
dy = -(x-xc)*sin(theta) + (y-yc)*cos(theta)
g  = (dx/a)^2 + (dy/b)^2 - 1                    # dimensionless; 0 on boundary
grad_mag = 2*sqrt((dx/a^2)^2 + (dy/b^2)^2)       # |grad g|, units 1/meter
d_ellipse = g / (grad_mag + eps) - r_pad         # meters
```
(`nmpc/path_following.py:ellipse_distance_casadi`). `grad_mag`'s formula
comes directly from the chain rule — rotation is orthonormal, so cross terms
cancel exactly. Because `d_ellipse` is now genuinely meters-scaled, it drops
straight into the *same* `d_list`/`softmin_casadi` used by circles and walls
— no separate constraint row, no change to `SIGMA`/`W_SLACK`/the soft
constraint block. This is what actually resolves the deferred units
mismatch: not a new mechanism, just a term that finally speaks the same
units as everything else already there.

**Degenerate sanity check** (a=b=r, a circle): the formula reduces exactly
to `(D²-r²)/(2D)`, `D` = center-distance. First-order exact at the boundary
(`D=r`) — where the `SIGMA`-bounded constraint actually cares.

**The safety property that matters here is conservatism, not far-field
accuracy** — matching `softmin_casadi`'s own established philosophy (§3:
"never less conservative than the true nearest-obstacle distance"). A first
design draft claimed the approximation was also asymptotically *exact* far
from the ellipse; checked by direct derivation and a 960-point numerical
sweep (5 eccentricities × 24 angles × 8 range factors) and found **wrong** —
the far-field ratio to true distance settles around 0.5 for a circle (not
1.0), and as low as ~0.26–0.36 off-axis for a 10:1 aspect-ratio ellipse. What
the same sweep *did* confirm, with zero violations across all 960 points:
`d_ellipse` never overestimates clearance. That — not far-field precision —
is the property this design actually needs, since the constraint only
meaningfully acts near the point of closest approach; the far-field
degradation only ever makes the soft-min react a little earlier than
strictly necessary to a distant, eccentric ellipse, never later. For
scenarios needing tight long-range accuracy against a very elongated
ellipse, prefer a chain of capsules instead — `tests/test_ellipse_distance.py`
pins both properties (a `check_conservative_vs_bruteforce` numerical sweep,
and a `check_far_field_degradation_characterized` test recording the ~0.5
ratio, so a future silent change to either is caught).

**One real divergence from the capsule/wall calling convention, not an
oversight:** circles and walls pass `r_c = object_radius + config.R_ASV` as
their padding — but an ellipse's own size is already fully encoded in `a, b`,
so `ellipse_distance_casadi`'s padding argument is `config.R_ASV` alone
(ship-only). `r_pad` is subtracted from the *normalized* value, not added to
`a`/`b` before normalizing — Minkowski-padding an ellipse by a disk isn't
itself an ellipse (no simple closed form), so post-hoc subtraction (matching
`capsule_distance_casadi`'s own `sqrt(...) - r` pattern) is the right,
consistent choice.

Dummy padding (unused `MAX_ELLIPSES` slots) follows the same far-away-dummy
convention as `pad_obstacles`/`pad_walls`, with one adjustment: `a=b=1.0`,
not `0.0` — a zero-size ellipse would make `grad_mag`'s denominator `0/0`
at the dummy center.

## 7. What changed, file by file

- **`nmpc_interfaces`**: new `WallObstacle.msg` (`id, x0, y0, x1, y1, radius`)
  / `WallObstacleArray.msg`, a new `/map/walls` topic (latched, mirrors
  `/map/obstacles`), and a `walls` field on `GetScenario.srv`. New
  `Ellipse.msg` (`id, x, y, a, b, theta` — theta in radians, world frame) /
  `EllipseArray.msg`, `/map/ellipses`, and an `ellipses` field on
  `GetScenario.srv`. Circle messages (`Obstacle`/`ObstacleArray`) untouched
  throughout.
- **`nmpc/path_following.py`**: `pad_walls`/`pad_ellipses` (mirror
  `pad_obstacles`), `capsule_distance_casadi`, `ellipse_distance_casadi`,
  `softmin_casadi`.
- **`nmpc/params.py` / `sim_params.yaml`**: `MAX_WALLS`/`MAX_ELLIPSES`
  (default 5 each) and `SOFTMIN_K` (default 3.0) config fields, loaded the
  same way as `MAX_OBSTACLES`.
- **`nmpc/nmpc_acados.py`**: the parameter vector grows to
  `5 + 3*MAX_OBSTACLES + 5*MAX_WALLS + 5*MAX_ELLIPSES`; the old per-circle
  `h_list` loop is replaced by the unified soft-min row described above,
  now looping circles, then walls, then ellipses into the same `d_list`;
  `AcadosNMPC.solve()` takes `walls=`/`ellipses=` kwargs alongside
  `obstacles=`.
- **`map_node.py` / `nmpc_node.py`**: load/publish/cache `walls`/`ellipses`
  the same way `obstacles` already are.
- **`scenario_editor.py`**: a "Wall" mode (click sets one endpoint, click
  again sets the other — a wall has no natural drag distance, radius comes
  from a `wallr_box`) and an "Ellipse" mode (a single press-drag-release,
  like Obstacle mode — drag direction becomes `theta`, drag length becomes
  semi-major axis `a`, in one motion; semi-minor axis `b` is fixed, from an
  `ellipseb_box`). Rendering uses matplotlib's native `Ellipse` patch (unlike
  walls, which needed a custom polygon builder — matplotlib has no capsule
  primitive but does have an ellipse one), with `angle` converted via the
  same `plot_yaw = pi/2 - theta` axis-swap compensation `rviz_node.py`'s
  `_yaw_quat` uses. Saved under `"walls"`/`"ellipses"` keys in
  `scenario.json`, alongside the untouched `"obstacles"` key.
- **`rviz_node.py`**: wall markers (`LINE_STRIP`, `scale.x = 2*radius`) —
  RViz doesn't round the line's end caps the way the solver's exact capsule
  distance does; a documented visualization-only simplification, it doesn't
  affect the constraint math itself. Ellipse markers (`CYLINDER`, independent
  `scale.x=2a`/`scale.y=2b`, orientation via `_yaw_quat(theta)` reused
  exactly as its own docstring anticipates for any world-frame angle).
- **`tests/test_capsule_distance.py`**: headless (no ROS/acados) checks —
  the degenerate-circle case matches the old formula exactly, endpoint
  clamping, perpendicular mid-segment distance, the soft-min safety
  property and its convergence as `K` grows, and the all-dummy-slots
  underflow regression described in §4.
- **`tests/test_ellipse_distance.py`**: headless checks mirroring the above
  — degenerate-circle self-consistency, boundary first-order-exactness, the
  brute-force numerical conservatism sweep (§6, no closed form exists to
  check against analytically, unlike the capsule case), dummy-padding
  no-underflow, and the far-field-degradation characterization.
- **`tests/test_nmpc.py`**: `test7_real_obstacle_collision_regression`, an
  acados closed-loop check (its own fresh solver, per §4b) that reproduces
  the exact real collision scenario and asserts the obstacle is actually
  cleared, not just that the solver reports success.
  `test8_ellipse_obstacle_avoidance`, a new-feature acceptance test (own
  fresh solver, same rationale) checking a synthetic elliptical-obstacle
  scenario clears the padded boundary via brute-force parametric sampling,
  not just `success=True`.

[^1]: [Collision avoidance MPC in acados — dynamic obstacles and signed distance fields](https://discourse.acados.org/t/collision-avoidance-mpc-in-acados-dynamic-obstacles-and-signed-distance-fields/908), acados discourse forum.
[^2]: Jacquet, M., Harms, M., Alexis, K. **"Neural NMPC through Signed Distance Field Encoding for Collision Avoidance."** [arXiv:2511.21312](https://arxiv.org/abs/2511.21312), 2025.
[^3]: **"GPU-Accelerated Polygonal Signed Distance Functions for Real-Time Collision Avoidance."** [arXiv:2607.04310](https://arxiv.org/abs/2607.04310).
[^4]: Lutz, M., Meurer, T. **"Efficient Formulation of Collision Avoidance Constraints in Optimization Based Trajectory Planning and Control."** [arXiv:2104.12641](https://arxiv.org/abs/2104.12641), 2021.
