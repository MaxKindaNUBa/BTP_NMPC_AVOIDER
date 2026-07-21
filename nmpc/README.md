# nmpc

The actual NMPC controller: a path-following + (eventually) obstacle-avoidance
Nonlinear Model Predictive Controller built on top of the MMG vessel dynamics
(`casadi_mmg_solver/`), adapted from the two papers in `research_papers/`.
Two solver backends share one formulation — CasADi/IPOPT for
debugging, Acados/SQP-RTI for anything close to real time.

## Package layout

| File | Role |
|---|---|
| `config.py` | Single source of truth for every tunable parameter (horizon, actuator bounds, cost weights, obstacle/slack settings, solver options). Nothing is hardcoded anywhere else. |
| `path_following.py` | Pure NumPy guidance geometry: cross-track error, course angle/error, waypoint switching, the distance-scaled speed-reference ramp, obstacle padding. Also holds `wrap180_casadi`, the one CasADi-symbolic helper this file needs for the solvers' cost functions. |
| `state_augmentation.py` | The CasADi symbolic augmented-state ODE (`augmented_dynamics_casadi`) wrapping `casadi_mmg_solver`'s MMG dynamics with the new guidance/actuator-rate rows, plus an RK4 step and a validation harness against the raw MMG model. |
| `nmpc_casadi.py` | **Step A** — `CasadiNMPC`, a full multiple-shooting NLP solved with IPOPT. Slow, but every intermediate value is inspectable — this is where the formulation itself gets debugged. |
| `nmpc_acados.py` | **Step B** — `AcadosNMPC`, the same formulation compiled to a real-time SQP-RTI OCP via acados. This is the one meant to actually run in a control loop. |
| `test_nmpc_open_loop.py` | Closed-loop validation harness (plant + solver, no obstacles) — 4 scenarios, results plotted to `results/`. |
| `run_live_open_loop.py` | Same rollouts as above, but streamed live into `mpc_visualization/`'s dashboard instead of saved as static plots. |
| `compare_qu.py` | Ad-hoc diagnostic script comparing `Q[u]=0` vs the tuned weight on an open-loop run. |
| `results/` | Output plots from `test_nmpc_open_loop.py`. |

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

Both solvers share:
- **Cost**: quadratic tracking (`Q`) + control-rate penalty (`R`) + terminal
  cost (`Qe = QE_SCALE * Q`), summed over the horizon.
- **Dynamics constraint**: RK4-discretized `augmented_dynamics_casadi`,
  enforced via multiple shooting (CasADi) / the OCP's own integrator (Acados).
- **Obstacle avoidance**: soft, slack-relaxed quadratic distance constraints,
  one per fixed obstacle slot (`config.MAX_OBSTACLES`, unused slots padded
  with a harmless far-away dummy via `pad_obstacles`) — never hard, since a
  hard constraint would make the NLP infeasible the moment an obstacle gets
  close.
- **Actuator bounds**: rudder angle/rate and propeller speed/rate, applied as
  state/control bounds.

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
   chi_p)` residual itself in both solvers' cost (`wrap180_casadi`); Acados
   had to switch from `LINEAR_LS` to `NONLINEAR_LS` cost type to express this,
   since a wrapped residual isn't affine in the state.
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
6. **No braking near a target.** `Q[x]`/`Q[y]` (position error, meters²)
   dwarfs `Q[u]` (speed error, (m/s)²) at any real distance, so a constant
   speed reference never gets "discovered" as needing to shrink — the ship
   kept cruising at full reference speed until nearly on top of the target,
   then couldn't decelerate fast enough and overshot. Fixed by explicitly
   shrinking the *speed reference itself* with distance
   (`compute_effective_u_ref`, a linear ramp inside `config.BRAKE_DISTANCE`),
   rather than relying on the cost weights to find braking on their own.

## Known open issue

Under a low-speed pivot maneuver (e.g. correcting heavily near a target with
the braking ramp active), `u` and `v` can both approach zero simultaneously.
`casadi_mmg_solver`'s `u_val = ca.fmax(u, 1e-5)` floor only guards the
propeller/rudder terms — it does **not** prevent the resultant speed
`U = sqrt(u_val^2 + v^2)` itself from collapsing, and `r_ndm = r*Lpp/U` (used
throughout the hull force terms) blows up for any nonzero yaw rate as `U→0`.
This has been observed to make Acados' SQP-RTI QP solver fail outright during
a live run, and because SQP-RTI's warm-started iterate persists across calls,
a single poisoned solve can freeze the controller permanently rather than
recovering on the next step. A hard lower bound on `u` (e.g. via `idxbx`/
`lbx` in this project's own OCP, not touching the read-only MMG model) has
been identified as the fix but is not yet implemented — `U_REF_MIN=0.05` in
`config.py` is deliberately kept just above zero specifically to steer clear
of this, but nothing currently makes it a hard constraint the optimizer must
respect.

## Running things

```bash
# Self-tests (no solver needed)
python nmpc/path_following.py
python nmpc/state_augmentation.py

# Static-plot open-loop validation (4 scenarios, results/*.png)
python -m nmpc.test_nmpc_open_loop

# Live-visualized open-loop run
python nmpc/run_live_open_loop.py --solver acados --test 3
```

## Dependencies

`numpy`, `casadi`, `acados_template` (for `nmpc_acados.py` and
`run_live_open_loop.py`'s default solver), `matplotlib` (test/demo plotting).
