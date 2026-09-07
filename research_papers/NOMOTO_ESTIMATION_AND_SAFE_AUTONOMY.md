# Better target-motion prediction and an independent own-ship failsafe (theory only, no code changes)

## Context

The moving-obstacle companion doc (`COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md`)
extrapolates a tracked target ship's future position with plain constant-velocity
extrapolation, and lists three "protections" against a non-compliant/rogue target
(the hard/soft distance constraint, fast RTI replanning, and a growing safety
margin). This document speculates on two follow-on improvements raised in
discussion: (1) replacing constant-velocity extrapolation with a model-based
estimator built on a Nomoto-style first-order steering response, and (2) an
own-ship-side failsafe that is architecturally independent of trusting the target
vessel's compliance *or* the primary NMPC's own correctness — addressing a fair
objection that the three earlier "protections" all live inside the same NMPC loop
and share its failure modes. Both are speculative extensions, more exploratory than
the other two docs in this directory; grounded where a source was found, flagged
as reasoning where not. No codebase changes are described here.

## Part A — Nomoto-informed target motion prediction

### 1. Why constant-velocity/constant-turn are inadequate for a maneuvering ship

Constant-velocity assumes zero yaw rate forever (the target goes dead straight).
Constant-turn-rate assumes whatever yaw rate it currently has, forever (a perfect
circle). Neither matches how a real ship's yaw rate actually evolves while
steering: it *relaxes* toward a commanded value over a time constant, which is
exactly Nomoto's classical first-order ship-steering model,

```
r_dot = (K * delta - r) / T
```

(`r` = yaw rate, `delta` = rudder angle, `K`/`T` = vessel-specific maneuvering
gain/time-constant). Real-time online identification of exactly this kind of model
fused with a nonlinear Kalman filter, for ship trajectory prediction, is an
established approach [1][2] — so the shape of this idea (not the specific
application to a passively-tracked *target*, see below) is grounded in existing
work, not invented here.

### 2. The identifiability problem, and how to route around it

The target's actual rudder angle `delta` and its true hull-specific `K`/`T` are
not observable from a passive tracker. Genuine system identification of "the real
Nomoto model of that ship" isn't possible. What's proposed instead: don't identify
the real model, use the model's *form* as a smoothing prior. Fold `K*delta` into a
single hidden, estimated state — `r_ss`, "the yaw rate this ship currently appears
to be settling toward" — and either fix `T` at a generic value for the target's
apparent size class, or leave it as a modeling approximation rather than something
genuinely fitted. This is a speculative simplification, not confirmed against a
paper that does exactly this for a passively-observed target specifically (the
identification literature found [1] fits this model to *one's own* vessel, where
`delta` is known).

### 3. Reusing this project's own estimation pattern

This project's UKF already estimates a slowly-varying, unmeasured driving quantity
(ocean current) as an augmented hidden state (see `ukf_node`/`ukf_core.py` and this
repo's UKF design-decision memory) — a "hidden state + relaxation/process-noise
model" structure. The speculative target-motion filter is the same kind of
estimation problem, aimed at a different unknown:

```
state = [x_t, y_t, psi_t, u_t, r_t, r_ss]
x_dot   = u * cos(psi),   y_dot = u * sin(psi),   psi_dot = r
r_dot   = (r_ss - r) / T        (Nomoto relaxation, T fixed/assumed)
r_ss_dot = process noise         (hidden, slowly-varying, estimated online)
u_dot    = process noise
```

fed from noisy tracked `(x,y)` (and heading, if the tracker provides it) through an
EKF/UKF, mirroring the existing current-estimator's design rather than introducing
a new estimation paradigm to the project.

### 4. Where this plugs into the existing plumbing

The rollout `x_target(k)` for `k = 0..N` replaces the constant-velocity
extrapolation in the companion doc's section 2, feeding the *same* stage-varying
acados parameter mechanism [3] already designed there — only the function
computing `ox_i(k), oy_i(k)` changes, from linear extrapolation to integrating the
Nomoto-relaxation model forward.

### 5. A side benefit: principled uncertainty growth, for free

An EKF/UKF propagates a covariance alongside the state estimate. That covariance is
exactly the ingredient the companion doc's growing-margin / chance-constrained
mechanism needs (see its section 3, and Li, Sun, Liao, Weiland [4]) — instead of a
hand-tuned `growth(k)` schedule, the margin could scale off the filter's own
propagated covariance. Adopting a real estimator for the target yields a better
central prediction *and* a principled uncertainty estimate as two outputs of the
same filter, rather than two separately-tuned problems.

### 6. The rigorous upgrade: Interacting Multiple Model (IMM) filtering

Since it isn't known in advance whether a target is holding straight, turning
steadily, or actively maneuvering, the standard rigorous solution in maneuvering-
target tracking is an IMM filter: run several models in parallel (constant-velocity,
constant-turn, Nomoto-relaxation), each producing a likelihood against the observed
track, and probabilistically blend/switch between them. A single fixed
Nomoto-relaxation filter is a reasonable starting point; IMM is the "properly done
later" version, in the same relationship as DOGMa was to a simple tracker in the
companion doc.

### 7. What this does not solve

No model — Nomoto-based or IMM — protects against a target starting a genuinely new
maneuver at the exact instant of prediction, before any trend is visible in its
track history. A better predictor narrows how *often* an emergency response is
needed; it doesn't remove the need for one. That's the motivation for Part B.

## Part B — An own-ship failsafe independent of trusting the target or the NMPC

### 1. Why the companion doc's three "protections" aren't a real failsafe

The hard/soft distance constraint, fast RTI replanning, and the growing margin
(`COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md`, section 9) all live *inside* the same
NMPC solve. They only help if the NMPC keeps solving correctly, on time, with good
inputs. None of them protect against the NMPC's own model being wrong (bad track,
misclassified encounter, a bug), a poor step from RTI's single-SQP-iteration
scheme at a bad linearization point, or a stale/glitched perception input reaching
the solver undetected. A genuine failsafe needs to be architecturally independent
of the primary controller being correct.

### 2. The established pattern: Simplex Architecture / Runtime Assurance

Running a high-performance primary controller alongside a much simpler,
independently-verified backup, with a decision layer between them, is a
well-established safety-engineering pattern — the Simplex Architecture and its
Runtime Assurance descendants [5][6][7][8]. This is not something invented for
this project; it's the standard answer to "what if the sophisticated controller is
wrong."

### 3. A concrete instantiation, applied to this project

A specific, modern two-layer design — a primary MPC-based planner cascaded into a
faster CBF-QP safety filter [9] — maps directly onto this project:

- **Layer 1 (already designed): the NMPC.** Optimizes path-following plus
  COLREGS-aware avoidance using the grid/DCPA-TCPA/margin machinery from the other
  two docs. Allowed to be wrong sometimes (bad classification, solver hiccup) *as
  long as Layer 2 exists.*
- **Layer 2 (the actual failsafe, new): a fast CBF-QP filter between the NMPC's
  output and the actuators.** Every cycle, take the NMPC's proposed
  `(delta_nmpc, n_nmpc)` and solve a small QP: find the control *closest* to what
  the NMPC wants, subject to `h_dot(x,u) >= -alpha(h(x))` for every currently-
  tracked obstacle, using only the *current measured state* — no trust in the
  NMPC's internal model, no trust in the target's compliance, no dependency on the
  grid or COLREGS logic being correct. Per [9], this needs **no separate
  escalation/trigger logic**: it runs unconditionally every cycle and is a no-op
  whenever the NMPC's own output is already safe, only clipping it when it isn't.
  This is a materially different use of a CBF from the one in the companion doc's
  section 4 (CBF *inside* the NMPC's own constraint set, still subject to whatever
  else that optimization does) — here it's a wholly separate computation that
  doesn't depend on what's inside the NMPC at all.
- **Layer 0 (last resort, matching Simplex's "verified baseline controller"):** if
  the CBF-QP is itself infeasible (actuator limits mean *no* control satisfies the
  safety condition — genuinely unavoidable geometry), fall back to a fixed,
  maximally simple maneuver (full reverse + max rudder away from the nearest
  threat) — the automotive-AEB-style last-resort reflex.

### 4. A second, distinct trigger: the ship's own pipeline health

Beyond external obstacle geometry, a proper failsafe should also watch its *own*
health — acados's solver-status flag, the UKF's innovation/covariance blowing up,
stale tracking timestamps — and treat "my own state estimate might be garbage" as
its own trigger into Layer 0, not just "something got too close." This is a
distinct half of "own-ship failsafe," separate from anything about the target's
behavior.

### 5. Why this actually answers the objection

This layer doesn't depend on the target being COLREGS-compliant, and it doesn't
depend on the primary NMPC being correct either — it's the one layer in this whole
design that is safe by construction, independently of the grid, the tracker, the
encounter classification, and the NMPC's own solve quality.

## References

1. "Real-time parameter identification of ship maneuvering response model based
   on nonlinear Gaussian Filter." ScienceDirect S0029801821017522.
2. "The Ship Movement Trajectory Prediction Algorithm Using Navigational Data
   Fusion." PMC5492365.
3. Frey, J. et al. "Multi-Phase Optimal Control Problems for NMPC with acados."
   arXiv:2408.07382. (Same citation as in the companion docs — the stage-varying
   parameter mechanism this rollout would reuse.)
4. Li, R., Sun, S., Liao, F., Weiland, S. "Moving Obstacle Collision Avoidance via
   Chance-Constrained MPC with CBF." arXiv:2304.01639.
5. "Mission-Level Runtime Assurance Framework for Autonomous Driving."
   arXiv:2606.06996.
6. "Perception Simplex: Verifiable Collision Avoidance in Autonomous Vehicles
   Amidst Obstacle Detection Faults." arXiv:2209.01710 (also published in
   *Software Testing, Verification and Reliability*, Bansal et al. 2024).
7. "The Use of the Simplex Architecture to Enhance Safety in Deep-Learning-Powered
   Autonomous Systems." arXiv:2509.21014.
8. "The Black-Box Simplex Architecture for Runtime Assurance of Autonomous CPS."
   arXiv:2102.12981.
9. "Layered Safety: Enhancing Autonomous Collision Avoidance via Multistage CBF
   Safety Filters." arXiv:2603.00338. Source of the concrete two-layer
   (predictive-MPC + real-time CBF-QP) architecture in Part B, section 3.
