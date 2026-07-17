# research_papers

The two papers that directly shaped the NMPC formulation in `nmpc/`. Both
study the **VTec S-III** autonomous surface vehicle (ASV) at Tecnologico de
Monterrey; this project adapts their path-following + obstacle-avoidance
NMPC structure to a different vessel (an MMG-modeled, rudder-and-propeller
actuated hull, instead of the VTec S-III's twin-thruster differential
actuation).

The PDFs themselves are **not** committed to this repository — they're
copyrighted Elsevier journal articles, and redistributing the full text in
a git repo (especially one that might end up on a public GitHub remote)
isn't something to do casually. Both are easy to find via their DOI at a
library/institutional proxy or on ScienceDirect.

## 1. Path-following and obstacle avoidance (the primary source)

> Gonzalez-Garcia, A., Collado-Gonzalez, I., Cuan-Urquizo, R., Sotelo, C.,
> Sotelo, D., Castañeda, H. (2022). **"Path-following and LiDAR-based
> obstacle avoidance via NMPC for an autonomous surface vehicle."**
> *Ocean Engineering*, 266, 112900.
> https://doi.org/10.1016/j.oceaneng.2022.112900

This is the main structural source for `nmpc/`:

- The **augmented state vector** idea — folding path-following error terms
  (cross-track error, sine/cosine of course-angle error) directly into the
  NMPC's own state, so a single quadratic cost tracks both the guidance
  objective and the vehicle dynamics simultaneously, instead of running a
  separate outer guidance loop.
- The **cross-track error formula**, `e_y = -(x-x_d)sin(chi_p) + (y-y_d)cos(chi_p)`,
  measuring signed perpendicular distance to the line through the active
  waypoint at path angle `chi_p`.
- Representing course-angle error as `(sin(psi_e), cos(psi_e))` rather than
  a raw angle, specifically so the cost function never has to differentiate
  through a +-180 degree wraparound discontinuity.
- The **soft obstacle-avoidance constraint** structure: a quadratic distance
  constraint per obstacle, relaxed by a slack variable penalized heavily in
  the cost, rather than a hard constraint that would make the NLP infeasible
  whenever an obstacle briefly gets close.
- Real-time solution via **acados** (SQP-RTI), validated in both simulation
  and field experiments against physical buoys — the same solver stack used
  here for `nmpc/nmpc_acados.py`.

This project's `nmpc/config.py` and `nmpc/nmpc_acados.py` cite this paper by
its `main.pdf` filename in several comments (e.g. `Q[psi]=30` in the weight
matrix, and the note on why `psi` is wrapped in the cost residual).

## 2. Adaptive sliding-mode control with NMPC-based obstacle avoidance

> Collado-Gonzalez, I., Gonzalez-Garcia, A., Cuan-Urquizo, R., Sotelo, C.,
> Sotelo, D., Castañeda, H. (2024). **"Adaptive sliding mode control with
> nonlinear MPC-based obstacle avoidance using LiDAR for an autonomous
> surface vehicle under disturbances."** *Ocean Engineering*, 311, 118998.
> https://doi.org/10.1016/j.oceaneng.2024.118998

A follow-up from an overlapping author group, on the same VTec S-III
platform, using NMPC purely as a **guidance** layer (outputting a desired
heading rate) paired with an adaptive sliding-mode inner controller, to
explicitly handle environmental disturbances. This project doesn't adopt
its sliding-mode/disturbance-adaptive architecture (the MMG dynamics model
already fully replaces the guidance-only kinematic model this paper uses),
but its cross-track-error and obstacle-constraint formulas (its Eq. 51
and Eq. 54-55) independently corroborate the ones taken from paper 1, and
its explicit statement that the path angle is **piecewise-constant per leg**
(only recomputed at waypoint transitions, not continuously along a
parametric curve) directly informed how `nmpc/path_following.py`'s
`compute_path_angle()` / waypoint-switching logic is structured.

## How these map onto this codebase

| Paper concept | This repo |
|---|---|
| Augmented state `[chi, sin(chi), cos(chi), y_e, x, y, psi, u, v, thrusters...]` | `nmpc/config.py` `STATE_DIM=11`, `xi = [e_y, sin(psi_e), cos(psi_e), r, x, y, psi, u, v, delta, n]` |
| Twin-thruster control `[T_port, T_stbd]` | Rudder + propeller rate `u_aug = [delta_dot, n_dot]` (MMG actuation, not the papers' thrusters) |
| Cost weight matrix `Q`/`R`/terminal `Qe` | `nmpc/config.py` `Q_DIAG`/`R_DIAG`/`QE_SCALE`, started from the paper's published values and re-tuned for this vessel/formulation (see `nmpc/README.md`) |
| Soft slack-relaxed obstacle constraint | `nmpc/nmpc_casadi.py` / `nmpc/nmpc_acados.py` obstacle terms, `config.SIGMA`/`W_SLACK` |
| acados SQP-RTI real-time solve | `nmpc/nmpc_acados.py` |
