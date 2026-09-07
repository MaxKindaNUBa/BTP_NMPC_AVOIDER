# Mixing moving and static obstacles in the NMPC (theory only, no code changes)

## Context

This is the moving-obstacle companion to `GRID_AWARE_NMPC_OBSTACLE_AVOIDANCE.md` in
this directory, which designs the static side: a LIDAR-derived occupancy grid
turned into a distance field, `D̂(x,y)`, feeding a single per-stage constraint. That
design treats every obstacle as static across the horizon — fine for permanent
structure, wrong for anything actually moving (another vessel, a drifting object).
This document is a **theory-only** design note on what changes when some obstacles
move, researched broadly (general robotics/controls literature, not just marine)
and then specifically for the maritime/COLREGS case. No codebase changes are
described here.

## 1. The architecture doesn't change — static and moving stay separate

Every source found — Nav2's layered costmaps [1] (a dedicated static layer plus a
separate obstacle-tracking layer fed via a distinct message API, explicitly because
"costmap information does not contain any time information"), TEB local planner's
[2] separate `include_dynamic_obstacles` code path (constant-velocity motion
prediction for flagged dynamic obstacles), and Dynamic Occupancy Grid Maps' [3]
(DOGMa) own offline object-extraction step [4] (which clusters "dynamic" cells back
into discrete tracked objects before handing them to a planner) — lands on the same
split: static structure stays in the grid/distance-field exactly as designed in the
companion doc; moving obstacles are tracked as discrete objects (position +
velocity estimate) and handled as a **separate constraint block**, not merged into
the grid's math. No source unifies both into one representation for planning
purposes, even the ones (DOGMa) that unify them for *perception*.

## 2. Mechanism for the moving block

Standard treatment across UAV/mobile-robot/ASV NMPC literature (e.g. arXiv:2208.03529
[5], the GP-motion-forecast UAV-MPC work at ECC 2024 [6]): constant-velocity
(occasionally constant-acceleration) extrapolation per stage,

```
obstacle_pos(k) = obstacle_pos(0) + k * dt * obstacle_vel
```

feeding a **stage-varying** parameter rather than the one-shared-value-per-horizon
pattern used for static obstacles. acados supports this natively — `set(k, "p",
p_k)` called with a different value per stage `k`, documented concretely in Frey et
al.'s multi-phase-OCP paper [7] — with no codegen change, since
the parameter *count* per stage is still fixed (a `MAX_MOVING_OBS`-style slot cap,
same padding-with-dummies idea as the static side's `MAX_OBSTACLES`); only the
*values* now differ stage-to-stage instead of being identical. The constraint
itself stays **circle-shaped**, not grid-shaped:

```
h_{i,k}(x_k, y_k) = (x_k - ox_i(k))^2 + (y_k - oy_i(k))^2 - r_c_i(k)^2
```

soft-constrained the same way as the original static circles. This is deliberate,
not a gap: a discrete tracked object with a known/estimated velocity is exactly
what the original circle formulation fits — the grid/distance-field exists for
everything else (arbitrary-shaped, no velocity estimate, "just structure").
Practically: **the circle-based constraint math predating the grid switch doesn't
get thrown away — it comes back specifically for the moving-obstacle block**, just
with per-stage-varying `(ox_i(k), oy_i(k))` instead of constant values.

## 3. Growing margin with `k`, and chance-constrained MPC

Constant-velocity extrapolation error compounds with how far into the horizon you
look, so `r_c_i(k) = r_c_i(0) + growth(k)` is the standard mitigation — found
consistently across sources, though with no single universal formula (it's tuned
per application). The principled version is **chance-constrained MPC**: model the
obstacle's future position as Gaussian with covariance that grows stage over stage,
then convert the probabilistic constraint `P(collision at k) <= epsilon` into a
deterministic one by scaling the margin by the propagated covariance (a
Mahalanobis-distance/moment-matching step — see Li, Sun, Liao, Weiland,
arXiv:2304.01639 [8], combining this with a CBF; also arXiv:2403.06222 [9] on
learning the obstacle-uncertainty model this growth is derived from). This is the
moving-obstacle analogue
of the companion doc's `w_obs=0`-pending-tuning decision, except here it's the
**constraint margin itself** (safety-critical), not an optional potential term —
worth flagging as a decision this project will need to make (fixed margin vs. a
growth schedule vs. full chance-constraints) once real tracked-object data exists,
not something resolved in this theory pass.

## 4. Alternative/complementary constraint form: VO and CBF

Velocity Obstacles (VO) compute a cone of relative velocities that would lead to
collision and forbid the ship's chosen velocity from entering it — cheaper
(linear-in-velocity halfspace) than a nonlinear circle-per-stage constraint, and
several papers embed it directly as an NMPC constraint rather than treating it as a
competing planner (e.g. VO-derived CBFs, arXiv:2503.00606 [10], arXiv:2303.15871
[11]; VO folded directly into an MPC's constraint set for mobile robots [12]; a
velocity-space NMPC constraint for agile multi-agent avoidance, arXiv:2512.08574
[13]). One caveat found repeatedly: *reciprocal* VO (RVO/ORCA) assumes the other
agent also gives way, which doesn't hold for an uncooperative/unknown vessel — seen
concretely in multi-USV work that pairs RVO with reinforcement learning assuming
mutual cooperation [14] — plain (non-reciprocal) VO is the defensible default here,
not RVO.

## 5. The maritime literature validates a design choice already made

Li et al. 2023 [15] (*J. Mar. Sci. Eng.* 11(7):1408, DOI 10.3390/jmse11071408) —
also an MMG-based NMPC, structurally the closest match to this project found in
either search — folds COLREGS compliance directly into the **NMPC's cost function**
via an Improved Artificial Potential Field shaped by DCPA/TCPA (Distance/Time to
Closest Point of Approach, the standard maritime collision-risk metrics used
throughout this literature, e.g. [18][19], both inherently *relative-velocity*
quantities) and encounter-geometry classification (head-on/crossing/overtaking),
rather than a separate rule-based supervisory layer overriding the NMPC (a minority
pattern in the literature, e.g. arXiv:1907.00198 [16]).
This is structurally the same idea as the companion doc's `conf(x,y)`-weighted
`L_obs` potential term, just with DCPA/TCPA as the moving-obstacle version of the
confidence weight — the potential-term mechanism already designed for the static
case generalizes to moving obstacles by swapping what the weight is a function of.

## 6. DCPA/TCPA in the constraint vs. in the cost — precisely

It's tempting to read section 2 as "the constraint doesn't see velocity, the cost
does" — that's not quite right. The per-stage circle constraint *does* depend on
the obstacle's velocity — that's how `ox_i(k)` gets computed — but only indirectly:
velocity is used once, offline/upstream, to extrapolate a future position, and the
constraint the solver actually evaluates is pure position-vs-position, blind to the
ship's *own* velocity entirely. DCPA/TCPA is a different kind of object: a
closed-form function of *relative* position and *relative* velocity, evaluated live,
inside the optimization, at every stage:

```
v_rel = v_obstacle - v_ship        (v_ship is the NMPC's own decision variable here)
TCPA  = -(p_rel . v_rel) / |v_rel|^2
DCPA  = |p_rel + TCPA * v_rel|
```

Both formulas contain `v_rel` — including the ship's *own* predicted velocity at
that stage, which the position-only circle constraint never sees. That's why
DCPA/TCPA answers "if I speed up/slow down/turn now, does my eventual closest
approach improve?", a question the circle constraint structurally can't ask, and
why it's the natural fit for a cost term (steer gently) rather than a hard
constraint (wall off).

## 7. How give-way/stand-on is actually sensed and classified

Setting aside sensor noise/limited-range concerns entirely for this section —
assume own-ship and target-ship position/heading/velocity are all known exactly,
no noise, no delay, purely to isolate the classification math itself. Own-ship
state `(x_o,y_o,psi_o)` is already known exactly (it's the NMPC's own state, not
sensed). Target state `(x_t,y_t,psi_t,v_t)` is whatever the tracked-object pipeline
provides (out of scope here — see section 10). Two angles are computed from that:

```
beta_ot = atan2(y_t - y_o, x_t - x_o) - psi_o     (relative bearing: where the target is, from own bow)
psi_ot  = psi_t - psi_o                            (relative heading: closing head-on vs. crossing)
```

This exact `beta_ot`/`psi_ot` formulation is used directly in a fetchable, checked
source, the turning-circle CBF paper [17]. `beta_ot` is then binned into the
standard sectors used across this literature — the four-sector division of the
region around own-ship traces to Eriksen et al.'s branching-course MPC work [18],
and is summarized alongside CPA/TCPA-based classification more broadly in the
"Autonomous Collision Avoidance at Sea" survey [19]:

| `beta_ot` (relative bearing of target) | Situation |
|---|---|
| 345 to 15 (+-15 deg dead ahead) | Head-on |
| 15 to 112.5 | Crossing, target on own starboard -> own ship is give-way |
| 247.5 to 345 | Crossing, target on own port -> own ship is stand-on |
| 112.5 to 247.5 | Overtaking (abaft the beam) |

(the exact numeric boundaries above reflect the general convention reported across
this literature's secondary summaries, not a word-for-word quote verified against
Eriksen et al.'s primary text — [18] and [19] are the best available attribution
found, not a fully confirmed source for these specific numbers.)

Overtaking additionally needs relative *speed*, not just bearing, since the sector
alone doesn't say which vessel is closing on which: whichever vessel is coming up
on the other from within that abaft-the-beam sector is the overtaking (give-way)
one, regardless of which one started ahead. One implementation detail that matters
even under ideal sensing: **this classification is not a stateless function of the
instantaneous bearing** — COLREGS Rule 13 explicitly says a vessel deemed
overtaking stays the overtaking vessel "until she is finally past and clear," even
if later bearing drift would look like a crossing situation if evaluated fresh. So
the classification must be latched/tracked across solves, not recomputed from
scratch every solve — the same hysteresis principle as the track-chatter
mitigation in section 9, except here it's mandated by the rule text itself, not
just a numerical nicety.

## 8. What COLREGS actually mandates once classified (Rules 13-17)

Rule text and thresholds below confirmed directly against the official convention
and an IALA/eColregs commentary source [20], not just secondary summaries.

- **Rule 13 (Overtaking)** — deemed overtaking when approaching from more than
  22.5 deg abaft the other vessel's beam. The overtaking vessel must keep clear
  (give-way); no side is mandated — unlike head-on/crossing, Rule 13 doesn't say
  which way to pass, only that the overtaking vessel stays clear until past.
- **Rule 14 (Head-on)** — both vessels alter course to **starboard**, so each
  passes on the other's port side. The one situation with no give-way/stand-on
  split: both have the identical, mutually-consistent obligation.
- **Rule 15 (Crossing)** — whichever vessel has the other on **her own starboard
  side** gives way, and should avoid crossing ahead (i.e. pass astern) if
  circumstances allow. The other is stand-on.
- **Rule 16** — the give-way vessel takes early, substantial action.
- **Rule 17** — the stand-on vessel maintains course/speed, but may act alone if
  it becomes apparent the give-way vessel isn't acting appropriately, and if it
  does act, must **not turn to port for a vessel on her own port side** — an
  explicit rule against turning into the danger if forced to break "hold course."

**What this means for the overtaking cost term specifically, since Rule 13 assigns
no side:** unlike head-on (mandatory starboard bias) or crossing-give-way (biased
toward passing astern), the DCPA/TCPA-weighted potential term for an
*overtaking*-classified target should be **isotropic** — no directional bias, just
plain distance/time-based repulsion, whichever side is geometrically cheaper. And
because of Rule 13's persistence requirement above, that weighting must key off the
*latched* classification, not a bearing recomputed fresh every solve — otherwise a
trajectory sitting near the 112.5 deg boundary would flicker between an isotropic
overtaking potential and an asymmetric crossing one, the same chatter problem as
grid/track churn (section 9), just triggered by a discrete classification flip
instead of continuous sensor noise.

## 9. What happens if the obstacle ship doesn't follow COLREGS — goes rogue

Two different claims need separating here, same distinction as section 6:
- The **risk metric** (DCPA/TCPA itself) doesn't assume compliance at all — it's
  purely kinematic, computed from actual measured/tracked relative motion, and
  stays correct no matter what the other vessel does.
- The **role-based directional bias** (e.g. "I'm give-way here, so bias toward
  passing astern") *does* implicitly assume the other vessel plays its assigned
  role (a stand-on vessel holding course). If it doesn't — rogue, or another
  autonomous system reasoning differently, or it panics — both vessels can end up
  turning the same way and closing the gap the maneuver was meant to open. This is
  the identical failure mode already flagged for reciprocal Velocity Obstacles
  (RVO/ORCA) in section 4, just showing up in the COLREGS framing.

Three things protect against this, none of which requires trusting the other
vessel's compliance:
1. **The hard/soft distance constraint from section 2 doesn't go away.** The
   DCPA/TCPA cost term is a *preference* layered on top of it, not a replacement —
   it should never be the only thing standing between the ship and a collision,
   compliant target or not.
2. **Fast replanning, not correct prediction, is the actual safety net.** At RTI
   (`dt=0.1s`), a non-compliant target's updated tracked velocity feeds straight
   into the very next solve's `v_rel` — the system doesn't need to have predicted
   the rogue behavior, only to react within one control cycle. Same principle as
   the receding-horizon-safety point in the companion doc's partial-observability
   section, generalized to "don't trust the other ship's future intent, trust your
   own replanning speed."
3. **The growing-margin/uncertainty mechanism from section 3 is the natural place
   to encode distrust.** A tracked vessel suspected of being non-compliant or
   erratic just gets a faster-growing margin with `k`, rather than needing a
   different mechanism entirely.

(Caveat: neither research pass behind this document specifically searched for
"NMPC robustness against non-compliant give-way vessels" as its own topic — the
three protections above are reasoned from what the searches did surface (the
RVO-reciprocity caveat, the constraint/cost division of labor), not a cited
finding on this exact question.)

## 10. Pitfalls specific to moving obstacles

- **Track loss / ID switches cause constraint chatter** — a tracked object
  discontinuously appearing/disappearing/jumping reads as trajectory jitter,
  exactly like the companion doc's grid-churn concern. Same class of fix: track
  hysteresis, a minimum track age before feeding the NMPC, rate-limiting the
  position/velocity parameters between solves.
- **False "moving" classification from sensor/localization noise** — a stationary
  object misclassified as slow-moving inflates the moving-obstacle count and can
  trigger unnecessary maneuvers. Mitigation: velocity-magnitude thresholding with
  hysteresis before a track is classified "moving" at all, rather than reacting to
  any nonzero velocity estimate.

**Where the tracked-object data itself would come from** is out of scope for this
theory pass, same boundary already drawn for the grid in the companion doc ("assume
it's already present"): either a lightweight point-cloud-clustering-plus-Kalman-
tracker pipeline on points not explained by the static grid, or (heavier, a
later-stage option) a full Dynamic Occupancy Grid Map, whose per-cell particle
filter yields both an occupancy value (feeds the static path) and a velocity
estimate (feeds the moving path) — DOGMa is the "properly done later" version of
the same dual-representation split, not a different architecture.

## References

1. Nav2 (ROS2 Navigation) documentation: "Costmap 2D"
   (https://docs.nav2.org/configuration/packages/configuring-costmaps.html),
   "Static Layer Parameters"
   (https://docs.nav2.org/configuration/packages/costmap-plugins/static.html),
   "Obstacle Layer Parameters"
   (https://docs.nav2.org/configuration/packages/costmap-plugins/obstacle.html).
2. `teb_local_planner` documentation and source (https://index.ros.org/p/teb_local_planner/);
   Rösmann, C. et al., "Online Trajectory Optimization and Navigation in Dynamic
   Environments in ROS" (ResearchGate 326233414) — the `include_dynamic_obstacles`
   constant-velocity motion-prediction path referenced in section 1.
3. Nuss, D., Reuter, S., Thom, M., Yuan, T., Krehl, G., Maile, M., Gern, A.,
   Dietmayer, K. (2018). "A random finite set approach for dynamic occupancy grid
   maps with real-time application." *International Journal of Robotics
   Research*, 37(8), 841–866. arXiv:1605.02406.
4. "Offline Object Extraction from Dynamic Occupancy Grid Map Sequences."
   arXiv:1804.03933.
5. "Collision Avoidance for Dynamic Obstacles with Uncertain Predictions using
   Model Predictive Control." arXiv:2208.03529.
6. "Dynamic Obstacle Avoidance for UAVs using MPC and GP-Based Motion Forecast."
   European Control Conference (ECC) 2024.
7. Frey, J. et al. "Multi-Phase Optimal Control Problems for NMPC with acados."
   arXiv:2408.07382.
8. Li, R., Sun, S., Liao, F., Weiland, S. "Moving Obstacle Collision Avoidance via
   Chance-Constrained MPC with CBF." arXiv:2304.01639.
9. "Robust Predictive Motion Planning by Learning Obstacle Uncertainty."
   arXiv:2403.06222.
10. "Dynamic Collision Avoidance Using Velocity-Obstacle-Based Control Barrier
    Functions." arXiv:2503.00606.
11. "Control Barrier Functions in Dynamic UAVs for Kinematic Obstacle Avoidance: A
    Collision Cone Approach." arXiv:2303.15871.
12. "MPC Based Motion Planning For Mobile Robots Using the Velocity Obstacle
    Paradigm." ResearchGate 372839371.
13. Kratky, V., Penicka, R., Gupta, A., Prochazka, O., Saska, M. "RVC-NMPC."
    arXiv:2512.08574.
14. "Proximal Policy Optimization with Reciprocal Velocity Obstacle-based
    collision avoidance for multi-USV." ScienceDirect S002980182300389X.
15. Li et al. (2023). "A COLREGs-Compliant Ship Collision Avoidance
    Decision-Making Support Scheme Based on Improved APF and NMPC." *Journal of
    Marine Science and Engineering*, 11(7), 1408. DOI 10.3390/jmse11071408. (Full
    text not directly accessible for this document — MDPI blocked automated
    fetches on every attempt; citation is from the paper's indexed abstract and
    secondary summaries, not a verified full-text read.)
16. "Hybrid Collision Avoidance for ASVs Compliant with COLREGs Rules 8 and
    13-17." arXiv:1907.00198.
17. "Efficient COLREGs-Compliant Collision Avoidance using Turning Circle-based
    Control Barrier Function." arXiv:2504.19247. Source of the `beta_ot`/`psi_ot`
    bearing/heading formulation and the own-ship/target-ship symmetric
    classification requirement in section 7.
18. Eriksen, B.-O.H., Breivik, M., Wilthil, E.F., Flåten, A.L., Brekke, E.F.
    (2019). "The branching-course model predictive control algorithm for
    maritime collision avoidance." *Journal of Field Robotics*.
    https://doi.org/10.1002/rob.21900. Origin of the four-sector relative-bearing
    encounter classification referenced in section 7 (exact numeric boundaries
    per secondary summaries, not confirmed from this paper's own full text).
19. "Autonomous Collision Avoidance at Sea: A Survey." *Frontiers in Robotics and
    AI*, 2021. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8481591/. General
    survey covering CPA/TCPA-based encounter classification and COLREGS-compliant
    collision avoidance approaches.
20. Official COLREGS rule text and commentary: IALA/eColregs, "COLREG Rule 13 —
    Overtaking" (https://ialacolreg.com/en/colreg/rule-13); "COLREGs course - Rule
    17 (Action by stand-on vessel)"
    (https://ecolregs.com/index.php?Itemid=390&id=57&lang=en&layout=item&option=com_k2&view=item);
    IMO, "Convention on the International Regulations for Preventing Collisions
    at Sea, 1972 (COLREGs)" (https://www.imo.org/en/about/conventions/pages/colreg.aspx).
