# OPEN ISSUE: propeller never actually reduces RPS near a sharp-turn waypoint

Not yet fixed. Documented here (rather than folded into `README.md`'s
known-issues log) because it's still open, per this project's convention
that the README's numbered list only records fixes that have actually been
verified — see items 8/8b/9 there for the (fixed, verified) work this issue
was found while spot-checking.

## Context

Found while checking whether the vessel slows down approaching the sharp
turn in `scenario.json` (wp0→wp1→wp2, ~117°), *after* README items 8
(multi-segment horizon preview), 8b (heading/position decoupling fix), and
9 (waypoint-passage weight boost) were already verified fixed for
heading/position tracking through that same turn. Two distinct sub-bugs,
confirmed via a fine-grained (every-step, not every-10s) closed-loop
rollout and by inspecting `build_horizon_references`'s `u_ref_arr` /
`dist_to_corner_arr` directly at each step from a live Python session (the
one-off diagnostic scripts used weren't kept — re-derive similarly if
picked up again, e.g. call `build_horizon_references` inline inside a
`run_scenario`-style loop and print `u_ref_arr[0]` alongside `n`/`u`).

## (A) `n` stays pinned at `RPS_MAX=18.2` throughout the entire approach, crossing, and departure, even though `u_ref` is correctly computed low

At `t=88.0-88.9s` in a `dt=0.1s` rollout (distance to wp1 shrinking from
2.54m to 2.04m, well inside both `BRAKE_DISTANCE=8.0m` and
`WAYPOINT_PASSAGE_DIST=4.0m`), `u_ref_arr[0]` ramps down exactly as
designed (0.256 → 0.216 m/s), yet the actual commanded `n` never leaves
18.20 rps the entire time, and true surge speed `u` only creeps down slowly
(0.569 → 0.544 m/s) — nowhere near tracking the much lower `u_ref`.

So the braking-ramp *reference* is provably correct and is reaching the
solver (confirmed by inspecting `u_ref_arr` directly); what's broken (or at
least unconfirmed as intentional) is that the QP never actually reduces
thrust to chase it.

**Working hypothesis, not yet verified:** during a hard turn, rudder
authority (`N_rudder` in `casadi_mmg_solver/casadi_mmg.py`) scales with the
propeller-driven inflow speed, so the solver may be implicitly "choosing"
to keep thrust high to preserve turning authority (heading/position
tracking — `Q[psi]=30`, and the boosted `Q[x]/Q[y]=20` near the corner from
item 9) at the cost of a large, ignored `u_ref` mismatch (`Q[u]=100` on a
~0.33 m/s error is cost `~11`, evidently cheap next to whatever
heading/position cost a slower turn would incur).

**Before concluding this is "intentional" rather than a genuine bug**
(e.g. a sign error or missing coupling somewhere in the cost/dynamics),
verify by temporarily zeroing `Q[psi]`/`Q[x]`/`Q[y]` for a diagnostic-only
run and checking whether `n` finally drops in response to the low `u_ref`.

## (B) `u_ref` snaps back to full cruise (0.70) the instant wp1 is crossed

Same pattern as the original pre-session bug (README item 6), still
present after items 8/8b/9. At `t=89.00s` — the step `target_idx` advances
to 2 and `SegmentQueue.pop_crossed` pops wp1 — `u_ref_arr[0]` jumps from
~0.21 straight to 0.700, and `dist_to_corner_arr[0]` jumps from ~2.0m to
~39.0m in the same step. This happens because the queue now only contains
the final waypoint (wp2), ~39m away, so there is no "nearby corner" left
for the braking ramp (`_brake_ramp` in `path_following.py`) to reference.

This reintroduces exactly the reactive/no-anticipation braking pattern
items 8/9 were built to fix — just for the leg *after* a crossing instead
of before it.

**Likely fix direction:** `_brake_ramp`'s "remaining distance to nearest
corner" should also consider braking for the corner just passed, for a few
meters past it (some hysteresis/tail), not only the next one ahead — or
more fundamentally, (A) may need solving first, since even a correctly
non-reset `u_ref` here wouldn't matter if the solver ends up ignoring it
anyway (per (A) above).

## Status

Not fixed. Flagged for follow-up rather than chased further in the session
that found it, per the project's usual pace of verifying one thing at a
time.
