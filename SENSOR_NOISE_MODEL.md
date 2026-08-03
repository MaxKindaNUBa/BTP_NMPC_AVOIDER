# Sensor Noise Model — Specification

## 0. Scope and status

This document is a **design specification only**. Nothing in the simulator currently
injects sensor noise — `casadi_mmg.py` integrates the true, noise-free MMG state, and
`run_live.py` feeds that true state straight back into the NMPC solver every step. No
code in this repository implements what follows yet.

The purpose of this file is to pin down, precisely and numerically, **what each sensor
noise process is and how it is generated**, so that whoever implements it (human or AI
assistant) does not have to re-derive the statistics from scratch. It intentionally does
not describe *where in the codebase* this should be wired in — that question was already
answered by the flowchart discussed earlier (a `sensor_model` step that sits between the
true plant state and the value fed back to the NMPC, so the controller only ever sees
corrupted measurements and must reject the corruption via feedback, never via internal
model knowledge) and by `ROS2_CONVERSION_PLAN.md`'s forward-compatibility notes.

## 1. What signals get sensed

The plant's true state is the 6-dim MMG vector `[u, v, r, x, y, psi]` (`mmg_state[0..5]`
in `run_live.py`), plus the two actuator states `delta` (rudder angle) and `n`
(propeller speed) tracked alongside it. Every one of these is a candidate for a
noise-corrupted measurement:

| Signal          | Symbol  | True source        | Real-world sensor analogue                  | Units |
|-----------------|---------|---------------------|----------------------------------------------|-------|
| Position        | x, y    | `mmg_state[3], [4]` | GNSS/GPS receiver                             | m     |
| Heading          | psi     | `mmg_state[5]`      | Magnetic compass / AHRS yaw                   | rad   |
| Yaw rate         | r       | `mmg_state[2]`      | MEMS/fiber-optic gyroscope                    | rad/s |
| Surge velocity   | u       | `mmg_state[0]`      | Speed log (EM/Doppler) or GPS-derived         | m/s   |
| Sway velocity    | v       | `mmg_state[1]`      | Speed log (rarely direct) or GPS-derived      | m/s   |
| Rudder angle     | delta   | actuator state      | Rudder angle encoder                          | rad   |
| Propeller speed  | n       | actuator state       | Shaft tachometer / encoder                    | rps   |

`psi` is an angle and must always be noise-perturbed *then* wrapped to `[-pi, pi]`
(or equivalently perturbed directly on the `sin(psi_e)`/`cos(psi_e)` pair used inside
the augmented state, so wrap-around is never explicitly needed downstream).

## 2. Building blocks (generic noise primitives)

Every per-signal model in Section 3 is assembled out of these primitives. Each is
described by exactly how the random signal is generated at each simulation step `k`
(step size `dt`, matching `config.dt`).

### 2.a White Gaussian noise (the measurement noise floor)

```
y_k = x_k + w_k,        w_k ~ N(0, sigma_w^2),  i.i.d. across k
```

Independent at every step — no memory, no correlation with the previous sample. This
is the baseline "how precise is the sensor right now" term that every signal has.

### 2.b Gauss-Markov / Ornstein-Uhlenbeck colored drift

Used for anything that should wander slowly rather than jitter step-to-step (GPS
multipath/ionospheric bias, gyro bias instability, compass hard-iron drift). Generated
by a first-order autoregressive recursion:

```
a = exp(-dt / tau)
q = sigma_b * sqrt(1 - a^2)
b_k = a * b_{k-1} + q * eps_k,     eps_k ~ N(0, 1)
b_0 ~ N(0, sigma_b^2)              (steady-state initial condition)
```

- `tau` = correlation time constant (seconds): how long the drift persists before
  "forgetting" its current value. Large `tau` → slow wandering bias; small `tau` →
  it behaves almost like white noise.
- `sigma_b` = steady-state standard deviation of the drift itself (the process is
  constructed so `Var(b_k) → sigma_b^2` regardless of `tau`).
- This is exactly a discretized OU process; `a` and `q` above are the exact (not
  Euler-approximated) discretization, so it is correct for any `dt`.

### 2.c Combined sensor model (white + bias)

```
y_k = x_k + b_k + w_k
```

`b_k` evolves per 2.b (once per signal, persists across steps), `w_k` is redrawn per
2.a every step. This is the default full model for GPS position and gyro/compass axes.

### 2.d Constant or random-walk actuator offset

For encoders (rudder angle, propeller speed) a full OU drift is overkill; a much
smaller, either fixed-for-the-whole-run or very-slow-random-walk offset suffices:

```
b_k = b_{k-1} + sigma_rw * eps_k     (random-walk, sigma_rw tiny)
```

or simply `b_k = b_0` (drawn once, held constant) if the encoder offset is effectively
static over a simulation run.

### 2.e Quantization

Encoders and low-resolution sensors report discretized values:

```
y_k = round(x_k / LSB) * LSB
```

`LSB` = the sensor's least-significant-bit resolution. Applied *after* 2.a–2.d, i.e.
quantization is the last step in the chain, not a substitute for the other noise terms.

### 2.f Outliers / heavy-tailed noise (dropouts, multipath spikes)

Modeled as a contaminated-Gaussian (Bernoulli-Gaussian) mixture rather than swapping the
whole distribution to something heavy-tailed, because outliers should be *rare*, not
the typical case:

```
w_k = { N(0, sigma_w^2)          with probability (1 - p)
      { N(0, sigma_outlier^2)    with probability p,   sigma_outlier >> sigma_w
```

`p` is small (e.g. 0.01–0.05). This is a general capability of the position noise
model (`pos_outlier_p`/`pos_outlier_sigma`); the only implemented preset (light, Section
4) sets `p = 0`, disabling it.

### 2.g Update-rate / zero-order hold (ZOH) — deferred, not used yet

Real sensors generally sample slower than the NMPC solver frequency, and that
mismatch would normally be handled with a zero-order hold (`y_k = y_{k - (k mod N)}`,
holding the last value between sensor updates, with the OU recursions in 2.b advancing
only once per sensor step rather than once per solver step). **This is explicitly out
of scope for now** — see Section 3's note that every sensor currently runs at the
solver/model frequency. This subsection is kept only as a placeholder for when
per-sensor update rates are reintroduced later.

### 2.h Differentiation noise amplification

Only relevant if a rate-like quantity (e.g. velocity) is derived by finite-differencing
a noisy position rather than measured directly:

```
v_hat_k = (y_k - y_{k-1}) / dt
```

If `y_k = x_k + w_k` with `w_k` i.i.d. `N(0, sigma_pos^2)`, then since the two noise
draws are independent:

```
sigma_vel = sigma_pos * sqrt(2) / dt
```

This is why differentiated velocity is dramatically noisier than a directly-sensed
one, especially at small `dt` — the division by `dt` amplifies the (already doubled)
position noise variance. `sensor_model` supports this as a general "derived" velocity
mode (Section 3.4), but the only implemented preset (light) uses the direct-sensor
mode instead; a preset built around "derived" mode was tried and removed after it
proved unusable in closed loop — see the note at the end of Section 3.4.

## 3. Per-signal specification

**Update rate note (applies to every signal below):** all sensors sample and are
regenerated at the same frequency as the NMPC solver — i.e. once per simulation step
`dt` (`config.dt`), with no zero-order hold and no per-sensor update-rate offset. In
reality sensors like GPS run much slower than the solver; that distinction (2.g) is
deliberately deferred and is not applied anywhere below.

### 3.1 Position (x, y) — GPS-like

Model: 2.c (white + OU bias), optionally 2.f (outliers). Regenerated every solver step.

```
x_meas_k = x_true_k + b_x_k + w_x_k
y_meas_k = y_true_k + b_y_k + w_y_k
```

`b_x`, `b_y` are two **independent** OU processes (2.b) — independence keeps the model
simple and is the correct first-pass choice; a shared/rotated 2D bias (to mimic
directional multipath geometry) is a possible later refinement but is not needed for a
first implementation.

- `tau_gps` ≈ 60–300 s — GPS multipath/ionospheric bias wanders on the timescale of
  minutes, much slower than the vessel's control loop, but the OU recursion (2.b)
  itself is still stepped once per solver step using `dt = config.dt`.
- Outliers (2.f) represent occasional multipath spikes near structures/other vessels;
  not used by the light preset (`pos_outlier_p = 0`).

### 3.2 Heading (psi) — compass / AHRS

Model: 2.c, angle-wrapped.

```
psi_meas_k = wrap(psi_true_k + b_psi_k + w_psi_k)
```

- `tau_psi` ≈ 100–600 s (hard-iron/soft-iron compass drift, or AHRS yaw-gyro bias
  instability integrated over time).
- Because the NMPC's augmented state actually carries `sin(psi_e)`, `cos(psi_e)`
  rather than a raw angle, the noise can equivalently be injected before that
  transformation (perturb `psi` then recompute sin/cos) — the wrap step then happens
  automatically and never needs an explicit `atan2` correction.

### 3.3 Yaw rate (r) — gyroscope

Model: 2.c, with the white and bias terms kept as clearly separate contributions
(a real gyro datasheet always splits "angular random walk" from "bias instability" —
collapsing them into one term would hide the fact that they have very different
correlation times):

```
r_meas_k = r_true_k + b_r_k + w_r_k
```

- `w_r`: white term, standard deviation ~0.05–0.2 deg/s (light preset).
- `b_r`: OU bias term, `tau_r` ≈ 100–600 s, representing gyro "bias instability" — this
  is the dominant error source in integrated heading if `r` were dead-reckoned, which
  is exactly why it must not be dropped even though its magnitude is small.

### 3.4 Surge / sway velocity (u, v)

Two mutually-exclusive generation strategies — pick one per preset, not both:

1. **Direct sensor** (speed log / Doppler log) — model 2.c with small `sigma_w`
   (~0.05–0.2 m/s) and a small/near-static bias. Used by the **light** preset (the only
   preset currently implemented), representing a vessel with real velocity
   instrumentation.
2. **GPS-derived** (finite-differenced from the already-noisy position measurement of
   3.1) — apply formula 2.h on top of the position noise. Representing a lower-cost
   sensor suite without a dedicated speed log, where velocity is inferred rather than
   measured and inherits amplified GPS noise.

   **Tried and removed:** an earlier "heavy" preset used this mode with GPS noise large
   enough (`pos_white_sigma = 6 m` at `dt = 0.1s`) that formula 2.h put `sigma_vel` at
   roughly 85 m/s — on a vessel whose true speed is ~1 m/s. Individual samples routinely
   landed in the tens-to-hundreds of m/s. Fed into the NMPC's initial-state equality
   constraint, this caused `AcadosNMPC`'s single-iteration SQP-RTI solve to fail
   (HPIPM "error status 3") almost immediately, and — since the solver's internal
   warm-start iterate is never reset on failure — it then failed permanently for the
   rest of the run. Root cause: feeding raw, unfiltered measurement noise straight into
   a hard `x0` constraint only works when the noise is small relative to the vehicle's
   own dynamics; a real system would reject an outlier like this with a state estimator
   (Kalman filter/EKF) between the sensor and the controller, which this simulator does
   not yet have. The "derived" mode itself remains a general capability of
   `sensor_model`; it's just not exercised by any preset right now.

### 3.5 Rudder angle (delta) / propeller speed (n)

These are actuator encoder feedback, not indirectly inferred quantities, so they get
the lightest treatment:

```
delta_meas_k = round((delta_true_k + b_delta) / LSB_delta) * LSB_delta + w_delta_k
n_meas_k     = round((n_true_k     + b_n)     / LSB_n)     * LSB_n     + w_n_k
```

- `b_delta`, `b_n`: static per-run offset (2.d, held constant — encoder mounting
  offset), not a full OU process.
- `w_delta`, `w_n`: small white term, e.g. `sigma_delta` ~0.1 deg, `sigma_n` ~0.5 rpm.
- `LSB_delta`, `LSB_n`: encoder resolution (2.e); can be set to an effectively-zero
  step size if quantization is not of interest yet.

## 4. "Light" noise preset

The only preset currently implemented (`sensor_model/config.py`'s `LIGHT`). An earlier
"heavy" preset was tried and removed — see the note at the end of Section 3.4 for why.
Concrete numeric starting points (tune once real hardware specs are known). Every
signal is regenerated once per solver step (`dt`); no column for update rate is needed
since all rows are identical on that axis.

| Signal | sigma_white | tau (bias) | sigma_bias | Outliers (p, sigma_outlier) |
|---|---|---|---|---|
| x, y (GPS) | 1.5 m | 120 s | 1.0 m | none |
| psi | 1.0 deg | 300 s | 0.5 deg | none |
| r | 0.1 deg/s | 300 s | 0.05 deg/s | none |
| u, v | 0.1 m/s (direct) | static | 0.02 m/s | none |
| delta | 0.1 deg | static | 0.05 deg | none |
| n | 0.5 rpm | static | 0.2 rpm | none |

## 5. Cross-signal consistency notes

- **Independence by default.** Each signal's noise is generated from its own
  independent random draws unless explicitly stated otherwise (x/y share no common
  process; psi and r share no common process). This is the simplest model that is
  still physically defensible, and should be the first implementation. Shared/coupled
  noise sources (e.g. a single clock-bias term coupling GPS x/y/time, or an IMU
  temperature drift coupling gyro+accelerometer axes together) are a possible later
  refinement, not a starting requirement.
- **Plant-model mismatch is mandatory, not optional.** All of the above corrupts the
  *measurement* fed back to the NMPC. The NMPC's internal prediction model
  (`casadi_mmg.py` / `nmpc_acados.py`/`nmpc_casadi.py`'s integrator) must keep
  integrating the clean, noise-free dynamics. If noise were injected into the
  prediction model itself, the controller would be "told the truth" about its own
  uncertainty and the exercise of testing feedback robustness would be meaningless.
- **Reproducibility.** Every OU/white-noise generator above should be seeded (one seed
  per signal, or one seed per run covering all signals) so that a given noise preset
  produces a repeatable trajectory across runs, which matters for comparing controller
  tuning changes fairly.

## 6. Relationship to other planning documents

This file defines *only* the statistical generation of sensor noise. It intentionally
does not repeat:
- The flowchart / block-diagram placement of the sensor-model step relative to the
  true plant and the NMPC (already established: sensor model sits strictly between
  true plant output and NMPC input/feedback).
- The ROS2 node/topic wiring this would eventually live in — see
  `ROS2_CONVERSION_PLAN.md`, Section 8 ("Forward-compatibility notes"), which already
  flags where a future `sensor_model` stage would attach without requiring
  re-architecture of the node graph.
