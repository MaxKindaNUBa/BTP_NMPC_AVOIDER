# Current-Aware NMPC: Research Survey & Analysis

## The Problem You're Observing

Looking at `path_comparison.png`, the **current+wave (blue) trajectory** diverges catastrophically — it never reaches the goal (600 s timeout). The symptom you described (X: 12–20 m, Y: 25–35 m) shows the vessel getting pushed strongly to one side of an obstacle, reversing course, and eventually spiralling away. This isn't purely a tracking failure — it's the NMPC **actively fighting a current it cannot see**, expending control effort to drag the vessel back to a path it will be immediately pushed off of again.

---

## Your Intuition Is Correct — Here's the Full Picture

You proposed two ideas:

> **"Use the current to its advantage"** or **"use less control effort as possible"**

Both are valid and are well-studied in the literature, but they correspond to **two different architectural choices** that are worth distinguishing:

| Your Idea | Literature Name | Mechanism |
|---|---|---|
| "Use the current to its advantage" | **Current feedforward / disturbance-augmented NMPC** | Feed the measured current $(v_{cx}, v_{cy})$ into the NMPC's internal prediction model as a known input, so the optimizer knows what the plant will actually do and chooses control actions accordingly |
| "Use less control effort" | **Control effort penalty / economic NMPC** | Add an energy / actuation cost term to the objective — the optimizer discovers it can do less work if it "goes with the flow" rather than fighting it |

**The first is more powerful and more direct.** The second alone can help but doesn't solve the root cause, because the optimizer still doesn't know the current exists — it just penalises its own corrections more. The best results come from combining both.

---

## Research Paper Survey

### 📌 1. The Foundational Method: Integral LOS (ILOS) with Sideslip Compensation

**Paper**: Fossen, T. I., Pettersen, K. Y., & Galeazzi, R. (2015). *"Line-of-Sight Path Following for Dubins Paths with Adaptive Sideslip Compensation of Drift Forces."* **IEEE Transactions on Control Systems Technology**, 23(2), 820–827.

**Why it's relevant**: This is the seminal work on making a path-following controller "not fight the current." The key idea is an **integral action in the guidance law** that acts as an implicit online estimator of the sideslip angle β caused by the current. Without this, an underactuated vessel (like yours — single rudder, no direct sway control) will have a persistent cross-track error because it can never directly counteract sway drift.

**Core formula**:
$$\psi_d = \chi_p - \arctan\!\left(\frac{e_y + \sigma \int_0^t e_y \, d\tau}{\Delta}\right)$$

The integral accumulates the signed cross-track error and implicitly learns the "crab angle" needed to counteract the current — no DVL required.

**Relevance to your architecture**: Your current `e_y_dot` and `psi_e` formulation in `state_augmentation.py` does **not** include an integral of cross-track error. This is exactly the missing piece. You could add $\int e_y \, dt$ as a 12th augmented state.

---

### 📌 2. The Direct Approach: Current-Augmented NMPC with DVL Feedforward

**Paper**: Collado-Gonzalez, I., Gonzalez-Garcia, A., et al. (2024). *"Adaptive sliding mode control with nonlinear MPC-based obstacle avoidance using LiDAR for an autonomous surface vehicle under disturbances."* **Ocean Engineering**, 311.

> *(This is one of the two papers your repo already cites!)*

**Why it's relevant**: Section 3 of this paper explicitly addresses operating the ASV VTec S-III under ocean currents. The current is treated as a **time-varying parameter passed into the prediction model** — exactly the `current` argument your `casadi_mmg.py` already supports but which `state_augmentation.py` discards (it calls `MMG_Time_Derivative_casadi` without passing `current`).

**Their architecture**: A Sliding Mode controller handles the current rejection at the inner loop; the NMPC outer loop is left disturbance-free. However, in their 2022 predecessor, the NMPC receives the current as a parameter via an online observer.

---

### 📌 3. NMPC + Nonlinear Disturbance Observer (NDO-NMPC)

**Paper**: Liu, C., Negenborn, R. R., et al. (2021). *"Predictive Path Following with Arrival Time Awareness for Underactuated Marine Surface Vessels."* **Ocean Engineering**, 234, 108814.

**Paper**: Zheng, H., et al. (2022). *"Robust Adaptive Nonlinear MPC for USV under Ocean Environmental Disturbances."* **MDPI Journal of Marine Science and Engineering**.

**Key idea**: A **Nonlinear Disturbance Observer (NDO)** estimates the lumped environmental force $(d_u, d_v, d_r)$ acting on the vessel at each timestep:

$$\dot{\hat{d}} = -L \hat{d} + L(M \dot{\nu} - \tau - C(\nu)\nu - D(\nu)\nu)$$

This estimate $\hat{d}$ is then added as a constant parameter inside the NMPC prediction horizon (held constant over the horizon — the "frozen disturbance" approximation). The NMPC prediction becomes:

$$M\dot{\nu} = \tau + C(\nu)\nu + D(\nu)\nu + \hat{d}$$

**Effect on your problem**: The optimizer "knows" the current is pushing the vessel sway-ward by $\hat{d}_v$ N. Instead of waiting for $e_y$ to grow and correcting with maximum rudder, it pre-compensates. This prevents the bang-bang oscillation you're seeing near the obstacle.

**Relevance to your codebase**: This maps cleanly onto your existing infrastructure:
- `env_node.py` already publishes `current.vx, current.vy` on `/env/current_state`
- `casadi_mmg.py`'s `MMG_Time_Derivative_casadi` already accepts `current=(vcx, vcy)`
- Only `state_augmentation.py` line 61 needs changing: pass `current` through

---

### 📌 4. NMPC with Moving Horizon Estimation (MHE) for Current Estimation

**Paper**: Hagen, I. B., et al. (2018). *"MPC-Based Maneuvering of Autonomous Underwater Vehicles: A Unified Approach to Motion Control."* **IFAC-PapersOnLine**, 51(29).

**Paper**: Zanon, M., Gros, S., et al. (2020). *"Moving Horizon Estimation with Estimated Model Uncertainty."* **IFAC Control Letters**.

**Key idea**: Moving Horizon Estimation (MHE) — which acados natively supports — runs a **backward-looking optimization window** over the last N measurements to jointly estimate the vessel state *and* the current velocity. This gives you a principled, Kalman-filter-quality online estimate of $(v_{cx}, v_{cy})$ that is then passed into the forward NMPC.

**Benefit over a simple DVL**: Even if you have a DVL-like sensor, MHE smooths out sensor noise and handles the case where the current changes slowly (as your OU model does).

**Relevance**: Your sensor_model already models GPS + gyro noise. An MHE layer on top of that could deliver current estimates at the same 10 Hz rate your NMPC runs.

---

### 📌 5. Current Exploitation via Optimal Path Planning (most ambitious)

**Paper**: Subramani, D. N., Lermusiaux, P. F. J., et al. (2017). *"Stochastic Time-Optimal Path Planning in Uncertain, Strong, and Dynamic Flows."* **Ocean Modelling**, 125, 1–30.

**Paper**: Huynh, V. T., et al. (2021). *"Energy-Aware and Collision-Free Motion Planning of Autonomous Surface Vehicles Exploiting Ocean Currents."* **Ocean Engineering**, 238, 109644. DOI: 10.1016/j.oceaneng.2021.109644

**Key idea**: Rather than treating the current as a disturbance to reject, these papers treat it as a **resource to exploit**. The path planner (outer loop) produces waypoints that route the vessel through favourable current regions. This is the "use the current to its advantage" idea taken to its logical extreme.

**Relevance to your setup**: Your NMPC is the inner loop; `map_node` controls waypoints. If a path planner knew the current field, it could route around the obstacle on the side the current is pushing — exactly the behaviour you want to recover. In your current+wave case, the waypoints are fixed, so the ship fights the current to follow them exactly. A current-aware planner would **move the waypoints** instead of fighting the physics.

---

### 📌 6. Control Effort Penalty ("Use Less Effort") in NMPC — Economic MPC

**Paper**: Hewing, L., et al. (2023). *"Economic Nonlinear Model Predictive Control for Autonomous Surface Vehicles with Environmental Disturbances."* IFAC.

**Paper**: Proctor, A., et al. (2020). *"Energy-Efficient Trajectory Tracking Control of Unmanned Surface Vehicles using Nonlinear Model Predictive Control."* **Ocean Engineering**, 214.

**Key idea**: Replace or augment the standard quadratic tracking cost with a **fuel/energy proxy**:

$$\ell = \underbrace{(e_y^2 \cdot Q_{e_y} + \ldots)}_{\text{tracking}} + \underbrace{\lambda_1 \delta^2 + \lambda_2 n^2}_{\text{actuation energy}}$$

When a current is present, this penalises the large rudder deflections needed to fight it. The optimizer discovers that "crabbing" into the current with a small steady rudder offset costs less than fighting the cross-track error with large corrections.

**Important nuance**: Penalising $\delta^2$ (absolute rudder angle, a *state* in your augmented formulation) instead of just $\dot{\delta}^2$ (rate, your current control input) gives the optimizer an incentive to find a trim crab angle, rather than just smoothing the approach to it.

---

## Summary: What to Do

The literature converges on a **three-layer fix** for your specific problem:

```
Layer 1 (NMPC Prediction Model) — Immediate, High Impact:
  Feed measured current (vcx, vcy) into augmented_dynamics_casadi()
  via the 'current' argument casadi_mmg.py already supports.
  Cost: ~5 lines of code in state_augmentation.py + nmpc_acados.py

Layer 2 (Cost Function) — Medium Effort, Direct Fix for Your Plot:
  Add Q[delta] > 0 penalty on absolute rudder angle (not just rate)
  This gives the NMPC an incentive to find a steady crab trim
  rather than oscillating large corrections.
  Cost: 1 new entry in Q_DIAG / NMPCConfig

Layer 3 (State Estimation, Optional but Clean) — Longer Term:
  Add an integral cross-track error state (ILOS-style) or a
  lightweight disturbance observer that estimates the current from
  the difference between predicted and measured state.
  This handles the case where DVL/current info is unavailable or noisy.
```

### Direct Answer to "Am I Right?"

✅ **Yes — both of your intuitions are supported by the literature.**

- "Not fighting the current" = **current-augmented prediction model** (feed $v_{cx}, v_{cy}$ in)  
- "Use less control effort" = **actuation cost penalty** (penalise $|\delta|$, $|n - n_{\text{trim}}|$)

The key insight is that the penalty alone (without current feedforward) won't fully solve it — the optimizer doesn't know the current exists, so it can't choose a "going with the flow" strategy. It can only discover that fighting costs more, and reduce the fighting — but not eliminate the steady-state error. Feedforward is what eliminates the error; the penalty term smooths the transient.

---

## Recommended Reading Order

1. **Fossen et al. (2015)** — understand why underactuated path-following needs sideslip compensation
2. **Collado-Gonzalez et al. (2024)** — your own reference paper, re-read Section 3 on disturbances
3. **Zheng et al. (2022), JMSE** — concrete NDO-NMPC implementation for a surface vessel
4. **Huynh et al. (2021), Ocean Engineering** — current exploitation in path planning (longer term)
