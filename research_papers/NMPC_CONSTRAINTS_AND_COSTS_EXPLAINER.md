# Theoretical Analysis: NMPC Constraints and Cost Formulations

This document provides a theoretical breakdown of all **cost function terms** and **constraints** in the Nonlinear Model Predictive Control (NMPC) formulations for Autonomous Surface Vehicles (ASVs), derived from the research papers in this directory:

* **Paper 1 (Gonzalez-Garcia et al., 2022)**: *"Path-following and LiDAR-based obstacle avoidance via NMPC for an autonomous surface vehicle"*, *Ocean Engineering*, 266, 112900.
* **Paper 2 (Collado-Gonzalez et al., 2024)**: *"Adaptive sliding mode control with nonlinear MPC-based obstacle avoidance using LiDAR for an autonomous surface vehicle under disturbances"*, *Ocean Engineering*, 311, 118998.
* **Survey & Extensions**: Disturbance-aware current integration, continuous distance-field obstacle avoidance, and COLREGS-compliant dynamic collision avoidance.

---

## 1. Optimal Control Problem (OCP) Overview

The standard discrete-time NMPC optimization problem over a prediction horizon of $N$ stages with sampling time $\Delta t$ is formulated as:

$$\min_{\mathbf{x}_0, \dots, \mathbf{x}_N, \; \mathbf{u}_0, \dots, \mathbf{u}_{N-1}, \; \mathbf{s}_0, \dots, \mathbf{s}_{N-1}} J = \sum_{k=0}^{N-1} \ell(\mathbf{x}_k, \mathbf{u}_k, \mathbf{s}_k) + E(\mathbf{x}_N)$$

$$\begin{aligned}
\text{subject to:} \quad & \mathbf{x}_0 = \mathbf{x}_{\text{init}} && \text{(Initial State Measurement)} \\
& \mathbf{x}_{k+1} = \mathbf{f}_{\text{disc}}(\mathbf{x}_k, \mathbf{u}_k, \mathbf{p}_k) && \text{(System Dynamics Continuity)} \\
& \mathbf{x}_{\min} \le \mathbf{x}_k \le \mathbf{x}_{\max} && \text{(State Bounds)} \\
& \mathbf{u}_{\min} \le \mathbf{u}_k \le \mathbf{u}_{\max} && \text{(Control / Rate Bounds)} \\
& h_j(\mathbf{x}_k, \mathbf{p}_k) + s_{j,k} \ge 0 && \text{(Soft Obstacle Avoidance)} \\
& 0 \le s_{j,k} \le s_{\max} && \text{(Slack Bounds)}
\end{aligned}$$

---

## 2. Explanation of All Cost Function Terms

The overall objective function is decomposed into:

$$J = \underbrace{\sum_{k=0}^{N-1} \left( \|\mathbf{x}_k - \mathbf{x}_{k,\text{ref}}\|_Q^2 + \|\mathbf{u}_k\|_R^2 + \|\mathbf{s}_k\|_S^2 \right)}_{\text{Stage Cost } \sum \ell(\mathbf{x}_k, \mathbf{u}_k, \mathbf{s}_k)} + \underbrace{\|\mathbf{x}_N - \mathbf{x}_{N,\text{ref}}\|_{Q_N}^2}_{\text{Terminal Cost } E(\mathbf{x}_N)}$$

```
                            TOTAL NMPC COST FUNCTION J
                                      │
         ┌────────────────────────────┼───────────────────────────┐
         ▼                            ▼                           ▼
  State Tracking Cost          Actuation & Slew Costs     Constraint Slack Penalty
  • Cross-track error (e_y)    • Control effort (u^2)     • Linear slack penalty (z_l)
  • Course error (sin/cos)     • Rate of change (u_dot^2) • Quadratic penalty (Z_l)
  • Surge speed (u - u_d)      • Smooth actuator commands • Guarantees NLP feasibility
  • Yaw rate damping (r^2)
```

### 2.1. State Tracking Stage Costs ($\|\mathbf{x}_k - \mathbf{x}_{k,\text{ref}}\|_Q^2$)

| Term / Weight | State Variable | Mathematical Expression | Theoretical Role & Physical Rationale |
| :--- | :--- | :--- | :--- |
| **Cross-Track Error Penalty** | $e_y$ | $Q_{e_y} \cdot e_{y,k}^2$ | Measures signed perpendicular deviation from the path segment: $e_y = -(x - x_d)\sin\chi_p + (y - y_d)\cos\chi_p$. Forces convergence back to the centerline. |
| **Course Alignment Penalty** | $\chi_s, \chi_c$ | $Q_{\chi_s}(\sin\chi_k - \sin\chi_p)^2 + Q_{\chi_c}(\cos\chi_k - \cos\chi_p)^2$ | Uses trigonometric coordinates $\sin\chi$ and $\cos\chi$ to eliminate $\pm\pi$ angle wraparound discontinuities, ensuring the vessel travels in the forward path direction. |
| **Surge Speed Tracking** | $u$ | $Q_u \cdot (u_k - u_d)^2$ | Regulates forward thrust to maintain desired transit speed $u_d$ while allowing speed drops when executing tight avoidance turns. |
| **Yaw Rate Damping** | $r$ | $Q_r \cdot r_k^2$ | Penalizes large rotational velocities, damping high-frequency heading oscillations and stabilizing straight-line motion. |
| **Inertial Coordinate Tracking** | $x, y$ | $Q_x(x_k - x_{k,\text{ref}})^2 + Q_y(y_k - y_{k,\text{ref}})^2$ | Used during waypoint transition zones to guide the ship through the exact turning corner. |

### 2.2. Actuation and Slew Rate Costs ($\|\mathbf{u}_k\|_R^2$)

1. **Actuator Effort Penalty:**
   $$\ell_{\text{effort}} = R_{\mathbf{u}} \|\mathbf{u}_k\|^2 = R_{T_1} T_{1,k}^2 + R_{T_2} T_{2,k}^2$$
   Penalizes unnecessary thruster energy consumption and prevents actuator saturation.
2. **Actuator Slew / Rate Penalty (Augmented Formulation):**
   When the actuator positions (thrusters or rudder/shaft) are states, the control input $\mathbf{u}_k$ is the rate of change ($\dot{\mathbf{u}}$):
   $$\ell_{\text{slew}} = R_{\dot{\mathbf{u}}} \|\dot{\mathbf{u}}_k\|^2 = R_{\dot{\delta}} \dot{\delta}_k^2 + R_{\dot{n}} \dot{n}_k^2$$
   Enforces smooth, continuous control signals, preventing high-frequency chattering and mechanical wear on steering gears and thruster motors.

### 2.3. Slack Variable Penalties ($\|\mathbf{s}_k\|_S^2$)

$$\ell_{\text{slack}} = \sum_{j=1}^{n_{\text{obs}}} \left( z_{l,j} s_{j,k} + Z_{l,j} s_{j,k}^2 \right)$$

* $s_{j,k} \ge 0$ is the relaxation distance for obstacle $j$ at stage $k$.
* $Z_{l,j} \gg Q$ (e.g., $Z_l = 50 - 1000$).
* **Theoretical Purpose:** Prevents the quadratic programming (QP) solver from returning an infeasible status if the ship is momentarily pushed into the collision buffer by inertia or currents.

### 2.4. Terminal State Cost ($E(\mathbf{x}_N)$)

$$E(\mathbf{x}_N) = \|\mathbf{x}_N - \mathbf{x}_{N,\text{ref}}\|_{Q_N}^2$$

* $Q_N \approx (2 \sim 5) \cdot Q$.
* **Theoretical Purpose:** Emulates an infinite-horizon cost-to-go, ensuring asymptotic stability and avoiding greedy/short-sighted horizon behavior.

### 2.5. Extended Theoretical Cost Formulations

* **Smooth Exponential Potential Cost (Grid Fields):**
  $$L_{\text{obs}}(x_k, y_k) = w_{\text{obs}} \exp\left(-\frac{\hat{D}(x_k, y_k) - r_{\text{safety}}}{\lambda}\right)$$
  Provides early, smooth repulsion from obstacles before constraints become active.
* **COLREGS Encounter Risk Cost (Dynamic Vessels):**
  $$L_{\text{COLREGS}} = w_{\text{cpa}} \left(\frac{1}{\text{DCPA}^2 + \epsilon}\right) \exp\left(-\frac{\text{TCPA}}{\tau}\right) \cdot f_{\text{sector}}(\beta_{ot}, \psi_{ot})$$
  Penalizes close Distance/Time to Closest Point of Approach ($\text{DCPA}/\text{TCPA}$) and enforces right-of-way passing rules.

---

## 3. Explanation of All NMPC Constraints

```
                            NMPC CONSTRAINTS
                                   │
         ┌─────────────────────────┼────────────────────────┐
         ▼                         ▼                        ▼
  System Dynamics           Actuator & State         Collision Avoidance
  (Equality)                Bounds (Inequality)      (Inequality)
  • Discretized ODE model   • Surge speed [u_min, u_max] • Hard: dist^2 >= r_c^2
  • Initial state pinning   • Actuator limits [u_min, u_max] • Soft: dist^2 - r_c^2 + s >= 0
  • Kinematic equations     • Rate limits [u_dot_min, u_dot_max] • Distance field D_hat >= r_s
```

### 3.1. Dynamic Model Equality Constraints

$$\mathbf{x}_{k+1} = \mathbf{f}_{\text{disc}}(\mathbf{x}_k, \mathbf{u}_k, \mathbf{p}_k), \quad k = 0, \dots, N-1$$

* Enforces physical continuity across the prediction horizon using explicit integration (e.g., Runge-Kutta 4th Order, RK4).
* Incorporates hydrodynamic damping, added mass, thruster/rudder mechanics, and ocean current disturbances $\mathbf{v}_c = [v_{cx}, v_{cy}]^T$.
* **Initial State Pinning:** $\mathbf{x}_0 = \mathbf{x}(t)$ pins stage 0 to the actual estimated vessel state.

### 3.2. State and Actuator Box Inequality Constraints

| Constraint | Variable | Mathematical Form | Theoretical Purpose |
| :--- | :--- | :--- | :--- |
| **Surge Velocity Limits** | $u$ | $u_{\min} \le u_k \le u_{\max}$ | Prevents motor overspeed and avoids mathematical division singularities ($u \approx 0$) in hydrodynamic sideslip models. |
| **Yaw Rate Limits** | $r$ | $r_{\min} \le r_k \le r_{\max}$ | Bounds turning rates to avoid hull capsizing and sensor tracking loss. |
| **Actuator Position Limits** | $T$ or $\delta, n$ | $T_{R,\max} \le T_i \le T_{F,\max}$ <br> $\delta_{\min} \le \delta_k \le \delta_{\max}$ <br> $n_{\min} \le n_k \le n_{\max}$ | Physical bounds of the thrusters, rudder angle ($\pm 35^\circ$), and propeller RPM. |
| **Actuator Slew Rate Limits** | $\dot{T}$ or $\dot{\delta}, \dot{n}$ | $\dot{T}_{\min} \le \dot{T}_i \le \dot{T}_{\max}$ <br> $\dot{\delta}_{\min} \le \dot{\delta}_k \le \dot{\delta}_{\max}$ <br> $\dot{n}_{\min} \le \dot{n}_k \le \dot{n}_{\max}$ | Physical maximum velocity of steering pumps and thruster motor acceleration. |

### 3.3. Obstacle Avoidance Inequality Constraints

#### A. Circle-Based Hard Avoidance Constraint
For each obstacle $j$ at $(x_{o,j}, y_{o,j})$:

$$(x_k - x_{o,j})^2 + (y_k - y_{o,j})^2 \ge r_{c,j}^2$$

where $r_{c,j} = r_{\text{obs},j} + R_{\text{ASV}}$ ($R_{\text{ASV}}$ is vessel bounding radius, $r_{\text{obs},j}$ is detected obstacle radius).

#### B. Soft Slack-Relaxed Constraint
To guarantee mathematical feasibility under sudden disturbances:

$$h_j(x_k, y_k) + s_{j,k} \ge 0 \iff (x_k - x_{o,j})^2 + (y_k - y_{o,j})^2 - r_{c,j}^2 + s_{j,k} \ge 0$$
$$\text{with: } \quad 0 \le s_{j,k} \le s_{\max}$$

* When clear of obstacles: $h_j \ge 0 \implies s_{j,k} = 0$.
* If an unavoidable intrusion occurs: $s_{j,k} > 0$ absorbs the violation, penalized heavily by $Z_{l} s_{j,k}^2$ in the cost function.

#### C. Moving Obstacles (Stage-Varying Constraint)
For moving vessels with velocity $\mathbf{v}_{\text{obs},j}$:

$$(x_k - x_{o,j}(k))^2 + (y_k - y_{o,j}(k))^2 \ge (r_{c,j} + \sigma_k)^2 - s_{j,k}$$
$$x_{o,j}(k) = x_{o,j}(0) + k \Delta t \cdot v_{ox,j}, \quad y_{o,j}(k) = y_{o,j}(0) + k \Delta t \cdot v_{oy,j}$$

where $\sigma_k$ is an uncertainty growth margin expanding with lookahead stage $k$.

#### D. Continuous Grid Distance Field Constraint
When occupancy grids are transformed via Euclidean Distance Transform (EDT):

$$\hat{D}(x_k, y_k) + s_k \ge r_{\text{safety}}$$

where $\hat{D}(x, y)$ provides an Eikonal unit gradient ($\|\nabla \hat{D}\| = 1$) pointing directly away from the nearest obstacle surface.

---

## 4. Comparison Summary: Primary Literature

| Feature | Gonzalez-Garcia et al. (2022) | Collado-Gonzalez et al. (2024) |
| :--- | :--- | :--- |
| **Control Structure** | Single-Layer Unified NMPC (Guidance + Plant Dynamics). | Two-Layer: NMPC Guidance (Heading command) + Adaptive Sliding Mode Control (ASMC) Inner Loop. |
| **Prediction Model** | 3-DOF Non-linear Catamaran Dynamic Equations. | Kinematic Heading-Lag Model with sideslip compensation. |
| **State Vector $\mathbf{x}$** | $[\chi, \sin\chi, \cos\chi, e_y, x, y, u, v, r, T_{\text{port}}, T_{\text{stbd}}]^T$ | $[e_y, \chi_e, x, y, \psi, \psi_{d,\text{filt}}]^T$ |
| **Control Input $\mathbf{u}$** | $[\dot{T}_{\text{port}}, \dot{T}_{\text{stbd}}]^T$ (Thruster rate commands). | $[\omega_d]^T$ (Desired yaw rate command). |
| **Disturbance Handling** | Passive kinematic sideslip angle formulation $\beta = \arcsin(v/U)$. | Active robust cancellation via inner-loop ASMC. |
| **Obstacle Constraint** | Soft quadratic distance constraint with slack $s_{j,k}$. | Soft quadratic distance constraint with slack $s_{j,k}$. |
| **Solver Stack** | `acados` SQP-RTI (Gauss-Newton Hessian). | `acados` SQP-RTI + HPIPM QP Solver. |
