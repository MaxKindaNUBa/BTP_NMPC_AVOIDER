import os
import ctypes
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Set ACADOS_SOURCE_DIR programmatically
os.environ["ACADOS_SOURCE_DIR"] = "/home/chandran/acados"

# Pre-load acados shared libraries to bypass LD_LIBRARY_PATH requirement on Linux
acados_lib_dir = "/home/chandran/acados/lib"
if os.path.exists(acados_lib_dir):
    try:
        mode = ctypes.RTLD_GLOBAL
        ctypes.CDLL(os.path.join(acados_lib_dir, "libqdldl.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libosqp.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libqpOASES_e.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libblasfeo.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libhpipm.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libacados.so"), mode=mode)
    except Exception as e:
        print(f"Warning: programmatically loading acados libraries failed: {e}")

import scipy.interpolate as interp

# 3-Point Spline Knots in TH = d/h
# Knot 1: Deep water (h/d = 19.5 => TH = 0.0513)
# Knot 2: Shallow water (h/d = 1.5 => TH = 0.6667)
# Knot 3: Very shallow water (h/d = 1.2 => TH = 0.8333)
TH_KNOTS_3PT = np.array([0.0513, 0.6667, 0.8333])

# Shallow-water hydrodynamic scaling factors at the 3 depth anchor points
# Anchored to KVLCC2 captive-model test benchmarks (SIMMAN 2020 / Li et al. 2024 / Yoshimura 1990)
SHALLOW_FACTORS_3PT = {
    'F_Yv':   np.array([1.000, 2.160, 3.600]),
    'F_Yr':   np.array([1.000, 1.150, 1.350]),
    'F_Nv':   np.array([1.000, 1.850, 2.600]),
    'F_Nr':   np.array([1.000, 2.100, 3.000]),
    'F_Yvvv': np.array([1.000, 2.000, 3.000]),
    'F_Nvvr': np.array([1.000, 2.000, 3.000]),
    'F_Nvrr': np.array([1.000, 2.200, 3.200]),
    'F_my':   np.array([1.000, 1.614, 2.645]),
    'F_jz':   np.array([1.000, 1.614, 2.645]),
}


def casadi_pchip_3pt(th_sym, x_knots, y_knots):
    """
    Evaluates a 3-point Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
    symbolically in CasADi with strict Fritsch-Carlson monotonicity guarantees.
    Runs inside the CasADi computational graph and compiles to C-code for Acados.
    """
    x0, x1, x2 = float(x_knots[0]), float(x_knots[1]), float(x_knots[2])
    y0, y1, y2 = float(y_knots[0]), float(y_knots[1]), float(y_knots[2])
    h0 = x1 - x0
    h1 = x2 - x1

    p = interp.PchipInterpolator(x_knots, y_knots)
    d0, d1, d2 = [float(val) for val in p.derivative()(x_knots)]

    th_c = ca.fmin(ca.fmax(th_sym, x0), x2)

    # Subinterval 0: [x0, x1]
    t0 = (th_c - x0) / h0
    P0 = (y0 * (2.0 * t0**3 - 3.0 * t0**2 + 1.0)
          + h0 * d0 * (t0**3 - 2.0 * t0**2 + t0)
          + y1 * (-2.0 * t0**3 + 3.0 * t0**2)
          + h0 * d1 * (t0**3 - t0**2))

    # Subinterval 1: [x1, x2]
    t1 = (th_c - x1) / h1
    P1 = (y1 * (2.0 * t1**3 - 3.0 * t1**2 + 1.0)
          + h1 * d1 * (t1**3 - 2.0 * t1**2 + t1)
          + y2 * (-2.0 * t1**3 + 3.0 * t1**2)
          + h1 * d2 * (t1**3 - t1**2))

    return ca.if_else(th_c < x1, P0, P1)


def MMG_Time_Derivative_casadi(state, control, current=(0.0, 0.0), wave_force=(0.0, 0.0, 0.0),
                                h_over_d=19.5, smooth=True, scale=1000.0):
    """
    CasADi implementation of MMG_Time_Derivative with Inoue's Geometric Equations
    and 3-Point Monotonic Spline (PCHIP) Shallow-Water Hydrodynamic Formulations.

    Parameters:
    -----------
    state : ca.MX or ca.SX
        Current state of the ship [u, v, r, x, y, psi]
    control : ca.MX or ca.SX
        Control inputs [delta, rps] (rudder angle in rad, propeller speed in rps)
    current : (float, float) or ca.MX/ca.SX pair, default (0.0, 0.0)
        Earth-frame current velocity (vcx, vcy) [m/s].
    wave_force : (float, float, float) or ca.MX/ca.SX triple, default (0.0, 0.0, 0.0)
        Body-frame (surge, sway, yaw) wave drift force/moment [N, N, N*m].
    h_over_d : float or ca.MX/ca.SX, default 19.5
        Water depth to draft ratio (h/d). h_over_d >= 10 represents deep water;
        values approaching 1.2 represent extremely shallow water.
    smooth : bool, default True
        If True, uses smooth tanh blending for continuous functions.
    scale : float, default 1000.0
        Steepness of the tanh transition for discontinuous functions when smooth=True.

    Returns:
    --------
    dstate : ca.MX or ca.SX (6x1)
        Time derivatives [u_dot, v_dot, r_dot, x_dot, y_dot, psi_dot]
    """
    u, v, r, x, y, psi = state[0], state[1], state[2], state[3], state[4], state[5]
    delta, rps = control[0], control[1]

    if smooth:
        u_val = ca.fmax(u, 0.00001)
        rps_val = ca.fmax(rps, 0.1)
    else:
        u_val = ca.if_else(u <= 0, 0.00001, u)
        rps_val = ca.if_else(rps <= 0.1, 0.1, rps)

    # Relative water velocities under environmental current
    vcx, vcy = current[0], current[1]
    uc = vcx * ca.cos(psi) + vcy * ca.sin(psi)
    vc = -vcx * ca.sin(psi) + vcy * ca.cos(psi)
    ur = u_val - uc
    vr = v - vc

    U = ca.sqrt((ur**2) + (vr**2))     # Resultant relative speed
    beta = ca.atan2(-vr, ur)           # Relative drift angle
    Np = rps_val

    ######################################################
    ####### Principal Particulars of the Vessel ##########
    ######################################################
    Lpp    = 2.902        # Length between perpendiculars [m]
    B      = 0.527        # Ship beam [m]
    d      = 0.189        # Ship draft [m]
    xG     = 0.102        # Longitudinal coordinate of center of gravity [m]
    rho    = 1025.0       # Water density [kg/m^3]
    Volume = 0.235        # Displacement volume [m^3]
    Dp     = 0.090        # Propeller diameter [m]
    HR     = 0.144        # Rudder span length [m]
    AR     = 0.00928      # Rudder profile area [m^2]

    # Non-dimensional geometric ratios
    CB = Volume / (Lpp * B * d)  # Block coefficient (~0.8130 for KVLCC2)
    d_prime = d / Lpp
    B_prime = B / Lpp
    L_over_B = Lpp / B
    xG_prime = xG / Lpp
    k = 2.0 * d_prime            # Aspect ratio parameter (2d / Lpp)

    ### Normalization Parameters ###
    ndm_force   = 0.5 * rho * Lpp * d * (U**2)
    ndm_moment  = 0.5 * rho * (Lpp**2) * d * (U**2)
    ndm_mass    = 0.5 * rho * d * (Lpp**2)
    ndm_massMoI = 0.5 * rho * d * (Lpp**4)

    ######################################################
    #### Inoue's Added Mass & Moment of Inertia ##########
    ######################################################
    # Non-dimensional displacement mass m'
    m_prime = 2.0 * CB * B_prime
    m = Volume * rho

    # Surge added mass m'_x (Eq. 22 in Vaidesh Report / Inoue 1981)
    mx_prime = 0.03 * m_prime

    # Sway added mass m'_y (Eq. 23 in Vaidesh Report / Inoue 1981)
    my_prime = m_prime * (
        0.882 - 0.54 * CB * (1.0 - 1.6 * (d / B))
        - 0.156 * (1.0 - 0.673 * CB) * L_over_B
        + 0.826 * (d / B) * L_over_B * (1.0 - 0.678)
        - 0.638 * CB * (d / B) * (1.0 - 0.669 * (d / B))
    )

    # Added moment of inertia J'_zz (Eq. 24 in Vaidesh Report / Inoue 1981)
    Izz_prime = 0.25 * m_prime
    Jzz_prime = 0.009 * Izz_prime

    # Non-dimensional depth parameter TH = d / h
    th = 1.0 / ca.fmax(h_over_d, 1.01)

    # Added mass shallow-water scaling factors from 3-point monotonic spline
    F_my = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_my'])
    F_jz = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_jz'])

    # Dimensional mass terms
    mx = mx_prime * ndm_mass
    my = (my_prime * F_my) * ndm_mass
    jz = (Jzz_prime * F_jz) * ndm_massMoI
    IzG = m * ((0.25 * Lpp)**2)

    #####################################################
    ####### Governing Mass Matrix & Coriolis Terms ######
    #####################################################
    M = ca.vertcat(
        ca.horzcat(m + mx, 0.0, 0.0),
        ca.horzcat(0.0, m + my, m * xG),
        ca.horzcat(0.0, m * xG, jz + ((xG**2) * m) + IzG)
    )

    LHS_r = ca.vertcat(
        -m * v * r - xG * m * (r**2) - my * vr * r,
        m * u_val * r + mx * ur * r,
        m * xG * u_val * r
    )

    #####################################################
    ####### Hydrodynamic Force on Bare Hull (F_hull) ####
    #####################################################
    v_ndm = vr / U
    r_ndm = r * Lpp / U

    R0 = 0.022  # Straight moving resistance coefficient

    # 1. Deep-Water Linear Baseline (Inoue 1981 / Vaidesh Eqs. 29-32)
    Y_v_deep = -(0.5 * ca.pi * k + 1.4 * CB * B_prime)
    Y_r_deep = mx_prime + 0.5 * CB * B_prime
    N_v_deep = -k
    N_r_deep = -(0.54 * k - (k**2))

    # 2. 3-Point Monotonic PCHIP Spline Shallow-Water Factors
    F_Yv   = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_Yv'])
    F_Yr   = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_Yr'])
    F_Nv   = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_Nv'])
    F_Nr   = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_Nr'])
    F_Yvvv = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_Yvvv'])
    F_Nvvr = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_Nvvr'])
    F_Nvrr = casadi_pchip_3pt(th, TH_KNOTS_3PT, SHALLOW_FACTORS_3PT['F_Nvrr'])

    # Shallow-water corrected linear derivatives
    Y_v = Y_v_deep * F_Yv
    Y_r = Y_r_deep * F_Yr
    N_v = N_v_deep * F_Nv
    N_r = N_r_deep * F_Nr

    # 3. Inoue's Nonlinear Hydrodynamic Derivatives (Vaidesh Eqs. 38-44)
    X_vv = 1.15 * CB / L_over_B - 0.18
    X_vvvv = -6.68 * CB / L_over_B + 1.1
    X_rr = -0.085 * CB / L_over_B + 0.008 - xG_prime * m_prime
    X_vr = -(my_prime - 1.91 * CB / L_over_B + 0.08)

    Y_vvv_deep = -(0.185 * L_over_B + 0.48)
    Y_rrr = 0.02
    Y_vrr = -(0.26 * (1.0 - CB * L_over_B) + 0.11)
    Y_vvr = -0.083

    N_vvv = -(-0.69 * CB + 0.66)
    N_rrr = 0.25 * CB / L_over_B - 0.056
    N_vrr_deep = -(0.075 * (1.0 - CB * L_over_B) - 0.098)
    N_vvr_deep = 1.55 * CB / L_over_B - 0.76

    # Shallow-water corrected nonlinear derivatives
    Y_vvv = Y_vvv_deep * F_Yvvv
    N_vvr = N_vvr_deep * F_Nvvr
    N_vrr = N_vrr_deep * F_Nvrr

    # Assemble non-dimensional hull forces
    X_hull = -R0 + (X_vv * (v_ndm**2)) + (X_vr * v_ndm * r_ndm) + (X_rr * (r_ndm**2)) + (X_vvvv * (v_ndm**4))
    Y_hull = (Y_v * v_ndm) + (Y_r * r_ndm) + (Y_vvv * (v_ndm**3)) + (Y_vvr * (v_ndm**2) * r_ndm) + (Y_vrr * v_ndm * (r_ndm**2)) + (Y_rrr * (r_ndm**3))
    N_hull = (N_v * v_ndm) + (N_r * r_ndm) + (N_vvv * (v_ndm**3)) + (N_vvr * (v_ndm**2) * r_ndm) + (N_vrr * v_ndm * (r_ndm**2)) + (N_rrr * (r_ndm**3))

    F_hull = ca.vertcat(ndm_force * X_hull, ndm_force * Y_hull, ndm_moment * N_hull)

    #####################################################
    ##### Propeller Force on Ship (F_propeller) #########
    #####################################################
    tp       = 0.220
    wp0      = 0.40
    k0,k1,k2 = 0.2931, -0.2753, -0.1385
    xP       = -0.48
    beta_P   = beta - (xP * r_ndm)

    wp          = wp0 * ca.exp(-4.0 * (beta_P**2))
    Jp          = ur * (1.0 - wp) / (Np * Dp)
    KT          = k0 + k1 * Jp + k2 * (Jp**2)
    T_prop      = rho * (Np**2) * (Dp**4) * KT
    X_propellar = (1.0 - tp) * T_prop

    F_propellar = ca.vertcat(X_propellar, 0.0, 0.0)

    #####################################################
    ######### Rudder Force on Ship (F_rudder) ###########
    #####################################################
    epsilon = 1.09
    k_rud   = 0.5
    eta     = Dp / HR
    uP      = (1.0 - wp) * ur

    uR1 = ca.sqrt(1.0 + ((8.0 * KT) / (ca.pi * (Jp**2))))
    uR2 = (1.0 + k_rud * (uR1 - 1.0))**2
    uR  = epsilon * uP * ca.sqrt((eta * uR2) + (1.0 - eta))

    lR     = -0.710
    beta_R = beta - (lR * r_ndm)

    if smooth:
        gamma_r = 0.396 + (0.64 - 0.396) * (1.0 + ca.tanh(scale * beta_R)) / 2.0
    else:
        gamma_r = ca.if_else(beta_R < 0, 0.396, 0.64)

    vR = U * gamma_r * beta_R

    f_alpha = 2.747
    alpha_R = delta - ca.atan2(vR, uR)
    Ur      = ca.sqrt((uR**2) + (vR**2))

    F_normal = 0.5 * rho * AR * (Ur**2) * f_alpha * ca.sin(alpha_R)

    tR = 0.387
    aH = 0.312
    xH = -0.464
    xR = -0.5 * Lpp

    X_rudder = -(1.0 - tR) * F_normal * ca.sin(delta)
    Y_rudder = -(1.0 + aH) * F_normal * ca.cos(delta)
    N_rudder = -(xR + (aH * xH)) * F_normal * ca.cos(delta)

    F_rudder = ca.vertcat(X_rudder, Y_rudder, N_rudder)
    F_Drift = ca.vertcat(*wave_force)

    # Solve dynamic accelerations
    RHS = F_hull + F_propellar + F_rudder - LHS_r + F_Drift
    X_acc = ca.solve(M, RHS)

    #####################################################
    ############## Kinematics of the Ship ###############
    #####################################################
    R_mat = ca.vertcat(
        ca.horzcat(ca.cos(psi), -ca.sin(psi), 0.0),
        ca.horzcat(ca.sin(psi), ca.cos(psi), 0.0),
        ca.horzcat(0.0, 0.0, 1.0)
    )
    Vel_Mom = ca.mtimes(R_mat, ca.vertcat(u_val, v, r))

    dstate = ca.vertcat(X_acc[0], X_acc[1], X_acc[2], Vel_Mom[0], Vel_Mom[1], Vel_Mom[2])
    return dstate


def make_casadi_integrator(h, method="rk4", smooth=True, scale=1000.0, sym_type=ca.MX,
                           with_env=False, h_over_d=19.5):
    """
    Creates a CasADi Function mapping (state, control) -> next_state, or, when
    with_env=True, (state, control, current, wave_force) -> next_state.
    """
    state = sym_type.sym("state", 6)      # [u, v, r, x, y, psi]
    control = sym_type.sym("control", 2)  # [delta, rps]

    if with_env:
        current = sym_type.sym("current", 2)          # [vcx, vcy], earth-frame
        wave_force = sym_type.sym("wave_force", 3)     # [fx, fy, fn], body-frame
        current_arg = (current[0], current[1])
        wave_force_arg = (wave_force[0], wave_force[1], wave_force[2])
    else:
        current_arg = (0.0, 0.0)
        wave_force_arg = (0.0, 0.0, 0.0)

    u, v, r, x, y, psi = state[0], state[1], state[2], state[3], state[4], state[5]

    if method.lower() == "euler":
        dstate = MMG_Time_Derivative_casadi(state, control, current_arg, wave_force_arg,
                                            h_over_d=h_over_d, smooth=smooth, scale=scale)
        r_dot_a = dstate[2]
        nxt_state = state + h * dstate

    elif method.lower() == "rk4":
        K1_ = MMG_Time_Derivative_casadi(state, control, current_arg, wave_force_arg,
                                         h_over_d=h_over_d, smooth=smooth, scale=scale)
        r_dot_a = K1_[2]
        K1 = h * K1_[:3]
        k1 = h * ca.vertcat(u, v, r)

        state2 = ca.vertcat(
            u + K1[0] / 2.0,
            v + K1[1] / 2.0,
            r + K1[2] / 2.0,
            x + k1[0] / 2.0,
            y + k1[1] / 2.0,
            psi + k1[2] / 2.0
        )
        K2_ = MMG_Time_Derivative_casadi(state2, control, current_arg, wave_force_arg,
                                         h_over_d=h_over_d, smooth=smooth, scale=scale)
        K2 = h * K2_[:3]
        k2 = h * (ca.vertcat(u, v, r) + 0.5 * K1)

        state3 = ca.vertcat(
            u + K2[0] / 2.0,
            v + K2[1] / 2.0,
            r + K2[2] / 2.0,
            x + k2[0] / 2.0,
            y + k2[1] / 2.0,
            psi + k2[2] / 2.0
        )
        K3_ = MMG_Time_Derivative_casadi(state3, control, current_arg, wave_force_arg,
                                         h_over_d=h_over_d, smooth=smooth, scale=scale)
        K3 = h * K3_[:3]
        k3 = h * (ca.vertcat(u, v, r) + 0.5 * K2)

        state4 = ca.vertcat(
            u + K3[0],
            v + K3[1],
            r + K3[2],
            x + k3[0],
            y + k3[1],
            psi + k3[2]
        )
        K4_ = MMG_Time_Derivative_casadi(state4, control, current_arg, wave_force_arg,
                                         h_over_d=h_over_d, smooth=smooth, scale=scale)
        K4 = h * K4_[:3]
        k4 = h * (ca.vertcat(u, v, r) + 0.5 * K3)

        del_u_v_r = (1.0 / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4)
        del_k = (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        R = ca.vertcat(
            ca.horzcat(ca.cos(psi), -ca.sin(psi), 0.0),
            ca.horzcat(ca.sin(psi), ca.cos(psi), 0.0),
            ca.horzcat(0.0, 0.0, 1.0)
        )
        rotated_del_k = ca.mtimes(R, del_k)

        nxt_state = ca.vertcat(
            u + del_u_v_r[0],
            v + del_u_v_r[1],
            r + del_u_v_r[2],
            x + rotated_del_k[0],
            y + rotated_del_k[1],
            psi + rotated_del_k[2]
        )

    else:
        raise ValueError(f"Unknown integration method: {method}")

    inputs, input_names = [state, control], ["state", "control"]
    if with_env:
        inputs += [current, wave_force]
        input_names += ["current", "wave_force"]

    hd_str = str(h_over_d).replace('.', '_')
    return ca.Function(f"{method}_step_hd_{hd_str}", inputs, [nxt_state, r_dot_a], input_names, ["next_state", "r_dot_a"])


def run_straight_to_steady_state(integrator, dt, rps=18.2, accel_tol=1e-3, hold_time=3.0, max_time=180.0):
    """Runs delta=0, rps=`rps` open-loop from rest until u/v/r have all reached steady state."""
    hold_steps = max(1, int(round(hold_time / dt)))
    control = ca.DM([0.0, rps])
    state_casadi = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    hist = [np.array(state_casadi).flatten()]
    prev_uvr = hist[0][:3]
    consecutive = 0
    n_steps = int(round(max_time / dt))

    for _ in range(n_steps):
        state_casadi, _ = integrator(state_casadi, control)
        cur = np.array(state_casadi).flatten()
        hist.append(cur)

        accel = np.abs(cur[:3] - prev_uvr) / dt
        prev_uvr = cur[:3]

        if np.all(accel < accel_tol):
            consecutive += 1
            if consecutive >= hold_steps:
                break
        else:
            consecutive = 0

    hist = np.vstack(hist)
    straight_time = (len(hist) - 1) * dt
    return hist, straight_time


def simulate_turning_circle(rudder_deg, sim_time, dt, integrator, rps=18.2, rudder_rate_deg_s=30.0):
    """IMO MSC.137(76) turning circle: straight steady approach run, then step rudder."""
    hist_straight, straight_time = run_straight_to_steady_state(integrator, dt, rps=rps)
    state_casadi = ca.DM(hist_straight[-1])

    rudder_target = np.deg2rad(rudder_deg)
    rate_rad_s = np.deg2rad(rudder_rate_deg_s) if rudder_rate_deg_s is not None else None
    max_delta_step = rate_rad_s * dt if rate_rad_s is not None else None

    n_turn_steps = int(round(sim_time / dt))
    hist_turn = []
    delta_turn_deg = []
    cur_delta = 0.0

    for _ in range(n_turn_steps):
        if max_delta_step is not None:
            diff = rudder_target - cur_delta
            if abs(diff) <= max_delta_step:
                cur_delta = rudder_target
            else:
                cur_delta += np.sign(diff) * max_delta_step
        else:
            cur_delta = rudder_target

        control = ca.DM([cur_delta, rps])
        state_casadi, _ = integrator(state_casadi, control)
        hist_turn.append(np.array(state_casadi).flatten())
        delta_turn_deg.append(np.rad2deg(cur_delta))

    hist_casadi = np.vstack([hist_straight] + ([np.vstack(hist_turn)] if hist_turn else []))
    time_steps = np.arange(len(hist_casadi)) * dt
    delta_deg_hist = np.concatenate([np.zeros(len(hist_straight)), delta_turn_deg])

    return time_steps, hist_casadi, delta_deg_hist, straight_time


def compute_turning_params(hist_casadi, straight_time, dt, Lpp=2.902):
    """Standard IMO turning-circle maneuver parameters."""
    idx0 = int(round(straight_time / dt))
    x0, y0, psi0 = hist_casadi[idx0, 3], hist_casadi[idx0, 4], hist_casadi[idx0, 5]
    psi = hist_casadi[:, 5]
    dpsi = np.unwrap(psi - psi0)

    def find_idx(deg_target):
        target = np.deg2rad(deg_target)
        cond = np.abs(dpsi[idx0:]) >= target
        return idx0 + int(np.argmax(cond)) if np.any(cond) else None

    idx90 = find_idx(90.0)
    idx180 = find_idx(180.0)

    def to_local(x, y):
        dx, dy = x - x0, y - y0
        fwd = dx * np.cos(psi0) + dy * np.sin(psi0)
        rgt = -dx * np.sin(psi0) + dy * np.cos(psi0)
        return fwd, rgt

    advance = transfer = tactical_diameter = None
    if idx90 is not None:
        advance, transfer = to_local(hist_casadi[idx90, 3], hist_casadi[idx90, 4])
        transfer = abs(transfer)
    if idx180 is not None:
        _, rgt180 = to_local(hist_casadi[idx180, 3], hist_casadi[idx180, 4])
        tactical_diameter = abs(rgt180)

    n_tail = max(1, int(0.1 * (len(hist_casadi) - idx0)))
    u_ss = hist_casadi[-n_tail:, 0].mean()
    v_ss = hist_casadi[-n_tail:, 1].mean()
    r_ss = hist_casadi[-n_tail:, 2].mean()
    U_ss = np.hypot(u_ss, v_ss)
    steady_radius = U_ss / abs(r_ss) if abs(r_ss) > 1e-9 else np.nan
    steady_diameter = 2.0 * steady_radius
    advance_ratio = advance / Lpp if advance is not None else np.nan
    tactical_ratio = tactical_diameter / Lpp if tactical_diameter is not None else np.nan

    return dict(advance=advance, transfer=transfer, tactical_diameter=tactical_diameter,
                steady_radius=steady_radius, steady_diameter=steady_diameter,
                advance_ratio=advance_ratio, tactical_ratio=tactical_ratio,
                u_ss=u_ss, v_ss=v_ss, r_ss=r_ss, U_ss=U_ss)


if __name__ == "__main__":
    print("================================================================================")
    print("  INOUE GEOMETRIC EQUATIONS & SHALLOW-WATER MMG TURNING CIRCLE SIMULATION")
    print("================================================================================")

    dt = 0.05
    sim_time = 160.0
    rps = 18.2
    Lpp = 2.902

    # We evaluate both Starboard (+35 deg) and Port (-35 deg) turns across different water depths
    rudder_angles = [35.0, -35.0]
    depth_ratios = [19.5, 3.0, 2.0, 1.5, 1.2]

    results = {}

    fig, axes = plt.subplots(len(rudder_angles), 2, figsize=(14, 10))

    for row_idx, rudder_deg in enumerate(rudder_angles):
        direction_label = "Starboard (+35°)" if rudder_deg > 0 else "Port (-35°)"
        print(f"\n>>> Running {direction_label} Turning Circle Tests across Water Depths...")

        ax_traj = axes[row_idx, 0]
        ax_uvr = axes[row_idx, 1]

        colors = ['navy', 'dodgerblue', 'forestgreen', 'darkorange', 'crimson']

        for hd, color in zip(depth_ratios, colors):
            integrator = make_casadi_integrator(dt, method="rk4", smooth=True, h_over_d=hd)
            t_steps, hist, delta_hist, straight_time = simulate_turning_circle(
                rudder_deg, sim_time, dt, integrator, rps=rps
            )
            params = compute_turning_params(hist, straight_time, dt, Lpp=Lpp)
            results[(rudder_deg, hd)] = params

            depth_tag = f"h/d={hd} (Deep)" if hd == 19.5 else f"h/d={hd}"
            print(f"  [{depth_tag:<14}] Approach: {hist[int(straight_time/dt), 0]:.3f} m/s | "
                  f"Advance: {params['advance']:6.2f} m ({params['advance_ratio']:4.2f} L) | "
                  f"Tactical: {params['tactical_diameter']:6.2f} m ({params['tactical_ratio']:4.2f} L) | "
                  f"Steady R: {params['steady_radius']:6.2f} m ({params['steady_radius']/Lpp:4.2f} L) | "
                  f"Steady U: {params['U_ss']:.3f} m/s | r: {params['r_ss']:+.4f} rad/s")

            idx0 = int(round(straight_time / dt))
            x_rel = hist[idx0:, 3] - hist[idx0, 3]
            y_rel = hist[idx0:, 4] - hist[idx0, 4]
            t_rel = t_steps[idx0:] - straight_time

            # Trajectory plot (Y vs X) starting from rudder execute at (0,0)
            ax_traj.plot(y_rel, x_rel, color=color, label=f"h/d = {hd} (R = {params['steady_radius']/Lpp:.2f} L)", linewidth=1.8)

            # Yaw rate plot vs time starting from rudder execute (t >= 0)
            ax_uvr.plot(t_rel, np.rad2deg(hist[idx0:, 2]), color=color, label=f"h/d = {hd} (r = {np.rad2deg(params['r_ss']):.2f}°/s)", linewidth=1.6)

        ax_traj.scatter([0], [0], color='black', marker='x', s=60, zorder=5, label='Rudder Execute (0,0)')

        ax_traj.set_title(f"Trajectory: {direction_label} Turn", fontsize=11, fontweight='bold')
        ax_traj.set_xlabel("Y [m] (East)", fontsize=10)
        ax_traj.set_ylabel("X [m] (North)", fontsize=10)
        ax_traj.grid(True, linestyle="--", alpha=0.6)
        ax_traj.axis("equal")
        ax_traj.legend(loc="best", fontsize=9)

        ax_uvr.set_title(f"Yaw Rate vs Time: {direction_label} Turn", fontsize=11, fontweight='bold')
        ax_uvr.set_xlabel("Time from Rudder Execute [s]", fontsize=10)
        ax_uvr.set_ylabel("Yaw Rate r [deg/s]", fontsize=10)
        ax_uvr.grid(True, linestyle="--", alpha=0.6)
        ax_uvr.legend(loc="best", fontsize=9)

    plt.tight_layout()
    out_dir = "/mnt/0BF1C240574D9C37/BTP_NMPC_AVOIDER/nmpc_sim_logs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inoue_shallow_turning_circle.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"\nPlot successfully saved to: {out_path}")
