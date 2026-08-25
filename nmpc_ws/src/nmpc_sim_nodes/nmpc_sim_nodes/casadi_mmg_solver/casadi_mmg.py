import os
import ctypes

# Set ACADOS_SOURCE_DIR programmatically
os.environ["ACADOS_SOURCE_DIR"] = "/home/chandran/acados"

# Pre-load acados shared libraries to bypass LD_LIBRARY_PATH requirement on Linux
acados_lib_dir = "/home/chandran/acados/lib"
if os.path.exists(acados_lib_dir):
    try:
        # Load in topological dependency order
        mode = ctypes.RTLD_GLOBAL
        ctypes.CDLL(os.path.join(acados_lib_dir, "libqdldl.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libosqp.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libqpOASES_e.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libblasfeo.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libhpipm.so"), mode=mode)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libacados.so"), mode=mode)
    except Exception as e:
        print(f"Warning: programmatically loading acados libraries failed: {e}")

import casadi as ca
import numpy as np


def MMG_Time_Derivative_casadi(state, control, current=(0.0, 0.0), wave_force=(0.0, 0.0, 0.0),
                                smooth=True, scale=1000.0):
    """
    CasADi implementation of MMG_Time_Derivative.

    Parameters:
    -----------
    state : ca.MX or ca.SX
        Current state of the ship [u, v, r, x, y, psi]
    control : ca.MX or ca.SX
        Control inputs [delta, rps] (rudder angle in rad, propeller speed in rps)
    current : (float, float) or ca.MX/ca.SX pair, default (0.0, 0.0)
        Earth-frame current velocity (vcx, vcy) [m/s]. Enters only the
        hydrodynamic force terms below (hull/propeller/rudder) via relative
        velocity substitution -- the mass matrix, LHS_r, and kinematics block
        stay in terms of the state's own (absolute) u_val, v, r. Standard
        MMG-plus-current approximation; default is disturbance-free.
    wave_force : (float, float, float) or ca.MX/ca.SX triple, default (0.0, 0.0, 0.0)
        Body-frame (surge, sway, yaw) second-order wave-drift force/moment
        [N, N, N*m], added directly into the RHS before the M^-1 solve.
        Default is disturbance-free (replaces the old always-zero F_Drift).
    smooth : bool, default True
        If True, uses smooth tanh blending for continuous functions.
        If False, uses ca.if_else for exact NumPy behavior replication.
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
        rps_val = ca.fmax(rps, 1.0)
    else:
        u_val = ca.if_else(u <= 0, 0.00001, u)
        rps_val = ca.if_else(rps <= 0, 1.0, rps)

    # Current: earth-frame (vcx, vcy) -> body-frame (uc, vc) via the state's
    # own psi, then relative velocity (ur, vr) substituted for (u_val, v) in
    # every hydrodynamic quantity below that depends on flow past the hull
    # (U/beta, v_ndm/r_ndm, beta_P/Jp, beta_R/uR). u_val, v themselves stay
    # untouched in the mass matrix/LHS_r/kinematics further down.
    vcx, vcy = current[0], current[1]
    uc = vcx * ca.cos(psi) + vcy * ca.sin(psi)
    vc = -vcx * ca.sin(psi) + vcy * ca.cos(psi)
    ur = u_val - uc
    vr = v - vc

    U             = ca.sqrt((ur**2)+(vr**2))     # Resultant speed, relative to water

    beta          = ca.atan2(-vr, ur)            # Hull drift angle at midship, relative
    Np            = rps_val                      # Just for notation

    ################################################
    ###### Principle Particulars of the ship ########
    ################################################
    Lpp     = 2.902        # length between perpendicular
    d       = 0.189        # ship draft
    xG      = 0.102        # Longitudinal coordinate of center of gravity of ship
    rho     = 1025         # Water density
    Volume  = 0.235        # Displacement volume of ship
    Dp      = 0.090        # Propeller diameter
    HR      = 0.144        # Rudder span length
    AR      = 0.00928      # Profile area of movable part of mariner rudder

    ######################################################
    #### Mass, Added Mass and Added Moment of inertia ####
    ######################################################
    mx, my, jz = 0.022, 0.223, 0.011

    ### Non Dimensionalizing(ndm) parameters ###
    ndm_force   = 0.5*rho*Lpp*d*(U**2)
    ndm_moment  = 0.5*rho*(Lpp**2)*d*(U**2)
    ndm_mass    = 0.5*rho*d*(Lpp**2)
    ndm_massMoI = 0.5*rho*d*(Lpp**4)

    ### Dimensional mass ###
    mx = mx*ndm_mass
    my = my*ndm_mass
    jz = jz*ndm_massMoI
    m  = Volume*rho
    IzG = m * ((0.25*Lpp)**2) # Moment of inertia of ship around center of gravity

    #####################################################
    ####### Governing Equation (Equation 4) #############
    #####################################################

    ### Mass Matrix ### (u_dot,v_dot,r_dot)
    M = ca.vertcat(
        ca.horzcat(m + mx, 0.0, 0.0),
        ca.horzcat(0.0, m + my, m * xG),
        ca.horzcat(0.0, m * xG, jz + ((xG**2)*m) + IzG)
    )

    ### LHS ### (Other than u_dot,v_dot,r_dot)
    # Split by physical origin, not lumped at a single velocity: the rigid-body
    # Coriolis/centripetal piece (mass m) is a purely kinematic consequence of
    # differentiating momentum in a frame rotating with the hull -- it exists
    # with no water at all, so it uses absolute (ground-relative) velocity,
    # same as Vel_Mom's kinematics below. The added-mass piece (mx, my) is a
    # hydrodynamic fluid-reaction term -- it uses water-relative velocity
    # (ur, vr), same convention as F_hull/F_propellar/F_rudder above. With no
    # current (ur=u_val, vr=v) this reduces exactly to the previous lumped form.
    LHS_r = ca.vertcat(
        -m*v*r - xG*m*(r**2) - my*vr*r,
        m*u_val*r + mx*ur*r,
        m*xG*u_val*r
    )

    #####################################################
    ####### Hydrodynamic Force on Hull(F_hull)  #########
    #####################################################
    v_ndm = vr/U
    r_ndm = r*Lpp/U

    R0 = 0.022 # Ship resistance coefficient in straight moving

    X_vv, X_vr ,X_rr ,X_vvvv = -0.04, 0.002, 0.011, 0.771
    Y_v, Y_R, Y_vvv, Y_vvr, Y_vrr, Y_rrr = -0.315, 0.083, -1.607, 0.379, -0.391, 0.008
    N_v, N_R, N_vvv, N_vvr, N_vrr, N_rrr = -0.137, -0.049, -0.03, -0.294, 0.055, -0.013

    X_hull = -R0 + (X_vv*(v_ndm**2)) + (X_vr*v_ndm*r_ndm) + (X_rr*(r_ndm**2)) + (X_vvvv*(v_ndm**4))
    Y_hull = (Y_v*v_ndm) + (Y_R*r_ndm) + (Y_vvv*(v_ndm**3)) + (Y_vvr*(v_ndm**2)*r_ndm) + (Y_vrr*v_ndm*(r_ndm**2)) + (Y_rrr*(r_ndm**3))
    N_hull = (N_v*v_ndm) + (N_R*r_ndm) + (N_vvv*(v_ndm**3)) + (N_vvr*(v_ndm**2)*r_ndm) + (N_vrr*v_ndm*(r_ndm**2)) + (N_rrr*(r_ndm**3))

    X_hull = ndm_force * X_hull
    Y_hull = ndm_force * Y_hull
    N_hull = ndm_moment * N_hull

    F_hull = ca.vertcat(X_hull, Y_hull, N_hull)

    #####################################################
    ##### Propeller Force on Ship(F_propeller)  #########
    #####################################################
    tp          = 0.220                    # Thrust deduction factor
    wp0         = 0.40                     # Effective wake in straight moving
    k0,k1,k2    = 0.2931, -0.2753, -0.1385 # 2nd order polynomial function coefficient
    xP          = -0.48                    # Longitudinal coordinate of propeller position
    beta_P      = beta - (xP * r_ndm)      # geometrical inflow angle

    wp          = wp0*ca.exp(-4*(beta_P**2))   # Equation 12
    Jp          = ur*(1-wp)/(Np*Dp)            # Equation 11
    KT          = k0 + k1*Jp + k2*(Jp**2)      # Equation 10
    T           = rho*(Np**2)*(Dp**4)*KT       # Equation 9
    X_propellar = (1 - tp)*T                   # Equation 8

    F_propellar = ca.vertcat(X_propellar, 0.0, 0.0)

    #####################################################
    ######### Rudder Force on Ship(F_rudder)  ###########
    #####################################################
    epsilon    = 1.09                   # Ratio of wake fraction
    k          = 0.5                    # An experimental constant for expressing uR
    eta        = Dp/HR                  # Ratio of propeller diameter to rudder span
    uP         = (1 - wp)*ur            # propeller inflow velocity

    uR1   = ca.sqrt(1 + ((8*KT)/(ca.pi * (Jp**2))))
    uR2   = (1 + k*(uR1 - 1))**2
    uR    = epsilon * uP * ca.sqrt((eta*uR2) + (1-eta))

    lR          = -0.710                        # Effective longitudinal coordinate
    beta_R      = beta - (lR*r_ndm)             # Effective inflow angle to rudder

    if smooth:
        gamma_r = 0.396 + (0.64 - 0.396) * (1.0 + ca.tanh(scale * beta_R)) / 2.0
    else:
        gamma_r = ca.if_else(beta_R < 0, 0.396, 0.64)

    vR = U * gamma_r * beta_R

    f_alpha     = 2.747                     # Rudder lift gradient coefficient
    alpha_R     = delta - ca.atan2(vR, uR)  # Effective inflow angle to rudder
    Ur          = ca.sqrt((uR**2)+(vR**2))  # Resultant inflow velocity to rudder

    F_normal = 0.5 * rho * AR * (Ur**2) * f_alpha * ca.sin(alpha_R)

    tR         = 0.387                      # Steering resistance deduction factor
    aH         = 0.312                      # Rudder force increase factor
    xH         = -0.464                     # Longitudinal coordinate of additional lateral force
    xR         = -0.5 * Lpp                 # Longitudinal coordinate of rudder position

    X_rudder = -(1 - tR) * F_normal * ca.sin(delta)
    Y_rudder = -(1 + aH) * F_normal * ca.cos(delta)
    N_rudder = -(xR + (aH*xH)) * F_normal * ca.cos(delta)

    F_rudder = ca.vertcat(X_rudder, Y_rudder, N_rudder)

    F_Drift = ca.vertcat(*wave_force)

    # Solve dynamics
    RHS = F_hull + F_propellar + F_rudder - LHS_r + F_Drift
    X_acc = ca.solve(M, RHS)

    #####################################################
    ############## Kinetics of the Ship   ###############
    #####################################################
    R_mat = ca.vertcat(
        ca.horzcat(ca.cos(psi), -ca.sin(psi), 0.0),
        ca.horzcat(ca.sin(psi), ca.cos(psi), 0.0),
        ca.horzcat(0.0, 0.0, 1.0)
    )
    Vel_Mom = ca.mtimes(R_mat, ca.vertcat(u_val, v, r))

    dstate = ca.vertcat(X_acc[0], X_acc[1], X_acc[2], Vel_Mom[0], Vel_Mom[1], Vel_Mom[2])
    return dstate

def make_casadi_integrator(h, method="rk4", smooth=True, scale=1000.0, sym_type=ca.MX, with_env=False):
    """
    Creates a CasADi Function mapping (state, control) -> next_state, or, when
    with_env=True, (state, control, current, wave_force) -> next_state --
    current/wave_force held constant across the RK4 substeps, the same
    convention already used for control within one integration step. Default
    (with_env=False) leaves the exported Function's input signature exactly
    as before -- every existing call site is unaffected.
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
        dstate = MMG_Time_Derivative_casadi(state, control, current_arg, wave_force_arg, smooth, scale)
        r_dot_a = dstate[2]
        nxt_state = state + h * dstate

    elif method.lower() == "rk4":
        K1_ = MMG_Time_Derivative_casadi(state, control, current_arg, wave_force_arg, smooth, scale)
        r_dot_a = K1_[2]
        K1 = h * K1_[:3]
        k1 = h * ca.vertcat(u, v, r)

        state2 = ca.vertcat(
            u + K1[0]/2.0,
            v + K1[1]/2.0,
            r + K1[2]/2.0,
            x + k1[0]/2.0,
            y + k1[1]/2.0,
            psi + k1[2]/2.0
        )
        K2_ = MMG_Time_Derivative_casadi(state2, control, current_arg, wave_force_arg, smooth, scale)
        K2 = h * K2_[:3]
        k2 = h * (ca.vertcat(u, v, r) + 0.5 * K1)

        state3 = ca.vertcat(
            u + K2[0]/2.0,
            v + K2[1]/2.0,
            r + K2[2]/2.0,
            x + k2[0]/2.0,
            y + k2[1]/2.0,
            psi + k2[2]/2.0
        )
        K3_ = MMG_Time_Derivative_casadi(state3, control, current_arg, wave_force_arg, smooth, scale)
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
        K4_ = MMG_Time_Derivative_casadi(state4, control, current_arg, wave_force_arg, smooth, scale)
        K4 = h * K4_[:3]
        k4 = h * (ca.vertcat(u, v, r) + 0.5 * K3)

        del_u_v_r = (1.0/6.0) * (K1 + 2.0*K2 + 2.0*K3 + K4)
        del_k = (1.0/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

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

    return ca.Function(f"{method}_step", inputs, [nxt_state, r_dot_a], input_names, ["next_state", "r_dot_a"])

def make_casadi_accel_function(sym_type=ca.MX):
    """(state, control, current, wave_force) -> accel [u_dot, v_dot], body-frame.

    Exposes MMG_Time_Derivative_casadi's otherwise-discarded X_acc[0:2] -- the
    surge/sway acceleration terms that make_casadi_integrator computes
    internally on the way to `dstate` but never returns (its second output,
    r_dot_a, is only the yaw-rate acceleration). Used to synthesize an IMU
    accelerometer reading: mmg_node evaluates this at the true state/control/
    current/wave_force each tick to get the "true" (ax, ay) an accelerometer
    would sense; ukf_node's measurement model evaluates it at each sigma
    point's predicted state to get the corresponding nonlinear ax/ay rows.
    Purely additive -- does not touch MMG_Time_Derivative_casadi or
    make_casadi_integrator's existing signatures/behavior.
    """
    state = sym_type.sym("state", 6)
    control = sym_type.sym("control", 2)
    current = sym_type.sym("current", 2)
    wave_force = sym_type.sym("wave_force", 3)
    dstate = MMG_Time_Derivative_casadi(state, control, (current[0], current[1]),
                                         (wave_force[0], wave_force[1], wave_force[2]))
    return ca.Function("mmg_accel", [state, control, current, wave_force], [dstate[:2]],
                        ["state", "control", "current", "wave_force"], ["accel"])


def make_acados_integrator(h, smooth=True, scale=1000.0):
    """
    Creates an acados SimSolver for the MMG model using RK4 (explicit Runge-Kutta).

    Parameters:
    -----------
    h : float
        Integration time step (s)
    smooth : bool, default True
        If True, uses smooth tanh blending for continuous functions.
    scale : float, default 1000.0
        Steepness of the tanh transition.

    Returns:
    --------
    integrator_fun : function
        A function mapping (state, control) -> (next_state, r_dot_a)
    """
    import os
    from acados_template import AcadosModel, AcadosSim, AcadosSimSolver

    # Ensure environment variables are set
    os.environ["ACADOS_SOURCE_DIR"] = "/home/chandran/acados"
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/home/chandran/acados/lib"

    model = AcadosModel()
    model.name = f"vessel_mmg_acados_{'smooth' if smooth else 'exact'}"

    # States and controls
    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 2)

    # Dynamics (explicit ODE)
    f_expl = MMG_Time_Derivative_casadi(x, u, smooth=smooth, scale=scale)

    model.f_expl_expr = f_expl
    model.x = x
    model.u = u

    sim = AcadosSim()
    sim.model = model

    # Set RK4 options
    sim.solver_options.T = h
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1

    # Clean output directory config
    sim.code_export_directory = os.path.abspath(f"c_generated_code_sim_{'smooth' if smooth else 'exact'}")

    sim_solver = AcadosSimSolver(sim)

    # CasADi function to evaluate derivatives (e.g. for r_dot_a)
    f_expl_fun = ca.Function('f_expl_fun', [x, u], [f_expl])

    def integrator_fun(state_in, control_in):
        x0 = np.array(state_in).flatten()
        u0 = np.array(control_in).flatten()

        sim_solver.set("x", x0)
        sim_solver.set("u", u0)
        sim_solver.solve()

        nxt_state = sim_solver.get("x")

        # Evaluate derivative for r_dot_a (3rd element, i.e. r_dot)
        dstate0 = f_expl_fun(x0, u0)
        r_dot_a = float(dstate0[2])

        return ca.DM(nxt_state), r_dot_a

    return integrator_fun

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # Ensure environment variables are set for acados
    os.environ["ACADOS_SOURCE_DIR"] = "/home/chandran/acados"
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/home/chandran/acados/lib"

    print("==============================================================")
    print("Running standalone CasADi vs Acados SimSolver Comparison...")
    print("==============================================================")

    sim_time = 200.0
    dt = 0.05
    time_steps = np.arange(0, sim_time, dt)

    # 1. Instantiate both integrators
    f_casadi = make_casadi_integrator(dt, method="rk4", smooth=True)
    f_acados = make_acados_integrator(dt, smooth=True)

    # Initial state: [u, v, r, x, y, psi]
    state_casadi = ca.DM([0.78, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_acados = ca.DM([0.78, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Constant control: rudder 35 deg, propeller 18.2 rps
    control = ca.DM([np.deg2rad(35.0), 18.2])

    hist_casadi_x, hist_casadi_y, hist_casadi_r = [], [], []
    hist_acados_x, hist_acados_y, hist_acados_r = [], [], []

    print("Running simulations...")
    for t in time_steps:
        # CasADi histories
        hist_casadi_x.append(float(state_casadi[3]))
        hist_casadi_y.append(float(state_casadi[4]))
        hist_casadi_r.append(float(state_casadi[2]))

        # Acados histories
        hist_acados_x.append(float(state_acados[3]))
        hist_acados_y.append(float(state_acados[4]))
        hist_acados_r.append(float(state_acados[2]))

        # Step both dynamics
        state_casadi, _ = f_casadi(state_casadi, control)
        state_acados, _ = f_acados(state_acados, control)

    print("Simulations completed successfully. Calculating differences...")

    diff_x = np.abs(np.array(hist_casadi_x) - np.array(hist_acados_x))
    diff_y = np.abs(np.array(hist_casadi_y) - np.array(hist_acados_y))
    diff_r = np.abs(np.array(hist_casadi_r) - np.array(hist_acados_r))

    print(f"Max absolute trajectory differences between CasADi RK4 and Acados SimSolver:")
    print(f"  x: {np.max(diff_x):e} m")
    print(f"  y: {np.max(diff_y):e} m")
    print(f"  r: {np.max(diff_r):e} rad/s")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist_casadi_y, hist_casadi_x, 'b-', label='CasADi RK4', linewidth=2)
    plt.plot(hist_acados_y, hist_acados_x, 'r--', label='Acados SimSolver', linewidth=1.5)
    plt.scatter(0, 0, marker='P', color='green', s=100, label='Start')
    plt.title('Ship Trajectory (Turning Circle)', fontsize=12, fontweight='bold')
    plt.xlabel('Y Coordinate (m)', fontsize=10)
    plt.ylabel('X Coordinate (m)', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.plot(time_steps, hist_casadi_r, 'b-', label='CasADi RK4', linewidth=2)
    plt.plot(time_steps, hist_acados_r, 'r--', label='Acados SimSolver', linewidth=1.5)
    plt.title('Yaw Rate vs Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Yaw Rate r (rad/s)', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plot_path = "casadi_turning_circle_test.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved to: {os.path.abspath(plot_path)}")
