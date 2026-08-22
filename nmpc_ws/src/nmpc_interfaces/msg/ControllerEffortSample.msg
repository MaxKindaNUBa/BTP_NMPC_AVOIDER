# One raw sample per accepted /nmpc/solve call, published by nmpc_node purely
# so controller-effort/cost-breakdown metrics can be computed OUTSIDE the NMPC
# control loop (see logger_node.py + controller_effort_logger.py). Deliberately
# carries only raw numbers nmpc_node already has in hand at zero extra solver
# cost -- no metric here is computed in the control loop; J_u, J_du, TV,
# saturation stats, peak/RMS, and the Q/R cost breakdown are all derived
# downstream from these fields.
std_msgs/Header header

uint64 step             # monotonically increasing solve counter (nmpc_node-owned; more reliable for ordering than stamp alone under QoS drops)
float64 dt               # this solve's timestep [s] (nmpc_node.config.dt)

float64[] u              # applied control this step (CONTROL_DIM,): u_opt[:,0], the augmented rate command [delta_dot, n_dot] actually integrated into (delta, n)
float64[] x              # augmented state used as xi_0 for this solve (STATE_DIM,): xi_traj[:,0]
float64[] x_ref          # augmented reference state used in this solve's stage cost (STATE_DIM,), unwrapped (psi row = chi_p, not zeroed)

float64 solve_time       # solver wall time [s], echoes SolverStatus.solve_time
bool success              # echoes SolverStatus.success

uint64 upstream_drops    # cumulative count of samples nmpc_node's internal queue has dropped (queue.Full backpressure) since startup -- lets the downstream logger report data loss without nmpc_node ever doing extra I/O on the solve thread to announce it
