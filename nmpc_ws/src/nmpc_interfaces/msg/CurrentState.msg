std_msgs/Header header
float64 vx        # earth-frame current velocity x-component [m/s]
float64 vy        # earth-frame current velocity y-component [m/s]
float64 speed     # derived: sqrt(vx^2+vy^2) [m/s], for convenience
float64 heading   # derived: atan2(vy,vx) [rad], for convenience
bool enabled       # echoes env_node's current_enabled param
