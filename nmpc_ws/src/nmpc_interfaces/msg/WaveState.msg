std_msgs/Header header
float64 hs             # significant wave height [m], scaled-model units
float64 tp             # peak period [s]
float64 mean_heading   # earth-fixed mean wave direction [rad]
float64 fx             # body-frame surge drift force applied this tick [N]
float64 fy             # body-frame sway drift force applied this tick [N]
float64 fn             # body-frame yaw drift moment applied this tick [N*m]
bool enabled            # echoes env_node's wave_enabled param
