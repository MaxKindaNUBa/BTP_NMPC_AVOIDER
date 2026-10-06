string id
float64 x
float64 y
float64 radius
float64 vx        # constant-velocity, earth-frame [m/s]; 0.0 = stationary (the default
                   # for every obstacle placed before this field existed)
float64 vy         # see research_papers/COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md --
                    # prediction/visualization only, NEVER read by the NMPC solver
