string id
float64 x0
float64 y0
float64 x1
float64 y1
float64 radius
float64 vx        # constant-velocity, earth-frame [m/s] -- the WHOLE wall translates
                   # rigidly (both endpoints move by the same vx/vy); 0.0 = stationary
float64 vy         # see research_papers/COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md --
                    # prediction/visualization only, NEVER read by the NMPC solver
