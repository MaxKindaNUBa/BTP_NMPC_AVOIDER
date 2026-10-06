string id
float64 x
float64 y
float64 a
float64 b
float64 theta
float64 vx        # constant-velocity, earth-frame [m/s]; 0.0 = stationary. theta (the
                   # ellipse's own orientation) is independent of vx/vy -- setting a
                   # velocity does NOT rotate the ellipse to face its heading of travel.
float64 vy         # see research_papers/COLREGS_AWARE_NMPC_MOVING_OBSTACLES.md --
                    # prediction/visualization only, NEVER read by the NMPC solver
