string id
float64[] x
float64[] y
float64[] psi   # optional live/predicted heading, same length as x/y when present;
                # empty for a producer with no heading to report (e.g. a plain
                # CV circle/wall) -- see nmpc/README.md's moving-obstacles note
float64 a       # semi-major axis (or radius for circle)
float64 b       # semi-minor axis
