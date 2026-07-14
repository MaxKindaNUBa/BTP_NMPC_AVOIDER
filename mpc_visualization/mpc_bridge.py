
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import threading

@dataclass
class Obstacle:
    id: str
    x: float
    y: float
    radius: float
    static: bool = True

@dataclass
class MPCBridgeState:
    ship_state: np.ndarray            # [u, v, r, x, y, psi]
    timestamp: float
    predicted_trajectory: np.ndarray  # shape (N+1, 6), full horizon states
    control_horizon: np.ndarray       # shape (N, 2), [delta, rps] over horizon
    reference_path: np.ndarray        # (M, 2) waypoints being tracked, optional
    obstacles: List[Obstacle]
    start: Tuple[float, float]
    goal: Tuple[float, float]

class MPCBridge:
    """
    Thread-safe shared state between the NMPC solver and the Visualizer.
    This acts as a clean data bridge.
    """
    def __init__(self, start: Tuple[float, float] = (0.0, 0.0), goal: Tuple[float, float] = (50.0, 30.0)):
        self._lock = threading.Lock()

        # Initialize default state
        self._ship_state = np.array([0.78, 0.0, 0.0, start[0], start[1], 0.0]) # [u, v, r, x, y, psi]
        self._timestamp = 0.0
        self._predicted_trajectory = np.zeros((1, 6))
        self._control_horizon = np.zeros((1, 2))
        self._reference_path = np.array([start, goal])
        self._obstacles: List[Obstacle] = []
        self._start = start
        self._goal = goal

    def update_ship_state(self, state: np.ndarray, t: float):
        with self._lock:
            self._ship_state = np.array(state, dtype=np.float64)
            self._timestamp = float(t)

    def update_prediction(self, predicted_trajectory: np.ndarray, control_horizon: np.ndarray):
        with self._lock:
            self._predicted_trajectory = np.array(predicted_trajectory, dtype=np.float64)
            self._control_horizon = np.array(control_horizon, dtype=np.float64)

    def set_obstacles(self, obstacles: List[Obstacle]):
        with self._lock:
            self._obstacles = list(obstacles)

    def set_start_goal(self, start: Tuple[float, float], goal: Tuple[float, float]):
        with self._lock:
            self._start = start
            self._goal = goal
            self._reference_path = np.array([start, goal])

    def get_obstacles_relative(self, ship_x: float, ship_y: float, ship_psi: float) -> List[Dict]:
        """
        Returns each obstacle as {id, range, bearing, radius} in the ship's body frame.
        This represents the standard interface that LiDAR or sensory inputs would map to.
        """
        with self._lock:
            obstacles_copy = list(self._obstacles)

        relative_obstacles = []
        for obs in obstacles_copy:
            dx = obs.x - ship_x
            dy = obs.y - ship_y

            # Rotate into body frame (x_b is forward, y_b is port/starboard)
            xb = dx * np.cos(ship_psi) + dy * np.sin(ship_psi)
            yb = -dx * np.sin(ship_psi) + dy * np.cos(ship_psi)

            r_val = np.sqrt(xb**2 + yb**2)
            bearing = np.arctan2(yb, xb)

            relative_obstacles.append({
                "id": obs.id,
                "range": float(r_val),
                "bearing": float(bearing),
                "radius": float(obs.radius)
            })
        return relative_obstacles

    def snapshot(self) -> MPCBridgeState:
        """
        Returns an immutable copy of the current state for the visualizer or logger.
        """
        with self._lock:
            return MPCBridgeState(
                ship_state=self._ship_state.copy(),
                timestamp=self._timestamp,
                predicted_trajectory=self._predicted_trajectory.copy(),
                control_horizon=self._control_horizon.copy(),
                reference_path=self._reference_path.copy(),
                obstacles=list(self._obstacles),
                start=self._start,
                goal=self._goal
            )
