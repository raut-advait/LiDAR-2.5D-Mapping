import numpy as np


class SimulatedHazardVehicle:
    """Simulated hazard vehicle state for BEV demonstration."""

    def __init__(
        self,
        x: float = 38.0,
        y: float = 3.0,
        vx: float = -1.5,
        vy: float = -0.6,
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def step(self, dt: float = 0.5):
        """Advance vehicle state by dt seconds."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def get_state(self) -> dict:
        """Return current vehicle state and risk corridor check."""
        dist = float(np.sqrt(self.x**2 + self.y**2))
        in_corridor = bool(abs(self.y) < 2.5 and 0.0 < self.x <= 40.0)
        return {
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "distance": dist,
            "in_corridor": in_corridor,
        }
