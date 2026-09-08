import math
import numpy as np
from demo.sim_vehicle import SimulatedHazardVehicle


class HazardSimulator:
    """
    Deterministic hazard simulator for Project Balerion demo.
    Wraps the main project's SimulatedHazardVehicle backend module.
    Default canonical crossing hazard: start=(38.0, 3.0), velocity=(-1.5, -0.6), dt=0.5.
    """

    def __init__(
        self,
        x: float = 38.0,
        y: float = 3.0,
        vx: float = -1.5,
        vy: float = -0.6,
        dt: float = 0.5,
    ):
        self.initial_x = x
        self.initial_y = y
        self.initial_vx = vx
        self.initial_vy = vy
        self.dt = dt
        self.tick = 0
        self.vehicle = SimulatedHazardVehicle(x=x, y=y, vx=vx, vy=vy)

    def reset(self):
        """Reset vehicle state to initial canonical scenario configuration."""
        self.tick = 0
        self.vehicle = SimulatedHazardVehicle(
            x=self.initial_x,
            y=self.initial_y,
            vx=self.initial_vx,
            vy=self.initial_vy,
        )

    def step(self):
        """Advance hazard vehicle by dt seconds."""
        self.vehicle.step(self.dt)
        self.tick += 1

    def get_state(self) -> dict:
        """Return vehicle state dict with trajectory prediction and heading."""
        st = self.vehicle.get_state()
        st["time"] = float(self.tick * self.dt)
        st["heading_rad"] = float(math.atan2(st["vy"], st["vx"]))
        
        # 3-second predicted path line
        t_future = np.arange(0.0, 3.1, 0.5)
        st["predicted_path"] = [
            (st["x"] + st["vx"] * tf, st["y"] + st["vy"] * tf)
            for tf in t_future
        ]
        return st
