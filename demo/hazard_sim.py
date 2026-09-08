"""
Hazard Simulation module (Track D / Integration).

Simulates dynamic hazard state {x, y, vx, vy} and tick-based position stepping
for interactive or scripted evaluation scenarios.

Mathematically Designed Default Scenario:
Start position: x0 = 40.0m, y0 = 5.0m
Velocity      : vx = -2.0 m/s (approaching ego), vy = -0.8 m/s (crossing corridor)
Time step dt  : 0.5s

Progression across risk levels:
- LOW      : t = 0.0s .. 3.0s (x = 40.0m .. 34.0m, y = 5.0m .. 2.6m, outside corridor)
- MEDIUM   : t = 3.5s (x = 33.0m, y = 2.2m, enters corridor)
- HIGH     : t = 4.0s .. 7.0s (x = 32.0m .. 26.0m, y = 1.8m .. -0.6m, inside corridor)
- CRITICAL : t = 7.5s .. 9.0s (x = 25.0m .. 22.0m, y = -1.0m .. -2.2m, inside corridor)
"""

import math


class HazardSimulator:
    """
    Deterministic dynamic hazard simulator for path-crossing vehicle or pedestrian scenarios.
    Exposes clean interface {x, y, vx, vy, active, tick} for consumption by risk engine and dashboard.
    """

    def __init__(
        self,
        x: float = 40.0,
        y: float = 5.0,
        vx: float = -2.0,
        vy: float = -0.8,
        dt: float = 0.5,
        active: bool = True,
    ):
        self.initial_x = float(x)
        self.initial_y = float(y)
        self.initial_vx = float(vx)
        self.initial_vy = float(vy)
        self.dt = float(dt)

        self.x = self.initial_x
        self.y = self.initial_y
        self.vx = self.initial_vx
        self.vy = self.initial_vy
        self.active = active
        self.tick = 0

    def start(self):
        """Activates the hazard simulation."""
        self.active = True

    def stop(self):
        """Deactivates/pauses the hazard simulation."""
        self.active = False

    def reset(self, x: float = None, y: float = None, vx: float = None, vy: float = None):
        """
        Resets hazard state back to initial values or custom specified parameters.
        """
        if x is not None:
            self.initial_x = float(x)
        if y is not None:
            self.initial_y = float(y)
        if vx is not None:
            self.initial_vx = float(vx)
        if vy is not None:
            self.initial_vy = float(vy)

        self.x = self.initial_x
        self.y = self.initial_y
        self.vx = self.initial_vx
        self.vy = self.initial_vy
        self.tick = 0
        self.active = True
        return self.get_state()

    def set_state(self, x: float, y: float, vx: float, vy: float):
        """Sets current hazard position and velocity explicitly."""
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        return self.get_state()

    def step(self, dt: float = None) -> dict:
        """
        Steps hazard position forward by time delta dt (using default self.dt if not passed).
        Returns updated hazard state dict.
        """
        if not self.active:
            return self.get_state()

        step_dt = float(dt) if dt is not None else self.dt

        self.x += self.vx * step_dt
        self.y += self.vy * step_dt
        self.tick += 1

        return self.get_state()

    def get_state(self) -> dict:
        """
        Returns clean, serializable dictionary of hazard state for risk engine and UI dashboard.
        """
        dist_to_ego = math.hypot(self.x, self.y)
        speed = math.hypot(self.vx, self.vy)
        in_corridor = abs(self.y) < 2.5 and (0.0 < self.x <= 40.0)

        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "active": self.active,
            "tick": self.tick,
            "time": round(self.tick * self.dt, 2),
            "dt": self.dt,
            "speed": round(speed, 3),
            "distance_to_ego": round(dist_to_ego, 3),
            "in_corridor": in_corridor,
        }

    def load_preset(self, preset_name: str = "default") -> dict:
        """
        Loads a pre-configured deterministic demo scenario.
        """
        presets = {
            "default": (40.0, 5.0, -2.0, -0.8, 0.5),      # Mathematically verified LOW -> MED -> HIGH -> CRITICAL
            "fast_crossing": (35.0, 6.0, -1.5, -2.0, 0.25),
            "corridor_head_on": (30.0, 0.5, -3.0, 0.0, 0.5),
            "pedestrian_near": (15.0, 3.5, -0.5, -1.0, 0.5),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Valid options: {list(presets.keys())}")

        x0, y0, vx, vy, dt = presets[preset_name]
        self.dt = dt
        return self.reset(x0, y0, vx, vy)


def mock_risk_evaluator(state: dict) -> tuple[float, str]:
    """
    Helper evaluation formula from Balerion execution plan to verify risk progression.
    Outside corridor (|y| >= 2.5m): capped at 0.58
    Inside corridor (|y| < 2.5m, 0 < x <= 40): full progression to 1.0
    """
    x, y = state["x"], state["y"]
    vx, vy = state["vx"], state["vy"]
    dist = state["distance_to_ego"]

    if abs(y) >= 2.5:
        base_risk = max(0.0, min(1.0, 1.0 - dist / 50.0))
        closing_lat_speed = -vy * (1.0 if y > 0 else -1.0)
        closing_factor = 0.2 if closing_lat_speed <= 0 else min(1.2, closing_lat_speed / 1.8)
        lat_factor = max(0.2, min(1.0, 1.0 - max(0.0, abs(y) - 2.5) / 8.0))
        final_risk = min(base_risk * closing_factor * lat_factor, 0.58)
    else:
        base_risk = max(0.0, min(1.0, 1.0 - dist / 50.0))
        vx_toward = -vx if vx < 0 else 0.0
        closing_factor = 1.0 + vx_toward / 4.0
        final_risk = max(0.0, min(1.0, base_risk * 1.15 * closing_factor))

    if final_risk < 0.35:
        level = "LOW"
    elif final_risk < 0.60:
        level = "MEDIUM"
    elif final_risk < 0.85:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return round(final_risk, 3), level


if __name__ == "__main__":
    print("=" * 70)
    print("      BALERION HAZARD SIMULATOR DEMO — DETERMINISTIC SCENARIO TEST")
    print("=" * 70)

    sim = HazardSimulator()
    state = sim.get_state()

    print(f"Initial Setup: x0={state['x']}m, y0={state['y']}m, vx={state['vx']}m/s, vy={state['vy']}m/s, dt={state['dt']}s\n")
    print(f"{'Tick':<6}{'Time':<8}{'X (m)':<8}{'Y (m)':<8}{'Dist (m)':<10}{'In Corridor':<13}{'Risk':<8}{'Level':<10}")
    print("-" * 70)

    for _ in range(20):
        st = sim.get_state()
        risk, level = mock_risk_evaluator(st)
        in_corr_str = "YES" if st["in_corridor"] else "NO"
        print(f"{st['tick']:<6}{st['time']:<8.1f}{st['x']:<8.2f}{st['y']:<8.2f}{st['distance_to_ego']:<10.2f}{in_corr_str:<13}{risk:<8.3f}{level:<10}")
        sim.step()

    print("-" * 70)
    print("Demo completed successfully. Resetting simulator...")
    sim.reset()
    print(f"Reset State: x={sim.x}m, y={sim.y}m, tick={sim.tick}")
