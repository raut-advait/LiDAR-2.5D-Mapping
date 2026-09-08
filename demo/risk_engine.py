"""
Risk Engine module (Track D / Integration - Owned by Advait).

Calculates risk score (0-1), risk level (LOW/MEDIUM/HIGH/CRITICAL),
local resolution refinement triggers, and safety actions (PROCEED/SLOW DOWN/BRAKE).
"""

import math
import numpy as np

from perception.cells import refine_local_region


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamps a floating point value to [min_val, max_val]."""
    return max(min_val, min(max_val, val))


def classify_risk(risk_score: float) -> str:
    """
    Classifies risk score into discrete levels:
    - LOW      : [0, 0.35)
    - MEDIUM   : [0.35, 0.60)
    - HIGH     : [0.60, 0.85)
    - CRITICAL : [0.85, 1.0]
    """
    if risk_score < 0.35:
        return "LOW"
    elif risk_score < 0.60:
        return "MEDIUM"
    elif risk_score < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


def get_safety_action(risk_level: str) -> str:
    """
    Maps risk level to safety response action:
    - LOW / MEDIUM -> PROCEED
    - HIGH         -> SLOW DOWN
    - CRITICAL     -> BRAKE
    """
    if risk_level in ("LOW", "MEDIUM"):
        return "PROCEED"
    elif risk_level == "HIGH":
        return "SLOW DOWN"
    elif risk_level == "CRITICAL":
        return "BRAKE"
    return "PROCEED"


def compute_ttc(hazard_state: dict) -> float | str:
    """
    Calculates path-relative Time To Collision (TTC).

    1. Before corridor entry (|y| >= 2.5m):
       Calculates estimated time until hazard laterally reaches corridor boundary (|y| = 2.5m).
       If moving away or not closing laterally, returns "SAFE".

    2. Inside corridor (|y| < 2.5m and 0 < x <= 40):
       Calculates forward TTC based on closing x speed (-vx if vx < 0 else 0).
       If closing toward ego (closing_x_speed > 0): ttc = x / closing_x_speed.
       Otherwise returns "SAFE".
    """
    x = float(hazard_state.get("x", 0.0))
    y = float(hazard_state.get("y", 0.0))
    vx = float(hazard_state.get("vx", 0.0))
    vy = float(hazard_state.get("vy", 0.0))

    is_in_corridor = (abs(y) < 2.5) and (0.0 < x <= 40.0)

    if is_in_corridor:
        closing_x_speed = -vx if vx < 0.0 else 0.0
        if closing_x_speed > 1e-6 and x > 0.0:
            ttc = x / closing_x_speed
            return round(float(ttc), 3)
        return "SAFE"
    else:
        # Outside corridor (|y| >= 2.5m or x not in (0, 40])
        if abs(y) >= 2.5:
            if y >= 2.5:
                dist_y = y - 2.5
                closing_y_speed = -vy
            else:  # y <= -2.5
                dist_y = -2.5 - y
                closing_y_speed = vy

            if closing_y_speed > 1e-6 and dist_y >= 0.0:
                ttc = dist_y / closing_y_speed
                return round(float(ttc), 3)
            return "SAFE"
        else:
            # |y| < 2.5 but x <= 0 or x > 40
            if x > 40.0:
                closing_x_speed = -vx if vx < 0.0 else 0.0
                if closing_x_speed > 1e-6:
                    ttc = (x - 40.0) / closing_x_speed
                    return round(float(ttc), 3)
            return "SAFE"


class RiskEngine:
    """Evaluates real-time hazard risk scores and safety actions based on exact corridor formulas."""

    def evaluate(self, hazard_state: dict) -> dict:
        """
        Evaluates hazard state using predefined corridor risk formulas from execution plan.

        Formulas:
        1. Outside corridor (|y| >= 2.5m):
           base_risk = clamp(1 - dist/50, 0, 1)
           closing_lateral_speed = -vy * sign(y)
           closing_factor = 0.2 if closing_lateral_speed <= 0 else min(1.2, closing_lateral_speed / 1.8)
           lat_factor = clamp(1 - max(0, |y|-2.5)/8, 0.2, 1.0)
           final_risk = min(base_risk * closing_factor * lat_factor, 0.58)

        2. Inside corridor (|y| < 2.5m and 0 < x <= 40):
           base_risk = clamp(1 - dist/50, 0, 1)
           vx_toward_ego = -vx if vx < 0 else 0
           closing_factor = 1 + vx_toward_ego / 4.0
           final_risk = clamp(base_risk * 1.15 * closing_factor, 0, 1)
        """
        x = float(hazard_state.get("x", 0.0))
        y = float(hazard_state.get("y", 0.0))
        vx = float(hazard_state.get("vx", 0.0))
        vy = float(hazard_state.get("vy", 0.0))

        dist = float(math.hypot(x, y))

        is_in_corridor = (abs(y) < 2.5) and (0.0 < x <= 40.0)

        if is_in_corridor:
            base_risk = clamp(1.0 - dist / 50.0, 0.0, 1.0)
            vx_toward_ego = -vx if vx < 0 else 0.0
            closing_factor = 1.0 + vx_toward_ego / 4.0
            final_risk = clamp(base_risk * 1.15 * closing_factor, 0.0, 1.0)
        else:
            base_risk = clamp(1.0 - dist / 50.0, 0.0, 1.0)

            if y > 0:
                sign_y = 1.0
            elif y < 0:
                sign_y = -1.0
            else:
                sign_y = 0.0

            closing_lateral_speed = -vy * sign_y

            if closing_lateral_speed <= 0:
                closing_factor = 0.2
            else:
                closing_factor = min(1.2, closing_lateral_speed / 1.8)

            lat_factor = clamp(1.0 - max(0.0, abs(y) - 2.5) / 8.0, 0.2, 1.0)

            final_risk = min(base_risk * closing_factor * lat_factor, 0.58)

        risk_level = classify_risk(final_risk)
        refinement_active = (final_risk >= 0.60)
        action = get_safety_action(risk_level)
        ttc = compute_ttc(hazard_state)

        return {
            "distance": round(dist, 3),
            "risk_score": float(final_risk),
            "risk_level": risk_level,
            "ttc": ttc,
            "refinement_active": refinement_active,
            "action": action,
            "in_corridor": is_in_corridor,
        }

    def apply_refinement(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        hazard_state: dict,
        radius: float = 5.0,
        res: float = 0.25,
    ) -> list:
        """
        Triggers local resolution refinement around hazard coordinates from raw source points
        if risk score is >= 0.60 (refinement_active is True).
        Re-bins raw points inside radius at fine tier resolution (0.25m).
        """
        evaluation = self.evaluate(hazard_state)
        if not evaluation["refinement_active"]:
            return []

        hx = float(hazard_state.get("x", 0.0))
        hy = float(hazard_state.get("y", 0.0))
        return refine_local_region(points, labels, center=(hx, hy), radius=radius, res=res)
