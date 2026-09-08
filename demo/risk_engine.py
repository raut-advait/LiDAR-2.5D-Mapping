import numpy as np


def compute_hazard_risk(
    x: float,
    y: float,
    vx: float,
    vy: float,
    corridor_half_width: float = 2.5,
    max_path_length: float = 40.0,
    max_distance_range: float = 50.0,
    ref_closing_speed: float = 1.8,
) -> dict:
    """Compute risk score, risk level, base risk, closing factor, and path-relative time metric.

    Rules & Progression:
        A. Outside Corridor (|y| >= 2.5m):
           - Driven by distance + lateral closing speed + lateral proximity.
           - base_risk = clamp(1.0 - distance / 50.0, 0, 1)
           - closing_factor = min(1.2, closing_lateral_speed / 1.8)
           - Capped at < 0.60 (allows LOW [0, 0.35) and MEDIUM [0.35, 0.60) risk levels).
           - Explicit time metric label: "Path Entry Time".

        B. Inside Corridor (|y| < 2.5m):
           - Driven by path occupancy + distance from ego + longitudinal closing speed.
           - Risk persists while inside corridor (never zeroed out by lateral velocity).
           - Uncapped, smoothly scaling through HIGH [0.60, 0.85) to CRITICAL [0.85, 1.0].
           - Explicit time metric label: "Ego TTC".
    """
    dist = float(np.sqrt(x**2 + y**2))
    in_corridor = bool(abs(y) < corridor_half_width and 0.0 < x <= max_path_length)

    sign_y = 1.0 if y > 0 else (-1.0 if y < 0 else 0.0)
    closing_lateral_speed = float(-vy * sign_y)

    if not in_corridor:
        # OUTSIDE CORRIDOR CALCULATIONS
        lat_dist_to_corr = max(0.0, abs(y) - corridor_half_width)
        base_risk = float(np.clip(1.0 - dist / max_distance_range, 0.0, 1.0))

        if closing_lateral_speed <= 0:
            closing_factor = 0.2
        else:
            closing_factor = float(min(1.2, closing_lateral_speed / ref_closing_speed))

        lat_factor = float(np.clip(1.0 - lat_dist_to_corr / 8.0, 0.2, 1.0))
        raw_risk = base_risk * closing_factor * lat_factor

        # Cap at 0.58 max outside corridor (stays in LOW/MEDIUM range)
        final_risk = float(min(raw_risk, 0.58))

        time_label = "Path Entry Time"
        if closing_lateral_speed > 0:
            ttc_val = lat_dist_to_corr / closing_lateral_speed
            ttc_str = f"{ttc_val:.2f}s"
        else:
            ttc_str = "SAFE"
    else:
        # INSIDE CORRIDOR CALCULATIONS (Hazard is directly in ego path)
        base_risk = float(np.clip(1.0 - dist / max_distance_range, 0.0, 1.0))

        vx_toward_ego = -vx if vx < 0 else 0.0
        closing_factor = float(1.0 + (vx_toward_ego / 2.65))

        # Occupancy multiplier for path presence (scales smoothly from MEDIUM at ~35m to CRITICAL near ego)
        occupancy_mult = 1.27
        raw_risk = base_risk * occupancy_mult * closing_factor

        # Inside corridor risk evaluated smoothly [0.0, 1.0]
        final_risk = float(np.clip(raw_risk, 0.0, 1.0))

        time_label = "Ego TTC"
        if vx_toward_ego > 0:
            ttc_val = x / vx_toward_ego
            ttc_str = f"{ttc_val:.2f}s"
        else:
            ttc_str = "SAFE"

    # Discrete Risk Level Mapping
    if final_risk < 0.35:
        risk_level = "LOW"
    elif final_risk < 0.60:
        risk_level = "MEDIUM"
    elif final_risk < 0.85:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "distance": dist,
        "closing_lateral_speed": closing_lateral_speed,
        "in_corridor": in_corridor,
        "base_risk": base_risk,
        "closing_factor": closing_factor,
        "raw_risk": raw_risk,
        "final_risk": final_risk,
        "risk_level": risk_level,
        "time_label": time_label,
        "ttc": ttc_str,
    }


class RiskEngine:
    """Object wrapper around compute_hazard_risk for class-based calls."""

    def compute(self, x: float, y: float, vx: float, vy: float, **kwargs) -> dict:
        return compute_hazard_risk(x, y, vx, vy, **kwargs)

    def __call__(self, x: float, y: float, vx: float, vy: float, **kwargs) -> dict:
        return compute_hazard_risk(x, y, vx, vy, **kwargs)

