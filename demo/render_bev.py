import numpy as np
from lidar_inference_pipeline.conversion.adaptive_2_5d_converter import Adaptive2_5DConverter
from demo.risk_engine import compute_hazard_risk
from demo.resolution_override import apply_local_refinement
from demo.decision_engine import build_safety_decision
from demo.dynamic_object_tracker import RealDynamicObjectTracker

_default_tracker = RealDynamicObjectTracker()


def process_track_d_tick(
    points: np.ndarray,
    labels: np.ndarray,
    hazard_state: dict = None,
    converter: Adaptive2_5DConverter = None,
    risk_engine=None,
    return_render_data: bool = True,
    ego_heading_rad: float = 0.0,
    dynamic_tracker: RealDynamicObjectTracker = None,
    dt: float = 0.5,
) -> dict:
    """
    Executes Track D tick computation using main project backend modules:
    1. Filter points to BEV range (r <= 60m).
    2. Runs Adaptive2_5DConverter.
    3. Detects & tracks REAL dynamic objects (labels == 2).
    4. Computes risk via compute_hazard_risk for primary dynamic object.
    5. Applies local refinement via apply_local_refinement.
    6. Builds safety decision via build_safety_decision.
    7. Returns dictionary conforming to Track D contract.
    """
    if converter is None:
        converter = Adaptive2_5DConverter()

    if dynamic_tracker is None:
        dynamic_tracker = _default_tracker

    dist_pts = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    mask = dist_pts <= 60.0
    points_clipped = points[mask]
    labels_clipped = labels[mask]

    base_res = converter.convert(points_clipped, labels_clipped)
    base_cells = base_res["cells"]

    # Baseline cell count estimate at fixed fine resolution 0.25m
    fine_res = getattr(converter, "near_resolution", 0.25)
    if len(points_clipped) > 0:
        grid_coords = np.floor(points_clipped[:, 0] / fine_res) + 1j * np.floor(points_clipped[:, 1] / fine_res)
        baseline_cells = len(np.unique(grid_coords))
    else:
        baseline_cells = 0

    # Detect REAL dynamic objects from labels == 2
    dynamic_objects = dynamic_tracker.process_frame(points_clipped, labels_clipped, dt=dt)

    if dynamic_objects:
        # Select primary dynamic object: prioritize in ego corridor (|y| <= 3.5m) then closest distance
        primary = min(dynamic_objects, key=lambda c: (abs(c["centroid"][1]) > 3.5, c["distance"]))
        hx, hy = float(primary["centroid"][0]), float(primary["centroid"][1])
        vx, vy = float(primary["vx"]), float(primary["vy"])
        has_track = bool(primary["has_track"])
    else:
        primary = None
        hx, hy, vx, vy = 0.0, 0.0, 0.0, 0.0
        has_track = False

    if primary is not None:
        v_eval_x = vx if has_track else 0.0
        v_eval_y = vy if has_track else 0.0

        if callable(risk_engine):
            risk_metrics = risk_engine(hx, hy, v_eval_x, v_eval_y)
        elif hasattr(risk_engine, "compute"):
            risk_metrics = risk_engine.compute(hx, hy, v_eval_x, v_eval_y)
        else:
            risk_metrics = compute_hazard_risk(hx, hy, v_eval_x, v_eval_y)

        if not has_track:
            risk_metrics["ttc"] = "N/A / TRACKING"
    else:
        risk_metrics = {
            "distance": 0.0,
            "final_risk": 0.0,
            "risk_level": "LOW",
            "ttc": "SAFE",
            "action": "PROCEED",
            "closing_lateral_speed": 0.0,
            "in_corridor": False,
            "base_risk": 0.0,
            "closing_factor": 1.0,
            "raw_risk": 0.0,
            "time_label": "Ego TTC",
        }

    hazard_info = {
        "x": hx,
        "y": hy,
        "vx": vx,
        "vy": vy,
        "has_track": has_track,
        "primary": primary,
        "dynamic_objects": dynamic_objects,
        "ego_heading_rad": ego_heading_rad,
    }

    refinement_info = apply_local_refinement(
        cells=base_cells,
        points=points_clipped,
        labels=labels_clipped,
        hazard_state=hazard_info,
        risk_metrics=risk_metrics,
        converter=converter,
        refinement_radius=5.0,
    )

    decision_info = build_safety_decision(
        risk_metrics=risk_metrics,
        refinement_info=refinement_info,
    )

    heading = float(ego_heading_rad)

    output = {
        "distance": float(risk_metrics["distance"]),
        "risk_score": float(risk_metrics["final_risk"]),
        "risk_level": str(risk_metrics["risk_level"]),
        "ttc": decision_info["time_value"],
        "action": str(decision_info["action"]),
        "refinement_active": bool(refinement_info["refinement_active"]),
        "cells_before": int(refinement_info["total_cells_before"]),
        "cells_after": int(refinement_info["total_cells_rendered"]),
        "total_cells": int(refinement_info["total_cells_rendered"]),
        "baseline_cells": int(baseline_cells),
        "ego_heading_rad": heading,
        "dynamic_objects": dynamic_objects,
        "primary_hazard": primary,
    }

    if return_render_data:
        output["points"] = points_clipped
        output["labels"] = labels_clipped
        output["cells"] = refinement_info["refined_cells"]
        output["hazard"] = hazard_info

    return output


class BEVRenderer:
    """Lightweight rendering helper wrapper for BEV processing."""

    def __init__(self, converter: Adaptive2_5DConverter = None):
        self.converter = converter if converter is not None else Adaptive2_5DConverter()
        self.dynamic_tracker = RealDynamicObjectTracker()

    def process_tick(self, points, labels, hazard_state=None, risk_engine=None, ego_heading_rad=0.0):
        return process_track_d_tick(
            points=points,
            labels=labels,
            hazard_state=hazard_state,
            converter=self.converter,
            risk_engine=risk_engine,
            return_render_data=True,
            ego_heading_rad=ego_heading_rad,
            dynamic_tracker=self.dynamic_tracker,
        )

