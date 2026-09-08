"""
BEV Rendering & Integration Pipeline module (Track D / Integration).

Renders semantic Bird's-Eye-View (BEV) visualizations, adaptive ring boundaries,
2.5D cell maps, local hazard refinement, ego vehicle marker, and predicted trajectories.

Coordinate Handoff Convention:
+X = forward/backward along ego heading
+Y = left/right
+Z = height
Ego vehicle is at (0, 0).

Visual Display Mapping:
Matplotlib plot_x = -y (left is negative plot_x, right is positive plot_x)
Matplotlib plot_y = +x (forward is positive plot_y, visually UP)
"""

import os
import sys
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.adaptive_2_5d import Adaptive2_5DConverter
from perception.cells import (
    clip_points_by_radius,
    compute_fixed_fine_baseline_cells,
    filter_cells_by_radius,
    get_cell_bounds,
    get_cell_dominant_label,
    merge_cells_with_refinement,
    refine_local_region,
)
from demo.risk_engine import RiskEngine

# Standard Semantic Label Colors & Names
SEMANTIC_COLORS = {
    0: "#27ae60",  # Drivable (Green)
    1: "#7f8c8d",  # Static (Gray)
    2: "#e74c3c",  # Dynamic (Red)
}

SEMANTIC_NAMES = {
    0: "Drivable",
    1: "Static",
    2: "Dynamic",
}


class BEVRenderer:
    """Bird's-Eye-View renderer for adaptive 2.5D cell maps and hazard trajectories."""

    def __init__(self, ax=None):
        self.ax = ax

    def set_ax(self, ax):
        """Assign or update the Matplotlib Axes object."""
        self.ax = ax

    def render_rings(
        self,
        near_dist: float = 15.0,
        med_dist: float = 35.0,
        far_dist: float = 60.0,
    ):
        """
        Renders distance-aware adaptive resolution rings around ego origin (0, 0).
        Ring boundaries are retrieved directly from the converter instance/config.
        """
        if self.ax is None:
            return

        rings = [
            (near_dist, f"NEAR ({near_dist:.0f}m)", "#2ecc71", ":"),
            (med_dist, f"MEDIUM ({med_dist:.0f}m)", "#f1c40f", "--"),
            (far_dist, f"FAR ({far_dist:.0f}m)", "#e67e22", "-."),
        ]

        for r, label_text, color, linestyle in rings:
            circle = plt.Circle(
                (0, 0),
                r,
                fill=False,
                edgecolor=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=0.7,
            )
            self.ax.add_patch(circle)

            # Label on the top-right diagonal (angle = pi/4)
            angle = np.pi / 4
            lbl_x = - (r * np.sin(angle))  # -y
            lbl_y = r * np.cos(angle)      # +x
            self.ax.text(
                lbl_x,
                lbl_y,
                f"  {r:.0f}m",
                color=color,
                fontsize=8,
                fontweight="bold",
                va="center",
                ha="left",
            )

    def render_points(self, points: np.ndarray, labels: np.ndarray, alpha: float = 0.5, s: float = 4.0):
        """
        Renders raw 3D LiDAR semantic points mapped to BEV plot coordinates (-y, x).
        """
        if self.ax is None or len(points) == 0:
            return

        x = points[:, 0]
        y = points[:, 1]

        plot_x = -y
        plot_y = x

        colors = [SEMANTIC_COLORS.get(int(lbl), "#95a5a6") for lbl in labels]
        self.ax.scatter(plot_x, plot_y, c=colors, s=s, alpha=alpha, edgecolors="none")

    def render_cells(self, cells: list, max_radius: float = 60.0):
        """
        Renders Adaptive2_5DConverter cell boxes mapped to BEV plot coordinates (-y, x).
        Cell centers are strictly filtered by max_radius.
        """
        if self.ax is None:
            return

        valid_cells = filter_cells_by_radius(cells, max_radius=max_radius)

        for cell in valid_cells:
            x_min, x_max, y_min, y_max = get_cell_bounds(cell)
            res = cell["resolution"]
            label = get_cell_dominant_label(cell)
            color = SEMANTIC_COLORS.get(label, "#95a5a6")
            zone = cell.get("zone", "unknown")

            # Plot mapping:
            # plot_x span: [-y_max, -y_min]  (width = res)
            # plot_y span: [x_min, x_max]    (height = res)
            plot_x_min = -y_max
            plot_y_min = x_min

            if zone == "refined":
                alpha = 0.85
                edgecolor = "#f39c12"  # Highlight refined fine cell borders
                linewidth = 0.8
            else:
                alpha = 0.55 if label == 0 else (0.65 if label == 1 else 0.8)
                edgecolor = "#2c3e50"
                linewidth = 0.3 if res <= 0.25 else (0.4 if res <= 0.50 else 0.5)

            rect = patches.Rectangle(
                (plot_x_min, plot_y_min),
                width=res,
                height=res,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=color,
                alpha=alpha,
            )
            self.ax.add_patch(rect)

    def render_ego(self, length: float = 4.5, width: float = 2.0):
        """
        Renders the ego vehicle marker at (0, 0) pointing UP (+X).
        Length = 4.5m, Width = 2.0m.
        """
        if self.ax is None:
            return

        half_w = width / 2.0
        rear_y = -1.0
        front_y = length - 1.0

        ego_rect = patches.Rectangle(
            (-half_w, rear_y),
            width=width,
            height=length,
            linewidth=1.5,
            edgecolor="#00d2d3",
            facecolor="#0984e3",
            alpha=0.85,
            zorder=10,
        )
        self.ax.add_patch(ego_rect)

        # Forward heading indicator wedge
        wedge_coords = np.array([
            [-half_w, front_y],
            [half_w, front_y],
            [0.0, front_y + 1.2],
        ])
        heading_wedge = patches.Polygon(
            wedge_coords,
            closed=True,
            facecolor="#00cec9",
            edgecolor="#ffffff",
            linewidth=1.0,
            zorder=11,
        )
        self.ax.add_patch(heading_wedge)

        self.ax.scatter([0], [0], color="#ffffff", s=25, zorder=12)
        self.ax.text(0, -2.5, "EGO", color="#00d2d3", fontsize=9, fontweight="bold", ha="center", va="top")

    def render_trajectory(self, trajectory_points: list = None, distance: float = 50.0):
        """
        Renders predicted-path trajectory line for ego vehicle or hazard object.
        """
        if self.ax is None:
            return

        if trajectory_points is not None and len(trajectory_points) > 0:
            pts = np.array(trajectory_points)
            traj_plot_x = -pts[:, 1]
            traj_plot_y = pts[:, 0]
            self.ax.plot(
                traj_plot_x,
                traj_plot_y,
                color="#e74c3c",
                linestyle="--",
                linewidth=2.5,
                marker="o",
                markersize=4,
                label="Hazard Trajectory",
                zorder=9,
            )
        else:
            path_x = np.linspace(0.0, distance, 50)
            path_y = np.zeros_like(path_x)

            plot_x = -path_y
            plot_y = path_x

            self.ax.plot(
                plot_x,
                plot_y,
                color="#00cec9",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                label="Predicted Ego Path",
                zorder=8,
            )

    def render_refinement_zone(self, hazard_state: dict, radius: float = 5.0):
        """
        Renders highlighted 5.0m refinement circle around hazard position if active.
        """
        if self.ax is None or hazard_state is None:
            return

        hx = hazard_state.get("x", 0.0)
        hy = hazard_state.get("y", 0.0)

        # Plot mapping (-y, x)
        plot_hx = -hy
        plot_hy = hx

        # 5m Refinement boundary circle
        circle = plt.Circle(
            (plot_hx, plot_hy),
            radius,
            fill=False,
            edgecolor="#f1c40f",
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            zorder=13,
            label="5.0m Local Refinement Zone",
        )
        self.ax.add_patch(circle)

        self.ax.text(
            plot_hx,
            plot_hy + radius + 1.2,
            "LOCAL FINE REFINEMENT (0.25m)",
            color="#f1c40f",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=14,
        )

    def render(
        self,
        cells: list = None,
        points: np.ndarray = None,
        labels: np.ndarray = None,
        hazard_state: dict = None,
        ego_trajectory: list = None,
        converter: Adaptive2_5DConverter = None,
        max_radius: float = 60.0,
        show_points: bool = False,
        refinement_active: bool = False,
        title: str = "Balerion — Adaptive 2.5D Semantic BEV Rendering",
    ):
        """
        Main rendering entrypoint.
        """
        if self.ax is None:
            fig, self.ax = plt.subplots(figsize=(9, 9), facecolor="#1e1e2e")

        self.ax.clear()
        self.ax.set_facecolor("#181825")

        # Read ring parameters directly from converter config
        near_d = getattr(converter, "near_distance", 15.0) if converter else 15.0
        med_d = getattr(converter, "medium_distance", 35.0) if converter else 35.0
        far_d = getattr(converter, "far_distance", 60.0) if converter else 60.0

        # 1. Distance Rings
        self.render_rings(near_dist=near_d, med_dist=med_d, far_dist=far_d)

        # 2. Render Cells
        if cells is not None:
            self.render_cells(cells, max_radius=max_radius)

        # Render raw points if requested
        if show_points and points is not None and labels is not None:
            self.render_points(points, labels, alpha=0.4, s=3.0)

        # 3. Ego Vehicle Marker
        self.render_ego()

        # 4. Predicted Path Trajectory
        self.render_trajectory(trajectory_points=ego_trajectory, distance=far_d)

        # 5. Hazard Marker & Refinement Circle
        if hazard_state is not None:
            hx, hy = hazard_state.get("x", 0.0), hazard_state.get("y", 0.0)
            h_plot_x, h_plot_y = -hy, hx

            self.ax.scatter(
                [h_plot_x],
                [h_plot_y],
                color="#e74c3c",
                s=130,
                edgecolors="#ffffff",
                linewidth=2.0,
                zorder=15,
                label="Hazard Object",
            )

            if refinement_active:
                self.render_refinement_zone(hazard_state, radius=5.0)

        # Axes Setup
        self.ax.set_xlim(-65.0, 65.0)
        self.ax.set_ylim(-15.0, 65.0)
        self.ax.set_aspect("equal")

        self.ax.grid(True, color="#313244", linestyle=":", linewidth=0.8, alpha=0.6)
        self.ax.set_xlabel("Lateral Position (-Y = Left, +Y = Right) [m]", color="#cdd6f4", fontsize=10)
        self.ax.set_ylabel("Forward Distance (+X = Forward) [m]", color="#cdd6f4", fontsize=10)
        self.ax.set_title(title, color="#cdd6f4", fontsize=12, fontweight="bold", pad=12)

        self.ax.tick_params(colors="#a6adc8")
        for spine in self.ax.spines.values():
            spine.set_color("#45475a")

        # Custom Legend
        legend_patches = [
            patches.Patch(color=SEMANTIC_COLORS[0], label="0: Drivable"),
            patches.Patch(color=SEMANTIC_COLORS[1], label="1: Static"),
            patches.Patch(color=SEMANTIC_COLORS[2], label="2: Dynamic"),
        ]
        if refinement_active:
            legend_patches.append(patches.Patch(color="#f1c40f", label="Refined (0.25m)"))

        self.ax.legend(
            handles=legend_patches,
            loc="upper right",
            facecolor="#1e1e2e",
            edgecolor="#45475a",
            labelcolor="#cdd6f4",
            fontsize=9,
        )


def process_track_d_tick(
    points: np.ndarray,
    labels: np.ndarray,
    hazard_state: dict,
    converter: Adaptive2_5DConverter = None,
    risk_engine: RiskEngine = None,
    ego_trajectory: list = None,
    max_radius: float = 60.0,
    return_render_data: bool = False,
) -> dict:
    """
    Integrates the complete Track D pipeline for a single tick:
    1. Clips raw source points to 60m radius.
    2. Runs baseline Adaptive2_5DConverter.
    3. Filters converter cell centers by 60m radius.
    4. Computes measured fixed fine-resolution baseline cell count.
    5. Evaluates risk score, TTC & refinement activation using RiskEngine.
    6. If risk >= 0.60 (refinement_active), re-bins raw source points inside 5.0m hazard radius at fine tier resolution.
    7. Merges baseline and refined cells, computing real cells_before, cells_after, total_cells.
    8. Returns clean contract output dict (with optional rendering data if return_render_data=True).
    """
    if converter is None:
        converter = Adaptive2_5DConverter()
    if risk_engine is None:
        risk_engine = RiskEngine()

    # 1. Raw points 60m radius clipping
    pts_clipped, lbls_clipped = clip_points_by_radius(points, labels, max_radius=max_radius)

    # 2. Baseline adaptive 2.5D converter execution
    converted_result = converter.convert(pts_clipped, lbls_clipped)
    raw_cells = converted_result["cells"]

    # 3. Filter cell centers by 60m center radius
    baseline_cells = filter_cells_by_radius(raw_cells, max_radius=max_radius)

    # 4. Measure fixed fine-resolution baseline cell count over clipped points
    fine_res = getattr(converter, "near_resolution", 0.25)
    baseline_cell_count = compute_fixed_fine_baseline_cells(pts_clipped, fine_res=fine_res, max_radius=max_radius)

    # 5. Evaluate risk score, TTC & refinement activation
    risk_eval = risk_engine.evaluate(hazard_state)
    refinement_active = risk_eval["refinement_active"]

    # 6. Raw-point local refinement around hazard center if active
    hx = float(hazard_state.get("x", 0.0))
    hy = float(hazard_state.get("y", 0.0))
    hazard_center = (hx, hy)
    refinement_radius = 5.0

    if refinement_active:
        refined_cells = refine_local_region(
            pts_clipped,
            lbls_clipped,
            center=hazard_center,
            radius=refinement_radius,
            res=fine_res,
        )
    else:
        refined_cells = []

    # 7. Merge cells & calculate exact measured counts
    merged_cells, cells_before, cells_after, total_cells = merge_cells_with_refinement(
        baseline_cells,
        refined_cells,
        hazard_center=hazard_center,
        radius=refinement_radius,
        refinement_active=refinement_active,
    )

    contract_output = {
        # D -> C Handoff Contract Fields (exact 10 keys)
        "distance": float(risk_eval["distance"]),
        "risk_score": float(risk_eval["risk_score"]),
        "risk_level": str(risk_eval["risk_level"]),
        "ttc": risk_eval.get("ttc", "SAFE"),
        "action": str(risk_eval["action"]),
        "refinement_active": bool(refinement_active),
        "cells_before": int(cells_before),
        "cells_after": int(cells_after),
        "total_cells": int(total_cells),
        "baseline_cells": int(baseline_cell_count),
    }

    if return_render_data:
        contract_output.update({
            "points": pts_clipped,
            "labels": lbls_clipped,
            "cells": merged_cells,
            "hazard": hazard_state,
            "trajectory": ego_trajectory,
            "risk": risk_eval,
            "converter": converter,
        })

    return contract_output


def process_frame(
    points: np.ndarray,
    labels: np.ndarray,
    hazard_state: dict,
    converter: Adaptive2_5DConverter = None,
    risk_engine: RiskEngine = None,
    ego_trajectory: list = None,
    max_radius: float = 60.0,
) -> dict:
    """
    Integrates the complete Track D pipeline for a single frame, including render data.
    """
    return process_track_d_tick(
        points=points,
        labels=labels,
        hazard_state=hazard_state,
        converter=converter,
        risk_engine=risk_engine,
        ego_trajectory=ego_trajectory,
        max_radius=max_radius,
        return_render_data=True,
    )


if __name__ == "__main__":
    from demo.hazard_sim import HazardSimulator

    print("Running BEVRenderer & Integration Pipeline smoke test...")
    mock_path = "data/predictions/synthetic_mock.npz"
    if not os.path.exists(mock_path):
        from data.generate_mock_data import generate_synthetic_mock_npz
        generate_synthetic_mock_npz(mock_path)

    data = np.load(mock_path)
    points = data["points"]
    labels = data["labels"]

    converter = Adaptive2_5DConverter()
    risk_engine = RiskEngine()
    sim = HazardSimulator()

    # 1. Normal State Test (Tick 0: Risk LOW)
    sim.reset()
    state_low = sim.get_state()
    output_low = process_frame(points, labels, state_low, converter=converter, risk_engine=risk_engine)

    print("\n--- Normal State (Tick 0) ---")
    print(f"Risk Level       : {output_low['risk']['risk_level']} (Score: {output_low['risk']['risk_score']:.4f})")
    print(f"Refinement Active: {output_low['refinement_active']}")
    print(f"Cells Before     : {output_low['cells_before']}")
    print(f"Cells After      : {output_low['cells_after']}")
    print(f"Total Cells      : {output_low['total_cells']}")
    print(f"Baseline Cells   : {output_low['baseline_cells']}")

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#1e1e2e")
    renderer = BEVRenderer(ax=ax)
    renderer.render(
        cells=output_low["cells"],
        hazard_state=output_low["hazard"],
        converter=converter,
        refinement_active=output_low["refinement_active"],
        title="Balerion BEV — Normal State (No Refinement)",
    )
    plt.savefig("demo/bev_render_normal.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. HIGH/CRITICAL State Test (Tick 8: Risk HIGH)
    for _ in range(8):
        sim.step()
    state_high = sim.get_state()
    output_high = process_frame(points, labels, state_high, converter=converter, risk_engine=risk_engine)

    print("\n--- HIGH Risk State (Tick 8) ---")
    print(f"Hazard Pos       : x={state_high['x']}m, y={state_high['y']}m")
    print(f"Risk Level       : {output_high['risk']['risk_level']} (Score: {output_high['risk']['risk_score']:.4f})")
    print(f"Refinement Active: {output_high['refinement_active']}")
    print(f"Cells Before     : {output_high['cells_before']}")
    print(f"Cells After      : {output_high['cells_after']}")
    print(f"Total Cells      : {output_high['total_cells']}")
    print(f"Baseline Cells   : {output_high['baseline_cells']}")

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#1e1e2e")
    renderer = BEVRenderer(ax=ax)
    renderer.render(
        cells=output_high["cells"],
        hazard_state=output_high["hazard"],
        converter=converter,
        refinement_active=output_high["refinement_active"],
        title="Balerion BEV — HIGH Risk State (5m Local Fine Refinement)",
    )
    plt.savefig("demo/bev_render_refined.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSmoke test rendered images saved to:")
    print("  - demo/bev_render_normal.png")
    print("  - demo/bev_render_refined.png")
