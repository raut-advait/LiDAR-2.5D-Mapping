import argparse
import sys
from pathlib import Path
import matplotlib.collections as mc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is on sys.path to import modules
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from lidar_inference_pipeline.conversion.adaptive_2_5d_converter import (
    Adaptive2_5DConverter,
)
from demo.sim_vehicle import SimulatedHazardVehicle
from demo.risk_engine import compute_hazard_risk
from demo.resolution_override import apply_local_refinement
from demo.decision_engine import build_safety_decision, format_decision_line, make_event_text


def get_dominant_label(cell: dict) -> int:
    """Extract the dominant semantic label for an adaptive 2.5D cell."""
    layers = cell.get("layers", [])
    if not layers:
        return 0
    if len(layers) == 1:
        return layers[0]["label"]

    label_counts = {}
    for layer in layers:
        lbl = layer["label"]
        cnt = layer.get("point_count", 1)
        label_counts[lbl] = label_counts.get(lbl, 0) + cnt
    return max(label_counts, key=label_counts.get)


def render_bev(
    npz_path: str,
    mode: str = "cells",
    show_rings: bool = True,
    trajectory_length: float = 40.0,
    hazard_state: dict = None,
    risk_metrics: dict = None,
    output_override: Path = None,
    quiet_metrics: bool = False,
):
    """Load prediction .npz file and render top-down (bird's-eye view) scatter/cell plot.

    Expected npz keys:
        - points: (N, 5) float32 (col 0 = X = forward/backward, col 1 = Y = left/right, col 2 = Z = height)
        - model_points: (N, 5) float32
        - labels: (N,) int64 (0: Drivable Surface, 1: Static Obstacle, 2: Dynamic Object)
        - source_file: str / scalar array

    Coordinate mapping for matplotlib axes:
        - Horizontal screen axis (X_screen) = -Y  (left = left, right = right)
        - Vertical screen axis (Y_screen)   =  X  (forward = up, backward = down)
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    # 1. Load npz prediction file
    data = np.load(path, allow_pickle=True)
    if "points" not in data or "labels" not in data:
        raise KeyError(f"File {path} must contain 'points' and 'labels' keys.")

    points = data["points"]
    labels = data["labels"]

    # Retrieve source_file if present, otherwise default to input filename
    if "source_file" in data:
        src_val = data["source_file"]
        source_file_str = str(src_val.item() if hasattr(src_val, "item") else src_val)
    else:
        source_file_str = path.name

    # 2. EXPLICIT 60M RADIUS FILTERING ON RAW POINTS BEFORE CONVERSION
    len_before = len(points)
    dist_origin = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    mask_60m = dist_origin <= 60.0
    points_clipped = points[mask_60m]
    labels_clipped = labels[mask_60m]
    len_after = len(points_clipped)

    if not quiet_metrics:
        print("=" * 60)
        print("60M RADIUS CLIPPING FILTER (RAW POINTS)")
        print("=" * 60)
        print(f"Points Before 60m Filter: {len_before}")
        print(
            f"Points After  60m Filter: {len_after} (Clipped {len_before - len_after} points beyond 60.0m)"
        )

    # 3. RUN ADAPTIVE CONVERTER AND FILTER CELLS TO <= 60.0M RADIUS
    converter = Adaptive2_5DConverter()
    conv_result = converter.convert(points_clipped, labels_clipped)
    raw_cells_list = conv_result["cells"]
    raw_cells_count = len(raw_cells_list)

    # Explicit cell center radius filter (sqrt(x_center**2 + y_center**2) <= 60.0)
    cells = [
        c
        for c in raw_cells_list
        if np.sqrt(
            ((c["x_index"] + 0.5) * c["resolution"]) ** 2
            + ((c["y_index"] + 0.5) * c["resolution"]) ** 2
        )
        <= 60.0
    ]

    # 3.5 APPLY LOCAL RESOLUTION REFINEMENT IF HAZARD IS PRESENT
    if risk_metrics is None and hazard_state is not None:
        risk_metrics = compute_hazard_risk(
            x=hazard_state["x"],
            y=hazard_state["y"],
            vx=hazard_state["vx"],
            vy=hazard_state["vy"],
            max_path_length=trajectory_length,
        )

    refinement_info = apply_local_refinement(
        cells=cells,
        points=points_clipped,
        labels=labels_clipped,
        hazard_state=hazard_state,
        risk_metrics=risk_metrics or {},
        converter=converter,
        refinement_radius=5.0,
    )
    cells = refinement_info["refined_cells"]
    total_cells = len(cells)

    # Directly index Adaptive2_5DConverter's per-tier cell size config (meters)
    fine_size_m = (converter.near_resolution, converter.near_resolution)
    med_size_m = (converter.medium_resolution, converter.medium_resolution)
    coarse_size_m = (converter.far_resolution, converter.far_resolution)

    # Count breakdown per resolution tier AFTER cell center filtering
    fine_cells = [c for c in cells if c.get("zone") == "near"]
    medium_cells = [c for c in cells if c.get("zone") == "medium"]
    coarse_cells = [c for c in cells if c.get("zone") == "far"]

    fine_count = len(fine_cells)
    medium_count = len(medium_cells)
    coarse_count = len(coarse_cells)

    if not quiet_metrics:
        print("\n" + "=" * 60)
        print("ADAPTIVE 2.5D CONVERTER CONFIG & CELL METRICS")
        print("=" * 60)
        print("Converter Configured Cell Sizes (width, height in meters):")
        print(
            f"  - Fine Tier (0-15m)   : ({fine_size_m[0]:.2f}m, {fine_size_m[1]:.2f}m)"
        )
        print(
            f"  - Medium Tier (15-35m): ({med_size_m[0]:.2f}m, {med_size_m[1]:.2f}m)"
        )
        print(
            f"  - Coarse Tier (35-60m): ({coarse_size_m[0]:.2f}m, {coarse_size_m[1]:.2f}m)"
        )
        print(f"\nRaw Converter Cells Generated : {raw_cells_count}")
        print(
            f"Total Adaptive 2.5D Cells (<= 60m): {total_cells} (Filtered {raw_cells_count - total_cells} cell > 60m center radius)"
        )
        print(
            f"  - Fine Tier (0-15m, {fine_size_m[0]:.2f}m res)   : {fine_count:5d} cells"
        )
        print(
            f"  - Medium Tier (15-35m, {med_size_m[0]:.2f}m res) : {medium_count:5d} cells"
        )
        print(
            f"  - Coarse Tier (35-60m, {coarse_size_m[0]:.2f}m res) : {coarse_count:5d} cells"
        )
        print("=" * 60)

    # 4. Setup matplotlib figure with dark background
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")

    # Color map for 3 semantic classes
    class_info = [
        (0, "Drivable Surface", "lightgray", (0.8, 0.8, 0.8)),
        (1, "Static Obstacle", "orange", (1.0, 0.65, 0.0)),
        (2, "Dynamic Object", "red", (1.0, 0.2, 0.2)),
    ]

    # 5. Render content based on mode (using exact variables passed to plotting calls)
    if mode == "points":
        if not quiet_metrics:
            dist_pts = np.sqrt(points_clipped[:, 0] ** 2 + points_clipped[:, 1] ** 2)
            max_pt_radius = float(np.max(dist_pts)) if len(dist_pts) > 0 else 0.0

            print("\n" + "=" * 60)
            print("PLOTTING VERIFICATION (RAW POINTS)")
            print("=" * 60)
            print(f"Plotted points array shape : {points_clipped.shape}")
            print(f"Plotted points total count : {len(points_clipped)}")
            print(f"Maximum plotted radius     : {max_pt_radius:.4f}m")
            print("=" * 60)

        x_screen = -points_clipped[:, 1]
        y_screen = points_clipped[:, 0]

        for label_id, class_name, color_str, _ in class_info:
            mask = labels_clipped == label_id
            if np.any(mask):
                ax.scatter(
                    x_screen[mask],
                    y_screen[mask],
                    c=color_str,
                    s=1.5,
                    label=class_name,
                    alpha=0.8,
                    edgecolors="none",
                    zorder=2,
                )
            else:
                ax.scatter([], [], c=color_str, label=class_name)

    elif mode == "cells":
        if not quiet_metrics:
            cell_centers_radii = [
                float(
                    np.sqrt(
                        ((c["x_index"] + 0.5) * c["resolution"]) ** 2
                        + ((c["y_index"] + 0.5) * c["resolution"]) ** 2
                    )
                )
                for c in cells
            ]
            max_cell_radius = max(cell_centers_radii) if cell_centers_radii else 0.0

            print("\n" + "=" * 60)
            print("PLOTTING VERIFICATION (ADAPTIVE 2.5D CELLS)")
            print("=" * 60)
            print(f"Plotted cells total count  : {len(cells)}")
            print(f"Maximum cell center radius : {max_cell_radius:.4f}m")
            print("Raw-point plotting         : N/A in cells mode")
            print("=" * 60)

        for label_id, class_name, color_str, rgb in class_info:
            class_rects = []
            class_rgbas = []

            for cell in cells:
                dom_label = get_dominant_label(cell)
                if dom_label != label_id:
                    continue

                res = cell["resolution"]
                gx = cell["x_index"]
                gy = cell["y_index"]

                x_min = -(gy + 1) * res
                y_min = gx * res

                point_count = cell["point_count"]
                density = point_count / (res**2)
                alpha = 0.3 + 0.65 * min(1.0, density / 200.0)

                rect = mpatches.Rectangle((x_min, y_min), res, res)
                class_rects.append(rect)
                class_rgbas.append((*rgb, alpha))

            if class_rects:
                pc = mc.PatchCollection(
                    class_rects,
                    facecolors=class_rgbas,
                    edgecolors="none",
                    zorder=2,
                )
                ax.add_collection(pc)

    # 6. Render Predicted Ego Trajectory (straight-line along +X / forward)
    if trajectory_length > 0:
        ax.plot(
            [0, 0],
            [0, trajectory_length],
            color="cyan",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=9,
            label="Predicted Path",
        )

    # 7. Render Simulated Hazard Vehicle if present
    if hazard_state is not None:
        hx = hazard_state["x"]
        hy = hazard_state["y"]
        # Screen coords: X_screen = -Y, Y_screen = X
        h_x_screen = -hy
        h_y_screen = hx

        ax.scatter(
            h_x_screen,
            h_y_screen,
            c="magenta",
            marker="s",
            s=140,
            edgecolors="white",
            linewidths=1.5,
            zorder=12,
            label="Hazard Vehicle",
        )

        if refinement_info and refinement_info.get("refinement_active"):
            ref_circle = mpatches.Circle(
                (h_x_screen, h_y_screen),
                radius=refinement_info.get("refinement_radius", 5.0),
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
                linestyle="--",
                zorder=11,
            )
            ax.add_patch(ref_circle)

    # 8. Adaptive resolution rings and cyan ego marker (rendered in BOTH modes if show_rings=True)
    r1 = converter.near_distance      # 15m
    r2 = converter.medium_distance    # 35m
    r3 = converter.far_distance       # 60m

    if show_rings:
        theta = np.linspace(0, 2 * np.pi, 360)
        for r in [r1, r2, r3]:
            circle_x = r * np.cos(theta)
            circle_y = r * np.sin(theta)
            ax.plot(
                circle_x,
                circle_y,
                color="lightgray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.4,
                zorder=5,
            )

        angle_rad = np.pi / 4
        band_labels = [
            ("FINE (0-15m)", r1 / 2.0),
            ("MEDIUM (15-35m)", (r1 + r2) / 2.0),
            ("COARSE (35-60m)", (r2 + r3) / 2.0),
        ]
        for text_str, mid_r in band_labels:
            tx = mid_r * np.cos(angle_rad)
            ty = mid_r * np.sin(angle_rad)
            ax.text(
                tx,
                ty,
                text_str,
                color="white",
                alpha=0.8,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#111111",
                    edgecolor="none",
                    alpha=0.7,
                ),
                zorder=6,
            )

        ax.scatter(
            0,
            0,
            c="cyan",
            marker="^",
            s=120,
            zorder=10,
            label="Ego Vehicle",
        )

    ax.set_xlim(-r3 - 2, r3 + 2)
    ax.set_ylim(-r3 - 2, r3 + 2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    mode_title = "Adaptive 2.5D Cells" if mode == "cells" else "Raw Points"
    title_str = f"BEV Plot ({mode_title}) - {source_file_str}"
    if show_rings:
        title_str += " (Adaptive Rings)"
    ax.set_title(title_str, color="white", fontsize=12, pad=10)

    legend_handles = [
        mpatches.Patch(color="lightgray", label="Drivable Surface"),
        mpatches.Patch(color="orange", label="Static Obstacle"),
        mpatches.Patch(color="red", label="Dynamic Object"),
    ]
    if show_rings:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="cyan",
                markersize=10,
                linestyle="None",
                label="Ego Vehicle",
            )
        )
    if trajectory_length > 0:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color="cyan",
                linestyle="--",
                linewidth=1.5,
                label="Predicted Path",
            )
        )
    if hazard_state is not None:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="magenta",
                markersize=10,
                linestyle="None",
                label="Hazard Vehicle",
            )
        )
        if refinement_info and refinement_info.get("refinement_active"):
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="cyan",
                    linestyle="--",
                    linewidth=1.2,
                    label="Refinement Region (5m)",
                )
            )

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="#1e1e1e",
        edgecolor="#444444",
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color("white")

    # 9. Determine output filename and save figure
    if output_override is not None:
        output_path = output_override
    else:
        clean_name = source_file_str
        for ext in [".pcd.bin", ".bin", ".npz", ".pcd", ".png"]:
            if clean_name.endswith(ext):
                clean_name = clean_name[:-len(ext)]
                break

        mode_suffix = "_cells" if mode == "cells" else ""
        rings_suffix = "_rings" if show_rings else ""
        output_dir = Path("demo") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"bev_{clean_name}{mode_suffix}{rings_suffix}.png"

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if not quiet_metrics:
        print(f"Saved BEV rendering to: {output_path}")

    if sys.flags.interactive or hasattr(sys, "ps1") or plt.isinteractive():
        plt.show()

    plt.close(fig)
    return output_path, refinement_info


def main():
    parser = argparse.ArgumentParser(
        description="Render a top-down (BEV) scatter or adaptive cell plot from a prediction .npz file."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the prediction .npz file.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to the prediction .npz file (alternative flag).",
    )
    parser.add_argument(
        "--mode",
        choices=["points", "cells"],
        default="cells",
        help="Rendering mode: 'points' for raw point cloud, 'cells' for Adaptive2_5DConverter cells (default: cells).",
    )
    parser.add_argument(
        "--trajectory-length",
        type=float,
        default=40.0,
        help="Length of predicted ego trajectory line in meters (default: 40.0).",
    )
    parser.add_argument(
        "--spawn-hazard",
        action="store_true",
        default=False,
        help="Spawn a simulated crossing hazard vehicle.",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=10,
        help="Number of simulation ticks to run when --spawn-hazard is enabled (default: 10).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Time step in seconds per simulation tick (default: 0.5s).",
    )

    parser.add_argument(
        "--scenario",
        choices=["crossing", "stress-cap", "far-slow-entry"],
        default="crossing",
        help="Simulated hazard vehicle scenario: 'crossing' (default: (30,8), vel=(-1,-2)), 'stress-cap' ((10,6), vel=(-1,-3)), or 'far-slow-entry' ((35,6), vel=(-1,-0.5)).",
    )

    # CLI flag --show-rings (default True) / --no-rings (False)
    parser.add_argument(
        "--show-rings",
        dest="show_rings",
        action="store_true",
        default=True,
        help="Show adaptive resolution rings and ego vehicle marker (default: True).",
    )
    parser.add_argument(
        "--no-rings",
        dest="show_rings",
        action="store_false",
        help="Hide adaptive resolution rings and ego vehicle marker.",
    )

    args = parser.parse_args()
    target_path = args.input_path or args.input

    if not target_path:
        parser.error("Please specify a prediction .npz file path.")

    if args.spawn_hazard:
        # Initialize hazard vehicle based on scenario
        if args.scenario == "stress-cap":
            hazard_veh = SimulatedHazardVehicle(x=10.0, y=6.0, vx=-1.0, vy=-3.0)
        elif args.scenario == "far-slow-entry":
            hazard_veh = SimulatedHazardVehicle(x=35.0, y=6.0, vx=-1.0, vy=-1.0)
        else:
            # Primary canonical crossing scenario: (38.0, 3.0), vel=(-1.5, -0.6)
            hazard_veh = SimulatedHazardVehicle(x=38.0, y=3.0, vx=-1.5, vy=-0.6)

        print("=" * 145)
        print(f"SIMULATED HAZARD VEHICLE TRAJECTORY & RISK ENGINE METRICS (Scenario: '{args.scenario}', {args.sim_steps} Ticks, dt={args.dt}s)")
        print("=" * 145)

        output_dir = Path("demo") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        decision_log = []
        previous_decision = None

        for tick in range(args.sim_steps):
            st = hazard_veh.get_state()
            risk_metrics = compute_hazard_risk(
                x=st["x"],
                y=st["y"],
                vx=st["vx"],
                vy=st["vy"],
                corridor_half_width=2.5,
                max_path_length=args.trajectory_length,
            )

            frame_path = output_dir / f"sim_frame_{tick}.png"
            output_path, refinement_info = render_bev(
                target_path,
                mode=args.mode,
                show_rings=args.show_rings,
                trajectory_length=args.trajectory_length,
                hazard_state=st,
                risk_metrics=risk_metrics,
                output_override=frame_path,
                quiet_metrics=(tick > 0), # Print full conversion metrics on tick 0
            )

            decision = build_safety_decision(
                risk_metrics=risk_metrics,
                refinement_info=refinement_info,
                previous_decision=previous_decision,
            )

            decision_log.append({
                "tick": tick,
                "hazard": {
                    "x": float(st["x"]),
                    "y": float(st["y"]),
                    "vx": float(st["vx"]),
                    "vy": float(st["vy"]),
                },
                "risk": float(risk_metrics["final_risk"]),
                "risk_level": risk_metrics["risk_level"],
                "action": decision["action"],
                "time_label": decision["time_label"],
                "time_value": decision["time_value"],
                "refinement_active": bool(refinement_info["refinement_active"]),
                "local_points": int(refinement_info["pts_in_local_region"]),
                "fine_cells_added": int(refinement_info["fine_cells_added"]),
                "source_point_difference": int(refinement_info["source_point_difference"]),
                "remaining_coarse_cells": int(
                    refinement_info["remaining_coarse_cells_inside_refine_radius"]
                ),
                "event": make_event_text(decision),
            })

            corridor_str = "True " if risk_metrics["in_corridor"] else "False"

            print("-" * 96)
            print(
                f"Tick {tick:2d} | Hazard Pos: ({st['x']:6.2f}m, {st['y']:6.2f}m) | "
                f"Vel: ({st['vx']:5.2f}m/s, {st['vy']:5.2f}m/s) | Corridor: {corridor_str}"
            )
            print(f"  {format_decision_line(decision)}")
            print(f"  JUDGE ACTION                                      : {decision['action']}")
            print(f"  EVENT                                             : {make_event_text(decision)}")
            print(f"  Refinement active (bool)                         : {refinement_info['refinement_active']}")
            print(f"  Refinement radius                                : {refinement_info['refinement_radius']:.1f}m")
            print(f"  Source points before re-binning (5m)             : {refinement_info['pts_in_local_region']}")
            print(f"  Source points represented by fine cells          : {refinement_info['pts_represented_by_fine_cells']}")
            print(f"  Source point conservation difference             : {refinement_info['source_point_difference']}")
            print(f"  Min source-point dist to hazard                  : {refinement_info['min_pt_distance']:.2f}m")
            print(f"  [Converter Output] Cells in local region (5m)    : {refinement_info['cells_before_local']}")
            print(f"  [Balerion Refinement] Cells removed from frame   : {refinement_info['cells_removed']}")
            print(f"  [Balerion Refinement] Fine cells added to frame  : {refinement_info['fine_cells_added']}")
            print(f"  Remaining coarse cells inside refinement radius  : {refinement_info['remaining_coarse_cells_inside_refine_radius']}")
            print(f"  Total cells before (Converter representation)    : {refinement_info['total_cells_before']}")
            print(f"  Final total cells rendered (Balerion override)   : {refinement_info['total_cells_rendered']}")
            print("-" * 80)

            # Advance state for next tick
            previous_decision = decision
            hazard_veh.step(args.dt)

        # Persist compact Stage 7 decisions for the final dashboard.
        import json
        decision_log_path = output_dir / "stage7_decision_log.json"
        decision_log_path.write_text(
            json.dumps(decision_log, indent=2),
            encoding="utf-8",
        )

        print("=" * 80)
        print(f"Saved {args.sim_steps} simulation frames to: {output_dir}/sim_frame_<0..{args.sim_steps-1}>.png")
        print(f"Stage 7 decision log: {decision_log_path}")
    else:
        render_bev(
            target_path,
            mode=args.mode,
            show_rings=args.show_rings,
            trajectory_length=args.trajectory_length,
        )


if __name__ == "__main__":
    main()
