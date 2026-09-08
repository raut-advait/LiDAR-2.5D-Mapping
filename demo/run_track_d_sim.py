"""
Full Track D Deterministic Simulation Demo.

Runs the default hazard scenario through the complete Track D pipeline across 20 ticks
and prints per-tick results matching all required dashboard fields.
"""

import os
import sys
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.adaptive_2_5d import Adaptive2_5DConverter
from demo.hazard_sim import HazardSimulator
from demo.risk_engine import RiskEngine
from demo.render_bev import process_track_d_tick


def run_simulation():
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

    history = []

    for tick in range(20):
        st = sim.get_state()
        tick_output = process_track_d_tick(
            points, labels, st, converter=converter, risk_engine=risk_engine, return_render_data=False
        )

        ttc_str = f"{tick_output['ttc']:.1f}s" if isinstance(tick_output['ttc'], (int, float)) else str(tick_output['ttc'])
        ref_str = "ON" if tick_output["refinement_active"] else "OFF"

        history.append({
            "tick": tick,
            "time": st["time"],
            "x": st["x"],
            "y": st["y"],
            "distance": tick_output["distance"],
            "risk_score": tick_output["risk_score"],
            "risk_level": tick_output["risk_level"],
            "ttc": ttc_str,
            "action": tick_output["action"],
            "refinement": ref_str,
            "cells_before": tick_output["cells_before"],
            "cells_after": tick_output["cells_after"],
            "total_cells": tick_output["total_cells"],
            "baseline_cells": tick_output["baseline_cells"],
        })

        sim.step()

    # Print Table
    header = (
        f"{'Tick':<6}{'Time':<7}{'X':<7}{'Y':<7}{'Dist':<9}{'Risk':<9}"
        f"{'Risk Level':<12}{'TTC':<9}{'Action':<11}{'Refine':<8}"
        f"{'Before':<8}{'After':<8}{'Total':<8}{'Baseline':<9}"
    )
    separator = "=" * len(header)

    print("\n" + separator)
    print("          BALERION TRACK D — FULL DETERMINISTIC SIMULATION")
    print(separator)
    print(header)
    print("-" * len(header))

    for h in history:
        print(
            f"{h['tick']:<6}{h['time']:<7.1f}{h['x']:<7.2f}{h['y']:<7.2f}"
            f"{h['distance']:<9.2f}{h['risk_score']:<9.4f}{h['risk_level']:<12}"
            f"{h['ttc']:<9}{h['action']:<11}{h['refinement']:<8}"
            f"{h['cells_before']:<8}{h['cells_after']:<8}{h['total_cells']:<8}{h['baseline_cells']:<9}"
        )

    print(separator + "\n")
    return history


if __name__ == "__main__":
    run_simulation()
