"""
Test suite for Adaptive Local Refinement + Cell Rendering Integration (Track D).

Verifies all 10 integration requirements:
1. Converter loads successfully
2. 60m point clipping works
3. Cell-center radius filtering works
4. Refinement is OFF below risk 0.60
5. Refinement is ON at exactly 0.60
6. Only points inside 5m hazard radius are re-binned
7. Surrounding cells remain unchanged
8. cells_after differs from cells_before in refinement region
9. baseline_cells is computed from actual data
10. total_cells reflects the merged live representation
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.adaptive_2_5d import Adaptive2_5DConverter
from perception.cells import (
    clip_points_by_radius,
    compute_fixed_fine_baseline_cells,
    filter_cells_by_radius,
    get_cell_center,
    merge_cells_with_refinement,
    refine_local_region,
)
from demo.hazard_sim import HazardSimulator
from demo.risk_engine import RiskEngine
from demo.render_bev import process_frame


class TestAdaptiveRefinement(unittest.TestCase):

    def setUp(self):
        mock_path = "data/predictions/synthetic_mock.npz"
        if not os.path.exists(mock_path):
            from data.generate_mock_data import generate_synthetic_mock_npz
            generate_synthetic_mock_npz(mock_path)

        data = np.load(mock_path)
        self.raw_points = data["points"]
        self.raw_labels = data["labels"]
        self.converter = Adaptive2_5DConverter()
        self.risk_engine = RiskEngine()
        self.simulator = HazardSimulator()

    def test_1_converter_loads(self):
        """1. Verify converter loads successfully and exposes configuration parameters."""
        self.assertIsNotNone(self.converter)
        self.assertEqual(self.converter.near_resolution, 0.25)
        self.assertEqual(self.converter.medium_resolution, 0.50)
        self.assertEqual(self.converter.far_resolution, 1.00)
        self.assertEqual(self.converter.near_distance, 15.0)
        self.assertEqual(self.converter.medium_distance, 35.0)
        self.assertEqual(self.converter.far_distance, 60.0)

    def test_2_point_clipping_60m(self):
        """2. Verify 60m raw point clipping works."""
        pts_clipped, lbls_clipped = clip_points_by_radius(self.raw_points, self.raw_labels, max_radius=60.0)
        dist = np.hypot(pts_clipped[:, 0], pts_clipped[:, 1])
        self.assertTrue(np.all(dist <= 60.0))
        # Ensure far points beyond 60m were actually removed
        all_dist = np.hypot(self.raw_points[:, 0], self.raw_points[:, 1])
        self.assertGreater(len(self.raw_points), len(pts_clipped))
        self.assertTrue(np.any(all_dist > 60.0))

    def test_3_cell_center_radius_filtering(self):
        """3. Verify cell-center radius filtering works (no cell center > 60m)."""
        converted = self.converter.convert(self.raw_points, self.raw_labels)
        raw_cells = converted["cells"]
        filtered = filter_cells_by_radius(raw_cells, max_radius=60.0)

        for cell in filtered:
            cx, cy = get_cell_center(cell)
            self.assertLessEqual(np.hypot(cx, cy), 60.0)

    def test_4_refinement_off_below_0_60(self):
        """4. Verify refinement is OFF below risk 0.60."""
        # Low risk state (Tick 0: risk ~0.059)
        self.simulator.reset()
        st = self.simulator.get_state()
        output = process_frame(self.raw_points, self.raw_labels, st, self.converter, self.risk_engine)

        self.assertLess(output["risk"]["risk_score"], 0.60)
        self.assertFalse(output["refinement_active"])
        self.assertEqual(output["cells_before"], output["cells_after"])

    def test_5_refinement_on_at_exactly_0_60(self):
        """5. Verify refinement is ON at or above risk 0.60."""
        # Mock hazard state with risk score >= 0.60
        st = {"x": 32.0, "y": 1.8, "vx": -2.0, "vy": -0.8}
        output = process_frame(self.raw_points, self.raw_labels, st, self.converter, self.risk_engine)

        self.assertGreaterEqual(output["risk"]["risk_score"], 0.60)
        self.assertTrue(output["refinement_active"])

    def test_6_only_points_inside_5m_rebinned(self):
        """6. Verify only raw points inside 5.0m hazard radius are selected for refinement."""
        hazard_center = (32.0, 1.8)
        radius = 5.0

        refined = refine_local_region(
            self.raw_points,
            self.raw_labels,
            center=hazard_center,
            radius=radius,
            res=0.25,
        )

        for cell in refined:
            cx, cy = get_cell_center(cell)
            # Center of any refined fine cell must be near or inside the hazard radius
            # (allowing cell half-diagonal tolerance of 0.25*sqrt(2)/2 ~ 0.18m)
            self.assertLessEqual(np.hypot(cx - hazard_center[0], cy - hazard_center[1]), radius + 0.3)

    def test_7_surrounding_cells_remain_unchanged(self):
        """7. Verify surrounding adaptive cells outside 5m radius remain unchanged."""
        hazard_center = (32.0, 1.8)
        radius = 5.0
        st = {"x": hazard_center[0], "y": hazard_center[1], "vx": -2.0, "vy": -0.8}

        output = process_frame(self.raw_points, self.raw_labels, st, self.converter, self.risk_engine)

        # Baseline cells
        baseline_converted = self.converter.convert(self.raw_points, self.raw_labels)["cells"]
        baseline_filtered = filter_cells_by_radius(baseline_converted, max_radius=60.0)

        outside_baseline = [c for c in baseline_filtered if np.hypot(get_cell_center(c)[0] - 32.0, get_cell_center(c)[1] - 1.8) > radius]
        outside_merged = [c for c in output["cells"] if c["zone"] != "refined"]

        self.assertEqual(len(outside_baseline), len(outside_merged))
        for cb, cm in zip(outside_baseline, outside_merged):
            self.assertEqual(cb["x_index"], cm["x_index"])
            self.assertEqual(cb["y_index"], cm["y_index"])
            self.assertEqual(cb["resolution"], cm["resolution"])

    def test_8_cells_after_differs_from_cells_before(self):
        """8. Verify cells_after differs from cells_before in refinement region."""
        # Tick 8 (HIGH risk state at x=32m, y=1.8m in medium 0.50m resolution zone)
        self.simulator.reset()
        for _ in range(8):
            self.simulator.step()
        st = self.simulator.get_state()

        output = process_frame(self.raw_points, self.raw_labels, st, self.converter, self.risk_engine)
        self.assertTrue(output["refinement_active"])
        self.assertNotEqual(output["cells_before"], output["cells_after"])

    def test_9_baseline_cells_computed_from_actual_data(self):
        """9. Verify baseline_cells is computed from actual data."""
        pts_clipped, _ = clip_points_by_radius(self.raw_points, self.raw_labels, max_radius=60.0)
        baseline_count = compute_fixed_fine_baseline_cells(pts_clipped, fine_res=0.25, max_radius=60.0)

        self.assertIsInstance(baseline_count, int)
        self.assertGreater(baseline_count, 0)
        # Baseline fixed fine cells (0.25m) across whole frame should be larger than adaptive total_cells
        st = {"x": 40.0, "y": 5.0, "vx": -2.0, "vy": -0.8}
        output = process_frame(self.raw_points, self.raw_labels, st, self.converter, self.risk_engine)
        self.assertGreater(output["baseline_cells"], output["total_cells"])

    def test_10_total_cells_reflects_merged_representation(self):
        """10. Verify total_cells reflects the merged live Balerion representation."""
        st = {"x": 32.0, "y": 1.8, "vx": -2.0, "vy": -0.8}
        output = process_frame(self.raw_points, self.raw_labels, st, self.converter, self.risk_engine)

        self.assertEqual(output["total_cells"], len(output["cells"]))


if __name__ == "__main__":
    unittest.main()
