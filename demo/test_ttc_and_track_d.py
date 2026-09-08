"""
Unit test suite for TTC calculation, Track D per-tick integration dictionary,
and risk/refinement safety rules.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.adaptive_2_5d import Adaptive2_5DConverter
from demo.hazard_sim import HazardSimulator
from demo.risk_engine import RiskEngine, compute_ttc, classify_risk, get_safety_action
from demo.render_bev import process_track_d_tick


class TestTTCAndTrackD(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()
        mock_path = "data/predictions/synthetic_mock.npz"
        if not os.path.exists(mock_path):
            from data.generate_mock_data import generate_synthetic_mock_npz
            generate_synthetic_mock_npz(mock_path)

        data = np.load(mock_path)
        self.points = data["points"]
        self.labels = data["labels"]
        self.converter = Adaptive2_5DConverter()

    def test_1_hazard_moving_toward_corridor_finite_ttc(self):
        """1. Hazard moving toward corridor -> finite lateral-entry TTC."""
        # y = 5.0m, vy = -1.0m/s (moving toward corridor edge at y=2.5m)
        st_left = {"x": 40.0, "y": 5.0, "vx": 0.0, "vy": -1.0}
        ttc_left = compute_ttc(st_left)
        self.assertIsInstance(ttc_left, float)
        self.assertAlmostEqual(ttc_left, 2.5, places=3)

        # y = -5.0m, vy = 1.0m/s (moving toward corridor edge at y=-2.5m)
        st_right = {"x": 40.0, "y": -5.0, "vx": 0.0, "vy": 1.0}
        ttc_right = compute_ttc(st_right)
        self.assertIsInstance(ttc_right, float)
        self.assertAlmostEqual(ttc_right, 2.5, places=3)

    def test_2_hazard_moving_away_from_corridor_safe(self):
        """2. Hazard moving away from corridor -> 'SAFE'."""
        # y = 5.0m, vy = +1.0m/s (moving further away to the left)
        st_left_away = {"x": 40.0, "y": 5.0, "vx": -2.0, "vy": 1.0}
        self.assertEqual(compute_ttc(st_left_away), "SAFE")

        # y = -5.0m, vy = -1.0m/s (moving further away to the right)
        st_right_away = {"x": 40.0, "y": -5.0, "vx": -2.0, "vy": -1.0}
        self.assertEqual(compute_ttc(st_right_away), "SAFE")

    def test_3_hazard_inside_corridor_moving_toward_ego(self):
        """3. Hazard already inside corridor and moving toward ego -> x / closing_speed."""
        st = {"x": 30.0, "y": 1.0, "vx": -2.0, "vy": -0.8}
        ttc = compute_ttc(st)
        self.assertIsInstance(ttc, float)
        self.assertAlmostEqual(ttc, 15.0, places=3)

    def test_4_hazard_inside_corridor_not_closing_safe(self):
        """4. Hazard inside corridor but not closing -> 'SAFE'."""
        # Moving away from ego (vx > 0)
        st_receding = {"x": 30.0, "y": 1.0, "vx": 2.0, "vy": 0.0}
        self.assertEqual(compute_ttc(st_receding), "SAFE")

    def test_5_zero_closing_speed_safe(self):
        """5. Zero closing speed -> 'SAFE'."""
        # Stationary inside corridor
        st_static_in = {"x": 30.0, "y": 1.0, "vx": 0.0, "vy": 0.0}
        self.assertEqual(compute_ttc(st_static_in), "SAFE")

        # Stationary outside corridor
        st_static_out = {"x": 40.0, "y": 5.0, "vx": 0.0, "vy": 0.0}
        self.assertEqual(compute_ttc(st_static_out), "SAFE")

    def test_6_no_negative_ttc(self):
        """6. Verification that no negative TTC values are ever returned."""
        test_states = [
            {"x": 10.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
            {"x": -5.0, "y": 0.0, "vx": -2.0, "vy": 0.0},
            {"x": 40.0, "y": 5.0, "vx": -2.0, "vy": 2.0},
            {"x": 40.0, "y": -5.0, "vx": -2.0, "vy": -2.0},
            {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        ]
        for st in test_states:
            res = compute_ttc(st)
            if res != "SAFE":
                self.assertGreater(res, 0.0, f"State {st} returned negative or zero TTC: {res}")

    def test_7_per_tick_dictionary_exact_keys(self):
        """7. Complete per-tick dictionary contains exactly the required dashboard keys."""
        sim = HazardSimulator()
        st = sim.get_state()
        output = process_track_d_tick(
            self.points, self.labels, st, self.converter, self.engine, return_render_data=False
        )

        expected_keys = {
            "distance",
            "risk_score",
            "risk_level",
            "ttc",
            "action",
            "refinement_active",
            "cells_before",
            "cells_after",
            "total_cells",
            "baseline_cells",
        }
        self.assertEqual(set(output.keys()), expected_keys)

        # Verify key types
        self.assertIsInstance(output["distance"], float)
        self.assertIsInstance(output["risk_score"], float)
        self.assertIsInstance(output["risk_level"], str)
        self.assertTrue(isinstance(output["ttc"], (float, int)) or output["ttc"] == "SAFE")
        self.assertIsInstance(output["action"], str)
        self.assertIsInstance(output["refinement_active"], bool)
        self.assertIsInstance(output["cells_before"], int)
        self.assertIsInstance(output["cells_after"], int)
        self.assertIsInstance(output["total_cells"], int)
        self.assertIsInstance(output["baseline_cells"], int)

    def test_8_refinement_state_agrees_with_risk_score(self):
        """8. Refinement state agrees with risk_score >= 0.60."""
        # Low risk (< 0.60)
        st_low = {"x": 40.0, "y": 5.0, "vx": -2.0, "vy": -0.8}
        out_low = self.engine.evaluate(st_low)
        self.assertLess(out_low["risk_score"], 0.60)
        self.assertFalse(out_low["refinement_active"])

        # High risk (>= 0.60)
        st_high = {"x": 32.0, "y": 1.8, "vx": -2.0, "vy": -0.8}
        out_high = self.engine.evaluate(st_high)
        self.assertGreaterEqual(out_high["risk_score"], 0.60)
        self.assertTrue(out_high["refinement_active"])

    def test_9_action_agrees_with_risk_level(self):
        """9. Action agrees with risk level."""
        self.assertEqual(get_safety_action("LOW"), "PROCEED")
        self.assertEqual(get_safety_action("MEDIUM"), "PROCEED")
        self.assertEqual(get_safety_action("HIGH"), "SLOW DOWN")
        self.assertEqual(get_safety_action("CRITICAL"), "BRAKE")

        # Test through RiskEngine evaluation
        st_low = {"x": 40.0, "y": 5.0, "vx": -2.0, "vy": -0.8}
        self.assertEqual(self.engine.evaluate(st_low)["action"], "PROCEED")

        st_high = {"x": 32.0, "y": 1.8, "vx": -2.0, "vy": -0.8}
        self.assertEqual(self.engine.evaluate(st_high)["action"], "SLOW DOWN")

        st_crit = {"x": 24.0, "y": -1.4, "vx": -2.0, "vy": -0.8}
        self.assertEqual(self.engine.evaluate(st_crit)["action"], "BRAKE")


if __name__ == "__main__":
    unittest.main()
