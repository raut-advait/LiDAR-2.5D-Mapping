"""
Test suite for RiskEngine module (Track D / Integration).

Tests boundary classification, safety action mapping, refinement triggers,
and deterministic tick progression using HazardSimulator default scenario.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demo.hazard_sim import HazardSimulator
from demo.risk_engine import RiskEngine, classify_risk, get_safety_action


class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()
        self.simulator = HazardSimulator()

    def test_boundary_values(self):
        """Explicit boundary tests for risk level classification."""
        self.assertEqual(classify_risk(0.0), "LOW")
        self.assertEqual(classify_risk(0.349999), "LOW")
        self.assertEqual(classify_risk(0.35), "MEDIUM")
        self.assertEqual(classify_risk(0.599999), "MEDIUM")
        self.assertEqual(classify_risk(0.60), "HIGH")
        self.assertEqual(classify_risk(0.849999), "HIGH")
        self.assertEqual(classify_risk(0.85), "CRITICAL")
        self.assertEqual(classify_risk(1.0), "CRITICAL")

    def test_safety_actions(self):
        """Tests safety action mapping from risk level."""
        self.assertEqual(get_safety_action("LOW"), "PROCEED")
        self.assertEqual(get_safety_action("MEDIUM"), "PROCEED")
        self.assertEqual(get_safety_action("HIGH"), "SLOW DOWN")
        self.assertEqual(get_safety_action("CRITICAL"), "BRAKE")

    def test_refinement_activation(self):
        """Tests refinement_active is True iff risk >= 0.60."""
        # Risk < 0.60 (e.g. LOW/MEDIUM)
        low_res = self.engine.evaluate({"x": 40.0, "y": 5.0, "vx": -2.0, "vy": -0.8})
        self.assertFalse(low_res["refinement_active"])

        # Risk >= 0.60 (e.g. HIGH)
        high_res = self.engine.evaluate({"x": 32.0, "y": 1.8, "vx": -2.0, "vy": -0.8})
        self.assertTrue(high_res["refinement_active"])

    def test_simulator_tick_progression(self):
        """
        Verifies risk engine against HazardSimulator default scenario across 20 ticks.
        Expected progression:
        - Ticks 0..6   : LOW
        - Tick 7       : MEDIUM
        - Ticks 8..14  : HIGH
        - Ticks 15..18 : CRITICAL
        - Tick 19+     : LOW
        """
        self.simulator.reset()
        history = []

        for tick in range(20):
            st = self.simulator.get_state()
            eval_res = self.engine.evaluate(st)
            history.append((tick, st["time"], st["x"], st["y"], eval_res["risk_score"], eval_res["risk_level"], eval_res["action"]))
            self.simulator.step()

        # Print detailed tick table for verification report
        print("\n" + "=" * 78)
        print("          RISK ENGINE TICK PROGRESSION VERIFICATION TEST")
        print("=" * 78)
        print(f"{'Tick':<6}{'Time':<8}{'X (m)':<8}{'Y (m)':<8}{'Risk Score':<12}{'Risk Level':<12}{'Action':<12}")
        print("-" * 78)
        for t, tm, x, y, r, lvl, act in history:
            print(f"{t:<6}{tm:<8.1f}{x:<8.2f}{y:<8.2f}{r:<12.4f}{lvl:<12}{act:<12}")
        print("-" * 78)

        # Assert exact tick expectations
        for t in range(0, 7):
            self.assertEqual(history[t][5], "LOW", f"Tick {t} should be LOW, got {history[t][5]}")

        self.assertEqual(history[7][5], "MEDIUM", f"Tick 7 should be MEDIUM, got {history[7][5]}")

        for t in range(8, 15):
            self.assertEqual(history[t][5], "HIGH", f"Tick {t} should be HIGH, got {history[t][5]}")

        for t in range(15, 19):
            self.assertEqual(history[t][5], "CRITICAL", f"Tick {t} should be CRITICAL, got {history[t][5]}")

        self.assertEqual(history[19][5], "LOW", f"Tick 19 should be LOW (exited corridor), got {history[19][5]}")


if __name__ == "__main__":
    unittest.main()
