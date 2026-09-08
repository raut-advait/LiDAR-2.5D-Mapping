"""Judge-facing safety decision layer for Project Balerion.

This module intentionally does NOT change the risk/refinement mathematics.
It only maps the existing risk level to an interpretable vehicle action and
formats a presentation-friendly time metric.
"""

from __future__ import annotations

from typing import Dict, Optional


_ACTIONS = {
    "LOW": {
        "action": "PROCEED",
        "short_action": "PROCEED",
        "severity": 0,
    },
    "MEDIUM": {
        "action": "PROCEED WITH CAUTION",
        "short_action": "CAUTION",
        "severity": 1,
    },
    "HIGH": {
        "action": "SLOW DOWN",
        "short_action": "SLOW DOWN",
        "severity": 2,
    },
    "CRITICAL": {
        "action": "BRAKE",
        "short_action": "BRAKE",
        "severity": 3,
    },
}


def build_safety_decision(
    risk_metrics: Dict,
    refinement_info: Optional[Dict] = None,
    previous_decision: Optional[Dict] = None,
) -> Dict:
    """Build a judge-facing decision from existing risk/refinement outputs.

    The function does not recalculate or alter risk. The risk level emitted by
    ``compute_hazard_risk`` is the sole source of the discrete decision.
    """
    level = str(risk_metrics.get("risk_level", "LOW")).upper()
    if level not in _ACTIONS:
        raise ValueError(f"Unsupported risk level: {level!r}")

    spec = _ACTIONS[level]
    refinement_active = bool(
        (refinement_info or {}).get(
            "refinement_active",
            float(risk_metrics.get("final_risk", 0.0)) >= 0.60,
        )
    )

    # Preserve the existing engine metric verbatim. For a hazard in the ego
    # corridor, the existing engine calls this "Ego TTC". For presentation,
    # we expose the same numeric value as "Path Conflict Time" without
    # changing how it was computed.
    source_time_label = str(risk_metrics.get("time_label", "Time Metric"))
    source_time_value = str(risk_metrics.get("ttc", "SAFE"))

    if risk_metrics.get("in_corridor", False):
        judge_time_label = "Path Conflict Time"
    else:
        judge_time_label = "Path Entry Time"

    # Human-readable event associated with the current state.
    if level == "CRITICAL":
        event = "CRITICAL RISK → BRAKE"
    elif level == "HIGH":
        event = "HIGH RISK → SLOW DOWN"
    elif level == "MEDIUM":
        event = "MEDIUM RISK → PROCEED WITH CAUTION"
    else:
        event = "LOW RISK → PROCEED"

    if refinement_active and level in ("HIGH", "CRITICAL"):
        event += " | LOCAL REFINEMENT ACTIVE"

    decision = {
        "risk": float(risk_metrics.get("final_risk", 0.0)),
        "risk_level": level,
        "action": spec["action"],
        "short_action": spec["short_action"],
        "severity": spec["severity"],
        "in_corridor": bool(risk_metrics.get("in_corridor", False)),
        "time_label": judge_time_label,
        "time_value": source_time_value,
        "source_time_label": source_time_label,
        "source_time_value": source_time_value,
        "refinement_active": refinement_active,
        "event": event,
    }

    if previous_decision is not None:
        previous_level = previous_decision.get("risk_level")
        previous_action = previous_decision.get("action")
        if previous_level != level:
            decision["transition"] = f"{previous_level or 'UNKNOWN'} → {level}"
        else:
            decision["transition"] = None

        previous_refinement = bool(previous_decision.get("refinement_active"))
        if previous_refinement != refinement_active:
            decision["refinement_transition"] = (
                "REFINEMENT ON" if refinement_active else "REFINEMENT OFF"
            )
        else:
            decision["refinement_transition"] = None

        if previous_action != spec["action"]:
            decision["action_transition"] = (
                f"{previous_action or 'UNKNOWN'} → {spec['action']}"
            )
        else:
            decision["action_transition"] = None
    else:
        decision["transition"] = None
        decision["refinement_transition"] = (
            "REFINEMENT ON" if refinement_active else None
        )
        decision["action_transition"] = f"START → {spec['action']}"

    return decision


def format_decision_line(decision: Dict) -> str:
    """Compact terminal/dashboard log line."""
    refinement = "ON" if decision["refinement_active"] else "OFF"
    return (
        f"RISK {decision['risk']:.3f} [{decision['risk_level']}] | "
        f"ACTION: {decision['action']} | "
        f"{decision['time_label']}: {decision['time_value']} | "
        f"REFINEMENT: {refinement}"
    )


def make_event_text(decision: Dict) -> str:
    """Return the most useful event text for a judge-facing log."""
    parts = []
    if decision.get("transition"):
        parts.append(f"RISK {decision['transition']}")
    if decision.get("action_transition"):
        parts.append(f"ACTION {decision['action_transition']}")
    if decision.get("refinement_transition"):
        parts.append(decision["refinement_transition"])

    if not parts:
        parts.append(decision["event"])

    return " | ".join(parts)


__all__ = ["build_safety_decision", "format_decision_line", "make_event_text"]
