"""Explainable irrigation recommendations.

Explanations are generated ONLY from the computed scenario metrics -- no
fabricated reasons. Every number in the text comes from the decision engine's
output for the current situation.
"""
from __future__ import annotations

from src.counterfactual.simulator import CandidateAction

LABEL_TEXT = {
    "no_irrigation": "do not irrigate now",
    "small": "apply a small amount",
    "medium": "apply a medium amount",
    "large": "apply a larger amount",
    "delay_6h": "wait 6 h, then apply a medium amount",
}


def _action_text(c: CandidateAction) -> str:
    if c.amount_mm <= 0:
        return "no irrigation"
    if c.delay_h > 0:
        return f"wait {c.delay_h} h, then apply {c.amount_mm:g} mm"
    return f"apply {c.amount_mm:g} mm"


def main_factors(components: dict) -> list[str]:
    named = {"Water use": components["water"], "Crop stress": components["stress"],
             "Overwatering": components["overwater"], "Risk/uncertainty": components["risk"]}
    ranked = sorted(named.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for name, val in ranked:
        if val > 0.3:
            out.append(f"{name} (score {val:.2f})")
    return out[:3]


def explain_recommendation(rec, theta0: float, target_low: float, target_high: float,
                           soil_label: str, crop_label: str) -> str:
    c = rec.chosen
    m = c.metrics
    lines = []
    if rec.fallback:
        lines.append("FALLBACK: no irrigation is recommended right now.")
        lines.append(
            f"Because the sensor forecast is uncertain (uncertainty score "
            f"{c.components['risk']:.2f}) and/or the sensor reliability is low, acting on a "
            f"point prediction could over- or under-water. The system recommends observing "
            f"for another cycle before making a confident decision.")
        return "\n".join(lines)

    best_cmp = c.components
    lines.append(
        f"For {crop_label} on {soil_label}, the recommended action is: {_action_text(c.candidate)}. "
        f"This would apply {m['total_water_applied_mm']:g} mm, keep moisture within "
        f"[{target_low:.3f}, {target_high:.3f}] m3/m3, and end the horizon at {m['final_theta']:.3f}.")

    # compare against the runner-up
    others = [s for s in rec.alternatives if s is not c and s.feasible]
    if others:
        runner = sorted(others, key=lambda s: s.total)[0]
        lines.append(
            f"Compared with {_action_text(runner.candidate)} (score {runner.total:.3f}), "
            f"this option scores {c.total:.3f}: it uses {m['total_water_applied_mm']:g} mm "
            f"versus {runner.metrics['total_water_applied_mm']:g} mm, keeps the crop below target "
            f"{m['time_below_range'] * 24:.0f} h versus {runner.metrics['time_below_range'] * 24:.0f} h, "
            f"and produces {m['runoff_mm'] + m['drainage_mm']:.1f} mm of drainage/runoff versus "
            f"{runner.metrics['runoff_mm'] + runner.metrics['drainage_mm']:.1f} mm.")

    if rec.conservative:
        lines.append(
            "This decision used CONFIDENCE-AWARE evaluation: stricter crop-safety constraints were "
            "applied because the forecast or the sensor signal was not fully reliable.")

    reasons = main_factors(best_cmp)
    if reasons:
        lines.append("Main driving factors: " + "; ".join(reasons) + ".")
    lines.append("Numbers above are simulated outcomes from the soil-water model, not field measurements.")
    return "\n".join(lines)