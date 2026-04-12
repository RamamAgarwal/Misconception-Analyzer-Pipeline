"""
src/scoring.py
==============
Pure scoring logic — no I/O, no LLM calls, no file paths.
Imports: standard library + pandas only.

Contains:
  • SEVERITY_DEDUCTION  — how many points each severity level subtracts
  • INTERVENTION_THRESHOLDS — score → teacher action label mapping
  • time_weighted_base()  — exponential recency-weighted accuracy score
  • get_intervention()    — maps a score to its intervention label
"""

import math

import pandas as pd

# ── Severity → mastery deduction ───────────────────────────────────────────────
#
# Teacher rationale:
#   low  (-5):  Surface slip — the student understands the concept; barely affects mastery.
#   medium (-12): Procedural gap — targeted practice on a specific step is needed.
#   high  (-22): Conceptual misunderstanding — teacher must re-teach from the ground up.
#
# The deductions are calibrated so a single high-severity error drops a 50%-accurate
# student below the 40-point re-teaching threshold, surfacing the intervention signal
# that a flat accuracy percentage would silently miss.

SEVERITY_DEDUCTION: dict[str, int] = {
    "low":    5,
    "medium": 12,
    "high":   22,
}

# ── Score -> intervention label ─────────────────────────────────────────────────
#
# Ordered highest -> lowest so the first matching threshold is used.

INTERVENTION_THRESHOLDS: list[tuple[int, str]] = [
    (85, "Mastery achieved"),
    (65, "Minor reinforcement suggested"),
    (40, "Targeted practice recommended"),
    ( 0, "Immediate re-teaching needed"),
]


def get_intervention(score: float) -> str:
    """Return the teacher-facing intervention label for a given mastery score."""
    for threshold, label in INTERVENTION_THRESHOLDS:
        if score >= threshold:
            return label
    return "🔴 Immediate re-teaching needed"


def time_weighted_base(group: pd.DataFrame) -> float:
    """
    Compute a recency-weighted accuracy score for one student × concept group.

    Why this beats a flat percentage (explained in teacher terms):
    ─────────────────────────────────────────────────────────────
    Recent attempts are stronger evidence of current understanding than old ones.
    A student who failed the first three attempts but succeeded in the last two is
    on an upward trajectory — their mastery score should reflect that progress,
    not be dragged down by early struggles.

    The exponential weight e^(i/n) gives the last attempt ~2.7× more influence
    than the first, without over-weighting any single result.

    Formula:
        weight_i = e^(i / n)     where i = sorted attempt index, n = total attempts
        base     = Σ(weight_i × is_correct_i) / Σ(weight_i) × 100
    """
    group = group.sort_values("timestamp")
    n = len(group)
    weights = [math.exp(i / n) for i in range(n)]
    weighted_correct = sum(
        w * int(r["is_correct"])
        for w, (_, r) in zip(weights, group.iterrows())
    )
    return (weighted_correct / sum(weights)) * 100
