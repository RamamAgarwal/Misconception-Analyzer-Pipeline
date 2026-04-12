"""
src/report.py
=============
Builds the final teacher_report.json structure from the clean DataFrame
and the LLM analysis results. No I/O — returns a plain dict ready for
json.dumps() in the pipeline orchestrator.

Produces:
  • Per-student, per-concept: mastery score, intervention badge, misconceptions
  • _concept_summaries: class-wide accuracy and top misconceptions per concept
"""

import pandas as pd

from src.scoring import SEVERITY_DEDUCTION, get_intervention, time_weighted_base


def build_teacher_report(df: pd.DataFrame, analysis: pd.DataFrame) -> dict:
    """
    Aggregate student attempts and LLM analysis into a structured report.

    Mastery score formula (per student × concept):
        base    = time_weighted_base(attempts)        # recency-weighted accuracy
        penalty = Σ SEVERITY_DEDUCTION[severity]      # LLM-informed deductions
        score   = max(0, base − penalty)

    The severity deductions ensure a student with a high-severity misconception
    cannot falsely appear "on track" even with decent raw accuracy.
    """
    report: dict = {}

    for student_id, s_df in df.groupby("student_id"):
        report[student_id] = {"concepts": {}}

        for concept, c_df in s_df.groupby("concept"):
            base = time_weighted_base(c_df)

            # Pull LLM rows for this student × concept pair
            mask    = (analysis["student_id"] == student_id) & (analysis["concept"] == concept)
            rows    = analysis[mask]
            penalty = sum(SEVERITY_DEDUCTION.get(s, 10) for s in rows["severity"])
            score   = max(0.0, round(base - penalty, 1))

            report[student_id]["concepts"][concept] = {
                "mastery_score": score,
                "intervention":  get_intervention(score),
                "misconceptions": [
                    {
                        "text":       m,
                        "severity":   s,
                        "confidence": c,
                        "hint":       h,
                    }
                    for m, s, c, h in zip(
                        rows["misconception"],
                        rows["severity"],
                        rows["confidence"],
                        rows["hint"],
                    )
                ],
            }

    # ── Class-wide concept summaries ──────────────────────────────────────────
    concept_summaries: dict = {}

    for concept, c_df in df.groupby("concept"):
        total   = len(c_df)
        correct = int(c_df["is_correct"].sum())
        a_rows  = analysis[analysis["concept"] == concept]

        top_misconceptions = (
            a_rows["misconception"].value_counts().head(3).index.tolist()
            if not a_rows.empty else []
        )

        concept_summaries[concept] = {
            "class_accuracy_pct": round(correct / total * 100, 1) if total else 0,
            "students_attempted": int(c_df["student_id"].nunique()),
            "top_misconceptions": top_misconceptions,
        }

    report["_concept_summaries"] = concept_summaries
    return report
