"""
pipeline.py  —  Misconception Analyzer (orchestrator)
======================================================
Problem  : Students make systematic errors teachers can't track at scale.
Solution : LLM-powered pipeline that diagnoses misconceptions per attempt and
           aggregates them into per-student, per-concept mastery scores.
Impact   : A teacher can now spot in one glance that S01 is repeatedly confusing
           denominator addition in Fractions — and intervene before the gap compounds.

Usage:
    pip install pandas google-genai          # google-genai only needed for Gemini
    export GEMINI_API_KEY="your-key-here"
    python pipeline.py

    Offline (no API key): set PROVIDER = "mock" in src/config.py
"""

import json
import logging
from pathlib import Path
from src.config import LOG_FILE, OUT_FILE, PROVIDER, STRATEGY
from src.data_loader import load_and_clean
from src.llm_analyzer import analyze_misconceptions
from src.report import build_teacher_report

log = logging.getLogger(__name__)


def main() -> None:
    log.info("═" * 55)
    log.info("  Misconception Analyzer Pipeline")
    log.info(f"  Provider : {PROVIDER}  |  Strategy : {STRATEGY}")
    log.info("═" * 55)

    df       = load_and_clean(LOG_FILE)
    analysis = analyze_misconceptions(df, strategy=STRATEGY)
    report   = build_teacher_report(df, analysis)

    Path(OUT_FILE).write_text(json.dumps(report, indent=2, default=str))

    student_count = len([k for k in report if not k.startswith("_")])
    log.info(f"\n✓ Report saved → {OUT_FILE}")
    log.info(f"  Students : {student_count}  |  Concepts : {df['concept'].nunique()}")


if __name__ == "__main__":
    main()
