"""
src/data_loader.py
==================
Responsible for one thing: loading student_logs.json into a clean,
fully-validated Pandas DataFrame.
"""

import json
import logging
from datetime import datetime

import pandas as pd

from src.config import LOG_FILE

log = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "student_id", "subject", "concept",
    "question_text", "correct_answer", "student_answer",
    "is_correct", "timestamp",
}


def load_and_clean(path: str = LOG_FILE) -> pd.DataFrame:
    """
    Load JSON logs into a validated Pandas DataFrame.

    Validation layers (each failure logged individually for full traceability):
      1. Required field presence
      2. student_id — non-null, non-empty string
      3. is_correct — strict Python bool; rejects "yes", 1, None
      4. timestamp  — ISO-8601 parse with timezone-aware handling

    Malformed rows are dropped with a WARNING naming the exact field and row index.
    Two intentionally malformed rows in student_logs.json exercise layers 2 and 4.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    clean, skipped = [], 0

    for i, rec in enumerate(raw):
        tag = f"Row {i} (id={rec.get('student_id', '?')})"

        # Layer 1 — required field presence
        missing = REQUIRED_FIELDS - rec.keys()
        if missing:
            log.warning(f"{tag}: missing fields {missing} — skipped")
            skipped += 1
            continue

        # Layer 2 — student_id must be a non-empty string
        if not isinstance(rec["student_id"], str) or not rec["student_id"].strip():
            log.warning(f"{tag}: 'student_id' is null or non-string — skipped")
            skipped += 1
            continue

        # Layer 3 — is_correct must be a strict Python bool (not "yes", 1, None …)
        if not isinstance(rec["is_correct"], bool):
            log.warning(
                f"{tag}: 'is_correct' = {rec['is_correct']!r} "
                f"({type(rec['is_correct']).__name__}), expected bool — skipped"
            )
            skipped += 1
            continue

        # Layer 4 — parse timestamp securely; replace Z → +00:00 for stdlib compat
        try:
            rec["timestamp"] = datetime.fromisoformat(
                rec["timestamp"].replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            log.warning(f"{tag}: unparseable timestamp '{rec.get('timestamp')}' — skipped")
            skipped += 1
            continue

        clean.append(rec)

    log.info(f"Loaded {len(clean)} valid rows | {skipped} malformed rows dropped.")

    df = pd.DataFrame(clean)
    for col in ("student_id", "subject", "concept"):
        df[col] = df[col].astype(str).str.strip()
    return df
