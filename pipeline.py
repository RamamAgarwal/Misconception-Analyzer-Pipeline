import json
import math
import os
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI, APIError, APITimeoutError

# Configuration

API_KEY   = os.getenv("OPENAI_API_KEY", "your-api-key-here")
MODEL     = "gpt-4.1-mini"                
STRATEGY  = "chain" # either "zero_shot" or "chain"
LOG_FILE  = "student_logs.json"
OUT_FILE  = "teacher_report.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

client = OpenAI(api_key=API_KEY)

# I. DATA PROCESSING

REQUIRED_FIELDS = {
    "student_id", "subject", "concept",
    "question_text", "correct_answer", "student_answer",
    "is_correct", "timestamp",
}

def load_and_clean(path: str = LOG_FILE) -> pd.DataFrame:

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    clean, skipped = [], 0

    for i, rec in enumerate(raw):
        tag = f"Row {i} (id={rec.get('student_id', '?')})"

        # 1. Checking whether required fields are present
        missing = REQUIRED_FIELDS - rec.keys()
        if missing:
            log.warning(f"{tag}: missing fields {missing} — skipped")
            skipped += 1
            continue

        # 2. student_id must be a non-empty string
        if not isinstance(rec["student_id"], str) or not rec["student_id"].strip():
            log.warning(f"{tag}: 'student_id' is null or non-string — skipped")
            skipped += 1
            continue

        # 3. is_correct must be a boolean (not "yes"/"no"/1/0)
        if not isinstance(rec["is_correct"], bool):
            log.warning(f"{tag}: 'is_correct' is {type(rec['is_correct']).__name__}, expected bool — skipped")
            skipped += 1
            continue

        # 4. Parse timestamp securely
        try:
            rec["timestamp"] = datetime.fromisoformat(
                rec["timestamp"].replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            log.warning(f"{tag}: invalid timestamp '{rec.get('timestamp')}' — skipped")
            skipped += 1
            continue

        clean.append(rec)

    log.info(f"Loaded {len(clean)} valid rows | {skipped} malformed rows dropped.")

    df = pd.DataFrame(clean)
    for col in ("student_id", "subject", "concept"):
        df[col] = df[col].astype(str).str.strip()
    return df


# II. MISCONCEPTION ANALYSIS

# System prompt forces structured JSON
ANALYSIS_SYSTEM = """You are an expert math and physics tutor.
Identify the specific misconception behind a student's wrong answer.
Reply ONLY with a valid JSON object with no markdown, no explanation outside it.
Required schema (use exactly these keys):
{
  "misconception": "<one concise sentence describing the error>",
  "severity": "low" | "medium" | "high",
  "hint": "<one corrective sentence for the student>"
}"""


def _call_llm(user_prompt: str, retries: int = 3) -> dict:

    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": ANALYSIS_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=300,
            )
            text = (getattr(resp, "output_text", "") or "").strip()
            if not text:
                raise ValueError("Empty response text from OpenAI")
            return json.loads(text)

        except json.JSONDecodeError:
            log.warning(f"Attempt {attempt + 1}: non-JSON response, retrying…")

        except APITimeoutError:
            wait = 2 ** attempt
            log.warning(f"Attempt {attempt + 1}: API timeout, waiting {wait}s…")
            time.sleep(wait)

        except APIError as exc:
            log.error(f"API error: {exc}")
            wait = 2 ** attempt
            if attempt < retries - 1:
                log.warning(f"Attempt {attempt + 1}: retrying in {wait}s…")
                time.sleep(wait)
            else:
                break

        except Exception as exc:
            log.error(f"Unexpected error: {exc}")
            break

    return {
        "misconception": "Analysis unavailable.",
        "severity": "medium",
        "hint": "Review the concept with your teacher.",
    }


# Strategy A: Zero-Shot

def _zero_shot_prompt(row: dict) -> str:
    return (
        f"Subject: {row['subject']}  |  Concept: {row['concept']}\n"
        f"Question: {row['question_text']}\n"
        f"Correct answer: {row['correct_answer']}\n"
        f"Student's answer: {row['student_answer']}\n\n"
        "Identify the student's misconception."
    )


# Strategy B: Prompt Chain (CoT)

def _chain_prompts(row: dict) -> dict:

    # Step 1 — solve
    step1 = (
        f"Solve this {row['subject']} problem step-by-step:\n"
        f"Concept: {row['concept']}\n"
        f"Question: {row['question_text']}\n"
        f"Expected answer: {row['correct_answer']}"
    )
    try:
        solution_resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": step1}],
            max_output_tokens=400,
        )
        solution = (getattr(solution_resp, "output_text", "") or "").strip() or \
                   f"The correct answer is {row['correct_answer']}."
    except Exception as exc:
        log.error(f"Chain step-1 failed: {exc}")
        solution = f"The correct answer is {row['correct_answer']}."

    # Step 2 — analyze student error against the solved problem
    step2 = (
        f"Here is the step-by-step solution to a problem:\n{solution}\n\n"
        f"The student answered: {row['student_answer']}\n"
        "Pinpoint exactly where and why the student went wrong."
    )
    return _call_llm(step2)


def analyze_misconceptions(df: pd.DataFrame, strategy: str = STRATEGY) -> pd.DataFrame:
    """
    
    """
    analyze = _chain_prompts if strategy == "chain" else \
              (lambda row: _call_llm(_zero_shot_prompt(row)))

    records = []
    for _, row in df[~df["is_correct"]].iterrows():
        label = f"{row['student_id']} | {row['concept']} | {row['question_text'][:45]}…"
        log.info(f"Analyzing [{strategy}]: {label}")

        result = analyze(row.to_dict())
        records.append({
            "student_id":   row["student_id"],
            "concept":      row["concept"],
            "timestamp":    row["timestamp"],
            **result,
        })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["student_id", "concept", "timestamp", "misconception", "severity", "hint"]
    )


# III. AGGREGATION

SEVERITY_DEDUCTION = {"low": 5, "medium": 12, "high": 22}


def _time_weighted_base(group: pd.DataFrame) -> float:
    """
    Exponential recency weighting: later attempts contribute more.
    Score = Σ(weight_i x correct_i) / Σ(weight_i) x 100
    weight_i = e^(rank / total_attempts)
    """
    group = group.sort_values("timestamp")
    n = len(group)
    weights = [math.exp(i / n) for i in range(n)]
    weighted_correct = sum(
        w * int(r["is_correct"]) for w, (_, r) in zip(weights, group.iterrows())
    )
    return (weighted_correct / sum(weights)) * 100


def build_teacher_report(df: pd.DataFrame, analysis: pd.DataFrame) -> dict:
    """
    Produce the final report keyed by student → concept.
    Mastery = time-weighted base score − Σ severity deductions (min 0).
    """
    report = {}

    for student_id, s_df in df.groupby("student_id"):
        report[student_id] = {"concepts": {}}

        for concept, c_df in s_df.groupby("concept"):
            base = _time_weighted_base(c_df)

            # Pull LLM analysis rows for this student+concept
            mask = (analysis["student_id"] == student_id) & \
                   (analysis["concept"] == concept)
            rows = analysis[mask]

            misconceptions = rows["misconception"].tolist()
            hints          = rows["hint"].tolist()
            penalty        = sum(SEVERITY_DEDUCTION.get(s, 10)
                                 for s in rows["severity"].tolist())

            report[student_id]["concepts"][concept] = {
                "mastery_score": max(0.0, round(base - penalty, 1)),
                "identified_misconceptions": misconceptions,
                "hints_for_student": hints,
            }

    return report


# MAIN

if __name__ == "__main__":
    log.info("═" * 55)
    log.info("  Misconception Analyzer Pipeline")
    log.info("═" * 55)

    # Step 1 — load & clean
    df = load_and_clean(LOG_FILE)

    # Step 2 — LLM analysis
    analysis = analyze_misconceptions(df, strategy=STRATEGY)

    # Step 3 — aggregate & report
    report = build_teacher_report(df, analysis)

    Path(OUT_FILE).write_text(json.dumps(report, indent=2))
    log.info(f"\n Report saved in {OUT_FILE}")
    log.info(f"  Students: {len(report)} | Strategy used: {STRATEGY}")