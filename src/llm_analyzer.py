"""
src/llm_analyzer.py
====================
Everything related to diagnosing student misconceptions:
  - The system prompt (used verbatim with any LLM)
  - Zero-shot and chain prompting strategies
  - Rule-based fallback (zero API cost, four deterministic patterns)
  - Provider dispatch: Gemini → mock (extensible to Ollama, OpenAI, etc.)
  - Resilient _call_llm() with retry + JSON enforcement
  - analyze_misconceptions() — the public entry point
"""

import json
import logging
import re
import time

import pandas as pd

from src.config import GEMINI_KEY, GEMINI_MODEL, PROVIDER, STRATEGY

log = logging.getLogger(__name__)

_client = None   # google.genai.Client, initialised on first import when PROVIDER=="gemini"

if PROVIDER == "gemini":
    try:
        from google import genai as _genai    # pip install google-genai
        _client = _genai.Client(api_key=GEMINI_KEY)
        log.info(f"Gemini client ready  (model: {GEMINI_MODEL})")
    except ImportError:
        log.error(
            "google-genai not installed.\n"
            "Run:  pip install google-genai\n"
            "Then set GEMINI_API_KEY or pass your key into src/config.py"
        )
        raise


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt — used verbatim with any LLM (Gemini, Ollama, OpenAI, …)
# ══════════════════════════════════════════════════════════════════════════════
#
# Design choices:
#   • JSON-only output -> no markdown fences, no prose to strip or regex away
#   • Severity + confidence anchored to rubric definitions → consistent scoring
#   • 20-word cap on `misconception` forces dashboard-ready brevity
#   • `confidence` field surfaces uncertainty for downstream review

ANALYSIS_SYSTEM_PROMPT = """You are an expert mathematics and physics tutor trained
in educational data mining. Identify the precise cognitive error a student made.

Reply ONLY with a valid JSON object — no markdown, no preamble, no text outside it.
Required schema (use exactly these keys):

{
  "misconception": "<one sentence, ≤20 words, suitable for a teacher dashboard>",
  "severity":      "low" | "medium" | "high",
  "confidence":    "low" | "medium" | "high",
  "hint":          "<one corrective sentence directed at the student>"
}

Severity rubric:
  low    - surface error (sign flip, arithmetic slip); student grasps the concept
  medium - procedural gap (wrong step order, incomplete formula use)
  high   - fundamental conceptual misunderstanding (entirely wrong mental model)

Confidence rubric:
  low    - multiple plausible explanations; exact error unclear
  medium - most likely cause, but other interpretations are possible
  high   - error pattern uniquely identifies the misconception"""


# ══════════════════════════════════════════════════════════════════════════════
# Prompting Strategies
# ══════════════════════════════════════════════════════════════════════════════

def _zero_shot_prompt(row: dict) -> str:
    """
    Strategy A — Zero-Shot.
    All facts in one prompt. Fast, sufficient for self-evident arithmetic errors.
    Weakness: the model reasons about the correct answer and the student's error
    simultaneously, which collapses multi-step mistakes into vague labels.

    EXACT PROMPT TEXT (used verbatim with any LLM):
    """
    return (
        f"Subject: {row['subject']}  |  Concept: {row['concept']}\n"
        f"Question: {row['question_text']}\n"
        f"Correct answer: {row['correct_answer']}\n"
        f"Student's answer: {row['student_answer']}\n\n"
        "Identify the specific cognitive error behind the student's wrong answer."
    )


def _chain_step1_prompt(row: dict) -> str:
    """
    Strategy B — Chain, Step 1: solve the problem step-by-step.
    Establishes an independent ground-truth solution before any comparison.

    EXACT PROMPT TEXT — Step 1 (solve):
    """
    return (
        f"Solve this {row['subject']} problem step-by-step, "
        f"showing each arithmetic operation:\n"
        f"Concept: {row['concept']}\n"
        f"Question: {row['question_text']}\n"
        f"Expected answer: {row['correct_answer']}"
    )


def _chain_step2_prompt(row: dict, solution: str) -> str:
    """
    Strategy B — Chain, Step 2: diagnose by comparing student answer to solution.
    The model now has the correct reasoning path before it sees the wrong answer,
    reducing hallucinated or vague diagnoses.

    EXACT PROMPT TEXT — Step 2 (diagnose):
    """
    return (
        f"A student solved the following problem incorrectly.\n\n"
        f"Step-by-step correct solution:\n{solution}\n\n"
        f"Student's answer: {row['student_answer']}\n\n"
        "Compare the correct solution to the student's answer and pinpoint "
        "the exact step where and why the student went wrong."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Rule-Based Fallback (zero API cost, four deterministic patterns)
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs BEFORE the LLM on every incorrect attempt. Catches obvious mistakes
# instantly and cheaply — makes the pipeline robust in zero-connectivity
# environments. Returns None if no pattern matched → control passes to LLM.

def _rule_based_fallback(row: dict) -> dict | None:
    q  = row["question_text"].lower()
    ca = str(row["correct_answer"]).strip()
    sa = str(row["student_answer"]).strip()

    # Pattern 1 — fraction addition: student added numerators AND denominators
    # e.g. 1/2 + 1/3 → 2/5  (1+1=2, 2+3=5) — high-severity conceptual error
    if "+" in q and re.search(r"\d/\d", ca) and re.search(r"\d/\d", sa):
        fracs = re.findall(r"(\d+)/(\d+)", q)
        if len(fracs) >= 2:
            try:
                n1, d1 = int(fracs[0][0]), int(fracs[0][1])
                n2, d2 = int(fracs[1][0]), int(fracs[1][1])
                s_parts = [int(x) for x in sa.split("/")]
                if s_parts == [n1 + n2, d1 + d2]:
                    return {
                        "misconception": "Added numerators and denominators separately instead of finding a common denominator.",
                        "severity": "high", "confidence": "high",
                        "hint": "Find the LCD first, convert both fractions, then add only the numerators.",
                    }
            except (ValueError, ZeroDivisionError):
                pass

    # Pattern 2 — kinematic formula: student answer is exactly half the correct value
    # e.g. v = u+at, t=2 → student writes 9.8 instead of 19.6
    try:
        if abs(float(sa) * 2 - float(ca)) < 0.01:
            return {
                "misconception": "Only computed half the formula — forgot to multiply by the full variable value.",
                "severity": "medium", "confidence": "medium",
                "hint": "Re-read the formula and substitute every variable before calculating.",
            }
    except ValueError:
        pass

    # Pattern 3 — algebra: wrong inverse operation when isolating x
    if re.search(r"solve for x|find x", q):
        try:
            if abs(float(sa) - float(ca)) > 1:
                return {
                    "misconception": "Applied the wrong inverse operation when isolating the variable.",
                    "severity": "medium", "confidence": "medium",
                    "hint": "After moving constants, divide both sides by the coefficient of x.",
                }
        except ValueError:
            pass

    # Pattern 4 — Newton's law mix-up: answered "Second" for a First Law question
    if "newton" in q and "first" in ca.lower() and "second" in sa.lower():
        return {
            "misconception": "Confused Newton's First Law (inertia) with the Second Law (F = ma).",
            "severity": "medium", "confidence": "high",
            "hint": "First Law: objects stay at rest or in motion without net force. Second Law: F = ma.",
        }

    return None    # no pattern matched → pass to LLM


# ══════════════════════════════════════════════════════════════════════════════
# LLM Dispatch
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK_RESULT = {
    "misconception": "Analysis unavailable — requires manual review.",
    "severity":      "medium",
    "confidence":    "low",
    "hint":          "Ask your teacher to walk through the correct solution.",
}


def _call_llm(user_prompt: str, retries: int = 3) -> dict:
    """
    Call the configured provider with the given user prompt.
    Handles JSON parse failures and API errors with exponential back-off.
    Falls back to a safe default dict after all retries are exhausted.
    """
    if PROVIDER == "gemini":
        from google import genai as _genai
        full_prompt = f"{ANALYSIS_SYSTEM_PROMPT}\n\n{user_prompt}"
        for attempt in range(retries):
            try:
                resp = _client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=full_prompt,
                )
                text = resp.text.strip().lstrip("```json").rstrip("```").strip()
                return json.loads(text)
            except json.JSONDecodeError:
                log.warning(f"Attempt {attempt + 1}: non-JSON response — retrying…")
            except Exception as exc:
                wait = 2 ** attempt
                log.warning(f"Attempt {attempt + 1}: {exc} — waiting {wait}s…")
                time.sleep(wait)
        return _FALLBACK_RESULT

    if PROVIDER == "mock":
        return _mock_llm(user_prompt)

    return _FALLBACK_RESULT


def _mock_llm(prompt: str) -> dict:
    """
    Keyword-based LLM simulation for offline / demo use.
    The ANALYSIS_SYSTEM_PROMPT above is what would be prepended to each
    user_prompt if a real provider were active.
    Covers the specific patterns in student_logs.json; not a general substitute.
    """
    p = prompt.lower()

    if "v = u + at" in p and "9.8" in p:
        return {
            "misconception": "Substituted t=1 implicitly; forgot to multiply a by the actual t=2.",
            "severity": "medium", "confidence": "high",
            "hint": "Substitute every value: v = 0 + 9.8 × 2 = 19.6 m/s.",
        }
    if "0.5*a*t" in p and "29.4" in p:
        return {
            "misconception": "Used t instead of t² in the displacement formula — treated quadratic as linear.",
            "severity": "high", "confidence": "high",
            "hint": "s = 0.5 × 9.8 × 3² = 0.5 × 9.8 × 9 = 44.1 m.",
        }
    if "newton" in p:
        return {
            "misconception": "Confused Newton's First Law (inertia) with the Second Law (F = ma).",
            "severity": "medium", "confidence": "high",
            "hint": "First Law: no net force needed to maintain constant motion.",
        }
    if "9.8 m/s" in p:
        return {
            "misconception": "Rounded g to 10 m/s² — common approximation, but not the precise value.",
            "severity": "low", "confidence": "high",
            "hint": "Use g = 9.8 m/s² unless the problem explicitly allows rounding.",
        }
    return {
        "misconception": "Error pattern not recognised — requires manual review.",
        "severity": "medium", "confidence": "low",
        "hint": "Review this problem with your teacher.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def analyze_misconceptions(df: pd.DataFrame, strategy: str = STRATEGY) -> pd.DataFrame:
    """
    Iterate over every incorrect attempt and return a diagnostic DataFrame.

    Order of operations per row:
      1. Rule-based fallback  — zero API cost, instant
      2. LLM (zero-shot or chain) — only when no rule matched
    """
    records = []

    for _, row in df[~df["is_correct"]].iterrows():
        label = f"{row['student_id']} | {row['concept']}"

        result = _rule_based_fallback(row.to_dict())

        if result:
            log.info(f"Rule-based match : {label}")
        else:
            log.info(f"LLM [{strategy:9s}]: {label}")
            if strategy == "chain":
                step1 = _call_llm(_chain_step1_prompt(row.to_dict()), retries=2)
                # step1 may return a dict (mock/error) — normalise to a string
                sol_text = (
                    step1.get("misconception", str(step1))
                    if isinstance(step1, dict) else str(step1)
                )
                result = _call_llm(_chain_step2_prompt(row.to_dict(), sol_text))
            else:
                result = _call_llm(_zero_shot_prompt(row.to_dict()))

        records.append({
            "student_id":     row["student_id"],
            "concept":        row["concept"],
            "subject":        row["subject"],
            "timestamp":      row["timestamp"],
            "question":       row["question_text"],
            "student_answer": row["student_answer"],
            "correct_answer": row["correct_answer"],
            **result,
        })

    cols = [
        "student_id", "concept", "subject", "timestamp", "question",
        "student_answer", "correct_answer",
        "misconception", "severity", "confidence", "hint",
    ]
    return pd.DataFrame(records, columns=cols) if records else pd.DataFrame(columns=cols)
