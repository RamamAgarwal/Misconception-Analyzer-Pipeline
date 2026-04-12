"""
src/config.py
=============
Single source of truth for every tuneable constant.
Change PROVIDER, STRATEGY, or file paths here
"""

import os
import logging

# ── LLM Provider ───────────────────────────────────────────────────────────────
PROVIDER     = "gemini"          # "gemini" | "mock"
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
GEMINI_MODEL = "gemini-2.5-flash"   # current free-tier model; swap to gemini-2.5-pro for higher quality

# ── Prompting strategy ─────────────────────────────────────────────────────────
STRATEGY = "chain"             # "zero_shot" | "chain"

# ── File paths ─────────────────────────────────────────────────────────────────
LOG_FILE = "student_logs.json"
OUT_FILE = "teacher_report.json"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
