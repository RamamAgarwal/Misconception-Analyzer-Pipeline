# Misconception Analyzer Pipeline

> **Problem -> Solution -> Impact** 

Students make systematic errors that teachers can't track at scale -> an LLM-powered pipeline diagnoses each error, scores concept mastery, and surfaces it in a teacher dashboard -> a teacher can now spot in one glance that *S01 is repeatedly confusing denominator addition in Fractions* and intervene before the gap compounds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    student_logs.json                            │
│          (10 valid records + 2 intentionally malformed)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Validator  │  Layer 1: required fields
                    │  (Section I)│  Layer 2: student_id type
                    │             │  Layer 3: is_correct bool
                    │             │  Layer 4: timestamp ISO-8601
                    └──────┬──────┘
                           │ clean DataFrame
                           │
              ┌────────────▼─────────────┐
              │   Rule-Based Fallback    │  zero API cost
              │   (4 error patterns)     │  deterministic & instant
              └────────────┬─────────────┘
                           │ unmatched rows only
                           │
              ┌────────────▼─────────────┐
              │  src/llm_analyzer.py     │  Gemini 2.5 Flash (free tier)
              │                          │  OR mock provider (offline)
              │  ┌─────────┐ ┌────────┐  │
              │  │Zero-Shot│ │ Chain  │  │  toggle via STRATEGY= in config.py
              │  └─────────┘ └────────┘  │
              └────────────┬─────────────┘
                           │ misconception, severity, confidence, hint
                           │
              ┌────────────▼─────────────┐
              │  src/scoring.py          │  time_weighted_base()
              │  src/report.py           │  severity deductions
              │                          │  per-student + class summaries
              └────────┬─────────────────┘
                       │
          ┌────────────┼────────────────┐
          │                            │
   ┌──────▼──────┐           ┌─────────▼───────┐
   │teacher_     │           │  dashboard.py   │
   │report.json  │           │  (Streamlit UI) │
   └─────────────┘           └─────────────────┘
```

---

## Setup

**Dependencies:**

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
```

**Run the pipeline:**

```bash
python pipeline.py
```

**Run the dashboard:**

```bash
streamlit run dashboard.py
```

> **No API key?** Set `PROVIDER = "mock"` in `pipeline.py`. The pipeline runs fully offline using rule-based and keyword-matching logic. The exact prompts that would be sent to a real LLM are preserved as comments in the code.

---

## I. Data Processing

### Input: `student_logs.json`

10 valid records covering 4 students, 2 subjects, 4 concepts along with **2 intentionally malformed rows** to demonstrate validation:

| Row    | Field          | Flaw                         | How it's caught                                    |
| ------ | -------------- | ---------------------------- | -------------------------------------------------- |
| Row 10 | `is_correct` | `"yes"` (string, not bool) | Strict `isinstance(val, bool)` check             |
| Row 11 | `timestamp`  | `"not-a-real-date"`        | `datetime.fromisoformat()` raises `ValueError` |

Each dropped row emits a `WARNING` log naming the exact field and row index making issues traceable in production.

### Sample Input

```json
{
  "student_id": "S01",
  "subject": "Math",
  "concept": "Fractions",
  "question_text": "1/2 + 1/3 = ?",
  "correct_answer": "5/6",
  "student_answer": "2/5",
  "is_correct": false,
  "timestamp": "2023-10-27T10:00:00Z"
}
```

---

## II. LLM Misconception Analysis

### System Prompt (exact text, used verbatim)

```
You are an expert mathematics and physics tutor trained in educational data mining.
Identify the precise cognitive error a student made.

Reply ONLY with a valid JSON object — no markdown, no preamble, no text outside it.
Required schema:
{
  "misconception": "<one sentence, ≤20 words, suitable for a teacher dashboard>",
  "severity":      "low" | "medium" | "high",
  "confidence":    "low" | "medium" | "high",
  "hint":          "<one corrective sentence directed at the student>"
}

Severity rubric:
  low    — surface error; student grasps the concept
  medium — procedural gap (wrong step order, incomplete formula)
  high   — fundamental conceptual misunderstanding (wrong mental model)
```

**Design rationale:** JSON-only output eliminates parsing ambiguity. The 20-word cap on `misconception` forces dashboard-ready brevity. The `confidence` field surfaces uncertainty so teachers know when to investigate further.

### Provider & API Key

Set at the top of `pipeline.py`:

```python
PROVIDER   = "gemini"          # "gemini" | "mock"
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "your-key-here")
GEMINI_MODEL = "gemini-2.5-flash"    # swap to gemini-2.5-pro for higher quality
```

Substituting any other provider requires only replacing the `_call_llm` function, the prompt text and output schema remain identical.

### Rule-Based Fallback (no API cost)

Four deterministic patterns run *before* the LLM and catch the majority of common errors instantly:

| Pattern                | Trigger                                  | Example                      |
| ---------------------- | ---------------------------------------- | ---------------------------- |
| Fraction addition      | numerator + denominator added separately | `1/2 + 1/3 = 2/5`         |
| Kinematic formula      | student answer = correct answer ÷ 2     | `v=19.6, student says 9.8` |
| Algebra isolation      | wrong inverse operation                  | `2x+3=11, student says 7`  |
| Newton's Law confusion | "Second" for First Law                   | answered "Second"            |

This makes the pipeline robust in zero-connectivity environments and keeps API costs low.

---

## III. Prompt Strategy Comparison

Two strategies are implemented, toggle with `STRATEGY = "zero_shot" | "chain"` in `pipeline.py`.

### Zero-Shot (Strategy A)

Sends all facts in one prompt. The model must reason about the correct answer and the student's error simultaneously.

**Zero-shot prompt (exact text):**

```
Subject: Math  |  Concept: Fractions
Question: 1/2 + 1/3 = ?
Correct answer: 5/6
Student's answer: 2/5

Identify the specific cognitive error behind the student's wrong answer.
```

### Chain (Strategy B)

Step 1 makes the model solve the problem step-by-step first (independent ground truth). Step 2 compares that solution to the student's answer.

**Chain Step 1 prompt (exact text):**

```
Solve this Math problem step-by-step, showing each arithmetic operation:
Concept: Fractions
Question: 1/2 + 1/3 = ?
Expected answer: 5/6
```

**Chain Step 2 prompt (exact text):**

```
A student solved the following problem incorrectly.

Step-by-step correct solution:
[output from Step 1]

Student's answer: 2/5

Compare the correct solution to the student's answer and pinpoint
the exact step where and why the student went wrong.
```

### Comparison Results (5 test cases)

| Question             | Student Answer | Zero-Shot Diagnosis                  | Chain Diagnosis                                                                                 |
| -------------------- | -------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `1/2 + 1/3 = ?`    | `2/5`        | "Incorrect fraction addition method" | "Added numerators (1+1) and denominators (2+3) as separate whole numbers"                       |
| `v = u+at, t=2`    | `9.8`        | "Arithmetic error in formula"        | "Substituted t=1 implicitly; forgot to multiply a by the full t=2"                              |
| `s = 0.5at², t=3` | `29.4`       | "Wrong formula application"          | "Used t instead of t²; treated the quadratic displacement formula as linear"                   |
| `2x + 3 = 11`      | `7`          | "Error isolating variable"           | "Subtracted from 11 correctly but then subtracted 1 instead of dividing by 2"                   |
| Newton's 1st Law     | "Second"       | "Misidentified Newton's law"         | "Confused the condition for inertia (no force needed) with the force-acceleration relationship" |

**Verdict: Chain outperforms zero-shot.** The independent solve-step prevents the model from anchoring on the student's wrong answer, producing diagnoses specific enough to guide re-teaching (e.g., "treated quadratic as linear" vs "formula error"). Trade-off: 2 API calls instead of 1 — acceptable for a batch pipeline.

---

## III. Aggregation - Why This Score Helps a Teacher

### Mastery Score Formula

```
mastery_score = max(0, time_weighted_base − Σ severity_deductions)
```

**Time-weighted base (why it beats a flat percentage):**

Recent attempts are stronger evidence of current understanding than old ones. A student who failed the first 3 attempts but succeeded in the last 2 is on an upward trajectory, their score should reflect that progress, not be dragged down by early struggles.

```
weight_i = e^(i / n)    [later attempts weighted ~2.7× more than the first]
base = Σ(weight_i × is_correct_i) / Σ(weight_i) × 100
```

**Severity deductions (LLM-informed):**

| Severity | Deduction | Teacher interpretation                         |
| -------- | --------- | ---------------------------------------------- |
| low      | −5 pts   | Surface slip - one targeted reminder fixes it |
| medium   | −12 pts  | Procedural gap - targeted practice needed     |
| high     | −22 pts  | Wrong mental model - re-teach from scratch    |

A student with 50% accuracy but one high-severity misconception scores below 40 and triggers the "immediate re-teaching" flag. A flat accuracy percentage would miss this.

**Concrete example:** A teacher can now see that S01 scored 50% accuracy on Fractions but their mastery score is 28% because the single wrong attempt reveals a *high-severity* misconception (treating fraction addition like whole-number addition). That's the intervention signal a flat percentage cannot give.

### Intervention Thresholds

| Score   | Label                 | Teacher action                         |
| ------- | --------------------- | -------------------------------------- |
| 85–100 | Mastery achieved      | No action needed                       |
| 65–84  | Minor reinforcement   | Assign 1–2 practice problems          |
| 40–64  | Targeted practice     | Assign structured worksheet            |
| 0–39   | Immediate re-teaching | One-on-one session or re-teach concept |

---

## Sample Output (`teacher_report.json`)

```json
{
  "S01": {
    "concepts": {
      "Fractions": {
        "mastery_score": 28.0,
        "intervention": "Immediate re-teaching needed",
        "misconceptions": [
          {
            "text": "Added numerators and denominators separately instead of finding a common denominator.",
            "severity": "high",
            "confidence": "high",
            "hint": "Find the LCD first, convert both fractions, then add only the numerators."
          }
        ]
      },
      "Algebra": {
        "mastery_score": 0.0,
        "intervention": "Immediate re-teaching needed",
        "misconceptions": [
          {
            "text": "Applied the wrong inverse operation when isolating the variable.",
            "severity": "medium",
            "confidence": "medium",
            "hint": "After moving constants, divide both sides by the coefficient of x."
          }
        ]
      }
    }
  },
  "_concept_summaries": {
    "Fractions": {
      "class_accuracy_pct": 40.0,
      "students_attempted": 3,
      "top_misconceptions": [
        "Added numerators and denominators separately instead of finding a common denominator."
      ]
    }
  }
}
```

---

## Dashboard (`dashboard.py`)

Run with `streamlit run dashboard.py`. Shows:

- **KPI cards** - students tracked, concepts covered, avg mastery, at-risk count
- **Concept heatmap** - student × concept mastery matrix (colour-coded red → green)
- **Mastery bar chart** - per-student scores grouped by concept
- **Misconception frequency** - most common errors, colour-coded by severity
- **Class accuracy by concept** - where the whole class is struggling
- **Per-student drill-down** - intervention badge, misconception details, hints, recent wrong attempts

---

## Project Structure

```
├── pipeline.py              Thin orchestrator - imports from src/, runs main()
├── dashboard.py             Streamlit teacher dashboard
├── student_logs.json        Input: 10 valid + 2 intentionally malformed records
├── teacher_report.json      Generated output (created by running pipeline.py)
├── README.md
└── src/
    ├── config.py            All constants: provider, model, strategy, file paths
    ├── data_loader.py       load_and_clean() - JSON -> validated Pandas DataFrame
    ├── llm_analyzer.py      Prompts, rule-based fallback, LLM dispatch, analyze_misconceptions()
    ├── scoring.py           time_weighted_base(), get_intervention(), SEVERITY_DEDUCTION
    └── report.py            build_teacher_report() - aggregates scores + misconceptions
```

Each module has a single, named responsibility. A reviewer can open `llm_analyzer.py` to audit prompts, `scoring.py` to inspect the mastery formula, or `report.py` to understand the output shape without reading anything else.

---

## Future Work

- **Longitudinal tracking** - store reports across sessions to plot mastery trajectories over time
- **Alert system** - email/Slack notification when a student drops below the re-teaching threshold
- **Adaptive question generation** - feed misconceptions back into the AI Tutor to auto-generate targeted remediation questions
- **Multi-subject extension** - add Chemistry and Biology concept taxonomies to the rule-based fallback
- **Student-facing hints** - expose the `hint` field directly in the AI Tutor UI after each wrong attempt

ChatGPT LLM Chat Thread: https://chatgpt.com/share/69dbb1e0-78fc-83e8-908e-e7f22a8dfb70
