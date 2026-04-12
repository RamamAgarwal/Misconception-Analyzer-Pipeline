"""
Teacher Dashboard — Misconception Analyzer
==========================================
Run with:  streamlit run dashboard.py
Reads:     teacher_report.json  +  student_logs.json  (must exist first — run pipeline.py)
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Teacher Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    report_path = Path("teacher_report.json")
    logs_path   = Path("student_logs.json")

    if not report_path.exists():
        st.error("teacher_report.json not found. Run `python pipeline.py` first.")
        st.stop()

    with open(report_path) as f:
        report = json.load(f)

    logs = []
    if logs_path.exists():
        with open(logs_path) as f:
            logs = json.load(f)

    return report, logs

report, raw_logs = load_data()

# Separate student data from concept summaries
students         = {k: v for k, v in report.items() if not k.startswith("_")}
concept_summaries = report.get("_concept_summaries", {})

# Build flat DataFrames for charting
records = []
for sid, sdata in students.items():
    for concept, cdata in sdata["concepts"].items():
        records.append({
            "Student":        sid,
            "Concept":        concept,
            "Mastery Score":  cdata["mastery_score"],
            "Intervention":   cdata["intervention"],
            "# Misconceptions": len(cdata["misconceptions"]),
        })
df_scores = pd.DataFrame(records)

# Build misconception frequency list
all_misconceptions = []
for sid, sdata in students.items():
    for concept, cdata in sdata["concepts"].items():
        for m in cdata["misconceptions"]:
            all_misconceptions.append({
                "Student": sid,
                "Concept": concept,
                "Text":    m["text"],
                "Severity": m["severity"],
                "Confidence": m["confidence"],
            })
df_misc = pd.DataFrame(all_misconceptions) if all_misconceptions else pd.DataFrame()

# Parse valid logs
valid_logs = []
for rec in raw_logs:
    try:
        if isinstance(rec.get("student_id"), str) and isinstance(rec.get("is_correct"), bool):
            valid_logs.append(rec)
    except Exception:
        pass
df_logs = pd.DataFrame(valid_logs) if valid_logs else pd.DataFrame()

# ── Colour helpers ─────────────────────────────────────────────────────────────

SCORE_COLOUR = {
    "🟢 Mastery achieved":               "#2ecc71",
    "🟡 Minor reinforcement suggested":  "#f1c40f",
    "🟠 Targeted practice recommended":  "#e67e22",
    "🔴 Immediate re-teaching needed":   "#e74c3c",
}

SEV_COLOUR = {"low": "#2ecc71", "medium": "#e67e22", "high": "#e74c3c"}

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📊 Teacher Dashboard - Misconception Analyzer")
st.caption("Real-time view of student mastery scores, misconceptions, and recommended interventions.")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Class-wide KPI cards
# ══════════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4 = st.columns(4)
k1.metric("Students tracked",   len(students))
k2.metric("Concepts covered",   df_scores["Concept"].nunique())
k3.metric("Avg mastery score",  f"{df_scores['Mastery Score'].mean():.1f}%")
at_risk = (df_scores["Mastery Score"] < 40).sum()
k4.metric("At-risk entries 🔴", int(at_risk), delta=None)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Concept Heatmap + Mastery bar chart
# ══════════════════════════════════════════════════════════════════════════════

col_heat, col_bar = st.columns([3, 2])

with col_heat:
    st.subheader("🗺️ Concept Mastery Heatmap")
    if not df_scores.empty:
        pivot = df_scores.pivot(index="Student", columns="Concept", values="Mastery Score")
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale=[[0, "#e74c3c"], [0.4, "#e67e22"],
                                     [0.65, "#f1c40f"], [1, "#2ecc71"]],
            zmin=0, zmax=100,
            text_auto=".0f",
            aspect="auto",
        )
        fig_heat.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_colorbar=dict(title="Score %"),
        )
        fig_heat.update_traces(textfont_size=14)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No score data available.")

with col_bar:
    st.subheader("📈 Student Mastery by Concept")
    if not df_scores.empty:
        fig_bar = px.bar(
            df_scores.sort_values("Mastery Score"),
            x="Mastery Score", y="Student", color="Concept",
            orientation="h", range_x=[0, 100],
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_bar.update_layout(margin=dict(l=0, r=0, t=20, b=0), legend_title="Concept")
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Most common misconceptions + Concept class accuracy
# ══════════════════════════════════════════════════════════════════════════════

col_misc, col_concept = st.columns([3, 2])

with col_misc:
    st.subheader("🧠 Most Common Misconceptions")
    if not df_misc.empty:
        freq = (
            df_misc.groupby(["Text", "Severity"])
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(8)
        )
        fig_misc = px.bar(
            freq,
            x="Count", y="Text", color="Severity",
            orientation="h",
            color_discrete_map=SEV_COLOUR,
            height=320,
        )
        fig_misc.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_misc, use_container_width=True)
    else:
        st.info("No misconceptions recorded.")

with col_concept:
    st.subheader("📚 Class Accuracy by Concept")
    if concept_summaries:
        cs_df = pd.DataFrame([
            {
                "Concept":       c,
                "Class Accuracy": v["class_accuracy_pct"],
                "Students":      v["students_attempted"],
            }
            for c, v in concept_summaries.items()
        ])
        fig_acc = px.bar(
            cs_df.sort_values("Class Accuracy"),
            x="Class Accuracy", y="Concept",
            orientation="h", range_x=[0, 100],
            color="Class Accuracy",
            color_continuous_scale=["#e74c3c", "#f1c40f", "#2ecc71"],
            text_auto=".1f",
            height=320,
        )
        fig_acc.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown("**Top class-wide misconceptions per concept:**")
        for concept, data in concept_summaries.items():
            if data["top_misconceptions"]:
                st.markdown(f"**{concept}**")
                for m in data["top_misconceptions"]:
                    st.markdown(f"- {m}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Per-student drill-down
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("🔍 Per-Student Detail")
selected = st.selectbox("Select a student", sorted(students.keys()))

s_data   = students[selected]
s_scores = df_scores[df_scores["Student"] == selected]

tab_overview, tab_misc, tab_recent = st.tabs(
    ["📋 Concept Overview", "🧠 Misconceptions & Hints", "📝 Recent Wrong Attempts"]
)

with tab_overview:
    for _, row in s_scores.iterrows():
        colour = SCORE_COLOUR.get(row["Intervention"], "#999")
        bar_w  = int(row["Mastery Score"])
        st.markdown(
            f"""
            <div style='padding:12px; margin-bottom:10px; border-radius:8px;
                        border-left:5px solid {colour}; background:#1e1e2e;'>
              <b>{row['Concept']}</b>
              <div style='font-size:0.85em; color:#aaa; margin:4px 0;'>
                {row['Intervention']}
              </div>
              <div style='background:#333; border-radius:4px; height:14px; margin-top:6px;'>
                <div style='background:{colour}; width:{bar_w}%; height:14px;
                            border-radius:4px; text-align:right; padding-right:4px;
                            font-size:0.7em; line-height:14px; color:#fff;'>
                  {bar_w}%
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_misc:
    s_misc = df_misc[df_misc["Student"] == selected] if not df_misc.empty else pd.DataFrame()
    if s_misc.empty:
        st.success("No misconceptions recorded — great work!")
    else:
        for _, row in s_misc.iterrows():
            sev_col = SEV_COLOUR.get(row["Severity"], "#999")
            with st.expander(f"**{row['Concept']}** — {row['Text'][:60]}…"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Severity:** <span style='color:{sev_col}'>{row['Severity'].upper()}</span>",
                            unsafe_allow_html=True)
                c2.markdown(f"**Confidence:** {row['Confidence'].upper()}")
                st.markdown(f"📌 **Misconception:** {row['Text']}")
                # Get hint from report
                concept_data = s_data["concepts"].get(row["Concept"], {})
                for m in concept_data.get("misconceptions", []):
                    if m["text"] == row["Text"]:
                        st.info(f"💡 **Hint for student:** {m['hint']}")

with tab_recent:
    if df_logs.empty:
        st.info("Log data unavailable.")
    else:
        s_logs = (
            df_logs[
                (df_logs["student_id"] == selected) &
                (~df_logs["is_correct"].astype(bool))
            ]
            .sort_values("timestamp", ascending=False)
            .head(5)
        )
        if s_logs.empty:
            st.success("No recent wrong attempts!")
        else:
            for _, row in s_logs.iterrows():
                with st.expander(f"❌ [{row['concept']}] {row['question_text'][:60]}"):
                    st.markdown(f"**Student answered:** `{row['student_answer']}`")
                    st.markdown(f"**Correct answer:** `{row['correct_answer']}`")
                    st.markdown(f"🕐 {row['timestamp']}")

st.divider()
st.caption("Misconception Analyzer Pipeline · Built for AI Tutor Teacher Dashboard")